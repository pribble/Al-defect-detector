"""
verification/generate/dump_vectors.py

Generates test vectors and fixture images for VHDL co-simulation verification.

Steps (run in order the first time, then only what changed):

  1. Save a fixture image (one-time, re-run only if you want a different image):
       python -m verification.generate.dump_vectors --save-fixture

  2. Dump binary test vectors from the fixture:
       python -m verification.generate.dump_vectors --dump-vectors

  3. Both steps at once:
       python -m verification.generate.dump_vectors --save-fixture --dump-vectors

All paths default to the latest ML run. Override with --run-id N.
Outputs:
    verification/fixtures/default/   image.png, image_raw_u8.bin, meta.json
    verification/vectors/default/    one *_in.bin, *_out.bin, ... per layer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.src.data.mnist64 import MNIST64Config, get_dataloaders
from ml.src.export.test_quantized_model import (
    QParams,
    conv2d_int_acc,
    linear_int_acc,
    quantize_input_from_normalized,
    requant_u8_from_acc,
)
from ml.src.models.alexnet64gray import AlexNet64Gray
from ml.src.utils import latest_run_id, ordered_conv_linear_modules, resolve_device


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def load_run(
    run_id: int,
    checkpoints_base: Path,
    outputs_base: Path,
    device: torch.device,
) -> Tuple[AlexNet64Gray, QParams, float]:
    """Load model weights and quantization parameters for a given run.

    Discovers all layers dynamically from the model.
    Layer names in QParams come from fpgaqparms.json and match PyTorch's
    named_modules() paths (e.g. 'features.0', 'classifier.1').

    Args:
        run_id:           ML run index.
        checkpoints_base: Root of ml/checkpoints/.
        outputs_base:     Root of ml/outputs/.
        device:           Torch device.

    Returns:
        (model, qp, s0) where s0 is the input quantisation scale.
    """
    ckpt_path = checkpoints_base / f"run{run_id}" / "best.pth"
    npz_path  = outputs_base     / f"run{run_id}" / "fpgaqparms.npz"
    meta_path = outputs_base     / f"run{run_id}" / "fpgaqparms.json"

    for p in (ckpt_path, npz_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    model = AlexNet64Gray(num_classes=10).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    qp = QParams(npz_path=npz_path, meta_path=meta_path, device=device)
    s0 = float(qp.meta["inputs"]["s0"])

    # Validate: every Conv2d and Linear in the model must have a QParams entry
    for name, _ in ordered_conv_linear_modules(model):
        if name not in qp.layer_meta:
            raise RuntimeError(
                f"Layer '{name}' found in model but missing from fpgaqparms.json. "
                f"Re-run the quantization export step."
            )

    return model, qp, s0


# ---------------------------------------------------------------------------
# Binary writers
# ---------------------------------------------------------------------------

def _write(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def _write_activation_conv(t: torch.Tensor, path: Path, dtype=np.uint8) -> None:
    """[1, C, H, W] → H×W×C raw bytes (channel-last for VHDL streaming)."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(dtype)
    _write(arr, path)


def _write_activation_fc(t: torch.Tensor, path: Path, dtype=np.uint8) -> None:
    """[1, C] → (C,) raw bytes."""
    arr = t.squeeze(0).cpu().numpy().astype(dtype)
    _write(arr, path)


def _write_weights(w: torch.Tensor, path: Path) -> None:
    _write(w.cpu().numpy().astype(np.int8), path)


def _write_i32(t: torch.Tensor, path: Path) -> None:
    _write(t.cpu().numpy().astype(np.int32), path)


def _write_requant(
    prefix: str,
    m: torch.Tensor,
    r: torch.Tensor,
    out_dir: Path,
) -> None:
    """Write requant params as JSON (human-readable) and binary (VHDL-readable)."""
    payload = {"layer": prefix, "m": m.cpu().tolist(), "r": r.cpu().tolist()}
    (out_dir / f"{prefix}_requant.json").write_text(json.dumps(payload, indent=2))
    # m as uint32, r as uint8 — directly consumable by VHDL testbench
    _write(m.cpu().numpy().astype(np.uint32), out_dir / f"{prefix}_requant_m.bin")
    _write(r.cpu().numpy().astype(np.uint8),  out_dir / f"{prefix}_requant_r.bin")


def _write_acc_sample_conv(acc: torch.Tensor, path: Path) -> None:
    """int32 accumulator at centre pixel, all output channels."""
    h, w = acc.shape[2] // 2, acc.shape[3] // 2
    _write_i32(acc[0, :, h, w].to(torch.int32), path)


def _write_acc_sample_fc(acc: torch.Tensor, path: Path) -> None:
    """int32 accumulator, full vector."""
    _write_i32(acc[0].to(torch.int32), path)


# ---------------------------------------------------------------------------
# Forward pass + vector dump
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_and_dump(
    model: AlexNet64Gray,
    qp: QParams,
    s0: float,
    fixture_dir: Path,
    vectors_dir: Path,
) -> int:
    """Run the quantized forward pass on the fixture image and write all vectors.

    Iterates the model dynamically — no hardcoded layer names.
    File prefixes are derived from PyTorch module paths (features.0 → features_0).

    Args:
        model:       Float AlexNet64Gray (architecture reference).
        qp:          QParams with int8 weights, int32 biases, m/r per layer.
        s0:          Input quantisation scale.
        fixture_dir: Source of image_raw_u8.bin.
        vectors_dir: Destination for all binary output files.

    Returns:
        Predicted class index (0–9).
    """
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Load and quantize fixture image
    img_u8 = np.frombuffer(
        (fixture_dir / "image_raw_u8.bin").read_bytes(), dtype=np.uint8
    ).reshape(1, 1, 64, 64)
    img_norm = torch.from_numpy(img_u8.astype(np.float32) / 255.0)
    x = quantize_input_from_normalized(img_norm, s0=s0)  # int8 [1,1,64,64]

    # Find the last Linear layer name so we know not to requantize it
    last_linear_prefix = None
    for name, mod in model.classifier.named_children():
        if isinstance(mod, nn.Linear):
            last_linear_prefix = f"classifier_{name}"

    def _prefix(parent: str, child_name: str) -> str:
        return f"{parent}_{child_name}"

    # ------------------------------------------------------------------
    # Feature extractor
    # ------------------------------------------------------------------
    for name, mod in model.features.named_children():
        pfx = _prefix("features", name)

        if isinstance(mod, nn.Conv2d):
            # Input: int8 for first layer, uint8 for the rest
            in_dtype = np.int8 if x.dtype == torch.int8 else np.uint8
            _write_activation_conv(x, vectors_dir / f"{pfx}_in.bin", dtype=in_dtype)

            W, B, m, r, _, _ = qp.get_layer_tensors(f"features.{name}")
            _write_weights(W, vectors_dir / f"{pfx}_weights.bin")
            if B is not None:
                _write_i32(B, vectors_dir / f"{pfx}_biases.bin")

            acc = conv2d_int_acc(x_u8_or_i8=x, w_i8=W, b_i32=B,
                                 stride=mod.stride, padding=mod.padding)
            acc = torch.clamp(acc, min=0)  # fused ReLU on accumulator
            _write_acc_sample_conv(acc, vectors_dir / f"{pfx}_acc_sample.bin")

            if m is None or r is None:
                raise RuntimeError(f"Missing requant params for features.{name}")
            _write_requant(pfx, m, r, vectors_dir)

            x = requant_u8_from_acc(acc, m=m, r=r)
            _write_activation_conv(x, vectors_dir / f"{pfx}_out.bin")

        elif isinstance(mod, nn.MaxPool2d):
            _write_activation_conv(x, vectors_dir / f"{pfx}_in.bin")
            x_pool = F.max_pool2d(
                x.to(torch.int32),
                kernel_size=mod.kernel_size,
                stride=mod.stride,
                padding=mod.padding,
                dilation=mod.dilation,
                ceil_mode=mod.ceil_mode,
            ).to(torch.uint8)
            _write_activation_conv(x_pool, vectors_dir / f"{pfx}_out.bin")
            x = x_pool

        elif isinstance(mod, nn.ReLU):
            continue  # fused into Conv accumulator step above

        else:
            raise NotImplementedError(f"Unhandled module in features: {name} {type(mod)}")

    # ------------------------------------------------------------------
    # Classifier
    # ------------------------------------------------------------------
    x = torch.flatten(x, 1)  # [1, 16384] uint8

    logits_float = None
    for name, mod in model.classifier.named_children():
        pfx = _prefix("classifier", name)

        if isinstance(mod, nn.Linear):
            _write_activation_fc(x, vectors_dir / f"{pfx}_in.bin")

            W, B, m, r, s_w, _ = qp.get_layer_tensors(f"classifier.{name}")
            _write_weights(W, vectors_dir / f"{pfx}_weights.bin")
            if B is not None:
                _write_i32(B, vectors_dir / f"{pfx}_biases.bin")

            acc = linear_int_acc(x_u8_or_i8=x, w_i8=W, b_i32=B)

            if pfx == last_linear_prefix:
                # Last FC: no ReLU, no requant — raw int32 accumulator → float logits
                _write_acc_sample_fc(acc, vectors_dir / f"{pfx}_acc.bin")
                s_x = qp.sx_for_layer(f"classifier.{name}")
                scale = (s_w.to(torch.float64) * float(s_x)).view(1, -1)
                logits_float = acc.to(torch.float64) * scale
                break

            acc = torch.clamp(acc, min=0)  # fused ReLU
            _write_acc_sample_fc(acc, vectors_dir / f"{pfx}_acc_sample.bin")

            if m is None or r is None:
                raise RuntimeError(f"Missing requant params for classifier.{name}")
            _write_requant(pfx, m, r, vectors_dir)

            x = requant_u8_from_acc(acc, m=m, r=r)
            _write_activation_fc(x, vectors_dir / f"{pfx}_out.bin")

        elif isinstance(mod, (nn.ReLU, nn.Dropout, nn.Flatten)):
            continue

        else:
            raise NotImplementedError(f"Unhandled module in classifier: {name} {type(mod)}")

    if logits_float is None:
        raise RuntimeError("Forward pass did not reach last Linear layer.")

    return int(logits_float.argmax(dim=1).item())


# ---------------------------------------------------------------------------
# Fixture saving
# ---------------------------------------------------------------------------

def save_fixture(
    model: AlexNet64Gray,
    test_loader: torch.utils.data.DataLoader,
    run_id: int,
    fixture_dir: Path,
) -> None:
    """Pick the first correctly-classified test image and save it as a fixture.

    Writes to fixture_dir/:
        image.png          — 64x64 grayscale PNG (human-readable)
        image_raw_u8.bin   — 4096 raw uint8 bytes (hardware input format)
        meta.json          — true class, predicted class, run_id, image index

    Args:
        model:       Float AlexNet64Gray, used only for finding a correct prediction.
        test_loader: MNIST64 test DataLoader (normalised, no augmentation).
        run_id:      ML run the model weights come from (recorded in meta.json).
        fixture_dir: Destination folder (created if missing).
    """
    fixture_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            for i in range(len(batch_x)):
                x = batch_x[i : i + 1]          # [1, 1, 64, 64] normalised float
                y_true = int(batch_y[i].item())

                logits = model(x)
                y_pred = int(logits.argmax(dim=1).item())

                if y_pred == y_true:
                    # Found a correctly-classified image — save it
                    img_np = x.squeeze().cpu().numpy()   # [64, 64], float in [0, 1]
                    img_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)

                    # PNG for human inspection
                    plt.imsave(
                        str(fixture_dir / "image.png"),
                        img_u8,
                        cmap="gray",
                        vmin=0,
                        vmax=255,
                    )

                    # Raw uint8 binary — 64×64 = 4096 bytes, row-major
                    img_u8.tofile(fixture_dir / "image_raw_u8.bin")

                    # Metadata
                    meta = {
                        "run_id":          run_id,
                        "true_class":      y_true,
                        "predicted_class": y_pred,
                        "correct":         True,
                        "image_shape":     [64, 64],
                        "dtype":           "uint8",
                        "layout":          "H x W, row-major",
                    }
                    (fixture_dir / "meta.json").write_text(json.dumps(meta, indent=2))

                    print(f"  Saved fixture: class={y_true}  predicted={y_pred}")
                    print(f"  -> {fixture_dir}")
                    return

    raise RuntimeError("No correctly-classified image found in test set.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate fixture images and test vectors for VHDL verification."
    )
    p.add_argument("--run-id",          type=int,   default=-1,
                   help="ML run ID (default: latest)")
    p.add_argument("--data-dir",        type=str,   default="ml/data")
    p.add_argument("--checkpoints-dir", type=str,   default="ml/checkpoints")
    p.add_argument("--outputs-dir",     type=str,   default="ml/outputs")
    p.add_argument("--fixtures-dir",    type=str,   default="verification/fixtures")
    p.add_argument("--vectors-dir",     type=str,   default="verification/vectors")
    p.add_argument("--device",          type=str,   default="",
                   help="cpu or cuda; empty = auto")
    p.add_argument("--seed",            type=int,   default=1234)
    p.add_argument("--val-ratio",       type=float, default=0.1)

    # Feature flags — only the enabled steps run
    p.add_argument("--save-fixture",  action="store_true",
                   help="Pick and save the default fixture image")
    p.add_argument("--dump-vectors",  action="store_true",
                   help="Run quantized forward pass on fixture and write binary vectors")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device           = resolve_device(args.device)
    checkpoints_base = Path(args.checkpoints_dir).expanduser().resolve()
    fixtures_base    = Path(args.fixtures_dir).expanduser().resolve()

    run_id       = args.run_id if args.run_id >= 0 else latest_run_id(checkpoints_base)
    outputs_base = Path(args.outputs_dir).expanduser().resolve()

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  dump_vectors   run{run_id}")
    print(f"  device       : {device}")
    print(f"{sep}\n")

    model, qp, s0 = load_run(
        run_id=run_id,
        checkpoints_base=checkpoints_base,
        outputs_base=outputs_base,
        device=device,
    )

    layers = [name for name, _ in ordered_conv_linear_modules(model)]
    print(f"  model   : AlexNet64Gray ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  s0      : {s0}")
    print(f"  layers  : {layers}\n")

    # ------------------------------------------------------------------
    # Feature: save fixture image  (needs test loader)
    # ------------------------------------------------------------------
    if args.save_fixture:
        cfg = MNIST64Config(
            data_dir=args.data_dir,
            batch_size=128,
            num_workers=0,
            val_ratio=args.val_ratio,
            seed=args.seed,
            normalize=True,
            augment=False,
        )
        _, _, test_loader = get_dataloaders(cfg)

        print("  [save-fixture]")
        save_fixture(
            model=model,
            test_loader=test_loader,
            run_id=run_id,
            fixture_dir=fixtures_base / "default",
        )

    # ------------------------------------------------------------------
    # Feature: dump binary vectors from fixture
    # ------------------------------------------------------------------
    if args.dump_vectors:
        fixture_dir = fixtures_base / "default"
        vectors_dir = Path(args.vectors_dir).expanduser().resolve() / "default"

        print("  [dump-vectors]")
        predicted = forward_and_dump(
            model=model,
            qp=qp,
            s0=s0,
            fixture_dir=fixture_dir,
            vectors_dir=vectors_dir,
        )

        files  = sorted(vectors_dir.iterdir())
        total  = sum(f.stat().st_size for f in files)
        print(f"\n  Files written to {vectors_dir}:\n")
        for f in files:
            print(f"    {f.name:<50} {f.stat().st_size:>9,} B")
        print(f"\n  Total: {total:,} bytes  |  predicted class: {predicted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
