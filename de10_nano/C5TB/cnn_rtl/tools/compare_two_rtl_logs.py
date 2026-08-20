#!/usr/bin/env python3
"""compare_two_rtl_logs.py — 对比两个 [LOUTD] 格式的 rtl_layers.log，
逐层报告第一个差异字节与差异数。用于：旧 rbf golden 输出 vs MAC II=1 输出。

用法：
  python3 compare_two_rtl_logs.py <golden_rtl_layers.log> <dut_rtl_layers.log>
"""
import sys

sys.path.insert(0, __import__('os').path.dirname(
    __import__('os').path.abspath(__file__)))
from compare_rtl_vs_blackbox import parse_rtl


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    golden = parse_rtl(sys.argv[1])
    dut = parse_rtl(sys.argv[2])
    print(f"golden layers={len(golden)}, dut layers={len(dut)}")
    nmis = 0
    for name, g in golden.items():
        if name not in dut:
            print(f"  {name:<20} MISSING in dut")
            nmis += 1
            continue
        d = dut[name]
        if len(g) != len(d):
            print(f"  {name:<20} size {len(g)} != {len(d)}")
            nmis += 1
            continue
        if g == d:
            print(f"  {name:<20} OK")
            continue
        first = next((i for i in range(len(g)) if g[i] != d[i]), -1)
        ndiff = sum(1 for i in range(len(g)) if g[i] != d[i])
        print(f"  {name:<20} mismatch first={first} ndiff={ndiff}/{len(g)} "
              f"gold=0x{g[first]:02x} dut=0x{d[first]:02x}")
        nmis += 1
    for name in dut:
        if name not in golden:
            print(f"  {name:<20} EXTRA in dut")
            nmis += 1
    print(f"mismatch layers: {nmis}")


if __name__ == '__main__':
    main()
