# SSD MobileNetV1 模型在 cnn_top 黑盒上的真实执行画像（设备实测）

来源：FPGA 板（172.16.68.110）`bash test.sh` 的 `[FPGA-DUMP]` 输出（paddle_lite + libvnna 带日志版本）。
用途：作为 cnn_top 复现的最终规格依据——算子集合、维度分布、行块切分、数据布局全部由实测确认。

## 1. 全图结构（360 ops）

```
[0-2]   feed ×3（image / im_shape / scale_factor）
[3]     calib fp32→int8（image → image/precision_trans0）
[4]     subgraph —— 47 个 conv/dw_conv 全部进 FPGA（下方第 2 节）
[5-40]  12 组 calib+transpose2+reshape2（SSD 12 个输出头：conv2d_69~80）
[41-58] 6 组 calib+prior_box+reshape2（6 个 feature map 的 prior box）
[59-352] SSD 后处理：slice / elementwise_sub|mul|add|div / scale / exp / stack（box decode + variance）
[353-354] concat ×2
[355]   softmax
[356]   transpose2
[357]   multiclass_nms3 → save_infer_model/scale_{0,1}.tmp_1
[358-359] fetch ×2
```

## 2. 进 FPGA 的 47 层（全部 bridge → FPGA，实测参数）

| # | type | in(C×H) | out(C×H) | k | s | pad | act | 说明 |
|---|------|---------|----------|---|---|-----|-----|------|
| 0  | CONV | 3×300 | 32×150 | 3 | 2 | 1 | relu | 首层 |
| 1  | DW   | 32×150 | 32×150 | 3 | 1 | 1 | relu | |
| 2  | CONV | 32×150 | 64×150 | 1 | 1 | 0 | relu | 1×1 |
| 3  | DW   | 64×150 | 64×75  | 3 | 2 | 1 | relu | |
| 4  | CONV | 64×75  | 128×75 | 1 | 1 | 0 | relu | |
| 5  | DW   | 128×75 | 128×75 | 3 | 1 | 1 | relu | |
| 6  | CONV | 128×75 | 128×75 | 1 | 1 | 0 | relu | |
| 7  | DW   | 128×75 | 128×38 | 3 | 2 | 1 | relu | |
| 8  | CONV | 128×38 | 256×38 | 1 | 1 | 0 | relu | |
| 9  | DW   | 256×38 | 256×38 | 3 | 1 | 1 | relu | |
| 10 | CONV | 256×38 | 256×38 | 1 | 1 | 0 | relu | |
| 11 | DW   | 256×38 | 256×19 | 3 | 2 | 1 | relu | |
| 12 | CONV | 256×19 | 512×19 | 1 | 1 | 0 | relu | |
| 13 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 14 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | |
| 15 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 16 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | |
| 17 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 18 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | |
| 19 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 20 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | |
| 21 | DW   | 512×19 | 512×19 | 3 | 1 | 1 | relu | |
| 22 | CONV | 512×19 | 512×19 | 1 | 1 | 0 | relu | 主干结束 |
| 23 | CONV | 512×19 | 16×19  | 1 | 1 | 0 | none | conv2d_69 分类头 |
| 24 | CONV | 512×19 | 20×19  | 1 | 1 | 0 | none | conv2d_70 位置头 |
| 25 | DW   | 512×19 | 512×10 | 3 | 2 | 1 | relu | |
| 26 | CONV | 512×10 | 1024×10| 1 | 1 | 0 | relu | |
| 27 | DW   | 1024×10| 1024×10| 3 | 1 | 1 | relu | |
| 28 | CONV | 1024×10| 1024×10| 1 | 1 | 0 | relu | |
| 29 | CONV | 1024×10| 24×10  | 1 | 1 | 0 | none | conv2d_71 |
| 30 | CONV | 1024×10| 30×10  | 1 | 1 | 0 | none | conv2d_72 |
| 31 | CONV | 1024×10| 256×10 | 1 | 1 | 0 | relu6 | |
| 32 | CONV | 256×10 | 512×5  | 3 | 2 | 1 | relu6 | |
| 33 | CONV | 512×5  | 24×5   | 1 | 1 | 0 | none | conv2d_73 |
| 34 | CONV | 512×5  | 30×5   | 1 | 1 | 0 | none | conv2d_74 |
| 35 | CONV | 512×5  | 128×5  | 1 | 1 | 0 | relu6 | |
| 36 | CONV | 128×5  | 256×3  | 3 | 2 | 1 | relu6 | |
| 37 | CONV | 256×3  | 24×3   | 1 | 1 | 0 | none | conv2d_75 |
| 38 | CONV | 256×3  | 30×3   | 1 | 1 | 0 | none | conv2d_76 |
| 39 | CONV | 256×3  | 128×3  | 1 | 1 | 0 | relu6 | |
| 40 | CONV | 128×3  | 256×2  | 3 | 2 | 1 | relu6 | |
| 41 | CONV | 256×2  | 24×2   | 1 | 1 | 0 | none | conv2d_77 |
| 42 | CONV | 256×2  | 30×2   | 1 | 1 | 0 | none | conv2d_78 |
| 43 | CONV | 256×2  | 64×2   | 1 | 1 | 0 | relu6 | |
| 44 | CONV | 64×2   | 128×1  | 3 | 2 | 1 | relu6 | |
| 45 | CONV | 128×1  | 16×1   | 1 | 1 | 0 | none | conv2d_79 |
| 46 | CONV | 128×1  | 20×1   | 1 | 1 | 0 | none | conv2d_80 |

**实测结论**：
- 算子集合：仅 `CONV(1)` + `DW_CONV(4)`；**无 leaky、无 dilation≠1、无 k≠1/3、无 pool**
- act 实际值：0（none，12 个分类/位置头）、1（relu，主干 29 层）、2（relu6，SSD head 6 层）——leaky(3) 从未使用
- 非 8 倍数输出通道：16/20/24/30 → `chn_block = up_round(out_c,8)` 实测为 2/3/3/4

## 3. 行块切分（intelfpga_subgraph 运行时计算，实测验证）

```
output_row_tile = min(OUTPUT_BUFF_SIZE / output_w, output_h)   # OUTPUT_BUFF_SIZE = 150*5 = 750
input_row_tile  = (output_row_tile-1)*stride + dilation*(kernel-1) + 1
output_channel_block_num = up_round(output_c, 8)
output_row_block_num     = up_round(output_h, output_row_tile)
```

| 输出尺寸 | row_tile | 例（k=3,s=1 → in_row_tile / k=3,s=2 → in_row_tile） |
|---------|----------|--------------------------------------------------|
| 150×150 | 5  | s1:7 / s2:11 |
| 75×75   | 10 | s1:12 / s2:21 |
| 38×38   | 19 | s1:21 / s2:39 |
| 19×19   | 19 | s1:21 / s2:39（受 output_h 限制） |
| 10×10   | 10 | s1:12 / s2:21 |
| 5×5     | 5  | s2:11 |
| 3×3     | 3  | s2:7 |
| 2×2     | 2  | s2:5 |
| 1×1     | 1  | s2:3 |

## 4. 数据布局（param 偏移为 64-bit 字，字节 = word×8，全部实测验证）

- 输入/输出：NHWC8 `[C/8, H, W, 8]`，字节数 = `ceil(C/8)*H*W*8`
  - 例：层 0 输入 3×300×300 = 720000 B；输出 32×150×150 = 720000 B（out_off 90000 word = 720000 B ✓）
  - 层间输出/输入缓冲**线性复用**（层 i out_off = 层 i-1 out_off + 层 i-1 输出字节/8）
- 权重：`[Co/8, Ci/8, 8, K², 8]`（K=1 或 3），字节数 = `ceil(Co/8)*ceil(Ci/8)*8*K²*8`
  - 例：层 0（3→32,k3）= 1*4*8*9*8 = 2304 B ✓；层 2（32→64,k1）= 8*4*8*1*8 = 2048 B ✓
  - 总权重 ≈ 5.9 MB（word 偏移 0 ~ 735104）
- scale 块：每输出通道 2 个 float（`ws` + `bias/os`），共 `2*out_c` word

## 5. 性能基线（黑盒实测，一帧 300×300）

```
input_organize_time: 64.3 ms（ARM 重排 NCHW→NHWC8）
fpga_time:          475.1 ms（47 层，平均 ~10.1 ms/层）
output_organize_time: 23.8 ms
```

## 6. 对 cnn_top 复现（阶段 2+）的约束

1. **硬件必须支持 k=1 与 k=3**（kernel 参数运行时给定，1×1 占 29/47 层）
2. **行块循环是硬件行为**：row_block（行方向）+ chn_block（通道方向 8 一组）双层循环，由参数驱动
3. act 需支持 0/1/2（relu6 clamp 边界即 `6·os/(ws·is)` 的 raw 域量化值）
4. depthwise = 对角权重（`ti==m` 才非零），复用同一数据通路（阶段 4 实现）
5. 无跨层融合，逐层启动；输入输出/权重偏移全由 param 提供
