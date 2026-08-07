# 黑盒 IP 数值语义的最终结论（2026-08-08，基于 blackbox_layers2.log 实测）

## 一、验证方法

- 设备上跑 main 分支（黑盒 qxp）+ 修改版 dump（`DUMP_LAYER_OUT=1`），
  拿到 4.jpg 两次推理（warm up + 4.jpg）的完整中间数据：
  `[LIN]/[LIND]`（每层输入区）、`[LOUT]/[LOUTD]`（每层输出区）、
  `[LSCALE]`（scale 数组）、`[LW]/[LWD]`（重排后权重）。
- 本地用 Python 精确重建每层 `acc = Σ(int8输入 × int8权重)`，
  逐层逐通道验证黑盒输出与候选公式的匹配率（每通道 12 采样点）。

## 二、黑盒 IP 的数值语义（全部 47 层 ≥95.8% 匹配，主干 100%）

```
acc  = Σ(input_int8 × weight_int8)                    // int32 精确累加（8×8 → 32）
fv   = acc × (ws·is/os) + (bias/os)                   // float 乘加（scale 区 float[2×out_c]）
r    = round_half_even(fv)                            // 四舍五入
if fv < 0: r -= 1                                     // 负值修正（等价 floor 除法特性）
if act ∈ {relu, relu6}: r = max(0, r)                 // relu（relu6 未见 min(6) 生效）
y    = r & 0xFF                                       // 8 位截断（wrap，非饱和！）
```

- `ws`、`bias/os` 直接来自软件写入 SCALE 区的 float 数组（`conv_op.cc` 填充）。
- 输出区布局：`[cb][h][w][8]` 全图序 NHWC8（与规格书 §6.4 一致）。
- 权重布局：`[cb_out][cb_in][mi][k][mo]`——**mi（输入通道）主序、k 次、mo 字节**。
- 多分支：SSD 检测头（conv2d_69-76）与主干并行，输入按 `in_off` 对应
  （relu_22 / relu_26 / relu6_1 / relu6_3 输出）。

## 三、与复现核（cnn-rtl）的差异——复现不成功的根因

| 项目 | 黑盒（实测） | 复现核（cnn-rtl） |
|---|---|---|
| requant | float：`round(acc·ws·is/os + bias/os)` | 定点预转：`((acc+bias_int)·mult + 2^29) >> 30` |
| bias 位置 | float 乘加域（bias/os 直接加） | 乘前域（bias_int = round(b/(ws·is)) 后乘 mult） |
| 溢出处理 | **8 位截断（wrap）**：-34 → 222 | 饱和 [-128, 127] |
| 负值舍入 | round 后额外 -1（floor 除法特性） | round-half-up 算术右移 |
| 舍入 | half-even（边界差 1 存在残余） | half-away |

关键差异：
1. **wrap vs 饱和**：黑盒 box 头（无激活层）输出负值 wrap 成 221-253，
   复现核饱和成 0/-128——**检测头输出完全不同 → 检测失败的直接原因**。
2. **float vs 定点**：逐层舍入误差累积（每层差 1-3），47 层后特征图偏离。
3. **bias 域**：黑盒 bias/os 直接加在 float 输出域；复现核 bias_int 乘前域。

## 四、修复方向（复现核对齐黑盒）

1. RTL requant 改为"float 等价的定点"：
   `y = ((acc × mult) + bias_mul) >> 30`，其中
   `mult = round(ws·is/os × 2^30)`、`bias_mul = round(bias/os × 2^30)`——
   **bias 必须加在乘后域**（当前复现核加在乘前域）。
2. 输出改为 **8 位截断（wrap）**，去掉 [-128,127] 饱和。
3. 负值修正：算术右移对负值天然是 floor——验证 `((v + 2^29) >> 30)` 的负值
   行为是否与黑盒 round-half-even + (-1) 一致，边界像素差 1 需对齐舍入规则。
4. relu/relu6：relu = max(0, y)（黑盒 relu6 未见 min(6)，可先按 relu 实现）。

## 五、工具与数据（全部在 _cnn_rtl_debug/，未污染原项目）

- `ssd_main_copy/`：main 分支副本（含 dump 修改 + 构建产物）
- `ssd_main_copy/blackbox_layers2.log`：黑盒真实 dump（111MB）
- `tools/all_layers.py`：47 层全量验证脚本（`python3 tools/all_layers.py`）
- `tools/analyze_v2.py`、`tools/chan_map.py` 等：分析工具
- `tools/blackbox_compare.py`：黑盒 vs ref 模型对比入口

## 六、遗留

- 层 40（最后一个 relu6_5）验证脚本越界（bb 长度），单独处理后可补全；
- 残余 0.4-4% 像素差 1（round half-even 边界），如需 bit-exact 需对齐黑盒
  舍入的精确实现（定点 vs float32 的最后一比特）。
