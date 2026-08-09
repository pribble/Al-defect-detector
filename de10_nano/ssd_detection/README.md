# ssd_detection/ — FPGA 推理服务（SSD MobileNet V1 + Paddle Lite + CNN 加速器）

> 导航：上一级 [de10_nano/](../README.md) · 相关 [kernel/](../kernel/README.md)、[C5TB/](../C5TB/README.md)

运行在 DE10-Nano（Cyclone V SoC）HPS 侧 Linux 上的 C++ 推理服务：加载量化后的
SSD MobileNet V1 模型（`.nb`），把卷积计算卸载到 FPGA 侧的 CNN 加速器
（[`C5TB/`](../C5TB/README.md)，通过 [`cmadrv`](../kernel/README.md) 驱动访问），
以 HTTP 服务（`:8080/predict`）或离线图片检测两种模式工作。

树莓派 GrabImage 把 300×300 灰度 raw 像素 POST 过来，本服务返回检测框 JSON。
坐标在 300×300 空间，由 Pi 侧 `_draw_defect_box` 缩放回 768×512 用于标注。

## 目录结构

| 路径 | 说明 |
|------|------|
| `ssd_detection.cc` | 主程序：HTTP server（`/predict`）+ 离线图片检测 |
| `intelfpga.cc` | FPGA SDK：寄存器映射（/dev/mem）、CMA 内存（/dev/cmadrv0）、权重/特征图重排、执行设备图 |
| `include/` | `intelfpga.h`（SDK 头）、`common.h`、`httplib.h`（单头文件 HTTP 库） |
| `paddle_lite/` | **精简版 Paddle-Lite**（api/core/lite/model_parser + 交叉编译产物 build/） |
| `lib/` | 构建产物：`libvnna.so`（intelfpga.cc）、`libpaddle_light_api_shared.so`、交叉编译的 `opencv/` |
| `build.sh` | 3 阶段一键交叉编译（x86_64 → armv7hf） |
| `upload.sh` | 增量同步 `deploy/` 到 FPGA 板 `172.16.68.110:/opt/paddle_frame` |
| `deploy/` | **运行目录**：二进制、模型、配置、库、测试图片 |

### deploy/ 运行目录

```
deploy/
├── ssd_detection           # 主程序（build.sh 拷贝）
├── ssd_mobilenet_v1_opt.nb # 量化模型（6.3MB，naive_buffer 格式）
├── config.txt              # 模型/标签/归一化配置
├── label_list              # 类别名 + 每类置信度阈值
├── run.sh                  # HTTP 模式启动（insmod cmadrv → ./ssd_detection config.txt）
├── test.sh                 # 离线图片目录检测（./ssd_detection config.txt data）
├── paddlelite_lib/         # libpaddle_light_api_shared.so + libvnna.so
├── cmadrv.ko               # kernel/build.sh 拷贝过来的内核驱动
├── images/  data/          # 测试输入（gitignore）
└── .last_time              # upload.sh 增量同步时间戳（gitignore）
```

## 模型与指令结构（关键！）

`ssd_mobilenet_v1_opt.nb` 是 naive_buffer 格式（meta_version + opt_version + topo
描述 + 参数），由 Paddle 官方 `opt` 工具对 MobileNet V1 + SSD 检测头量化生成。
主 block 共 **355 条指令**（见 `paddle_lite/api/paddle_api.cc`）：

```
[0]     calib       float32→int8，subgraph 输入转换
[1]     subgraph    FPGA 推理（MobileNet V1 backbone + SSD heads），内部桥接 47 个 conv/depthwise_conv 到 FPGA
[2..354]             SSD 后处理：prior_box、slice、elementwise_*、concat、softmax、multiclass_nms3
```

- **`PaddlePredictor::Run()` 循环执行全部指令**（`InferShape()` + `kernel->Launch()`），
  不要硬编码只跑 `[calib, subgraph]`——输出张量由最后的 `multiclass_nms3` 产生，
  跳过 post-processing 会让输出 dims 为空。
- **调用链**：`KernelBase::Launch()` 内部处理一次性 `PrepareForRun`、
  `ReInitWhenNeeded`、`WorkSpace::AllocReset`、`Run()`——优先通过 `Launch()`，
  不要直接调 `PrepareForRun()` + `Run()`。
- **subgraph kernel**：注册为 `kIntelFPGA / kInt8 / kNCHW`（输入输出绑
  `kHost / kInt8 / kNCHW`）。`SubgraphCompute::PrepareForRun()` 一次性做
  `BuildInstructions()` + `BuildDeviceProgram()`。
- **BuildDeviceProgram 桥接**：`SubgraphBridgeRegistry` 逐个 op 选 bridge
  （`USE_SUBGRAPH_BRIDGE(conv2d / depthwise_conv2d, kIntelFPGA)`），任一 op 桥接
  失败（`CHECK_FAILED`）则整体返回 false，`Run()` 退化为 **ARM fallback**（逐个
  指令在 CPU 上执行）。因此 `paddle_use_kernels.h` 中的 ARM conv2d /
  depthwise_conv2d kernel 必须保留。
- 桥接成功后 `ExecuteDeviceGraph()` → `intelfpga_subgraph()`（intelfpga.cc）：
  对每个 `DeviceGraphNode`（conv/dw_conv）写参数到 `uparam`、scale 到 `uscale`，
  `start_fpga()` 置位启动寄存器并轮询完成。

### 模型 I/O 名称

由 `PrepareFeedFetch()` 扫描 feed/fetch op 自动发现：

| 端口 | 名称 | 形状 | 说明 |
|------|------|------|------|
| feed 0 | `im_shape_0` | `[1,2]` | 固定 `{300, 300}`，构造时预填 |
| feed 1 | `image_0` | `[1,3,300,300]` | 每帧更新 |
| feed 2 | `scale_0` | `[1,2]` | 恒 `{1, 1}`（输入固定 300×300），构造时写死 |
| fetch 0 | `save_infer_model/scale_0.tmp_1` | `[N,6]` | 由 multiclass_nms3 产生；每行 `[class_id, confidence, x_min, y_min, x_max, y_max]` |

## 推理管线

HTTP 模式（`POST /predict`，multipart 字段 `image_file`）：

```
Pi: 768×512 gray → cv2.resize(300×300) → 90000 字节 raw → POST
FPGA: Mat(300×300, CV_8UC1) → preprocessImgGray（NEON 1ch→3ch NCHW，减 mean 127.5、乘 1/std）
    → 填 input[1]（input[0] im_shape、input[2] scale 构造时写死）→ PaddlePredictor::Run() → 解析 [N,6] 输出 → JSON
```

- `preprocessImgGray`：`CV_8UC1 → CV_32FC1`，NEON `vld1q/vmulq/vsubq` 一次算 4 像素，
  扩成 3 通道 NCHW（无 cvtColor、无冗余通道拷贝）。
- 类别阈值：`label_list` 每行 `<class_name> [threshold]`（缺省 0.45），在
  `Detector` 构造时解析；输出框按 `threshold < score` 过滤并裁剪到原图范围。
- 响应 JSON：

```json
{"len": 1, "action": "NG", "result": [
  {"class_name": "ca_shang", "loc": [x1,y1,x2,y2], "score": 0.95, "prediction_time": 45.2}]}
```

  空检测：`{"len": 0, "action": "OK", "result": []}`；非法图片大小返回
  `{"len":0,"action":"OK","result":[]}`。服务端 `ThreadPool(1)` 串行处理请求。

## 配置

`deploy/config.txt`：

```
model_file ssd_mobilenet_v1_opt.nb
label_path ./label_list
mean 127.5,127.5,127.5
std 127.502231,127.502231,127.502231
```

`deploy/label_list`（每类独立阈值）：

```
ca_shang 0.40
zang_wu 0.40
zhe_zhou 0.60
zhen_kong 0.45
```

## 构建（x86_64 主机 → armv7hf）

```bash
cd de10_nano/ssd_detection && bash build.sh
```

`build.sh` 三段式（依赖 `/opt/software/` 下的 cmake-3.10.3 与
gcc-linaro-5.4.1-2017.05，脚本内 export PATH）：

1. **[1/3] intelfpga_sdk → `lib/libvnna.so`**：`arm-linux-gnueabihf-g++ -shared intelfpga.cc`
   （`-DARCH_ABI_ARM32`、`-march=armv7-a -mfloat-abi=hard -mfpu=neon`）。
2. **[2/3] Paddle-Lite**：`paddle_lite/build_paddlelite.sh` 增量编译
   （cmake configure 一次 + `make publish_inference`；产物拷到 `lib/` 与
   `deploy/paddlelite_lib/`）。首次 configure 会把 `build/api/paddle_use_kernels.h`
   替换成 **25 条最小 kernel 集**（subgraph/feed/fetch/prior_box/multiclass_nms/
   reshape/calib/conv2d/depthwise_conv2d/scale/slice/softmax/transpose/layout 等，
   原本自动生成约 120 条）——SSD MobileNet FPGA 方案只需这些；新增算子要在此补充
   `USE_LITE_KERNEL(...)`。
3. **[3/3] ssd_detection**：`cmake`（`CMakeLists.txt` 交叉编译，链接
   `paddle_light_api_shared` + `vnna` + OpenCV）+ `make ssd_detection` → `deploy/`。

> 只重编内核模块：`cd de10_nano/kernel && bash build.sh`（自动把 `cmadrv.ko`
> 拷入 `deploy/`）。

## 部署到 FPGA 板

```bash
cd de10_nano/ssd_detection && bash upload.sh   # 增量同步 deploy/ → 172.16.68.110:/opt/paddle_frame
```

`upload.sh` 用 ssh + tar 推送 `deploy/` 中比 `.last_time` 新的文件（排除
`data/`、`images/`），并刷新 `.last_time`。首次全量推送。

板上运行：

```bash
cd /opt/paddle_frame
./run.sh        # insmod cmadrv.ko → LD_LIBRARY_PATH=./paddlelite_lib → ./ssd_detection config.txt（HTTP :8080）
./test.sh       # 同上，但以 data 目录为参数 → 离线检测（结果写 ./result/）
```

离线模式（`detect_image_file`）：接受单张图片或目录，`imread(IMREAD_GRAYSCALE)` →
`resize(300×300, INTER_CUBIC)` → `Detect()` → 画框存 `./result/*_result.jpg`。

## 调试技巧

- 在 `PaddlePredictor::Run()`（`paddle_lite/api/paddle_api.cc`）加
  `fprintf(stderr, ...)` 可看到指令数/类型，确认 355 条是否都在执行。
- 在 `SubgraphCompute::BuildDeviceProgram()`（`paddle_lite/lite/kernels/intel_fpga/subgraph_compute.h`）
  加日志可看到哪些 op 被桥接到 FPGA。
- 输出 tensor dims 为空 = post-processing 指令被跳过（见上文指令结构）。
- **CMA 分配失败**（推理报错）：`rmmod cmadrv && insmod cmadrv.ko` 重新加载驱动。
- 桥接/FPGA 执行失败时进程会打印错误并 `exit(-1)`（`fpga_release()` 释放资源），
  查看启动日志定位是哪个环节。

## 已知约定（勿改动）

- **`Detect()` 只接受灰度图（CV_8UC1）**：非 1 通道输入直接拒绝（WARN + 返回空），
  不做自动转灰度；离线模式 `imread` 固定 `IMREAD_GRAYSCALE`，HTTP 模式 Pi 发 300×300
  灰度 raw。
- 输入固定 300×300（`#define input_shape 300`，必须是 2 的整数倍），Pi 端保证。
- `intelfpga.h`：`REOGANIZE_TYPE = REOGANIZE_ARM`（输入/输出重组在 ARM 上用 NEON
  做，而非 FPGA reorganize IP）、`INPUT_CHANNEL_TILE = 8`（ARM32）、
  `image_h = 302`、`OUTPUT_BUFF_SIZE = 150*5`——这些与 C5TB 硬件参数强绑定。
- `SDK_EMULATE 0`（intelfpga.cc）：真实 FPGA 执行，非仿真。
