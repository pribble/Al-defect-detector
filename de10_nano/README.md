# de10_nano/ — DE10-Nano FPGA 推理平台

> 导航：上一级 [根 README](../README.md)

DE10-Nano（Cyclone V SoC `5CSEBA6U23I7`，HPS 双核 ARM Cortex-A9 + FPGA fabric）
是检测系统的**推理计算节点**：树莓派 GrabImage 把 300×300 灰度 raw 像素 POST 到
`172.16.68.110:8080/predict`，这里运行 SSD MobileNet V1 量化模型，卷积计算卸载到
FPGA 侧的自研 CNN 加速器，返回检测框 JSON。

## 三个子模块

| 模块 | 说明 | 运行位置 |
|------|------|----------|
| [ssd_detection/](ssd_detection/README.md) | C++ 推理服务：HTTP :8080 + 离线检测；Paddle Lite（精简版）+ intelfpga SDK | HPS Linux 用户态 |
| [kernel/](kernel/README.md) | `cmadrv` 内核驱动：CMA 连续物理内存分配 + DMA memcpy，字符设备 `/dev/cmadrv0` | HPS Linux 内核态 |
| [C5TB/](C5TB/README.md) | Quartus SoC 工程：FPGA fabric 侧，含自定义 CNN 加速器 IP（`cnn_top`） | FPGA 逻辑 |

### 调用关系

```
ssd_detection（用户态）
  ├── /dev/mem ──────────────► 映射 CNN 寄存器区（FPGAREG_CNN_*），启动/轮询加速器
  ├── /dev/cmadrv0 ──────────► kernel/cmadrv：分配 5 块 CMA（data/weight/param/scale/organize）
  │                             + mmap 到用户态；权重重排后写入，DMA 搬运
  └── 寄存器写物理地址 ──────► C5TB/cnn_top 加速器执行卷积（Avalon 从接口 hps2cnn_avs）
```

推理数据流：`image_0`（[1,3,300,300] int8）→ subgraph 桥接 47 个
conv/depthwise_conv 到 FPGA → 输出经 reorganize 回 NCHW → SSD 后处理
（prior_box/slice/concat/softmax/multiclass_nms3）→ `[N,6]` 检测框。

## 构建与部署流程

所有构建在 x86_64 主机上交叉编译，产物同步到 FPGA 板（root@172.16.68.110，目标目录
`/opt/paddle_frame`）：

```bash
# 1) 内核驱动（可选，独立构建；自动把 cmadrv.ko 拷入 ssd_detection/deploy/）
cd de10_nano/kernel && bash build.sh

# 2) 推理服务（3 阶段：libvnna.so → Paddle-Lite → ssd_detection 二进制）
cd de10_nano/ssd_detection && bash build.sh

# 3) 增量同步 deploy/ 到 FPGA 板
cd de10_nano/ssd_detection && bash upload.sh

# 4) 板上运行
cd /opt/paddle_frame
insmod cmadrv.ko                  # 或 ./run.sh（内含 insmod）
export LD_LIBRARY_PATH=./paddlelite_lib:$LD_LIBRARY_PATH
./ssd_detection config.txt        # HTTP 模式，监听 0.0.0.0:8080
```

- **C5TB/（硬件）变更流程独立**：Quartus 18.1 全编译 → `tools/sof_to_rbf.bat` 生成
  `soc_system.rbf` → `tools/gen_dtb.bat` 生成设备树 → 部署到 HPS 启动分区
  （详见 [C5TB/README.md](C5TB/README.md)）。
- 详细构建/调试见各子模块 README。

## .gitignore 说明

`de10_nano/.gitignore` 忽略的是**构建产物**，源码全部入库：

```
build/  *.o                     # 编译中间产物
lib/*.so                        # 预编译库：有意提交、勿误重建（build 时覆盖）
C5TB/db/  C5TB/incremental_db/  C5TB/output_files/   # Quartus 输出
C5TB/hps_isw_handoff/  C5TB/.qsys_edit/  C5TB/soc_system/  C5TB/*.qws
C5TB/c5_pin_model_dump.txt  C5TB/hps_sdram_p0_summary.csv
ssd_detection/deploy/.last_time # upload.sh 增量时间戳
ssd_detection/deploy/images/  ssd_detection/deploy/data/   # 测试输入
```

`C5TB/ip/`、`C5TB/tools/`（含生成的 `soc_system.rbf/.dtb/.dts`）与
`ssd_detection/deploy/`（含二进制与模型）均为提交内容。
