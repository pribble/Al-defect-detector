# kernel/ — cmadrv 内核模块（CMA 分配器 + DMA memcpy 引擎）

> 导航：上一级 [de10_nano/](../README.md) · 相关 [ssd_detection/](../ssd_detection/README.md)

`cmadrv` 是运行在 DE10-Nano（Cyclone V SoC）HPS 侧 Linux 中的字符设备驱动，为
FPGA 加速器提供两块基础能力：

1. **CMA 连续物理内存分配/释放** —— 通过 `dma_alloc_coherent()` 在 CMA 区域分配
   连续的物理内存，供 FPGA 侧 DMA 访问（FPGA 只能访问连续的物理地址）。
2. **DMA memcpy 引擎** —— 复用 HPS 的 dmaengine（`DMA_MEMCPY`）通道在内存间做
   DMA 拷贝，避免 CPU 搬运大块数据。

用户空间通过 `/dev/cmadrv0` 的 `ioctl`/`mmap` 使用，调用方是
[`ssd_detection/intelfpga.cc`](../ssd_detection/intelfpga.cc)（FPGA 推理服务）。

## 文件清单

| 文件 | 说明 |
|------|------|
| `cmadrv.c` | 模块主实现：字符设备、ioctl、mmap、DMA 通道管理 |
| `cmadrv.h` | 头文件：`struct cma_mblk_s` / `struct cma_mcpy_s`、ioctl 命令定义 |
| `Makefile` | 内核模块构建（`KSRC_DIR` 交叉编译内核树） |
| `build.sh` | 一键编译，并把 `cmadrv.ko` 拷贝到 `../ssd_detection/deploy/` |
| `load.sh` | 加载模块 + 创建 `/dev/cmadrv0` 设备节点 |
| `unload.sh` | 卸载模块 + 删除设备节点 |

## 构建

前置条件：

- 交叉编译好的内核源码树，路径配置在 `Makefile` 的 `KSRC_DIR`（默认
  `/opt/software/linux-4.9.78`，即内核 4.9.78，对应 DE10-Nano 官方 BSP）。
- Linaro 交叉编译器
  `gcc-linaro-5.4.1-2017.05-x86_64_arm-linux-gnueabihf`（`build.sh` 会 export 到 `PATH`）。

在 x86_64 主机上执行：

```bash
cd de10_nano/kernel
bash build.sh
```

`build.sh` 依次执行：`make all`（交叉编译 `cmadrv.ko`）→ 把
`cmadrv.ko` 移动到 `../ssd_detection/deploy/` → `make clean`。
因此构建内核模块的同时也就完成了推理服务的部署目录更新。

如果只想手动构建（不拷贝到 deploy/）：

```bash
make all    # 生成 cmadrv.ko
make clean  # 清理构建产物
```

## 在 FPGA 板上加载

```bash
cd /opt/paddle_frame            # 推理服务部署目录（含 cmadrv.ko）
insmod cmadrv.ko                # 或使用 kernel/load.sh
# 若 /dev/cmadrv0 不存在：按 load.sh 逻辑，依据 /proc/devices 中的主设备号
# mknod /dev/cmadrv0 c <major> 0
```

卸载：

```bash
rmmod cmadrv
rm -f /dev/cmadrv0              # unload.sh 会一并处理
```

> **已知问题**：CMA 分配失败时（`dma_alloc_coherent` 返回 NULL 导致推理报错），
> 重新加载模块通常能恢复：`rmmod cmadrv && insmod cmadrv.ko`。

## 设备接口（对 intelfpga.cc 等用户空间调用方）

| 操作 | 说明 |
|------|------|
| `open` | 记录调用进程 PID；`filp->private_data` 指向驱动状态 |
| `mmap` | 按 `vma->vm_pgoff`（物理地址）映射到用户空间，`pgprot_noncached` + `VM_IO` |
| `ioctl(CMA_CMD_MGET, struct cma_mblk_s*)` | 分配 size 字节连续物理内存，回填 `phys`（物理地址）、`virt`（内核地址）、`addr` |
| `ioctl(CMA_CMD_FREE, struct cma_mblk_s*)` | 释放 MGET 分配的块 |
| `ioctl(CMA_CMD_MCPY, struct cma_mcpy_s*)` | DMA memcpy：从 `src` 拷贝 `len` 字节到 `dst`（均为物理地址），同步等待完成 |

ioctl 命令定义在 `cmadrv.h`：

```c
#define CMA_CMD_MGET       0x00 // 分配：struct cma_mblk_s
#define CMA_CMD_FREE       0x01 // 释放：struct cma_mblk_s
#define CMA_CMD_MCPY       0x10 // DMA 拷贝：struct cma_mcpy_s
```

DMA 通道在模块加载时请求（`DMA_MEMCPY` 能力），拷贝完成后通过 completion 回调
`cma_dma_callback` 通知等待方；`ioctl` 全程持有 `mutex` 互斥，避免并发搬运冲突。
