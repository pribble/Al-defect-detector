#include "intelfpga.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>

#include "common.h"
#include "arm_neon.h"

using namespace std;

#define SDK_EMULATE 0

// ---- DUMP_LAYER_OUT：逐层 dump 输出区（NHWC8 原始布局）到 rtl_layers.log，
// 与黑盒 blackbox_layers2.log（_cnn_rtl_debug/ssd_main_copy/，47 层 × 2 次
// 推理：warm up + 4.jpg）逐层对比，定位 RTL 与黑盒的数值差异层。
// 只 dump 第 2 次推理（g_pass==2，即 warm up 后的第一张真实图 4.jpg），
// 与黑盒 log 的 4.jpg 段对齐（data/ 共 10 张图 + warm up = 11 次推理，
// 全 dump 会达数百 MB 且对比错位）（2026-08-10 诊断用）----
static FILE *g_dump_f = nullptr;
static int g_pass = 0;   // 推理序号：warm up=1，4.jpg=2，后续图片 3..11
static void dbg_dump_layer_output(const char *name, const int8_t *base,
                                  int byte_off, int nbytes) {
  if (!g_dump_f) {
    g_dump_f = fopen("rtl_layers.log", "wb");
    if (!g_dump_f) return;
  }
  fprintf(g_dump_f, "[LOUTD] %s off=%d n=%d\n",
          name ? name : "?", byte_off, nbytes);
  const uint8_t *p = (const uint8_t *)base + byte_off;
  for (int i = 0; i < nbytes; i += 32) {
    fprintf(g_dump_f, "%08x:", i);
    int jmax = (nbytes - i < 32) ? (nbytes - i) : 32;
    for (int j = 0; j < jmax; j++) fprintf(g_dump_f, " %02x", p[i + j]);
    fprintf(g_dump_f, "\n");
  }
  fflush(g_dump_f);
}

// ---- DUMP_LAYER_REF：与 DUMP_LAYER_OUT 配套，dump 每层输入/权重/定点 scale，
// 供 rtl_ref_check.py 用 numpy 重算参考输出，不依赖旧黑盒 log（新模型权重
// 变化后仍可用）。输出到 rtl_ref.log ----
static FILE *g_ref_f = nullptr;
static void dbg_dump_hex(FILE *f, const uint8_t *p, int nbytes) {
  for (int i = 0; i < nbytes; i += 32) {
    fprintf(f, "%08x:", i);
    int jmax = (nbytes - i < 32) ? (nbytes - i) : 32;
    for (int j = 0; j < jmax; j++) fprintf(f, " %02x", p[i + j]);
    fprintf(f, "\n");
  }
}
static void dbg_dump_layer_ref(const char *name,
                               const FpgaConvParam *argp) {
  if (!g_ref_f) {
    g_ref_f = fopen("rtl_ref.log", "wb");
    if (!g_ref_f) return;
  }
  const struct parameter &pm = argp->param;
  const int in_cb = (pm.in_c + 7) / 8;
  const int out_cb = (pm.output_c + 7) / 8;
  const int in_bytes = in_cb * pm.in_h * pm.in_w * 8;
  // DW（type=4）权重区只分配对角块（dw_conv2d_weight_reorganize：
  // out_cb × 1 × 8 × k² × 8），不是 conv 的 in_cb 倍
  const int w_bytes = out_cb * (pm.type == 4 ? 1 : in_cb) * 8 *
                      pm.kernel * pm.kernel * 8;
  const int scale_bytes = 4 * pm.output_c * 4;
  fprintf(g_ref_f,
          "[RINFO] %s type=%d in=%d,%d,%d out=%d,%d,%d k=%d s=%d p=%d act=%d"
          " in_off=%d w_off=%d out_off=%d\n",
          name, pm.type, pm.in_c, pm.in_h, pm.in_w, pm.output_c, pm.output_h,
          pm.output_w, pm.kernel, pm.stride, pm.in_pad, pm.relu,
          pm.input_offset, pm.weight_offset, pm.output_offset);
  fprintf(g_ref_f, "[RSCALE] %s n=%d\n", name, scale_bytes);
  dbg_dump_hex(g_ref_f, (const uint8_t *)argp->scale, scale_bytes);
  fprintf(g_ref_f, "[RIN] %s off=0 n=%d\n", name, in_bytes);
  dbg_dump_hex(g_ref_f, (const uint8_t *)udata + pm.input_offset * 8, in_bytes);
  fprintf(g_ref_f, "[RWD] %s off=0 n=%d\n", name, w_bytes);
  dbg_dump_hex(g_ref_f, (const uint8_t *)uweight + pm.weight_offset * 8,
               w_bytes);
  fflush(g_ref_f);
}

static int fpga_fd = -1;
static bool fpga_init_status = false;
static int weight_offset = 0;
static int output_offset = 0;

struct mem_cfg {
  int8_t *src;
  int8_t *dst;
  int size;
  bool valid;
};
static struct mem_cfg global_mem_cfg;

char const *op_type[] = {
    "",
    "INTELFPGA_Conv2D",
    "",
    "INTELFPGA_CALIB",
    "INTELFPGA_DW_Conv2D",
    "INTELFPGA_Pool2D_MAX",
    "INTELFPGA_Pool2D_AVG",
    "INTELFPGA_ELE_ADD",
};

int cma_alloc(int fd, struct cma_blk_s *pcb) {
  pcb->size = ROUND_UP(pcb->size, sysconf(_SC_PAGE_SIZE));
  if (ioctl(fd, CMA_IOCTL_MAKE(CMA_CMD_ALLOC), pcb)) {
    printf("CMA_CMD_ALLOC failed\n");
    return -1;
  }
  pcb->addr = mmap(0, pcb->size,
                   PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED,
                   fd, pcb->phys);
  if (pcb->addr == MAP_FAILED) {
    ioctl(fd, CMA_IOCTL_MAKE(CMA_CMD_FREE), pcb);
    return -1;
  }
  return 0;
}

int cma_free(int fd, struct cma_blk_s *pcb) {
  if (munmap(pcb->addr, pcb->size) == -1)
    return -1;
  if (ioctl(fd, CMA_IOCTL_MAKE(CMA_CMD_FREE), pcb)) {
    printf("CMEM_CMD_FREE failed\n");
    return -1;
  }
  return 0;
}

int up_round(int a, int b) {
  return (a - 1) / b + 1;
}

void intelfpga_free(void *ptr) {
  free(ptr);
}

void *intelfpga_malloc(size_t size) {
  return malloc(size);
}

void memorymap(int fd, uint32_t **addr, size_t length, off_t offset) {
  *addr = (uint32_t *)mmap(0, length, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, offset);
  if (*addr == (void *)-1) {
    printf("Can't map the memory:%lX to user space.\n", offset);
    fpga_release();
    exit(-1);
  }
}

void memoryunmap(void *addr, size_t len) {
  if (munmap(addr, len) == -1) {
    printf("memmory unmap failed\n");
  }
}

void foo_set(uint32_t *addr, int offset, uint32_t value) {
  addr[offset >> 2] = value;
}

uint32_t foo_get(uint32_t *addr, int offset) {
  return addr[offset >> 2];
}

void fpga_data_address_cmamap(int fd, struct cma_blk_s *cb, uint32_t **data_addr) {
  if (cma_alloc(fd, cb)) {
    close(fd);
    printf("cma_alloc fail!\n");
    exit(-1);
  }
  *data_addr = (uint32_t *)cb->addr;
}

void fpga_reg_address_map(int fd) {
  memorymap(fd, &foo, FPGAREG_MAP_SIZE, FPGAREG_CNN_BASE_ADDR);
}

int devmem_fd = 0;
int devcma_fd = 0;   // ARM32 平台（DE10-Nano）：CMA 设备与 5 块共享内存
cma_blk_s cb_data, cb_weight, cb_scale, cb_param, cb_org;

void fpga_release(void) {
  if (fpga_init_status) {
    fpga_init_status = false;
    printf("fpga release\n");
    cma_free(devcma_fd, &cb_data);
    cma_free(devcma_fd, &cb_weight);
    cma_free(devcma_fd, &cb_scale);
    cma_free(devcma_fd, &cb_param);
    cma_free(devcma_fd, &cb_org);
    close(devcma_fd);
    memoryunmap((uint32_t *)foo, FPGAREG_MAP_SIZE);
    close(devmem_fd);
  }
  output_offset = 0;
  weight_offset = 0;
}

int fpga_init() {
  if (fpga_init_status)
    return 0;

  devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
  fpga_reg_address_map(devmem_fd);

  devcma_fd = open("/dev/cmadrv0", O_RDWR);
  if (devcma_fd < 0) {
    printf("open drvier failed\n");
    fpga_release();
    exit(-1);
  }

  cb_data.size = FPGADATA_CNN_DATA_SIZE;
  cb_weight.size = FPGADATA_CNN_WEIGHT_SIZE;
  cb_param.size = FPGADATA_CNN_PARAM_SIZE;
  cb_scale.size = FPGADATA_CNN_SCALE_SIZE;
  cb_org.size = FPGADATA_ORGANIZE_DATA_SIZE;

  fpga_data_address_cmamap(devcma_fd, &cb_data, &udata);
  fpga_data_address_cmamap(devcma_fd, &cb_weight, &uweight);
  fpga_data_address_cmamap(devcma_fd, &cb_param, &uparam);
  fpga_data_address_cmamap(devcma_fd, &cb_scale, &uscale);
  fpga_data_address_cmamap(devcma_fd, &cb_org, &uorganize);

  printf("cb_data.phy:%x\r\n", cb_data.phys);
  printf("cb_weight.phy:%x\r\n", cb_weight.phys);
  printf("cb_param.phy:%x\r\n", cb_param.phys);
  printf("cb_scale.phy:%x\r\n", cb_scale.phys);
  printf("cb_org.phy:%x\r\n", cb_org.phys);

  foo_set(foo, FPGAREG_CNN_DDROUT, cb_data.phys);
  foo_set(foo, FPGAREG_CNN_DDRIN, cb_data.phys);
  foo_set(foo, FPGAREG_CNN_DDRW, cb_weight.phys);
  foo_set(foo, FPGAREG_CNN_PARAM, cb_param.phys);
  foo_set(foo, FPGAREG_CNN_SCALE, cb_scale.phys);

  fpga_init_status = true;
  return 0;
}

// FPGA 调试开关：FPGA_DEBUG=1 时打印每层 scale 定点参数溢出诊断（纯软件侧，
// 不读 RTL 寄存器）
static int fpga_debug = -1;

int start_fpga(uint32_t *ip, uint32_t start_reg_addr) {
  uint32_t status;
  status = foo_get(ip, start_reg_addr);
  status |= 0x1;
  foo_set(ip, start_reg_addr, status);
  status = foo_get(ip, start_reg_addr);
  auto waitip_start = std::chrono::steady_clock::now();

  while (status & 1) {
    status = foo_get(ip, start_reg_addr);
    std::chrono::duration<float> wait_ip_time =
        std::chrono::steady_clock::now() - waitip_start;
    if (wait_ip_time.count() >= 5) {
      printf("wait ip fail.\n");
      fpga_release();
      exit(-1);
    }
  }
}

void tran_8(uint8_t *gbild_, uint8_t *gbild_t_, size_t gx, size_t gy) {
  uint8_t **gbild = (uint8_t **)malloc(sizeof(char *) * gy);
  uint8_t **gbild_t = (uint8_t **)malloc(sizeof(char *) * gx);

  for (int i = 0; i < gy; i++)
    gbild[i] = gbild_ + i * gx;
  for (int i = 0; i < gx; i++)
    gbild_t[i] = gbild_t_ + i * gy;

  uint8x8x2_t reg882_0, reg882_1, reg882_2, reg882_3;
  uint16x4x2_t reg1642_0, reg1642_1, reg1642_2, reg1642_3;
  uint32x2x2_t reg3222_0, reg3222_1, reg3222_2, reg3222_3;
  int gx_r = gx % 8;
  int gy_r = gy % 8;
  int gx_l = gx - 7;
  int gy_l = gy - 7;
  int gx_k = gx - gx_r;
  int gy_k = gy - gy_r;
  int x, y;

  for (y = 0; y < gy_l; y += 8) {
    for (x = 0; x < gx_l; x += 8) {
      reg882_0.val[0] = vld1_u8(&gbild[y][x]);
      reg882_0.val[1] = vld1_u8(&gbild[y + 1][x]);
      reg882_1.val[0] = vld1_u8(&gbild[y + 2][x]);
      reg882_1.val[1] = vld1_u8(&gbild[y + 3][x]);
      reg882_2.val[0] = vld1_u8(&gbild[y + 4][x]);
      reg882_2.val[1] = vld1_u8(&gbild[y + 5][x]);
      reg882_3.val[0] = vld1_u8(&gbild[y + 6][x]);
      reg882_3.val[1] = vld1_u8(&gbild[y + 7][x]);

      reg882_0 = vtrn_u8(reg882_0.val[0], reg882_0.val[1]);
      reg882_1 = vtrn_u8(reg882_1.val[0], reg882_1.val[1]);
      reg882_2 = vtrn_u8(reg882_2.val[0], reg882_2.val[1]);
      reg882_3 = vtrn_u8(reg882_3.val[0], reg882_3.val[1]);

      reg1642_0 = vtrn_u16(vreinterpret_u16_u8(reg882_0.val[0]),
                           vreinterpret_u16_u8(reg882_1.val[0]));
      reg1642_1 = vtrn_u16(vreinterpret_u16_u8(reg882_0.val[1]),
                           vreinterpret_u16_u8(reg882_1.val[1]));
      reg1642_2 = vtrn_u16(vreinterpret_u16_u8(reg882_2.val[0]),
                           vreinterpret_u16_u8(reg882_3.val[0]));
      reg1642_3 = vtrn_u16(vreinterpret_u16_u8(reg882_2.val[1]),
                           vreinterpret_u16_u8(reg882_3.val[1]));

      reg3222_0 = vtrn_u32(vreinterpret_u32_u16(reg1642_0.val[0]),
                           vreinterpret_u32_u16(reg1642_2.val[0]));
      reg3222_1 = vtrn_u32(vreinterpret_u32_u16(reg1642_0.val[1]),
                           vreinterpret_u32_u16(reg1642_2.val[1]));
      reg3222_2 = vtrn_u32(vreinterpret_u32_u16(reg1642_1.val[0]),
                           vreinterpret_u32_u16(reg1642_3.val[0]));
      reg3222_3 = vtrn_u32(vreinterpret_u32_u16(reg1642_1.val[1]),
                           vreinterpret_u32_u16(reg1642_3.val[1]));

      reg882_0.val[0] = vreinterpret_u8_u32(reg3222_0.val[0]);
      reg882_0.val[1] = vreinterpret_u8_u32(reg3222_0.val[1]);
      reg882_1.val[0] = vreinterpret_u8_u32(reg3222_1.val[0]);
      reg882_1.val[1] = vreinterpret_u8_u32(reg3222_1.val[1]);
      reg882_2.val[0] = vreinterpret_u8_u32(reg3222_2.val[0]);
      reg882_2.val[1] = vreinterpret_u8_u32(reg3222_2.val[1]);
      reg882_3.val[0] = vreinterpret_u8_u32(reg3222_3.val[0]);
      reg882_3.val[1] = vreinterpret_u8_u32(reg3222_3.val[1]);

      vst1_u8(&gbild_t[x][y], reg882_0.val[0]);
      vst1_u8(&gbild_t[x + 1][y], reg882_2.val[0]);
      vst1_u8(&gbild_t[x + 2][y], reg882_1.val[0]);
      vst1_u8(&gbild_t[x + 3][y], reg882_3.val[0]);
      vst1_u8(&gbild_t[x + 4][y], reg882_0.val[1]);
      vst1_u8(&gbild_t[x + 5][y], reg882_2.val[1]);
      vst1_u8(&gbild_t[x + 6][y], reg882_1.val[1]);
      vst1_u8(&gbild_t[x + 7][y], reg882_3.val[1]);
    }
  }

  for (y = gy_k; y < gy; y++)
    for (x = 0; x < gx; x++)
      gbild_t[x][y] = gbild[y][x];
  for (x = gx_k; x < gx; x++)
    for (y = 0; y < gy_k; y++)
      gbild_t[x][y] = gbild[y][x];

  free(gbild);
  free(gbild_t);
}

void InputRearrange(int8_t *din, int8_t *dout,
                    const int c, const int h, const int w, const int pad) {
  int high = h + 2 * pad;
  int width = w + 2 * pad;
  int area = high * width;
  tran_8((uint8_t *)din, (uint8_t *)dout, area,
         up_round(c, INPUT_EXTEND_SCALE) * INPUT_EXTEND_SCALE);
  for (int cc = 0; cc < area; cc++)
    memset(dout + cc * INPUT_EXTEND_SCALE + c, 0, INPUT_EXTEND_SCALE - c);
}

void OutputRearrange(int8_t *din, int8_t *dout,
                     const int c, const int h, const int w) {
  int area = h * w;
  for (int i = 0; i < up_round(c, INPUT_EXTEND_SCALE); i++)
    tran_8((uint8_t *)din + i * area * INPUT_EXTEND_SCALE,
           (uint8_t *)dout + i * area * INPUT_EXTEND_SCALE,
           INPUT_EXTEND_SCALE, area);
}

struct device_output_config intelfpga_output_malloc(int8_t **dst,
                                                    int out_c,
                                                    int out_h,
                                                    int out_w) {
  fpga_init();
  int output_channel_block = (out_c - 1) / INPUT_EXTEND_SCALE + 1;
  int output_size = output_channel_block * INPUT_CHANNEL_TILE * out_h * out_w;
  if (*dst == nullptr)
    *dst = (int8_t *)udata + output_offset;
  struct device_output_config config;
  config.output_size = output_size;
  config.output_offset = output_offset / INPUT_CHANNEL_TILE;
  output_offset += output_size;
  return config;
}

int FpgaByte2WordOffset(int op_type, int byte_offset) {
  int offset;
  switch (op_type) {
  case INTELFPGA_Conv2D:
  case INTELFPGA_DW_Conv2D:
    offset = byte_offset / INPUT_CHANNEL_TILE;
    break;
  default:
    std::cout << "[ByteOffset2WordOffset] Unsupported op: " << op_type << "\n";
    fpga_release();
    exit(-1);
  }
  return offset;
}

int FpgaWord2ByteOffset(int op_type, int word_offset) {
  int offset;
  switch (op_type) {
  case INTELFPGA_Conv2D:
  case INTELFPGA_DW_Conv2D:
    offset = word_offset * INPUT_CHANNEL_TILE;
    break;
  default:
    std::cout << "[WordOffset2ByteOffset] Unsupported op: " << op_type << "\n";
    fpga_release();
    exit(-1);
  }
  return offset;
}

int FpgaGetOutputOffset(DeviceGraphNode *node) {
  int offset;
  if (node->op_type_ == INTELFPGA_Conv2D ||
      node->op_type_ == INTELFPGA_DW_Conv2D) {
    auto param = dynamic_cast<FpgaConvParam *>(node->node_param_);
    offset = param->param.output_offset;
  } else {
    std::cout << "[FpgaGetOutputOffset] Unsupported op: "
              << node->op_type_ << "\n";
    fpga_release();
    exit(-1);
  }
  return offset;
}

void FpgaReorganizeInput(DeviceGraphNode *node, int input_id, int layer) {
  if (node->op_type_ == INTELFPGA_Conv2D ||
      node->op_type_ == INTELFPGA_DW_Conv2D) {
    auto argp = dynamic_cast<FpgaConvParam *>(node->node_param_);
    // 输入重排（NEON 转置 → NHWC8）：写到 DDR 输入区（udata + input_offset×8）
    InputRearrange((int8_t *)argp->ia,
        (int8_t *)((int8_t *)udata + FpgaWord2ByteOffset(
            node->op_type_, argp->param.input_offset)),
        argp->param.in_c, argp->param.in_h, argp->param.in_w, 0);
  } else {
    std::cout << "[FpgaReorganizeInput] Unsupported op: "
              << node->op_type_ << "\n";
    fpga_release();
    exit(-1);
  }
}

void FpgaOutputReorganize(DeviceGraphNode *node, int layer) {
  if (node->op_type_ == INTELFPGA_Conv2D ||
      node->op_type_ == INTELFPGA_DW_Conv2D) {
    auto argp = dynamic_cast<FpgaConvParam *>(node->node_param_);
    // 输出重排（NHWC8 → NCHW）：从 DDR 输出区读回，经 uorganize 中转拷回张量
    OutputRearrange(
        (int8_t *)((int8_t *)udata + FpgaWord2ByteOffset(
            node->op_type_, argp->param.output_offset)),
        (int8_t *)uorganize,
        argp->param.output_c,
        argp->param.output_h,
        argp->param.output_w);
    global_mem_cfg.src = (int8_t *)uorganize;
    global_mem_cfg.dst = (int8_t *)argp->oa;
    global_mem_cfg.size =
        argp->param.output_c * argp->param.output_h * argp->param.output_w;
    global_mem_cfg.valid = true;
  } else {
    std::cout << "[FpgaOutputReorganize] Unsupported op: "
              << node->op_type_ << "\n";
    fpga_release();
    exit(-1);
  }
}

struct device_output_config FpgaMemMalloc(
    int op_type, int8_t *dst, int c, int h, int w) {
  auto config = device_output_config();
  switch (op_type) {
  case INTELFPGA_Conv2D:
  case INTELFPGA_DW_Conv2D:
    config = intelfpga_output_malloc(&dst, c, h, w);
    break;
  default:
    std::cout << "[DeviceMalloc] Unsupported op: " << op_type << "\n";
    fpga_release();
    exit(-1);
  }
  return config;
}

struct device_weight_config conv2d_weight_reorganize(int8_t *src,
                                                     int8_t **dst,
                                                     int out_c,
                                                     int in_c,
                                                     int kh,
                                                     int kw,
                                                     const char *filter_name) {
  fpga_init();
  int output_channel, input_channel;
  int block_of_input_channel = (in_c - 1) / INPUT_CHANNEL_TILE + 1;
  int block_of_output_channel = (out_c - 1) / OUTPUT_CHANNEL_TILE + 1;
  int kernel_size = kh * kw;
  int block_size = INPUT_CHANNEL_TILE * kernel_size * WEIGHT_EXTEND_SCALE;
  int8_t temp;
  int weight_size = block_of_output_channel * block_of_input_channel *
                    INPUT_CHANNEL_TILE * kernel_size * WEIGHT_EXTEND_SCALE;
  if (*dst == nullptr)
    *dst = (int8_t *)uweight + weight_offset;
  struct device_weight_config config;
  config.weight_size = weight_size;
  config.weight_offset = weight_offset / INPUT_CHANNEL_TILE;
  weight_offset += weight_size;
  for (int i = 0; i < block_of_output_channel; i++) {
    for (int j = 0; j < block_of_input_channel; j++) {
      for (int ti = 0; ti < INPUT_CHANNEL_TILE; ti++) {
        for (int k = 0; k < kernel_size; k++) {
          for (int m = 0; m < WEIGHT_EXTEND_SCALE; m++) {
            input_channel = j * INPUT_CHANNEL_TILE + ti;
            output_channel = i * OUTPUT_CHANNEL_TILE + m;
            if (output_channel >= out_c || input_channel >= in_c)
              temp = 0;
            else
              temp = src[(output_channel * in_c + input_channel) * kernel_size + k];
            (*dst)[(i * block_of_input_channel + j) * block_size +
                   (k + ti * kernel_size) * WEIGHT_EXTEND_SCALE + m] = temp;
          }
        }
      }
    }
  }
  return config;
}

struct device_weight_config dw_conv2d_weight_reorganize(
    int8_t *src, int8_t **dst, int out_c, int kh, int kw) {
  fpga_init();
  int block_of_output_channel = (out_c - 1) / OUTPUT_CHANNEL_TILE + 1;
  int kernel_size = kh * kw;
  int block_size = OUTPUT_CHANNEL_TILE * INPUT_CHANNEL_TILE * kernel_size;
  int weight_size = block_of_output_channel * INPUT_CHANNEL_TILE * kernel_size *
                    WEIGHT_EXTEND_SCALE;
  int8_t temp;
  if (*dst == nullptr)
    *dst = (int8_t *)uweight + weight_offset;

  struct device_weight_config config;
  config.weight_size = weight_size;
  config.weight_offset = weight_offset / INPUT_CHANNEL_TILE;
  weight_offset += weight_size;

  for (int i = 0; i < block_of_output_channel; i++) {
    for (int ti = 0; ti < INPUT_CHANNEL_TILE; ti++) {
      for (int k = 0; k < kernel_size; k++) {
        for (int m = 0; m < WEIGHT_EXTEND_SCALE; m++) {
          if (ti == m)
            temp = src[i * OUTPUT_CHANNEL_TILE * kernel_size + m * kernel_size + k];
          else
            temp = 0;
          (*dst)[i * block_size + (ti * kernel_size + k) * WEIGHT_EXTEND_SCALE + m] = temp;
        }
      }
    }
  }
  return config;
}

int intelfpga_subgraph(struct DeviceGraphNode *node) {
  fpga_init();
  if (fpga_debug < 0)
    fpga_debug = getenv("FPGA_DEBUG") ? atoi(getenv("FPGA_DEBUG")) : 0;
  if (node->is_input) g_pass++;   // 每次推理的图输入节点只出现一次
  struct timespec hw_start, hw_end;
  long long input_organize_time = 0, output_organize_time = 0;
  long long fpga_time = 0;

  static int num_node = 0;
  while (node != nullptr) {
    if (node->is_input) {
      clock_gettime(CLOCK_MONOTONIC, &hw_start);
      for (int i = 0; i < node->parent_vec_.size(); i++) {
        if (node->parent_vec_[i] == nullptr)
          FpgaReorganizeInput(node, i, num_node);
      }
      clock_gettime(CLOCK_MONOTONIC, &hw_end);
      input_organize_time +=
          (long long)(hw_end.tv_sec - hw_start.tv_sec) * 1000000000 +
          hw_end.tv_nsec - hw_start.tv_nsec;
    }
    if (node->op_type_ == INTELFPGA_Conv2D ||
        node->op_type_ == INTELFPGA_DW_Conv2D) {
      auto argp = dynamic_cast<FpgaConvParam *>(node->node_param_);

      clock_gettime(CLOCK_MONOTONIC, &hw_start);
      argp->param.output_row_tile = std::min(
          OUTPUT_BUFF_SIZE / argp->param.output_w, argp->param.output_h);
      argp->param.input_row_tile =
          (argp->param.output_row_tile - 1) * argp->param.stride +
          argp->param.dilation * (argp->param.kernel - 1) + 1;
      argp->param.output_channel_block_num =
          up_round(argp->param.output_c, OUTPUT_CHANNEL_TILE);
      argp->param.output_row_block_num =
          up_round(argp->param.output_h, argp->param.output_row_tile);

      memset(uparam, 0, sizeof(parameter));
      memcpy(uparam, (int *)(&(argp->param)), sizeof(struct parameter));
      // scale 区：每输出通道 4 个 int32（mult/bias_int/shift/rcl6），软件预转
      memcpy((int *)uscale, argp->scale,
             sizeof(int32_t) * (4 * argp->param.output_c));

      if (global_mem_cfg.valid) {
        memcpy(global_mem_cfg.dst, global_mem_cfg.src, global_mem_cfg.size);
        global_mem_cfg.valid = false;
      }
      start_fpga(foo, FPGAREG_CNN_START);
      // scale 溢出诊断（2026-08-10）：box 头 output_scale 极小，
      // mult=round(ws·is/os·2^30)、bias_mul=round(bias/os·2^22) 可能超
      // int32 回绕成负值 → 该通道 logits 系统性错误 → softmax 类别概率
      // 全偏 → 上板几百框。黑盒时代 scale 是 float 无此问题（定点化新引入）。
      if (fpga_debug && argp && argp->scale) {
        const int32_t *sc = argp->scale;
        const int oc = argp->param.output_c;
        long long min_m = 0, max_m = 0, min_b = 0, max_b = 0, min_r = 0, max_r = 0;
        for (int i = 0; i < oc; i++) {
          long long m = sc[i], b = sc[oc + i], r = sc[3 * oc + i];
          if (i == 0) {
            min_m = max_m = m;
            min_b = max_b = b;
            min_r = max_r = r;
          } else {
            if (m < min_m) min_m = m;
            if (m > max_m) max_m = m;
            if (b < min_b) min_b = b;
            if (b > max_b) max_b = b;
            if (r < min_r) min_r = r;
            if (r > max_r) max_r = r;
          }
        }
        const bool ovf =
            (max_m > 2147483647LL || min_m < -2147483648LL ||
             max_b > 2147483647LL || min_b < -2147483648LL);
        printf("[SCALE] %s out_c=%d mult[%lld,%lld] bias_mul[%lld,%lld]"
               " rcl6[%lld,%lld]%s\n",
               node->name_.c_str(), oc, min_m, max_m, min_b, max_b,
               min_r, max_r, ovf ? "  *** int32 OVERFLOW ***" : "");
      }
      // DUMP_LAYER_OUT=1：dump 本层输出区（NHWC8 原始布局，未做
      // OutputRearrange），与黑盒 blackbox_layers2.log 逐层对比。
      // 只 dump g_pass==2（4.jpg），与黑盒 log 的 4.jpg 段对齐
      if (getenv("DUMP_LAYER_OUT") && g_pass == 2) {
        const int oc = argp->param.output_c;
        const int oh = argp->param.output_h;
        const int ow = argp->param.output_w;
        const int obytes = ((oc + 7) / 8) * oh * ow * 8;
        dbg_dump_layer_output(node->name_.c_str(), (const int8_t *)udata,
                              argp->param.output_offset * 8, obytes);
        dbg_dump_layer_ref(node->name_.c_str(), argp);
      }

      clock_gettime(CLOCK_MONOTONIC, &hw_end);
      fpga_time +=
          (long long)(hw_end.tv_sec - hw_start.tv_sec) * 1000000000 +
          hw_end.tv_nsec - hw_start.tv_nsec;
    } else {
      std::cout << "Error: operator " << node->name_
                << " not supported in FPGA, exiting.\n";
      fpga_release();
      exit(-1);
    }
    if (node->is_output && !SDK_EMULATE) {
      clock_gettime(CLOCK_MONOTONIC, &hw_start);
      FpgaOutputReorganize(node, num_node);
      clock_gettime(CLOCK_MONOTONIC, &hw_end);
      output_organize_time +=
          (long long)(hw_end.tv_sec - hw_start.tv_sec) * 1000000000 +
          hw_end.tv_nsec - hw_start.tv_nsec;
    }
    node = node->next_;
    num_node++;
  }
  if (global_mem_cfg.valid) {
    memcpy(global_mem_cfg.dst, global_mem_cfg.src, global_mem_cfg.size);
    global_mem_cfg.valid = false;
  }
  printf("input_organize_time:%fms\n",
         (float)(input_organize_time / 1000000.0f));
  printf("fpga_time:%fms\n", (float)(fpga_time / 1000000.0f));
  printf("output_organize_time:%fms\n",
         (float)(output_organize_time / 1000000.0f));
  return 0;
}
