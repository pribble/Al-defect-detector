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

#include "common.h"
#include "arm_neon.h"

using namespace std;

#define SDK_EMULATE 0

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

// FPGA 调试开关：FPGA_DEBUG=1 时每层完成后打印观测寄存器快照
static int fpga_debug = -1;

// 层完成快照（观测寄存器 0x54-0x8C，复现核专用；位布局见 cnn_top_core.v）：
//   0x54 = {core_state[4:0], o_group[10:0], i_group[10:0], cmd_head, cmd_cnt}
//   0x58 = {rq_row, rq_col}  0x5C = {mac_row, mac_col}
//   0x60 = {mac_t, load_row, load_col, wf_cnt}
//   0x68-0x88 = 快照（core_done 自动锁存 lane0）：acc / v_act_l / v_rq64_l /
//               v_round_l / v_shifted / lb_q / w_q[0][0..3] / v_biased_l / v_rnd_delta
//   0x8C = {done_cnt, o_evt_cnt}
static void dbg_print_layer_snapshot(uint32_t *ip, const char *name) {
  uint32_t r54 = foo_get(ip, 0x54);
  uint32_t r58 = foo_get(ip, 0x58);
  uint32_t r5c = foo_get(ip, 0x5C);
  uint32_t r60 = foo_get(ip, 0x60);
  uint32_t r68 = foo_get(ip, 0x90);   // acc
  uint32_t r6c = foo_get(ip, 0x94);   // v_act_l[0]
  uint32_t r70 = foo_get(ip, 0x98);   // v_rq64_l[0] 低 32
  uint32_t r74 = foo_get(ip, 0x9C);   // v_round_l[0] 低 32
  uint32_t r78 = foo_get(ip, 0xA4);   // v_shifted[0] 低 32
  uint32_t r7c = foo_get(ip, 0xA8);   // lb_q 低 32
  uint32_t r80 = foo_get(ip, 0xAC);   // w_q[0][0..3]
  uint32_t r84 = foo_get(ip, 0xB0);   // v_biased_l[0]
  uint32_t r88 = foo_get(ip, 0xB4);   // v_rnd_delta[0] 低 32
  uint32_t r8c = foo_get(ip, 0xB8);   // {done_cnt, o_evt_cnt}
  printf("[DBG] %s: core_state=%d og=%d ig=%d cmd_h=%d cmd_c=%d"
         " rq=%d,%d mac=%d,%d t=%d lr=%d,%d wf=%d"
         " | acc=%08x vact=%08x vrq=%08x vrnd=%08x vshf=%08x"
         " | lb=%08x wq=%02x%02x%02x%02x vbias=%08x vdelta=%08x"
         " | done=%d oevt=%d\n",
         name ? name : "?",
         (r54 >> 27) & 0x1F, (r54 >> 16) & 0x7FF, (r54 >> 5) & 0x7FF,
         (r54 >> 3) & 0x3, r54 & 0x7,
         (r58 >> 16) & 0xFFFF, r58 & 0xFFFF,
         (r5c >> 16) & 0xFFFF, r5c & 0xFFFF,
         (r60 >> 28) & 0xF, (r60 >> 18) & 0x3FF, (r60 >> 8) & 0x3FF, r60 & 0xFF,
         r68, r6c, r70, r74, r78,
         r7c, (r80 >> 24) & 0xFF, (r80 >> 16) & 0xFF, (r80 >> 8) & 0xFF,
         r80 & 0xFF, r84, r88,
         (r8c >> 16) & 0xFFFF, r8c & 0xFFFF);
}

int start_fpga(uint32_t *ip, uint32_t start_reg_addr) {
  uint32_t status;
  status = foo_get(ip, start_reg_addr);
  status |= 0x1;
  foo_set(ip, start_reg_addr, status);
  status = foo_get(ip, start_reg_addr);
  auto waitip_start = std::chrono::steady_clock::now();
  float last_dbg_print = 0.0f;

  while (status & 1) {
    status = foo_get(ip, start_reg_addr);
    std::chrono::duration<float> wait_ip_time =
        std::chrono::steady_clock::now() - waitip_start;
    if (wait_ip_time.count() >= 5) {
      printf("wait ip fail.\n");
      fpga_release();
      exit(-1);
    }
    // 调试：等待期间每 500ms 打印一次 FPGA 状态机现场（0x44/0x48 是复现核
    // 的调试只读寄存器，非黑盒协议；正常完成时这里不会打印）
    if (wait_ip_time.count() - last_dbg_print >= 0.5f) {
      last_dbg_print = wait_ip_time.count();
      uint32_t dbg0 = foo_get(ip, 0x44);
      uint32_t dbg1 = foo_get(ip, 0x48);
      uint32_t dbg2 = foo_get(ip, 0x4C);  // lr 命令/返回计数
      uint32_t dbg3 = foo_get(ip, 0x50);  // wr 命令/返回计数
      uint32_t dbg4 = foo_get(ip, 0x54);  // core 状态/组指针
      uint32_t dbg5 = foo_get(ip, 0x58);  // requant 指针
      uint32_t dbg6 = foo_get(ip, 0x5C);  // MAC 指针
      uint32_t dbg7 = foo_get(ip, 0x60);  // DMA/权重进度
      printf("[DBG] t=%.1fs state=%x lr_p=%d wr_p=%d icb=%d ibeat=%d "
             "| wbeat=%d obeat=%d round_end=%d reset=%d i_ready=%d ow_ready=%d "
             "| lr_cmd=%d lr_rdv=%d wr_cmd=%d wr_rdv=%d "
             "| core_state=%d og=%d ig=%d rq=%d,%d mac=%d,%d t=%d lr=%d,%d wf=%d\n",
             wait_ip_time.count(),
             (dbg0 >> 28) & 0xF,   // 顶层 state
             (dbg0 >> 20) & 0xFF,  // lr_pending
             (dbg0 >> 12) & 0xFF,  // wr_pending
             (dbg0 >> 8) & 0xF,    // dma_icb（8-bit 高位被 ibeat 占用，仅低 4 位）
             (dbg0 >> 0) & 0xFFF,  // dma_ibeat[11:0]
             (dbg1 >> 12) & 0xFFFFF,  // dma_wbeat
             (dbg1 >> 0) & 0xFF,      // dma_obeat
             (dbg1 >> 11) & 1,      // lr_round_end
             (dbg1 >> 10) & 1,      // lr_round_reset
             (dbg1 >> 9) & 1,       // core_i_ready
             (dbg1 >> 8) & 1,       // core_ow_ready
             dbg2 & 0xFFFF, (dbg2 >> 16) & 0xFFFF,
             dbg3 & 0xFFFF, (dbg3 >> 16) & 0xFFFF,
             (dbg4 >> 27) & 0x1F, (dbg4 >> 16) & 0x7FF, (dbg4 >> 5) & 0x7FF,
             (dbg5 >> 16) & 0xFFFF, dbg5 & 0xFFFF,
             (dbg6 >> 16) & 0xFFFF, dbg6 & 0xFFFF,
             (dbg7 >> 28) & 0xF, (dbg7 >> 18) & 0x3FF, (dbg7 >> 8) & 0x3FF,
             dbg7 & 0xFF);
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
      if (fpga_debug)
        dbg_print_layer_snapshot(foo, node->name_.c_str());

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
