#include "lite/backends/intel_fpga/target_wrapper.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {

void* TargetWrapper<TARGET(kIntelFPGA)>::Malloc(size_t size) {
  return intelfpga_malloc(size);
}

void TargetWrapper<TARGET(kIntelFPGA)>::Free(void* ptr) { intelfpga_free(ptr); }

void TargetWrapper<TARGET(kIntelFPGA)>::MemcpySync(void* dst,
                                                   const void* src,
                                                   size_t size,
                                                   IoDirection dir) {
  memcpy(dst, src, size);
}

}  // namespace lite
}  // namespace paddle
