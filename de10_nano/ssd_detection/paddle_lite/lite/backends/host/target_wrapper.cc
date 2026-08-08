#include "core/target_wrapper.h"
#include <cstring>
#include <memory>

namespace paddle {
namespace lite {

const int MALLOC_ALIGN = 64;
const int MALLOC_EXTRA = 64;

void* TargetWrapper<TARGET(kHost)>::Malloc(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  CHECK(size);
  CHECK_GT(offset + size, size);
  size_t extra_size = sizeof(int8_t) * MALLOC_EXTRA;
  auto sum_size = offset + size;
  CHECK_GT(sum_size + extra_size, sum_size);
  char* p = static_cast<char*>(malloc(sum_size + extra_size));
  CHECK(p) << "Error occurred in TargetWrapper::Malloc period: no enough for "
              "mallocing "
           << size << " bytes.";
  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  return r;
}
void TargetWrapper<TARGET(kHost)>::Free(void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}
void TargetWrapper<TARGET(kHost)>::MemcpySync(void* dst,
                                              const void* src,
                                              size_t size,
                                              IoDirection dir) {
  if (size > 0) {
    CHECK(dst) << "Error: the destination of MemcpySync can not be nullptr.";
    CHECK(src) << "Error: the source of MemcpySync can not be nullptr.";
    memcpy(dst, src, size);
  }
}

}  // namespace lite
}  // namespace paddle
