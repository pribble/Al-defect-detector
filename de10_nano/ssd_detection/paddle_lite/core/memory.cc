#include "core/memory.h"

namespace paddle {
namespace lite {

void* TargetMalloc(TargetType target, size_t size) {
  void* data{nullptr};
  switch (target) {
    case TargetType::kHost:
    case TargetType::kX86:
    case TargetType::kARM:
      data = TargetWrapper<TARGET(kHost)>::Malloc(size);
      break;
    default:
      LOG(FATAL) << "Unknown supported target " << TargetToStr(target);
  }
  return data;
}

void TargetFree(TargetType target, void* data, std::string free_flag) {
  switch (target) {
    case TargetType::kHost:
    case TargetType::kX86:
    case TargetType::kARM:
      TargetWrapper<TARGET(kHost)>::Free(data);
      break;

    default:
      LOG(FATAL) << "Unknown type";
  }
}

void TargetCopy(TargetType target, void* dst, const void* src, size_t size) {
  switch (target) {
    case TargetType::kHost:
    case TargetType::kX86:
    case TargetType::kARM:
      TargetWrapper<TARGET(kHost)>::MemcpySync(
          dst, src, size, IoDirection::DtoD);
      break;

    default:
      LOG(FATAL) << "unsupported type";
  }
}

}  // namespace lite
}  // namespace paddle
