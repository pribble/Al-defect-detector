#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "api/paddle_place.h"
#include "core/dim.h"
#include "core/target_wrapper.h"
#include "lite/utils/log/logging.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

// Malloc memory for a specific Target. All the targets should be an element in
// the `switch` here.
LITE_API void* TargetMalloc(TargetType target, size_t size);

// Free memory for a specific Target. All the targets should be an element in
// the `switch` here.
void LITE_API TargetFree(TargetType target,
                         void* data,
                         std::string free_flag = "");

// Copy a buffer from host to another target.
void TargetCopy(TargetType target, void* dst, const void* src, size_t size);

template <TargetType Target>
void CopySync(void* dst, const void* src, size_t size, IoDirection dir) {
  switch (Target) {
    case TARGET(kX86):
    case TARGET(kHost):
    case TARGET(kARM):
      TargetWrapper<TARGET(kHost)>::MemcpySync(
          dst, src, size, IoDirection::HtoH);
      break;
    default:
      LOG(FATAL)
          << "The copy function of this target has not been implemented yet.";
  }
}

// Memory buffer manager.
class Buffer {
 public:
  Buffer(void* data, TargetType target, size_t size)
      : space_(size), data_(data), own_data_(false), target_(target) {}

  void* data() const { return data_; }
  TargetType target() const { return target_; }
  size_t space() const { return space_; }
  bool own_data() const { return own_data_; }

  virtual void ResetLazy(TargetType target, size_t size) {
    if (target != target_ || space_ < size) {
      CHECK_EQ(own_data_, true) << "Can not reset unowned buffer.";
      Free();
      data_ = TargetMalloc(target, size);
      target_ = target;
      space_ = size;
    }
  }

  void ResizeLazy(size_t size) { ResetLazy(target_, size); }

  virtual void Free() {
    if (space_ > 0 && own_data_) {
      TargetFree(target_, data_);
    }
    data_ = nullptr;
    target_ = TargetType::kHost;
    space_ = 0;
  }

  virtual void CopyDataFrom(const Buffer& other, size_t nbytes) {
    target_ = other.target_;
    ResizeLazy(nbytes);
    TargetCopy(target_, data_, other.data_, nbytes);
  }

  virtual ~Buffer() { Free(); }

  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer(Buffer&&) = default;

 protected:
  size_t space_{0};
  void* data_{nullptr};
  bool own_data_{true};
  TargetType target_{TargetType::kHost};
};

}  // namespace lite
}  // namespace paddle
