#pragma once
#include <memory>
#include "core/memory.h"
#include "core/types.h"

namespace paddle {
namespace lite {

/*
 * WorkSpace is a container that help to manage the temporary memory that are
 * shared across kernels during the serial execution.
 *
 * Due to the mobile library size limit, a complex allocator or GC algorithm is
 * not suitable here, one need to carefully manage the workspace inside a single
 * kernel.
 *
 * NOTE
 *
 * For kernel developers, one need to call the workspace as follows:
 *
 * - call `WorkSpace::Global().Alloc()` if needed to allocate some temporary
 * buffer.
 */
class WorkSpace {
 public:
  // Reset the workspace, and treat the workspace as empty.
  void AllocReset() { cursor_ = 0; }

  // Allocate a memory buffer.
  core::byte_t* Alloc(size_t size) {
    buffer_.ResetLazy(target_, cursor_ + size);
    auto* data = static_cast<core::byte_t*>(buffer_.data()) + cursor_;
    cursor_ += size;
    return data;
  }

  static WorkSpace& Global_Host() {
    static thread_local std::unique_ptr<WorkSpace> x(
        new WorkSpace(TARGET(kHost)));
    return *x;
  }

  static WorkSpace& Global_ARM() { return Global_Host(); }

 private:
  explicit WorkSpace(TargetType x) : target_(x) {}

  TargetType target_;
  Buffer buffer_;
  size_t cursor_;

  WorkSpace(const WorkSpace&) = delete;
  WorkSpace& operator=(const WorkSpace&) = delete;
};

}  // namespace lite
}  // namespace paddle
