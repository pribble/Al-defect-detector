#include "core/tensor.h"
#include <string>
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

void TensorLite::ShareDataWith(const TensorLite &other) {
  buffer_ = other.buffer_;
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  precision_ = other.precision_;
  offset_ = other.offset_;
}

void TensorLite::CopyDataFrom(const TensorLite &other) {
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  precision_ = other.precision_;
  persistable_ = other.persistable_;
  buffer_->CopyDataFrom(*other.buffer_, memory_size_);
}

void *TensorLite::mutable_data(size_t memory_size) {
  memory_size_ = memory_size;
  buffer_->ResetLazy(target_, memory_size_);
  return buffer_->data();
}

void *TensorLite::mutable_data(TargetType target, size_t memory_size) {
  target_ = target;
  return mutable_data(memory_size);
}


}  // namespace lite
}  // namespace paddle

