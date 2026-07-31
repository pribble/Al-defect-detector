#pragma once

#include <algorithm>
#include <functional>  // for multiplies
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "core/dim.h"
#include "core/memory.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {

class TensorLite;

using Tensor = lite::TensorLite;

using LoD = std::vector<std::vector<uint64_t>>;

// A light-weight tensor implementation.
class TensorLite {
 public:
  TensorLite() : buffer_(std::make_shared<Buffer>()) {}
  explicit TensorLite(std::shared_ptr<Buffer> buffer) : buffer_(buffer) {}

  // T is the data type and R is the return type
  // For OpenCL, the return type can be cl::Buffer
  // and the data type can be float/int8_t.
  // For other devices, T and R may be the same type.
  template <typename T, typename R = T>
  const R *data() const {
    return reinterpret_cast<const R *>(static_cast<char *>(buffer_->data()) +
                                       offset_);
  }

  void Resize(const DDimLite &ddim) { dims_ = ddim; }
  void Resize(const std::vector<int64_t> &x) { dims_.ConstructFrom(x); }

  const DDimLite &dims() const { return dims_; }
  int64_t numel() const { return dims_.production(); }

  const LoD &lod() const { return lod_; }
  LoD *mutable_lod() { return &lod_; }
  void set_lod(const LoD &lod) { lod_ = lod; }

  PrecisionType precision() const { return precision_; }
  void set_precision(PrecisionType precision) { precision_ = precision; }

  bool persistable() const { return persistable_; }
  void set_persistable(bool persistable) { persistable_ = persistable; }

  // T is the data type and R is the return type
  // For OpenCL, the return type can be cl::Buffer
  // and the data type can be float/int8_t.
  // For other devices, T and R may be the same type.
  template <typename T, typename R = T>
  R *mutable_data() {
    precision_ = lite_api::PrecisionTypeTrait<T>::Type();
    memory_size_ = dims_.production() * sizeof(T);
    buffer_->ResetLazy(target_, memory_size_);
    return reinterpret_cast<R *>(static_cast<char *>(buffer_->data()) +
                                 offset_);
  }

  // T is the data type and R is the return type
  // For OpenCL, the return type can be cl::Buffer
  // and the data type can be float/int8_t.
  // For other devices, T and R may be the same type.
  template <typename T, typename R = T>
  R *mutable_data(TargetType target) {
    target_ = target;
    return mutable_data<T, R>();
  }

  template <typename T, typename R = T>
  R *mutable_data(TargetType target, size_t memory_size) {
    precision_ = lite_api::PrecisionTypeTrait<T>::Type();
    memory_size_ = memory_size;
    buffer_->ResetLazy(target, memory_size_);
    target_ = target;
    return reinterpret_cast<R *>(static_cast<char *>(buffer_->data()) +
                                 offset_);
  }

  void *mutable_data(size_t memory_size);
  void *mutable_data(TargetType target, size_t memory_size);

  const void *raw_data() const {
    return static_cast<char *>(
        (static_cast<char *>(buffer_->data()) + offset_));
  }

  void *raw_data() {
    return static_cast<char *>(
        (static_cast<char *>(buffer_->data()) + offset_));
  }

  void clear() {
    buffer_->Free();
    offset_ = 0;
  }
  size_t data_size() const { return this->dims().production(); }

  size_t memory_size() const { return memory_size_; }

  size_t offset() const { return offset_; }

  bool IsInitialized() const { return buffer_->data(); }

  // Other share data to this.
  void ShareDataWith(const TensorLite &other);

  void CopyDataFrom(const TensorLite &other);

  TargetType target() const { return target_; }
  void set_target(TargetType target) { target_ = target; }

  template <typename T>
  TensorLite Slice(int64_t begin, int64_t end) const;

  friend STL::ostream &operator<<(STL::ostream &os, const TensorLite &tensor) {
    os << "Tensor:" << '\n';
    os << "dim: " << tensor.dims() << '\n';
    for (int i = 0; i < tensor.dims().production(); i++) {
      os << tensor.template data<float>()[i] << " ";
    }
    os << "\n";
    return os;
  }

 private:
  TargetType target_{TargetType::kHost};
  // precision_ and persistable_ are only used for persistable vars.
  // If your tensor wants to be saved and loaded correctly, you must
  // set values of precision_ and persistable_ after updating it.
  // If your tensor is just a temp tensor, such as activations,
  // you can ignore these two attributes.
  PrecisionType precision_{PrecisionType::kUnk};
  bool persistable_{false};

  DDimLite dims_;
  std::shared_ptr<Buffer> buffer_;
  LoD lod_;
  size_t memory_size_{};

  /// @brief Buffer may be shared with other tensors
  size_t offset_{0};
};

template <typename T>
TensorLite TensorLite::Slice(int64_t begin, int64_t end) const {
  CHECK_GE(begin, 0);
  CHECK_LE(end, dims_[0]);
  CHECK_LT(begin, end);
  if (dims_[0] == 1) {
    return *this;
  } else {
    int64_t base = numel() / dims_[0];
    TensorLite dst;
    dst.buffer_ = buffer_;
    dst.target_ = target_;
    auto dst_dims = dims_;
    dst_dims[0] = end - begin;
    dst.Resize(dst_dims);
    dst.offset_ = offset_ + static_cast<size_t>(begin * base) * sizeof(T);
    return dst;
  }
}

}  // namespace lite
}  // namespace paddle

