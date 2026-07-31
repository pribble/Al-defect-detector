#ifndef LITE_CORE_DIM_H_
#define LITE_CORE_DIM_H_

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {

// Fixed-capacity dim container — avoids heap alloc for typical 1-8 dims
class DDimLite {
 public:
  using value_type = int64_t;
  static constexpr int kMaxDims = 8;

  DDimLite() : ndim_(0) {}

  explicit DDimLite(const std::vector<value_type> &x) { ConstructFrom(x); }

  void ConstructFrom(const std::vector<value_type> &x) {
    ndim_ = std::min(static_cast<int>(x.size()), kMaxDims);
    for (int i = 0; i < ndim_; i++) data_[i] = x[i];
  }

  value_type operator[](int offset) const { return data_[offset]; }
  value_type &operator[](int offset) { return data_[offset]; }

  std::vector<int64_t> Vectorize() const {
    return std::vector<int64_t>(data_, data_ + ndim_);
  }

  size_t size() const { return ndim_; }
  bool empty() const { return ndim_ == 0; }

  value_type production() const {
    value_type prod = 1;
    for (int i = 0; i < ndim_; i++) prod *= data_[i];
    return prod;
  }

  const value_type *data() const { return data_; }
  value_type count(int start, int end) const;

  DDimLite Slice(int start, int end) const;

  DDimLite Flatten2D(int col) const {
    return DDimLite(
        std::vector<value_type>({Slice(0, col).production(),
                                 Slice(col, static_cast<int>(size())).production()}));
  }

  std::string repr() const;

  friend STL::ostream &operator<<(STL::ostream &os, const DDimLite &dims) {
    os << dims.repr();
    return os;
  }

  friend bool operator==(const DDimLite &a, const DDimLite &b) {
    if (a.ndim_ != b.ndim_) return false;
    for (int i = 0; i < a.ndim_; i++) {
      if (a.data_[i] != b.data_[i]) return false;
    }
    return true;
  }

  friend bool operator!=(const DDimLite &a, const DDimLite &b) {
    if (a.ndim_ != b.ndim_) return true;
    for (int i = 0; i < a.ndim_; i++) {
      if (a.data_[i] != b.data_[i]) return true;
    }
    return false;
  }

 private:
  int ndim_{0};
  value_type data_[kMaxDims]{};
};

using DDim = paddle::lite::DDimLite;
}  // namespace lite
}  // namespace paddle

#endif  // LITE_CORE_DIM_H_
