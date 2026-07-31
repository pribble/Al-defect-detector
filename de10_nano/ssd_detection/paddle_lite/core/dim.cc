#include "core/dim.h"
#include <string>

namespace paddle {
namespace lite {
using value_type = int64_t;

value_type DDimLite::count(int start, int end) const {
  start = std::max(start, 0);
  end = std::min(end, static_cast<int>(ndim_));
  if (end < start) return 0;
  value_type sum = 1;
  for (int i = start; i < end; ++i) sum *= data_[i];
  return sum;
}

DDimLite DDimLite::Slice(int start, int end) const {
  start = std::max(start, 0);
  end = std::min(end, ndim_);
  std::vector<value_type> new_dim(end - start);
  for (int i = start; i < end; i++) new_dim[i - start] = data_[i];
  return DDim(new_dim);
}

std::string DDimLite::repr() const {
  STL::stringstream ss;
  if (empty()) { ss << "{}"; return ss.str(); }
  ss << "{";
  for (size_t i = 0; i < this->size() - 1; i++) ss << (*this)[i] << ",";
  if (!this->empty()) ss << (*this)[size() - 1];
  ss << "}";
  return ss.str();
}

}  // namespace lite
}  // namespace paddle
