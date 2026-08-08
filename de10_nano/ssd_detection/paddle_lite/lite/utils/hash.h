#pragma once
#include <functional>

namespace paddle {
namespace lite {

// A simplified implementation of boost::hash_combine.
template <typename T>
inline void CombineHash(const T& from, size_t* to) {
  std::hash<T> h;
  *to ^= h(from) + 0x9e3779b9 + (*to << 6) + (*to >> 2);
}

}  // namespace lite
}  // namespace paddle
