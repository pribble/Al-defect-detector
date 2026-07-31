#pragma once

#include <stack>
#include <string>
#include <vector>
#include "api/paddle_place.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace core {

/*
 * Type representations used to represent standard types.
 */
// TODO(Superjomn) unify all the type representation across the lite framework.
enum class Type {
  UNK = -1,
  // primary typesINT32,
  INT32,
  INT64,
  FLOAT32,
  Float64,
  BOOL,
  STRING,
  // primary list type
  CHARLIST,
  // list types
  LIST,
  // enum type
  ENUM,
  FLOAT16,
  // number of types
  NUM,
};

enum class FluidType {
  // Pod Types
  BOOL = 0,
  INT16 = 1,
  INT32 = 2,
  INT64 = 3,
  FP16 = 4,
  FP32 = 5,
  FP64 = 6,
  // Tensor<size_t> is used in C++.
  SIZE_T = 19,
  UINT8 = 20,
  INT8 = 21,

  // Other types that may need additional descriptions
  LOD_TENSOR = 7,
  SELECTED_ROWS = 8,
  FEED_MINIBATCH = 9,
  FETCH_LIST = 10,
  STEP_SCOPES = 11,
  LOD_RANK_TABLE = 12,
  LOD_TENSOR_ARRAY = 13,
  PLACE_LIST = 14,
  READER = 15,
  // Any runtime decided variable type is raw
  // raw variables should manage their own allocations
  // in operators like nccl_op
  RAW = 17,
  TUPLE = 18,
};

template <typename T>
Type StdTypeToRepr() {
  return Type::UNK;
}
template <>
Type StdTypeToRepr<int32_t>();
template <>
Type StdTypeToRepr<int64_t>();
template <>
Type StdTypeToRepr<float>();
template <>
Type StdTypeToRepr<bool>();
template <>
Type StdTypeToRepr<double>();
template <>
Type StdTypeToRepr<std::vector<char>>();
template <>
Type StdTypeToRepr<std::string>();

// Factors that impact the kernel picking strategy. Multiple factors can be
// considered together by using statement like 'factor1 | factor2'
class KernelPickFactor {
 public:
  using value_type = unsigned char;
  enum class Factor : int {
    // The following factors are sorted by priority.
    TargetFirst = 1,
    PrecisionFirst = 1 << 1,
    DataLayoutFirst = 1 << 2,
    DeviceFirst = 1 << 3,
  };

  // Has any factors considered.
  bool any_factor_considered() const { return data_; }

  KernelPickFactor& ConsiderTarget();
  // Prefer a specific target, e.g. prefer CUDA kernels.
  KernelPickFactor& ConsiderPrecision();
  KernelPickFactor& ConsiderDataLayout();
  KernelPickFactor& ConsiderDevice();

  bool IsTargetConsidered() const;
  bool IsPrecisionConsidered() const;
  bool IsDataLayoutConsidered() const;
  bool IsDeviceConsidered() const;

  friend STL::ostream& operator<<(STL::ostream& os, const KernelPickFactor& k);

 private:
  unsigned char data_{};
};

struct dim2 {
  int x{};
  int y{};

  dim2(int x, int y) : x(x), y(y) {}
};

struct dim3 {
  int x{};
  int y{};
  int z{};

  dim3(int x, int y, int z) : x(x), y(y), z(z) {}
};

using byte_t = uint8_t;

}  // namespace core
}  // namespace lite
}  // namespace paddle
