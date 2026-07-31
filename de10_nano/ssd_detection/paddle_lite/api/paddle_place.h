#pragma once
#include <set>
#include <string>

// Shared library visibility (GCC ARM cross-compile only)
#define LITE_API __attribute__((visibility("default")))

namespace paddle {
namespace lite_api {

enum class TargetType : int {
  kUnk = 0,
  kHost = 1,
  kX86 = 2,
  kCUDA = 3,
  kARM = 4,
  kOpenCL = 5,
  kAny = 6,  // any target
  kFPGA = 7,
  kNPU = 8,
  kXPU = 9,
  kBM = 10,
  kMLU = 11,
  kRKNPU = 12,
  kAPU = 13,
  kHuaweiAscendNPU = 14,
  kImaginationNNA = 15,
  kIntelFPGA = 16,
  kMetal = 17,
  kNNAdapter = 18,
  NUM = 19,  // number of fields.
};
enum class PrecisionType : int {
  kUnk = 0,
  kFloat = 1,
  kInt8 = 2,
  kInt32 = 3,
  kAny = 4,  // any precision
  kFP16 = 5,
  kBool = 6,
  kInt64 = 7,
  kInt16 = 8,
  kUInt8 = 9,
  kFP64 = 10,
  NUM = 11,  // number of fields.
};

typedef enum {
  LITE_PRECISION_NORMAL = 0,
  LITE_PRECISION_LOW = 1,
} PrecisionMode;

enum class DataLayoutType : int {
  kUnk = 0,
  kNCHW = 1,
  kNHWC = 3,
  kImageDefault = 4,  // for opencl image2d
  kImageFolder = 5,   // for opencl image2d
  kImageNW = 6,       // for opencl image2d
  kAny = 2,           // any data layout
  kMetalTexture2DArray = 7,
  kMetalTexture2D = 8,
  NUM = 9,  // number of fields.
};

typedef enum {
  LITE_POWER_HIGH = 0,
  LITE_POWER_LOW = 1,
  LITE_POWER_FULL = 2,
  LITE_POWER_NO_BIND = 3,
  LITE_POWER_RAND_HIGH = 4,
  LITE_POWER_RAND_LOW = 5
} PowerMode;

typedef enum { MLU_220 = 0, MLU_270 = 1 } MLUCoreVersion;

enum class ActivationType : int {
  kIndentity = 0,
  kRelu = 1,
  kRelu6 = 2,
  kPRelu = 3,
  kLeakyRelu = 4,
  kSigmoid = 5,
  kTanh = 6,
  kSwish = 7,
  kExp = 8,
  kAbs = 9,
  kHardSwish = 10,
  kReciprocal = 11,
  kThresholdedRelu = 12,
  kElu = 13,
  kHardSigmoid = 14,
  kLog = 15,
  kSigmoid_v2 = 16,
  kTanh_v2 = 17,
  kGelu = 18,
  kErf = 19,
  kSign = 20,
  kSoftPlus = 21,
  kMish = 22,
  NUM = 23,
};

static size_t PrecisionTypeLength(PrecisionType type) {
  switch (type) {
    case PrecisionType::kFloat:
      return 4;
    case PrecisionType::kFP64:
      return 8;
    case PrecisionType::kUInt8:
      return 1;
    case PrecisionType::kInt8:
      return 1;
    case PrecisionType::kInt32:
      return 4;
    case PrecisionType::kInt64:
      return 8;
    case PrecisionType::kFP16:
      return 2;
    case PrecisionType::kInt16:
      return 2;
    default:
      return 0;
  }
}

enum class QuantType : int {
  QUANT_INT8,
  QUANT_INT16,
};

template <typename T>
struct PrecisionTypeTrait {
  constexpr static PrecisionType Type() { return PrecisionType::kUnk; }
};

#define _ForEachPrecisionTypeHelper(callback, cpp_type, precision_type) \
  callback(cpp_type, ::paddle::lite_api::PrecisionType::precision_type);

#define _ForEachPrecisionType(callback)                   \
  _ForEachPrecisionTypeHelper(callback, bool, kBool);     \
  _ForEachPrecisionTypeHelper(callback, float, kFloat);   \
  _ForEachPrecisionTypeHelper(callback, double, kFP64);   \
  _ForEachPrecisionTypeHelper(callback, uint8_t, kUInt8); \
  _ForEachPrecisionTypeHelper(callback, int8_t, kInt8);   \
  _ForEachPrecisionTypeHelper(callback, int16_t, kInt16); \
  _ForEachPrecisionTypeHelper(callback, int, kInt32);     \
  _ForEachPrecisionTypeHelper(callback, int64_t, kInt64);

#define DefinePrecisionTypeTrait(cpp_type, precision_type)           \
  template <>                                                        \
  struct PrecisionTypeTrait<cpp_type> {                              \
    constexpr static PrecisionType Type() { return precision_type; } \
  }

_ForEachPrecisionType(DefinePrecisionTypeTrait);

#undef _ForEachPrecisionTypeHelper
#undef _ForEachPrecisionType
#undef DefinePrecisionTypeTrait

#define TARGET(item__) paddle::lite_api::TargetType::item__
#define PRECISION(item__) paddle::lite_api::PrecisionType::item__
#define DATALAYOUT(item__) paddle::lite_api::DataLayoutType::item__

const std::string& ActivationTypeToStr(ActivationType act);

const std::string& TargetToStr(TargetType target);

const std::string& PrecisionToStr(PrecisionType precision);

const std::string& DataLayoutToStr(DataLayoutType layout);

const std::string& TargetRepr(TargetType target);

const std::string& PrecisionRepr(PrecisionType precision);

const std::string& DataLayoutRepr(DataLayoutType layout);

/*
 * Place specifies the execution context of a Kernel or input/output for a
 * kernel. It is used to make the analysis of the MIR more clear and accurate.
 */
struct LITE_API Place {
  TargetType target{TARGET(kUnk)};
  PrecisionType precision{PRECISION(kUnk)};
  DataLayoutType layout{DATALAYOUT(kUnk)};
  int16_t device{0};  // device ID

  Place() = default;
  Place(TargetType target,
        PrecisionType precision = PRECISION(kFloat),
        DataLayoutType layout = DATALAYOUT(kNCHW),
        int16_t device = 0)
      : target(target), precision(precision), layout(layout), device(device) {}

  bool is_valid() const {
    return target != TARGET(kUnk) && precision != PRECISION(kUnk) &&
           layout != DATALAYOUT(kUnk);
  }

  size_t hash() const;

  bool operator==(const Place& other) const {
    return target == other.target && precision == other.precision &&
           layout == other.layout && device == other.device;
  }

  bool operator!=(const Place& other) const { return !(*this == other); }

  friend bool operator<(const Place& a, const Place& b);

  std::string DebugString() const;
};

}  // namespace lite_api
}  // namespace paddle
