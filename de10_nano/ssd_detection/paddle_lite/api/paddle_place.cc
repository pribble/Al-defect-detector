#include "api/paddle_place.h"
#include "lite/utils/hash.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite_api {

size_t Place::hash() const {
  std::hash<int> h;
  size_t hash = h(static_cast<int>(target));
  lite::CombineHash(static_cast<int64_t>(precision), &hash);
  lite::CombineHash(static_cast<int64_t>(layout), &hash);
  lite::CombineHash(static_cast<int64_t>(device), &hash);
  return hash;
}

bool operator<(const Place& a, const Place& b) {
  if (a.target != b.target) return a.target < b.target;
  if (a.precision != b.precision) return a.precision < b.precision;
  if (a.layout != b.layout) return a.layout < b.layout;
  if (a.device != b.device) return a.device < b.device;
  return false;
}

std::string Place::DebugString() const {
  STL::stringstream os;
  os << TargetToStr(target) << "/" << PrecisionToStr(precision) << "/"
     << DataLayoutToStr(layout);
  return os.str();
}

const std::string& ActivationTypeToStr(ActivationType act) {
  static const std::string act2string[] = {"unk",
                                           "Relu",
                                           "Relu6",
                                           "PRelu",
                                           "LeakyRelu",
                                           "Sigmoid",
                                           "Tanh",
                                           "Swish",
                                           "Exp",
                                           "Abs",
                                           "HardSwish",
                                           "Reciprocal",
                                           "ThresholdedRelu",
                                           "Elu",
                                           "HardSigmoid",
                                           "log"};
  auto x = static_cast<int>(act);
  CHECK_LT(x, static_cast<int>(ActivationType::NUM));
  return act2string[x];
}

const std::string& TargetToStr(TargetType target) {
  static const std::string target2string[] = {"unk",
                                              "host",
                                              "x86",
                                              "cuda",
                                              "arm",
                                              "opencl",
                                              "any",
                                              "fpga",
                                              "npu",
                                              "xpu",
                                              "bm",
                                              "mlu",
                                              "rknpu",
                                              "apu",
                                              "huawei_ascend_npu",
                                              "imagination_nna",
                                              "intel_fpga",
                                              "metal",
                                              "nnadapter"};
  auto x = static_cast<int>(target);
  CHECK_LT(x, static_cast<int>(TARGET(NUM)));
  return target2string[x];
}

const std::string& PrecisionToStr(PrecisionType precision) {
  static const std::string precision2string[] = {"unk",
                                                 "float",
                                                 "int8_t",
                                                 "int32_t",
                                                 "any",
                                                 "float16",
                                                 "bool",
                                                 "int64_t",
                                                 "int16_t",
                                                 "uint8_t",
                                                 "double"};
  auto x = static_cast<int>(precision);
  CHECK_LT(x, static_cast<int>(PRECISION(NUM)));
  return precision2string[x];
}

const std::string& DataLayoutToStr(DataLayoutType layout) {
  static const std::string datalayout2string[] = {"unk",
                                                  "NCHW",
                                                  "any",
                                                  "NHWC",
                                                  "ImageDefault",
                                                  "ImageFolder",
                                                  "ImageNW",
                                                  "MetalTexture2DArray",
                                                  "MetalTexture2D"};
  auto x = static_cast<int>(layout);
  CHECK_LT(x, static_cast<int>(DATALAYOUT(NUM)));
  return datalayout2string[x];
}

const std::string& TargetRepr(TargetType target) {
  static const std::string target2string[] = {"kUnk",
                                              "kHost",
                                              "kX86",
                                              "kCUDA",
                                              "kARM",
                                              "kOpenCL",
                                              "kAny",
                                              "kFPGA",
                                              "kNPU",
                                              "kXPU",
                                              "kBM",
                                              "kMLU",
                                              "kRKNPU",
                                              "kAPU",
                                              "kHuaweiAscendNPU",
                                              "kImaginationNNA",
                                              "kIntelFPGA",
                                              "kMetal",
                                              "kNNAdapter"};
  auto x = static_cast<int>(target);
  CHECK_LT(x, static_cast<int>(TARGET(NUM)));
  return target2string[x];
}

const std::string& PrecisionRepr(PrecisionType precision) {
  static const std::string precision2string[] = {"kUnk",
                                                 "kFloat",
                                                 "kInt8",
                                                 "kInt32",
                                                 "kAny",
                                                 "kFP16",
                                                 "kBool",
                                                 "kInt64",
                                                 "kInt16",
                                                 "kUInt8",
                                                 "kFP64"};
  auto x = static_cast<int>(precision);
  CHECK_LT(x, static_cast<int>(PRECISION(NUM)));
  return precision2string[x];
}

const std::string& DataLayoutRepr(DataLayoutType layout) {
  static const std::string datalayout2string[] = {"kUnk",
                                                  "kNCHW",
                                                  "kAny",
                                                  "kNHWC",
                                                  "kImageDefault",
                                                  "kImageFolder",
                                                  "kImageNW",
                                                  "kMetalTexture2DArray",
                                                  "kMetalTexture2D"};
  auto x = static_cast<int>(layout);
  CHECK_LT(x, static_cast<int>(DATALAYOUT(NUM)));
  return datalayout2string[x];
}

}  // namespace lite_api
}  // namespace paddle
