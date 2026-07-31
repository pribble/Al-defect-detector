#include "core/kernel.h"
#include <cstdlib>
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

void KernelBase::ParseKernelType(const std::string &kernel_type,
                                 std::string *op_type,
                                 std::string *alias,
                                 Place *place) {
  auto parts = lite::SplitView(kernel_type, '/');
  CHECK_EQ(parts.size(), 5u);

  *op_type = parts[0];
  *alias = parts[1];

  const auto &target = parts[2];
  const auto &precision = parts[3];
  const auto &layout = parts[4];

  place->target = static_cast<TargetType>(target.to_digit<int>());
  place->precision = static_cast<PrecisionType>(precision.to_digit<int>());
  place->layout = static_cast<DataLayoutType>(layout.to_digit<int>());
}

std::string KernelBase::SerializeKernelType(const std::string &op_type,
                                            const std::string &alias,
                                            const Place &place) {
  STL::stringstream ss;
  ss << op_type << "/";
  ss << alias << "/";
  // We serialize the place value not the string representation here for
  // easier deserialization.
  ss << static_cast<int>(place.target) << "/";
  ss << static_cast<int>(place.precision) << "/";
  ss << static_cast<int>(place.layout);
  return ss.str();
}

bool ParamTypeRegistry::KeyCmp::operator()(
    const ParamTypeRegistry::key_t &a,
    const ParamTypeRegistry::key_t &b) const {
  return a.hash() < b.hash();
}

STL::ostream &operator<<(STL::ostream &os,
                         const ParamTypeRegistry::KernelIdTy &other) {
  std::string io_s = other.io == ParamTypeRegistry::IO::kInput ? "in" : "out";
  os << other.kernel_type << ":" << other.arg_name << ":" << io_s << ":"
     << other.place.DebugString();
  return os;
}

}  // namespace lite
}  // namespace paddle
