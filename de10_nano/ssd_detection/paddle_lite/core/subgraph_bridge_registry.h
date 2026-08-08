#pragma once

#include <functional>
#include <map>
#include <string>
#include "core/op_lite.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace subgraph {

const int FAILED = 1;
const int SUCCESS = 0;
const int REBUILD_WHEN_SHAPE_CHANGED = 2;
inline bool CHECK_FAILED(int status) { return status & FAILED; }
inline bool CHECK_SUCCESS(int status) { return !CHECK_FAILED(status); }
inline bool CHECK_REBUILD_WHEN_SHAPE_CHANGED(int status) {
  return status & REBUILD_WHEN_SHAPE_CHANGED;
}

using cvt_func_type =
    std::function<int(void* ctx, OpLite* op, KernelBase* kernel)>;
using cvt_map_type = std::map<int, std::map<std::string, cvt_func_type>>;
class SubgraphBridgeRegistry {
 public:
  static SubgraphBridgeRegistry& Instance();

  void Insert(const std::string& op_type,
              const TargetType& target,
              const cvt_func_type& cvt_func_name);
  const cvt_func_type& Select(const std::string& op_type,
                              const TargetType& target) const;
  bool Exists(const std::string& op_type, const TargetType& target) const;
  SubgraphBridgeRegistry() = default;

 private:
  cvt_map_type map_;
  SubgraphBridgeRegistry(const SubgraphBridgeRegistry&) = delete;
  SubgraphBridgeRegistry& operator=(const SubgraphBridgeRegistry&) = delete;
};

}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

#define STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(uniq_name, msg)         \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

#define REGISTER_SUBGRAPH_BRIDGE(op_type__, target__, cvt_func_name)      \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE_LITE(                          \
      __reg_subgraph_bridge_##op_type__##_##target__##__,                 \
      "REGISTER_SUBGRAPH_BRIDGE must be called in global namespace only " \
      "once!");                                                           \
  int __reg_subgraph_bridge_##op_type__##_##target__##_Insert() {         \
    paddle::lite::subgraph::SubgraphBridgeRegistry::Instance().Insert(    \
        #op_type__, TARGET(target__), cvt_func_name);                     \
    return 0;                                                             \
  }

#define USE_SUBGRAPH_BRIDGE(op_type__, target__, ...)                       \
  extern int __reg_subgraph_bridge_##op_type__##_##target__##_Insert();     \
  static int __reg_subgraph_bridge_##op_type__##_##target__##_Insert_return \
      UNUSED = __reg_subgraph_bridge_##op_type__##_##target__##_Insert();
