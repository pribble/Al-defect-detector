#include "core/subgraph_bridge_registry.h"
#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {

SubgraphBridgeRegistry& SubgraphBridgeRegistry::Instance() {
  static SubgraphBridgeRegistry x;
  return x;
}

void SubgraphBridgeRegistry::Insert(const std::string& op_type,
                                    const TargetType& target,
                                    const cvt_func_type& cvt_func_name) {
  int key = static_cast<int>(target);
  auto it = map_.find(key);
  if (it == map_.end()) {
    map_.insert(std::make_pair(key, std::map<std::string, cvt_func_type>()));
  }
  map_.at(key).insert(std::make_pair(op_type, cvt_func_name));
}

const cvt_func_type& SubgraphBridgeRegistry::Select(
    const std::string& op_type, const TargetType& target) const {
  int key = static_cast<int>(target);
  return map_.at(key).at(op_type);
}

bool SubgraphBridgeRegistry::Exists(const std::string& op_type,
                                    const TargetType& target) const {
  int key = static_cast<int>(target);
  bool found = map_.find(key) != map_.end();
  if (found) {
    found = map_.at(static_cast<int>(key)).find(op_type) != map_.at(key).end();
  }
  return found;
}

}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
