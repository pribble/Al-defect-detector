#include "core/scope.h"

namespace paddle {
namespace lite {

Scope::~Scope() = default;

Variable *Scope::Var(const std::string &name) {
  auto it = vars_.emplace(name, nullptr).first;
  if (!it->second) it->second.reset(new Variable);
  return it->second.get();
}

Variable *Scope::FindVar(const std::string &name) const {
  auto it = vars_.find(name);
  return (it != vars_.end()) ? it->second.get() : nullptr;
}

}  // namespace lite
}  // namespace paddle
