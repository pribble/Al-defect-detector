#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "core/variable.h"

namespace paddle {
namespace lite {

// Flat name→Variable store (no parent/child hierarchy).
class Scope final {
 public:
  Scope() = default;
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() = default;

  Variable* Var(const std::string &name) {
    auto it = vars_.emplace(name, nullptr).first;
    if (!it->second) it->second.reset(new Variable);
    return it->second.get();
  }

  Variable* FindVar(const std::string &name) const {
    auto it = vars_.find(name);
    return (it != vars_.end()) ? it->second.get() : nullptr;
  }

  const Tensor* FindTensor(const std::string& name) const {
    auto* var = FindVar(name);
    if (!var) return nullptr;
    return &var->GetTensor();
  }

  Tensor* FindMutableTensor(const std::string& name) {
    auto* var = FindVar(name);
    if (!var) return nullptr;
    return var->GetMutableTensor();
  }

  std::vector<Tensor>* FindMutableTensorList(const std::string& name) {
    auto* var = FindVar(name);
    if (!var) return nullptr;
    return var->GetMutableTensorList();
  }

  const std::vector<Tensor>* FindTensorList(const std::string& name) const {
    auto* var = FindVar(name);
    if (!var) return nullptr;
    return &var->GetTensorList();
  }

 private:
  std::map<std::string, std::unique_ptr<Variable>> vars_;
};

}  // namespace lite
}  // namespace paddle
