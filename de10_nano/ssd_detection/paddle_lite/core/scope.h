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
  ~Scope();

  Variable* Var(const std::string& name);
  Variable* FindVar(const std::string& name) const;

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
