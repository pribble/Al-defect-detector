#include "core/op_lite.h"
#include <set>
#include "core/op_registry.h"

namespace paddle {
namespace lite {

bool OpLite::Attach(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  CHECK(scope != nullptr);
  scope_ = scope;
  op_info_.reset(new OpInfo(opdesc));
  return AttachImpl(*op_info(), scope);
}

bool OpLite::InferShape() {
  return InferShapeImpl();
}

std::vector<std::unique_ptr<KernelBase>> OpLite::CreateKernels(
    const std::vector<Place>& places) {
  std::vector<std::unique_ptr<KernelBase>> kernels;
  CHECK(!op_type_.empty()) << "op_type_ should be set first";

  auto pick_kernel = [&](const Place& place) {
    auto ks = KernelRegistry::Global().Create(
        op_type_, place.target, place.precision, place.layout);
    for (auto&& it : ks) {
      AttachKernel(it.get());
      kernels.emplace_back(std::move(it));
    }
  };

  std::set<Place> expanded_places(places.begin(), places.end());
  for (auto& place : places) {
    expanded_places.insert(
        Place(place.target, place.precision, DATALAYOUT(kAny)));
    expanded_places.insert(Place(place.target, PRECISION(kAny), place.layout));
    expanded_places.insert(
        Place(place.target, PRECISION(kAny), DATALAYOUT(kAny)));
  }

  for (auto place : expanded_places) {
    pick_kernel(place);
  }
  return kernels;
}

// ── AttachImpl helpers ───────────────────────────────────────
// Used by operator files to resolve tensor pointers from scope.

Tensor* OpLite::GetMutableTensor(lite::Scope* scope,
                                 const std::string& name) const {
  return scope->FindMutableTensor(name);
}

Tensor* OpLite::GetMutableVarTensor(Scope* scope, const std::string& name) {
  auto* var = scope->FindVar(name);
  CHECK(var) << "No var found for " << name;
  return var->GetMutableTensor();
}

const Tensor* OpLite::GetVarTensor(Scope* scope, const std::string& name) {
  auto* var = scope->FindVar(name);
  CHECK(var) << "No var found for " << name;
  return &var->GetTensor();
}

// ── OpInfo helpers ────────────────────────────────────────────

bool OpInfo::GetInputArgname(const std::string& value_name,
                             std::string* out) const {
  for (auto& item : inputs()) {
    auto it = std::find(item.second.begin(), item.second.end(), value_name);
    if (it != item.second.end()) {
      *out = item.first;
      return true;
    }
  }
  return false;
}

bool OpInfo::GetOutputArgname(const std::string& value_name,
                              std::string* out) const {
  for (auto& item : outputs()) {
    auto it = std::find(item.second.begin(), item.second.end(), value_name);
    if (it != item.second.end()) {
      *out = item.first;
      return true;
    }
  }
  return false;
}

bool OpInfo::GetInputIndex(const std::string& input_name, int* out) const {
  for (auto& item : inputs()) {
    auto it = std::find(item.second.begin(), item.second.end(), input_name);
    if (it != item.second.end()) {
      *out = it - item.second.begin();
      return true;
    }
  }
  return false;
}

bool OpInfo::GetOutputIndex(const std::string& output_name, int* out) const {
  for (auto& item : outputs()) {
    auto it = std::find(item.second.begin(), item.second.end(), output_name);
    if (it != item.second.end()) {
      *out = it - item.second.begin();
      return true;
    }
  }
  return false;
}

bool OpInfo::HasInputScale(const std::string& name,
                           bool is_scale_name) const {
  if (is_scale_name) return HasAttr(name);
  std::string argname;
  int index;
  if (GetInputArgname(name, &argname) && GetInputIndex(name, &index))
    return HasAttr(argname + to_string(index) + "_scale");
  return false;
}

bool OpInfo::HasOutputScale(const std::string& name,
                            bool is_scale_name) const {
  if (is_scale_name) return HasAttr(name);
  std::string argname;
  int index;
  if (GetOutputArgname(name, &argname) && GetOutputIndex(name, &index))
    return HasAttr(argname + to_string(index) + "_scale");
  return false;
}

std::vector<float> OpInfo::GetInputScale(const std::string& name,
                                         bool is_scale_name) const {
  std::string scale_name;
  if (is_scale_name) {
    scale_name = name;
  } else {
    std::string argname;
    int index;
    CHECK(GetInputArgname(name, &argname));
    CHECK(GetInputIndex(name, &index));
    scale_name = argname + to_string(index) + "_scale";
  }
  return GetAttr<std::vector<float>>(scale_name);
}

std::vector<float> OpInfo::GetOutputScale(const std::string& name,
                                          bool is_scale_name) const {
  std::string scale_name;
  if (is_scale_name) {
    scale_name = name;
  } else {
    std::string argname;
    int index;
    CHECK(GetOutputArgname(name, &argname));
    CHECK(GetOutputIndex(name, &index));
    scale_name = argname + to_string(index) + "_scale";
  }
  return GetAttr<std::vector<float>>(scale_name);
}

}  // namespace lite
}  // namespace paddle
