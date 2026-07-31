#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "core/kernel.h"
#include "core/scope.h"
#include "model_parser/cpp_desc.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {

// ── OpLite ────────────────────────────────────────────────────
// Stripped from the original Paddle-Lite OpLite:
//   removed: valid_places_, CheckShape, InferShapeWithCache,
//   shape cache (last_input_shapes_, *lod*, *ptrs_cache_),
//   kernel_ storage, Run(), AttachInput/Output, OpDescWrite overload,
//   GetTensor, SetKernel, etc.

class OpInfo;

class OpLite {
 public:
  OpLite() = default;
  explicit OpLite(const std::string& type) : op_type_(type) {}

  std::string Type() const { return op_type_; }

  // Attach params from OpDesc. Each operator subclass implements.
  bool Attach(const cpp::OpDesc& opdesc, lite::Scope* scope);
  virtual bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) = 0;

  // Infer output tensor shapes.
  virtual bool InferShapeImpl() const { return true; }
  bool InferShape();

  // Debug name (used by registry).
  virtual std::string DebugString() const = 0;

  // ── kernel creation ──────────────────────────────────────
  virtual void AttachKernel(KernelBase* kernel) = 0;
  std::vector<std::unique_ptr<KernelBase>> CreateKernels(
      const std::vector<Place>& places);

  // ── subgraph bridge metadata ─────────────────────────────
  void set_op_info(std::unique_ptr<OpInfo> info) { op_info_ = std::move(info); }
  const OpInfo* op_info() const { return op_info_.get(); }
  OpInfo* mutable_op_info() { return op_info_.get(); }
  void set_scope(Scope* scope) { scope_ = scope; }
  Scope* scope() { return scope_; }

  virtual ~OpLite() = default;

 protected:
  // Helpers used by operator AttachImpl methods.
  Tensor* GetMutableTensor(lite::Scope* scope, const std::string& name) const;
  Tensor* GetMutableVarTensor(Scope* scope, const std::string& name);
  const Tensor* GetVarTensor(Scope* scope, const std::string& name);

  // Shape cache references (still used by reshape/unsqueeze AttachImpl).
  std::vector<const Tensor*> input_tensor_ptrs_cache_{};
  std::vector<Tensor*> output_tensor_ptrs_cache_{};

 private:
  Scope* scope_{nullptr};
  std::string op_type_;
  std::unique_ptr<OpInfo> op_info_;
};

// ── OpInfo ────────────────────────────────────────────────────
// Thin OpDesc wrapper.
class OpInfo : public cpp::OpDesc {
 public:
  OpInfo(const OpInfo&) = default;
  explicit OpInfo(const cpp::OpDesc& other) : cpp::OpDesc(other) {}

  std::vector<std::string> input_names() const {
    std::vector<std::string> res;
    for (auto& param : InputArgumentNames())
      for (auto& x : Input(param)) res.push_back(x);
    return res;
  }
  std::vector<std::string> output_names() const {
    std::vector<std::string> res;
    for (auto& param : OutputArgumentNames())
      for (auto& x : Output(param)) res.push_back(x);
    return res;
  }
  std::vector<std::string> input_argnames() const { return InputArgumentNames(); }
  std::vector<std::string> output_argnames() const { return OutputArgumentNames(); }

  void UpdateAllInputs(const std::string& from, const std::string& to) {
    for (auto& item : *mutable_inputs())
      for (auto& var : item.second)
        if (var == from) var = to;
  }
  void UpdateAllOutputs(const std::string& from, const std::string& to) {
    for (auto& item : *mutable_outputs())
      for (auto& var : item.second)
        if (var == from) var = to;
  }

  bool GetInputArgname(const std::string& value_name, std::string* out) const;
  bool GetOutputArgname(const std::string& value_name, std::string* out) const;
  bool GetInputIndex(const std::string& input_name, int* out) const;
  bool GetOutputIndex(const std::string& output_name, int* out) const;

  // Scale queries (used by subgraph_op).
  bool HasInputScale(const std::string& name, bool is_scale_name = false) const;
  bool HasOutputScale(const std::string& name, bool is_scale_name = false) const;
  std::vector<float> GetInputScale(const std::string& name,
                                   bool is_scale_name = false) const;
  std::vector<float> GetOutputScale(const std::string& name,
                                    bool is_scale_name = false) const;
};

}  // namespace lite
}  // namespace paddle
