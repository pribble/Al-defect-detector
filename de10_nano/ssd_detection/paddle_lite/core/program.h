#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "core/kernel.h"
#include "core/op_lite.h"
#include "core/op_registry.h"
#include "model_parser/cpp_desc.h"

namespace paddle {
namespace lite {

static const char kKernelTypeAttr[] = "__@kernel_type_attr@__";

struct Instruction {
  Instruction(const std::shared_ptr<OpLite>& op,
              std::unique_ptr<KernelBase>&& kernel)
      : op_(op), kernel_(std::move(kernel)) {}

  const OpLite* op() const { return op_.get(); }
  OpLite* mutable_op() { return op_.get(); }
  const KernelBase* kernel() const { return kernel_.get(); }
  KernelBase* mutable_kernel() { return kernel_.get(); }

 private:
  std::shared_ptr<OpLite> op_;
  std::unique_ptr<KernelBase> kernel_;
};

// Build a vector of Instruction from a ProgramDesc block.
// Used by PaddlePredictor (root block) and
// SubgraphCompute (subgraph block for ARM fallback).
std::vector<Instruction> BuildInstructions(
    const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
    Scope* scope,
    int block_idx);

}  // namespace lite
}  // namespace paddle
