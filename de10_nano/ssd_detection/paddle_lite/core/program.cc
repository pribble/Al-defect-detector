#include "core/program.h"

#include <algorithm>

#include "model_parser/cpp_desc.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {

std::vector<Instruction> BuildInstructions(
    const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
    Scope* exec_scope,
    int block_idx) {
  std::vector<Instruction> instructions;
  CHECK(program_desc);
  auto block_size = program_desc->BlocksSize();
  CHECK(block_size);
  CHECK(block_idx >= 0 && block_idx < block_size)
      << "Invalid block index " << block_idx;
  auto block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
  auto op_size = block_desc->OpsSize();

  for (size_t op_idx = 0; op_idx < op_size; op_idx++) {
    auto op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    if (op_type == "feed" || op_type == "fetch") continue;

    auto op = LiteOpRegistry::Global().Create(op_type);
    CHECK(op) << "\nError: operator '" << op_type << "' is not supported.\n";

    if (op_type == "subgraph") {
      static_cast<operators::SubgraphOp*>(op.get())->SetProgramDesc(
          program_desc);
    }
    op->Attach(*op_desc, exec_scope);

    std::unique_ptr<KernelBase> kernel;
    if (op_desc->HasAttr(kKernelTypeAttr)) {
      auto kernel_type = op_desc->GetAttr<std::string>(kKernelTypeAttr);
      std::string alias;
      Place place;
      KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      auto kernels = op->CreateKernels({place});
      if (kernels.size() == 0 && place.target == TargetType::kARM) {
        place.target = TargetType::kHost;
        kernels = op->CreateKernels({place});
      }
      CHECK_GT(kernels.size(), 0);
      auto it = std::find_if(kernels.begin(),
                             kernels.end(),
                             [&](std::unique_ptr<KernelBase>& k) {
                               return k->alias() == alias;
                             });
      CHECK(it != kernels.end());
      kernel = std::move(*it);
    } else {
      std::vector<std::unique_ptr<KernelBase>> kernels;
      kernels = op->CreateKernels({Place{TARGET(kARM)}, Place{TARGET(kHost)}});
      if (kernels.size() > 0) {
        kernel = std::move(kernels.front());
      }
    }
    instructions.emplace_back(std::move(op), std::move(kernel));
  }

  // Set kernel contexts (one-time init)
  for (auto& inst : instructions) {
    KernelBase* k = inst.mutable_kernel();
    if (k) k->SetContext(ContextScheduler::Global().NewContext(k->target()));
  }

  return instructions;
}

}  // namespace lite
}  // namespace paddle
