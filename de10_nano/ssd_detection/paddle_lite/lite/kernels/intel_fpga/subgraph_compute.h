#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <algorithm>

#include "api/paddle_place.h"
#include "core/kernel.h"
#include "core/op_lite.h"
#include "core/op_registry.h"
#include "core/program.h"
#include "core/subgraph_bridge_registry.h"
#include "core/tensor.h"
#include "core/type_system.h"
#include "core/types.h"
#include "lite/kernels/intel_fpga/graph.h"
#include "lite/utils/env.h"


namespace paddle {
namespace lite {
namespace kernels {
namespace intel_fpga {


// FPGA subgraph kernel — simplified for fixed-size SSD MobileNet V1 (300×300).
// Single device program, no input-shape-keyed map (always same input size).
class SubgraphCompute
    : public KernelLite<TARGET(kIntelFPGA), PRECISION(kInt8), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::SubgraphParam;

  void PrepareForRun() override {
    auto& param = this->Param<param_t>();
    block_idx_ = param.block_idx;
    program_desc_ = param.program_desc;
    exec_scope_ = param.exec_scope;
    input_names_ = param.input_data_names;
    output_names_ = param.output_data_names;
    std::stable_sort(input_names_.begin(), input_names_.end());
    std::stable_sort(output_names_.begin(), output_names_.end());

    // One-shot: build subgraph instructions + FPGA device graph.
    if (subgraph_instructions_.empty()) {
      subgraph_instructions_ =
          BuildInstructions(program_desc_, exec_scope_, block_idx_);
      BuildDeviceProgram();
    }
  }

  void Run() override {
    if (device_program_) {
      device_program_->ExecuteDeviceGraph();
    } else {
      // ARM fallback: run all subgraph ops on CPU.
      for (auto& inst : subgraph_instructions_) {
        inst.mutable_op()->InferShape();
        inst.mutable_kernel()->Launch();
      }
    }
  }

  virtual ~SubgraphCompute() = default;

 private:
  // --- Device program build (single program, no map) ---

  bool BuildDeviceProgram() {
    if (device_program_) return true;

    auto graph = std::make_shared<subgraph::intel_fpga::Graph>();
    graph->set_input_names(input_names_);
    graph->set_output_names(output_names_);

    const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
    for (size_t ii = 0; ii < subgraph_instructions_.size(); ii++) {
      auto& inst = subgraph_instructions_[ii];
      auto op = const_cast<OpLite*>(inst.op());
      op->InferShape();
      int status = bridges.Select(op->op_info()->Type(), TARGET(kIntelFPGA))(
          reinterpret_cast<void*>(graph.get()),
          const_cast<OpLite*>(op),
          const_cast<KernelBase*>(inst.kernel()));
      if (subgraph::CHECK_FAILED(status)) return false;
    }
    graph->BuildDeviceModel();
    device_program_ = graph;
    return true;
  }

  // --- Members ---

  int block_idx_{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc_{nullptr};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  Scope* exec_scope_{nullptr};
  std::vector<Instruction> subgraph_instructions_;
  std::shared_ptr<subgraph::intel_fpga::Graph> device_program_{nullptr};
};

}  // namespace intel_fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
