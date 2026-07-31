#include "api/paddle_api.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <utility>
#include "core/tensor.h"
#include "core/program.h"
#include "core/scope.h"
#include "core/variable.h"
#include "model_parser/flatbuffers/io.h"
#include "lite/utils/io.h"

namespace paddle {
namespace lite {

/*
 * Binary structure of naive_buffer model: model.nb
 * ----------------------------------------------------------
 * |       |    PART         |   Precision |   Length(byte) |
 * |   1   |  meta_version   |   uint16_t  |       2        |
 * |   2   |  opt_version    |   char[16]  |      16        |
 * |   3   |  topo_size      |   uint64_t  |       8        |
 * |   4   |  topo_data      |   char[]    | topo_size byte |
 * |   5   |  param_data     |   char[]    |                |
 * ----------------------------------------------------------
*/

void LoadModelNaiveFromFile(const std::string& filename,
                            Scope* scope,
                            cpp::ProgramDesc* cpp_prog) {
  model_parser::BinaryFileReader reader(filename, 0);

  uint16_t meta_version;
  reader.Read(&meta_version, sizeof(uint16_t));

  char opt_version[16];
  reader.Read(opt_version, 16 * sizeof(char));

  // get topo_size + topo_data
  uint64_t topo_size;
  reader.Read(&topo_size, sizeof(uint64_t));
  lite::model_parser::Buffer buf(topo_size);
  reader.Read(buf.data(), topo_size);
  cpp_prog->Init(std::move(buf));

  // load scope from params
  switch (meta_version) {
    case 1: {
      lite::model_parser::Buffer buf(reader.length() - reader.current());
      reader.Read(buf.data(), reader.length() - reader.current());
      fbs::CombinedParamsDescView params(std::move(buf));
      fbs::deprecated::SetScopeWithCombinedParams(scope, params);
      break;
    }
    case 2: {
      fbs::ParamDeserializer deserializer(&reader);
      deserializer.ForwardRead(scope);
      break;
    }
    default:
      LOG(FATAL) << "Unsupported model meta_version " << meta_version;
      break;
  }
}

}  // namespace lite

namespace lite_api {

PaddlePredictor::PaddlePredictor(const std::string& model_file) {
  scope_ = std::make_shared<lite::Scope>();
  auto prog = std::make_shared<lite::cpp::ProgramDesc>();
  lite::LoadModelNaiveFromFile(model_file, scope_.get(), prog.get());
  program_desc_ = std::move(prog);
  DequantizeWeight();

  // PrepareFeedFetch — scan feed/fetch ops for IO tensor names.
  PrepareFeedFetch();

  // ── BuildRuntimeProgram (inlined) ────────────────────────────
  // Pre-create all variables (ops call scope->FindVar during Attach).
  CHECK(program_desc_);
  auto block_size = program_desc_->BlocksSize();
  CHECK(block_size);
  for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
    auto block_desc = program_desc_->GetBlock<lite::cpp::BlockDesc>(block_idx);
    for (size_t var_idx = 0; var_idx < block_desc->VarsSize(); ++var_idx) {
      auto var_desc = block_desc->GetVar<lite::cpp::VarDesc>(var_idx);
      auto* var = scope_->Var(var_desc->Name());
      var->GetMutableTensor();  // ensure Tensor exists (ops call GetTensor)
    }
  }

  // Build instructions (calib + subgraph), then resolve IO pointers.
  instructions_ =
      lite::BuildInstructions(program_desc_, scope_.get(), lite::kRootBlockIdx);
  for (int i = 0; i < 3; i++) {
    CHECK(!input_names_[i].empty());
    input_ptrs_[i] = scope_->FindMutableTensor(input_names_[i]);
    CHECK(input_ptrs_[i]) << "no input tensor " << input_names_[i];
  }
  if (!output_name_.empty()) {
    output_ptr_ = scope_->FindMutableTensor(output_name_);
    CHECK(output_ptr_) << "no output tensor " << output_name_;
  }

  // program_desc_ no longer needed after instruction build.
  program_desc_.reset();
}

PaddlePredictor::~PaddlePredictor() = default;

void PaddlePredictor::Run() {
  // Execute all instructions in order:
  //   [0] calib          float32→int8  subgraph input conversion
  //   [1] subgraph       FPGA inference (MobileNet V1 + SSD heads)
  //   [2..354]           SSD post-processing (prior_box, slice,
  //                       elementwise_*, concat, softmax, multiclass_nms3)
  // KernelBase::Launch() handles one-time PrepareForRun, ReInitWhenNeeded,
  // workspace reset, and Run() internally.
  for (auto& inst : instructions_) {
    inst.mutable_op()->InferShape();
    inst.mutable_kernel()->Launch();
  }
}

void PaddlePredictor::InitInputs() {
  input_ptrs_[0]->Resize({1, 2});                           // im_shape
  input_ptrs_[1]->Resize({1, 3, 300, 300});                 // image
  input_ptrs_[2]->Resize({1, 2});                           // scale
}

// ── direct pointer access, no scope lookup ──────────────────

lite::Tensor* PaddlePredictor::GetInput(int i) {
  CHECK(i >= 0 && i < 3)
      << "SSD model has exactly 3 inputs (0:im_shape 1:image 2:scale)";
  return input_ptrs_[i];
}

const lite::Tensor* PaddlePredictor::GetOutput() const {
  return output_ptr_;
}

// Scans the main block for feed / fetch ops to discover the
// 3 input tensor names and 1 output tensor name.
void PaddlePredictor::PrepareFeedFetch() {
  auto main_block =
      program_desc_->GetBlock<lite::cpp::BlockDesc>(lite::kRootBlockIdx);
  for (size_t op_idx = 0; op_idx < main_block->OpsSize(); ++op_idx) {
    auto op_desc = main_block->GetOp<lite::cpp::OpDesc>(op_idx);
    if (op_desc->Type() == "feed") {
      int col = op_desc->GetAttr<int>("col");
      CHECK(col >= 0 && col < 3) << "feed col out of range: " << col;
      input_names_[col] = op_desc->Output("Out").front();
    } else if (op_desc->Type() == "fetch") {
      int col = op_desc->GetAttr<int>("col");
      if (col == 0) output_name_ = op_desc->Input("X").front();
    }
  }
}

void PaddlePredictor::DequantizeWeight() {
  CHECK(program_desc_ != nullptr);

  auto is_weight_quantized_op = [](const lite::cpp::OpDesc* op_desc) {
    CHECK(op_desc != nullptr);
    bool result = false;
    if (op_desc->HasAttr("quantization_type")) {
      std::string type = op_desc->GetAttr<std::string>("quantization_type");
      result = (type == "post_weight_abs_max") ||
               (type == "post_weight_channel_wise_abs_max");
    } else {
      result = op_desc->HasAttr("quantize_weight_bits");
    }
    return result;
  };
  lite::Tensor tmp_tensor;
  for (size_t i = 0; i < program_desc_->BlocksSize(); i++) {
    auto* block = program_desc_->GetBlock<lite::cpp::BlockDesc>(i);
    CHECK(block != nullptr);
    for (size_t k = 0; k < block->OpsSize(); ++k) {
      auto* op_desc = block->GetOp<lite::cpp::OpDesc>(k);
      CHECK(op_desc != nullptr);
      if (is_weight_quantized_op(op_desc)) {
        auto input_names = op_desc->input_vars();
        for (auto& input_name : input_names) {
          std::string input_scale_name = input_name + "_quant_scale";
          size_t found = input_name.find("/target_trans");
          std::string input_scale_name_alias = "";
          if (found != std::string::npos) {
            input_scale_name_alias =
                input_name.substr(0, found) + "_quant_scale";
          }
          if (op_desc->HasAttr(input_scale_name) ||
              (!input_scale_name_alias.empty() &&
               op_desc->HasAttr(input_scale_name_alias))) {
            if (!input_scale_name_alias.empty()) {
              input_scale_name = input_scale_name_alias;
              input_name = input_name.substr(0, found);
            }
            lite::Variable* scope_var = scope_->FindVar(input_name);
            CHECK(scope_var != nullptr);
            auto input_tensor = scope_var->GetMutableTensor();
            CHECK(input_tensor != nullptr);
            tmp_tensor.CopyDataFrom(*input_tensor);
            auto scale_list =
                op_desc->GetAttr<std::vector<float>>(input_scale_name);

            int quantize_weight_bits =
                op_desc->GetAttr<int>("quantize_weight_bits");
            CHECK(quantize_weight_bits == 8 || quantize_weight_bits == 16);
            float* fp_data = input_tensor->mutable_data<float>();
            CHECK(fp_data != nullptr);

            std::string op_type = op_desc->Type();
            if (op_type == "conv2d" || op_type == "depthwise_conv2d") {
              int64_t ch = input_tensor->dims()[0];
              int64_t offset = input_tensor->numel() / ch;
              CHECK_EQ(scale_list.size(), ch);
              if (quantize_weight_bits == 8) {
                const int8_t* int_data = tmp_tensor.data<int8_t>();
                CHECK(int_data != nullptr);
                for (int64_t i = 0; i < ch; ++i)
                  for (int64_t j = 0; j < offset; ++j)
                    fp_data[i * offset + j] =
                        scale_list[i] * int_data[i * offset + j];
              } else {
                const int16_t* int_data = tmp_tensor.data<int16_t>();
                CHECK(int_data != nullptr);
                for (int64_t i = 0; i < ch; ++i)
                  for (int64_t j = 0; j < offset; ++j)
                    fp_data[i * offset + j] =
                        scale_list[i] * int_data[i * offset + j];
              }
            }
          }
        }
      }
    }
  }

}

}  // namespace lite_api
}  // namespace paddle
