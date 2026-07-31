#pragma once
#include <memory>
#include <string>
#include <array>
#include <vector>
#include "paddle_place.h"
#include "core/tensor.h"
#include "core/types.h"
#include "core/program.h"
#include "model_parser/cpp_desc.h"

namespace paddle {
namespace lite_api {

// ── PaddlePredictor ──────────────────────────────────────────
// Hardcoded for SSD MobileNet V1 (3 inputs, 1 output).
//   input[0]: im_shape   [1, 2]
//   input[1]: image      [1, 3, 300, 300]  (NCHW)
//   input[2]: scale      [1, 2]
//   output:   detection  [N, 6]  (class_id, conf, x1, y1, x2, y2)
//
// GetInput / GetOutput return raw pointers to the underlying
// lite::Tensor (no wrapper).  The caller includes core/tensor.h
// to access Resize, mutable_data, data, dims, etc.

class LITE_API PaddlePredictor {
 public:
  explicit PaddlePredictor(const std::string& model_file);
  ~PaddlePredictor();

  // Pre-set all 3 input tensor shapes (always the same:
  // [1,2], [1,3,300,300], [1,2]).  Must be called once
  // after construction before any inference.
  void InitInputs();

  // Returns the i-th input tensor (0/1/2).
  lite::Tensor* GetInput(int i);

  // Returns the output tensor (index 0 only).
  const lite::Tensor* GetOutput() const;

  void Run();

 private:
  void PrepareFeedFetch();
  void DequantizeWeight();

  // ── fixed 3-in / 1-out ────────────────────────────────────
  std::string input_names_[3];      // from model file, 3 feed ops
  std::string output_name_;         // from model file, fetch op col 0
  lite::Tensor* input_ptrs_[3]{};   // resolved once in constructor
  const lite::Tensor* output_ptr_{};// (ditto)

  std::vector<lite::Instruction> instructions_;
  std::shared_ptr<lite::Scope> scope_;
  std::shared_ptr<const lite::cpp::ProgramDesc> program_desc_;
};

}  // namespace lite_api
}  // namespace paddle
