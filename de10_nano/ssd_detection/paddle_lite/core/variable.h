#pragma once
#include <memory>
#include <vector>
#include "core/tensor.h"

namespace paddle {
namespace lite {

using FeedFetchList = std::vector<lite::Tensor>;

class Variable {
 public:
  Tensor* GetMutableTensor() {
    if (!tensor_) tensor_.reset(new Tensor);
    return tensor_.get();
  }

  const Tensor& GetTensor() const {
    CHECK(tensor_);
    return *tensor_;
  }

  std::vector<Tensor>* GetMutableTensorList() {
    if (!tensor_list_) tensor_list_.reset(new std::vector<Tensor>);
    return tensor_list_.get();
  }

  const std::vector<Tensor>& GetTensorList() const {
    CHECK(tensor_list_);
    return *tensor_list_;
  }

  bool IsTensorList() const { return tensor_list_ != nullptr; }

 private:
  std::unique_ptr<Tensor> tensor_;
  std::unique_ptr<std::vector<Tensor>> tensor_list_;
};

}  // namespace lite
}  // namespace paddle
