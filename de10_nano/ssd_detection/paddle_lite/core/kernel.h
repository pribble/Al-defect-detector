#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "api/paddle_place.h"
#include "lite/backends/arm/math/type_trans.h"
#include "core/device_info.h"
#include "core/target_wrapper.h"
#include "core/type_system.h"
#include "core/types.h"
#include "lite/operators/op_params.h"
#include "lite/utils/all.h"
#include "lite/utils/stream.h"

namespace paddle {
namespace lite {

// An base with virtual functions to unify all the kernel implementation on
// different targets.
class KernelBase {
 public:
  // type_infer_handler is used to inference a output type by considering the
  // input types in the type system.
  using type_infer_handler_t = std::function<const Type*(
      const std::map<std::string, const Type*>& input_types,
      const std::string& out_arg)>;

  /// Attach params from OpDesc. Called once during instruction construction.
  virtual void Attach(const cpp::OpDesc& opdesc, Scope* scope) {}

  /// Infer output tensor shapes from input dims. Called every Run.
  virtual void InferShape() {}

  /// Run some initialization before `Run`, it will invoke after `SetParam` and
  /// `SetContext`, that is both the param_ and context_ are valid.
  virtual void PrepareForRun() {}

  /// Run kernel initialization if needed at every run (eg. input shape changed)
  virtual void ReInitWhenNeeded() {}

  /// Run the kernel. Before Run, both the param_ and context_ should be valid.
  virtual void Run() = 0;

  void Launch() {
    /// First run, init kernel, do weights transform once
    if (is_first_epoch_) {
      PrepareForRun();
      is_first_epoch_ = false;
    }
    /// re-init the kernel if needed (input shape should be checked in conv
    /// kernel)
    ReInitWhenNeeded();

    Run();
  }

  void SetContext(std::unique_ptr<KernelContext>&& ctx) {
    ctx_ = std::move(ctx);
  }
  template <typename T>
  void SetParam(T param) {
    param_.set(param);
  }
  template <typename P>
  P& Param() const {
    return *param_.get_mutable<P>();
  }

  void set_op_type(const std::string& type) { op_type_ = type; }
  const std::string& op_type() const { return op_type_; }

  void set_alias(const std::string& x) { alias_ = x; }
  const std::string& alias() const { return alias_; }

  virtual Place place() const = 0;
  virtual TargetType target() const = 0;
  virtual PrecisionType precision() const = 0;
  virtual DataLayoutType layout() const = 0;
  const KernelContext* context() const { return ctx_.get(); }
  KernelContext* mutable_context() { return ctx_.get(); }
  virtual std::string name() const = 0;

  static std::string SerializeKernelType(const std::string& op_type,
                                         const std::string& alias,
                                         const Place& place);

  static void ParseKernelType(const std::string& kernel_type,
                              std::string* op_type,
                              std::string* alias,
                              Place* place);

  virtual ~KernelBase() = default;

 protected:
  std::unique_ptr<KernelContext> ctx_{nullptr};
  mutable operators::param_t param_;
  // The corresponding op type.
  std::string op_type_{};
  // The extra identity to help defficiate a specific kernel, op_type_ + alias_
  // is the unique ID for the kernel.
  std::string alias_{};
  bool is_first_epoch_{true};
};

// Light-weight kernel implementation.
// The OpKernel is designed to implement the specific algorithm on a target
// device.
// TODO(Superjomn) Consider to add a Platform type to differentiate CUDNN,
// MKLDNN, plain CUDA C implementations.
template <TargetType Target,
          PrecisionType Precision,
          DataLayoutType DataLayout = DataLayoutType::kNCHW>
class KernelLite : public KernelBase {
 public:
  // Run the kernel.
  virtual void Run() override { CHECK(false) << "Not Implemented"; }

  TargetType target() const override { return Target; }
  PrecisionType precision() const override { return Precision; }
  DataLayoutType layout() const override { return DataLayout; }
  Place place() const override { return Place{Target, Precision, DataLayout}; }
  std::string name() const override;

  KernelLite() = default;
  virtual ~KernelLite() = default;
};

template <TargetType Target, PrecisionType Precision, DataLayoutType DataLayout>
std::string KernelLite<Target, Precision, DataLayout>::name() const {
  return op_type() + ":" + TargetToStr(Target) + "/" +
         PrecisionToStr(Precision) + "/" + DataLayoutToStr(DataLayout);
}

}  // namespace lite
}  // namespace paddle
