#include "model_parser/flatbuffers/block_desc.h"
#include <memory>

namespace paddle {
namespace lite {
namespace fbs {

template <>
proto::VarDesc const* BlockDescView::GetVar<proto::VarDesc>(int32_t idx) const {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(VarsSize())) << "idx >= vars.size()";
  return desc_->vars()->Get(idx);
}

template <>
proto::OpDesc const* BlockDescView::GetOp<proto::OpDesc>(int32_t idx) const {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(OpsSize())) << "idx >= ops.size()";
  return desc_->ops()->Get(idx);
}

template <>
VarDescView const* BlockDescView::GetVar<VarDescView>(int32_t idx) const {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(VarsSize())) << "idx >= vars.size()";
  return vars_[idx].get();
}

template <>
OpDescView const* BlockDescView::GetOp<OpDescView>(int32_t idx) const {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(OpsSize())) << "idx >= ops.size()";
  return ops_[idx].get();
}

#ifdef LITE_WITH_FLATBUFFERS_DESC
template <>
proto::VarDescT* BlockDesc::GetVar<proto::VarDescT>(int32_t idx) {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(VarsSize())) << "idx >= vars.size()";
  return vars_[idx]->raw_desc();
}

template <>
proto::VarDescT* BlockDesc::AddVar<proto::VarDescT>() {
  desc_->vars.push_back(std::unique_ptr<proto::VarDescT>(new proto::VarDescT));
  SyncVars();
  return vars_.back()->raw_desc();
}

template <>
proto::OpDescT* BlockDesc::GetOp<proto::OpDescT>(int32_t idx) {
  CHECK_GE(idx, 0)
      << "The index value should be greater than or equal to zero.";
  CHECK_LT(idx, static_cast<int32_t>(OpsSize())) << "idx >= vars.size()";
  return ops_[idx]->raw_desc();
}

template <>
proto::OpDescT* BlockDesc::AddOp<proto::OpDescT>() {
  desc_->ops.push_back(std::unique_ptr<proto::OpDescT>(new proto::OpDescT));
  SyncOps();
  return ops_.back()->raw_desc();
}
#endif  // LITE_WITH_FLATBUFFERS_DESC

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
