#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "api/paddle_place.h"
#include "core/apis.h"
#include "core/scope.h"
#include "core/tensor.h"
#include "core/types.h"
#include "model_parser/cpp_desc.h"
#include "lite/utils/all.h"
/*
 * This file contains all the argument parameter data structure for operators.
 */

namespace paddle {
namespace lite {
namespace operators {

struct ParamBase {};

using param_t = Any;
#define WITH_INT8_CONFIG             \
  bool enable_int8{false};           \
  float input_scale{1.0f};           \
  std::vector<float> weight_scale{}; \
  float output_scale{1.0f};          \
  int bit_length{8};

/// ----------------------- Functional operators ------------------------------
struct FeedParam : ParamBase {
  std::vector<lite::Tensor>* feed_list{};
  lite::Tensor* out{};
  int col;
};

struct FetchParam : ParamBase {
  const lite::Tensor* input{};
  std::vector<lite::Tensor>* fetch_list{};
  int col;
};

// Helper op for lite framework
struct IoCopyParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const std::vector<lite::Tensor>* x_array{nullptr};
  lite::Tensor* y{nullptr};
  std::vector<lite::Tensor>* y_array{nullptr};
  int process_type{0};
};

struct LayoutParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* y{};
  int process_type{0};
};

struct CalibParam : ParamBase {
  const lite::Tensor* input{};
  lite::Tensor* output{};
  float scale;
};

struct CalibInplaceParam : ParamBase {
  lite::Tensor* input{};
  lite::Tensor* output{};
  float scale;
};

struct SubgraphParam : ParamBase {
  std::vector<std::string> input_names{};
  std::vector<std::string> output_names{};
  std::vector<std::string> input_data_names{};
  std::vector<std::string> output_data_names{};
  std::vector<float> input_data_scales{};
  std::vector<float> output_data_scales{};
  int block_idx{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc{nullptr};
  Scope* exec_scope{nullptr};
};

/// -------------------------- NN operators ------------------------------------

struct FcParam : ParamBase {
  lite::Tensor* input{nullptr};
  lite::Tensor* w{nullptr};
  lite::Tensor* bias{nullptr};
  lite::Tensor* Prelu_alpha{nullptr};
  lite::Tensor* output{nullptr};
  lite::DDim in_mat_dims;
  // original dims of input weight
  lite::DDim w_dims;
  int in_num_col_dims{1};
  std::string activation_type{""};
  bool padding_weights{false};
  std::string Prelu_mode{
      "channel"};  // prelu param, can be "all", "channel" or "element"
  std::string op_type{"mul"};
  float alpha{6.f};
  // for int8
  WITH_INT8_CONFIG
};


// For Interpolate Op

// For Mul Op


// For Stack Op
struct StackParam : ParamBase {
  std::vector<lite::Tensor*> X;
  lite::Tensor* Out{};

  int axis{0};
};

// For Unstack Op

// For Power Op

// For Pow Op

// For Sign Op

struct ShuffleChannelParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  int group;
};

// For Yolobox
struct YoloBoxParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* ImgSize{};
  lite::Tensor* Boxes{};
  lite::Tensor* Scores{};

  std::vector<int> anchors{};
  int class_num{0};
  float conf_thresh{0.f};
  int downsample_ratio{0};
  bool clip_bbox{true};
  float scale_x_y{1.0f};
};

// For Scale Op
struct ScaleParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};

  float scale{1.f};
  float bias{0.f};
  bool bias_after_scale{true};
  std::string activation_type{""};
  bool fuse_relu{false};
  float alpha{6.f};

  bool fuse_scaleact{false};
  float scale1{1.f};
  float bias1{0.f};
};

// For Scatter OP

// For Softmax op
struct SoftmaxParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};
  int axis{-1};
  bool use_cudnn{true};
  bool eleminate_success{false};
};

// For LogSoftmax op

// For Reshape and Reshape2 Op
struct ReshapeParam : ParamBase {
  const lite::Tensor* x{};
  std::vector<const lite::Tensor*> shape_tensor_vct{};
  const lite::Tensor* shape_tensor{};
  std::vector<int> shape_vct{};
  lite::Tensor* output{};

  lite::Tensor* xshape{};
  bool inplace{false};

};

// For Concat op
struct ConcatParam : ParamBase {
  std::vector<lite::Tensor*> x{};
  lite::Tensor* output{};
  int axis{0};
  lite::Tensor* axis_tensor{};
};

/// ----------------------- activation operators ----------------------
struct ActivationParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite_api::ActivationType active_type{lite_api::ActivationType::kIndentity};
  bool has_active{false};
  float Leaky_relu_alpha{0.f};   // leaky_relu param
  float Relu_clipped_coef{6.f};  // relu_clipped param
  std::string Prelu_mode{
      "channel"};  // prelu param, can be "all", "channel" or "element"
  lite::Tensor* Prelu_alpha{};  // prelu param
  float Swish_beta;             // swish param
  // hard_sigmoid param
  float hard_sigmoid_slope{0.2f};
  float hard_sigmoid_offset{0.5f};
  // hard_swish param
  float hard_swish_threshold{6.0f};
  float hard_swish_scale{6.0f};
  float hard_swish_offset{3.0f};
  // swish param
  float swish_scale{6.0f};
  // thresholded_relu
  float relu_threshold{1.0f};
  // elu
  float Elu_alpha{1.0f};
  // relu6
  float threshold{6.0f};
  // gelu
  bool gelu_approximate{false};
  // softplus
  float softplus_beta{1.0f};
  float softplus_threshold{20.f};
};

struct ActivationGradParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Out{};
  // for backward
  lite::Tensor* X_grad{};
  const lite::Tensor* Out_grad{};
};

// For Sparse Convolution op

// For Convolution op
struct ConvParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* filter{};
  lite::Tensor* bias{nullptr};
  lite::Tensor* residualData{nullptr};
  lite::Tensor* second_x{nullptr};
  lite::Tensor* output{};
  std::vector<int> strides{1, 1};
  /* paddings type change
   * from std::vector<int> to std::shared_ptr<std::vector<int>>
   * to support dynamically modify padding
   * let kernel param and operator param Synchronous update
   */
  std::shared_ptr<std::vector<int>> paddings;
  int groups{1};
  /* dilations type change
   * from std::vector<int> to std::shared_ptr<std::vector<int>>
   * to support dynamically modify padding
   * let kernel param and operator param Synchronous update
   */
  std::shared_ptr<std::vector<int>> dilations;
  bool fuse_relu_before_depthwise_conv{false};
  bool use_mkldnn{false};
  bool fuse_relu{false};  // only used in mkldnn kernel
  bool fuse_sigmoid{false};
  bool fuse_tanh{false};
  bool fuse_swish{false};
  bool fuse_exp{false};
  bool fuse_abs{false};
  bool use_quantizer{
      false};  // set true for op that should be quantized, only used for cpu
  bool fuse_residual_connection{false};
  float scale_in{1.0f};           // only used with mkl-dnn int8
  float scale_out{1.0f};          // only used with mkl-dnn int8
  float scale_in_eltwise{1.0f};   // only used with mkl-dnn int8
  float scale_weights{1.0f};      // only used with mkl-dnn int8
  bool force_fp32_output{false};  // only used in mkl-dnn int8
  std::string data_format{"Anylayout"};
  // for activation
  ActivationParam activation_param;
  // for elementwise tree fuse
  std::string fuse_elementwise_op_type{""};
  // support var_length or not
  bool var_length{false};
  // only used in conv_transpose.
  std::vector<int> output_size;
  std::vector<int> output_padding;


  // for int8
  WITH_INT8_CONFIG
  // for Conv2d+Scale fusion
  std::string scale_activation_type{""};
};

// For BatchNorm op
struct BatchNormParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* bias{};
  lite::Tensor* scale{};
  lite::Tensor* mean{};
  lite::Tensor* variance{};
  lite::Tensor* y{};
  lite::Tensor* mean_out{};
  lite::Tensor* variance_out{};
  lite::Tensor* saved_mean{};
  lite::Tensor* saved_variance{};
  bool is_test{true};
  bool use_global_stats{false};
  float epsilon;
  float momentum;
  DataLayoutType data_layout{DATALAYOUT(kNCHW)};
};

// For Pooling op
struct PoolParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};
  lite::Tensor* mask{};
  std::string pooling_type{""};
  std::vector<int> ksize{};
  bool global_pooling{
      false};  // if true, knernel size and paddings will be ignored
  std::vector<int> strides{1, 1};
  /* paddings type change
   * from std::vector<int> to std::shared_ptr<std::vector<int>>
   * to support dynamically modify padding
   * let kernel param and operator param Synchronous update
   */
  std::shared_ptr<std::vector<int>> paddings;
  bool exclusive{true};
  bool adaptive{false};
  bool ceil_mode{false};
  bool use_quantizer{false};
  std::string padding_algorithm{"EXPLICIT"};
  std::string data_format{"AnyLayout"};
  // for int8
  WITH_INT8_CONFIG
};

// For Dropout op

// For PadConstantLike op

// For Split op


// For Transpose op
struct TransposeParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* output{};
  lite::Tensor* xshape{};

  std::vector<int> axis;
  bool use_mkldnn{false};
  std::string data_format{"AnyLayout"};
};

struct TrilTriuParam : ParamBase {
  const lite::Tensor* x{nullptr};
  lite::Tensor* out{nullptr};

  int diagonal{0};
  bool lower{true};
};

/// ----------------------- element wise operators ----------------------
struct ElementwiseParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  int axis{-1};  // for broadcasting.
  // for int8
  WITH_INT8_CONFIG
  float x_input_scale{1.0f};
  float y_input_scale{1.0f};
  // fuse ScaleParam
  bool fuse_scale{false};
  float scale{1.f};
  float bias{0.f};
  bool bias_after_scale{true};
  float alpha{6.f};
  std::string activation_type{""};
};

struct ElementwiseGradParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  const lite::Tensor* OutGrad{};
  lite::Tensor* XGrad{};
  lite::Tensor* YGrad{};
  int axis{-1};  // for broadcasting.
};

struct FusionElementwiseActivationParam : public ElementwiseParam {
  std::string act_type;
};

struct FusionElementwiseActivationGradParam : public ElementwiseGradParam {
  std::string act_type;
};

/// ----------------------- mean operators ----------------------


struct FillAnyLikeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  float value{0.0f};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
};

/// ----------------------- fill_constant operators ----------------------
struct FillConstantParam : ParamBase {
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  std::vector<int64_t> shape{};
  lite::Tensor* shape_tensor{nullptr};
  lite::Tensor* value_tensor{nullptr};
  std::vector<lite::Tensor*> shape_tensor_list{};

  float value{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu{false};
  lite::Tensor* in{};
  lite::Tensor* out{};
};

struct FillConstantBatchSizeLikeParam : ParamBase {
  const lite::Tensor* input{nullptr};
  lite::Tensor* out{nullptr};

  std::vector<int> shape{};
  int input_dim_idx{0};
  int output_dim_idx{0};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  float value{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu{false};
};

//






/// ----------------------- sgd operators ----------------------

/// ----------------------- uniform_random operators ----------------------
struct UniformRandomParam : ParamBase {
  const lite::Tensor* shape_tensor{nullptr};
  std::vector<lite::Tensor*> shape_tensor_list{};
  std::vector<int64_t> shape{};
  float min{-1.0f};
  float max{1.0f};
  int seed{0};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  lite::Tensor* Out{};
};
/// ----------------------- unfold operators ----------------------
/// ----------------------- negative operators --------------
/// ----------------------- pad2d operators ----------------------
struct Pad2dParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> paddings{0, 0, 0, 0};
  std::string mode{"constant"};
  float pad_value = 0.f;
  std::string data_format{"NCHW"};
};

/// ----------------------- Crop operators ----------------------
struct CropParam : ParamBase {
  const lite::Tensor* X{nullptr};
  const lite::Tensor* Y{nullptr};
  const lite::Tensor* Offsets{nullptr};
  lite::Tensor* Out{nullptr};
  std::vector<int> offsets;
  std::vector<int> shape;
};

/// ----------------------- CropTensor operators ----------------------
struct CropTensorParam : ParamBase {
  const lite::Tensor* X{nullptr};
  const lite::Tensor* Shape{nullptr};
  const lite::Tensor* Offsets{nullptr};
  const std::vector<lite::Tensor>* ShapeTensor{nullptr};
  const std::vector<lite::Tensor>* OffsetsTensor{nullptr};
  lite::Tensor* Out{nullptr};
  std::vector<int> offsets;
  std::vector<int> shape;
};

///----------------------- argmax operators ----------------------
struct ArgmaxParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Out{};
  int Axis{0};
  int dtype{-1};
  bool keepdims{false};
};

///----------------------- inverse operators ----------------------
struct InverseParam : ParamBase {
  lite::Tensor* Input{};
  lite::Tensor* Output{};
};

///----------------------- index_select operators ----------------------
struct Index_selectParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Index{};
  lite::Tensor* Out{};
  int dim{0};
};

///----------------------- reverse operators ----------------------
struct ReverseParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Out{};
  // for tensor_array
  std::vector<lite::Tensor>* X_array{nullptr};
  std::vector<lite::Tensor>* Out_array{nullptr};
  std::vector<int> Axis;
};

///----------------------- axpy operators ----------------------
/// ----------------------- GRU unit operators ----------------------f

/// ------------------------------ lrn operators ------------------------------

/// ----------------------- decode_bboxes operators ----------------------

/// ----------------------- box_coder operators ----------------------
struct BoxCoderParam : ParamBase {
  const lite::Tensor* prior_box{};
  const lite::Tensor* prior_box_var{};
  const lite::Tensor* target_box{};
  lite::Tensor* proposals{};
  // code_type: encode_center_size and decode_center_size
  std::string code_type{"encode_center_size"};
  bool box_normalized{true};
  int axis{0};
  std::vector<float> variance{};
};

/// ----------------------- multiclass_nms operators ----------------------
struct MulticlassNmsParam : ParamBase {
  const lite::Tensor* bboxes{};
  const lite::Tensor* scores{};
  lite::Tensor* out{};
  lite::Tensor* index{};
  int background_label{0};
  float score_threshold{};
  int nms_top_k{};
  float nms_threshold{0.3f};
  float nms_eta{1.0f};
  int keep_top_k;
  bool normalized{true};
  const lite::Tensor* rois_num{};
  lite::Tensor* nms_rois_num{};
};

/// ----------------------- matrix_nms operators ----------------------

/// ----------------------- priorbox operators ----------------------
struct PriorBoxParam : ParamBase {
  lite::Tensor* input{};
  lite::Tensor* image{};
  lite::Tensor* boxes{};
  lite::Tensor* variances{};

  bool flip{true};
  bool clip{true};
  std::vector<float> min_sizes;
  std::vector<float> max_sizes;
  std::vector<float> aspect_ratios;
  std::vector<float> variances_;
  int img_w{0};
  int img_h{0};
  float step_w{0.f};
  float step_h{0.f};
  float offset{0.5f};
  int prior_num{0};
  bool flatten_to_2d{false};
  // priortype: prior_min, prior_max, prior_com
  std::vector<std::string> order;
  bool min_max_aspect_ratios_order{false};
};

struct DensityPriorBoxParam : public PriorBoxParam {
  std::vector<float> fixed_sizes;
  std::vector<float> fixed_ratios;
  std::vector<int> density_sizes;
};
/// ----------------------- GRU operators ----------------------f


/// ----------------------- BeamSearchDecode operators ----------------------f
struct BeamSearchDecodeParam : ParamBase {
  std::vector<lite::Tensor>* ids{nullptr};
  std::vector<lite::Tensor>* scores{nullptr};
  lite::Tensor* sentence_ids{nullptr};
  lite::Tensor* sentence_scores{nullptr};
  int beam_size;
  int end_id;
};

/// ----------------------- LookupTable operators ----------------------f


struct Im2SequenceParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  std::vector<int> kernels{3, 3};
  std::vector<int> strides{1, 1};
  std::vector<int> paddings{0, 0, 0, 0};
  std::vector<int> out_strides{1, 1};
};

struct SequenceSoftmaxParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct NormParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* Norm{};
  int axis{1};
  float epsilon{1e-10f};
};



struct WhileParam : ParamBase {
  Tensor* cond{};
  int block_idx{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc{nullptr};
  Scope* exec_scope{nullptr};
};

struct TopkParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* KTensor{};
  lite::Tensor* Out{};
  lite::Tensor* Indices{};
  bool k_is_tensor{false};
  int K{1};
  int axis{-1};
};












struct SequencePadParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* PadValue{};
  lite::Tensor* Out{};
  lite::Tensor* Length{};
  int padded_length{-1};
};

struct SequenceUnpadParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Length{};
  lite::Tensor* Out{};
};

struct SequenceMaskParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* MaxLenTensor{nullptr};
  lite::Tensor* Y{};
  int maxlen{-1};
  int out_dtype;
};







struct LodResetParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  std::vector<int> target_lod;
  bool append;
};




/// ----------------------- shape operators ----------------------

struct CastParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  int out_dtype{2};
  int in_dtype{2};
};

struct SliceParam : ParamBase {
  const lite::Tensor* X{nullptr};
  lite::Tensor* Out{nullptr};
  const std::vector<lite::Tensor>* XTensorList{nullptr};
  std::vector<lite::Tensor>* OutTensorList{nullptr};
  std::vector<int> axes{};
  std::vector<int> starts{};
  std::vector<int> ends{};
  std::vector<int> decrease_axis{};
  std::vector<int> infer_flags{};
  std::vector<lite::Tensor*> StartsTensorList{};
  std::vector<lite::Tensor*> EndsTensorList{};
  const lite::Tensor* StartsTensor{nullptr};
  const lite::Tensor* EndsTensor{nullptr};
};



struct AnchorGeneratorParam : ParamBase {
  const lite::Tensor* Input{};
  std::vector<float> anchor_sizes{};
  std::vector<float> aspect_ratios{};
  std::vector<float> stride{};
  std::vector<float> variances{{0.1f, 0.1f, 0.2f, 0.2f}};
  float offset{0.5f};

  lite::Tensor* Anchors{};
  lite::Tensor* Variances{};
};

struct GenerateProposalsParam : ParamBase {
  // inputs
  const lite::Tensor* Scores{};
  const lite::Tensor* BboxDeltas{};
  const lite::Tensor* ImInfo{};
  lite::Tensor* Anchors{};
  lite::Tensor* Variances{};

  // attrs
  int pre_nms_topN{6000};
  int post_nms_topN{1000};
  float nms_thresh{0.5f};
  float min_size{0.1f};
  float eta{1.0f};

  // outputs
  lite::Tensor* RpnRois{};
  lite::Tensor* RpnRoiProbs{};
  lite::Tensor* RpnRoisLod{};
  lite::Tensor* RpnRoisNum{};
};

struct GenerateProposalsV2Param : ParamBase {
  // inputs
  const lite::Tensor* Scores{};
  const lite::Tensor* BboxDeltas{};
  const lite::Tensor* ImShape{};
  lite::Tensor* Anchors{};
  lite::Tensor* Variances{};

  // attrs
  int pre_nms_topN{6000};
  int post_nms_topN{1000};
  float nms_thresh{0.5f};
  float min_size{0.1f};
  float eta{1.0f};
  bool pixel_offset{true};

  // outputs
  lite::Tensor* RpnRois{};
  lite::Tensor* RpnRoiProbs{};
  lite::Tensor* RpnRoisLod{};
  lite::Tensor* RpnRoisNum{};
};

/// ----------------------- squeeze operators ----------------------
struct SqueezeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* XShape{};
  std::vector<int> axes{};
  bool inplace{false};
};

struct UnsqueezeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* XShape{};
  std::vector<int> axes{};
  const lite::Tensor* axes_tensor{};
  std::vector<const lite::Tensor*> axes_tensor_vct{};
  bool inplace{false};
};

/// ----------------------- expand operators ----------------------

/// ----------------------- expand v2 operators ----------------------

/// ----------------------- expand as operators ----------------------

/// ----------------------- matmul operators ----------------------





/// ----------------------- assign operators -----------------------
struct AssignParam : ParamBase {
  // for tensor
  const lite::Tensor* X{nullptr};
  lite::Tensor* Out{nullptr};

  // for tensor_array
  const std::vector<lite::Tensor>* X_array{nullptr};
  std::vector<lite::Tensor>* Out_array{nullptr};
};

/// ----------------------- roi_align operators -----------------------
struct RoiAlignParam : ParamBase {
  lite::Tensor* X{nullptr};
  lite::Tensor* ROIs{nullptr};
  lite::Tensor* RoisLod{nullptr};
  lite::Tensor* RoisNum{nullptr};
  lite::Tensor* Out{nullptr};
  float spatial_scale{1.0f};
  int pooled_height{1};
  int pooled_width{1};
  int sampling_ratio{-1};
  bool align{false};
};

/// ----------------------- box_clip operators -----------------------
struct BoxClipParam : ParamBase {
  const lite::Tensor* Input{};
  const lite::Tensor* ImInfo{};
  lite::Tensor* Output{};
};


/// ----------------------- assign_value operators -----------------------
struct AssignValueParam : ParamBase {
  std::vector<int> shape{};
  int dtype{};
  std::vector<float> fp32_values{};
  std::vector<int> int32_values{};
  std::vector<int64_t> int64_values{};
  std::vector<int> bool_values{};
  lite::Tensor* Out{};
};

/// --------------- sequence_topk_avg_pooling operators ------------------

/// --------------- topk_pooling operators ------------------

/// --------------- search_fc operators ------------------
/// --------------------- match_matrix_tensor operators --------------------

/// --------------------- search_seq_depadding operators --------------------

/// --------------------- search_grnn operators --------------------


struct MergeLodTensorParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* mask{};
  const lite::Tensor* in_true{};
  const lite::Tensor* in_false{};
  lite::Tensor* out{};
  int level{};
};

struct ConditionalBlockParam : ParamBase {
  const lite::Tensor* cond{};
  std::vector<lite::Tensor*> inputs{};
  std::vector<lite::Tensor*> outs{};
  int block_idx{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc{nullptr};
  Scope* exec_scope{nullptr};
  bool is_scalar_condition{};
};

struct CollectFpnProposalsParam : ParamBase {
  std::vector<lite::Tensor*> multi_level_rois{};
  std::vector<lite::Tensor*> multi_level_scores{};
  std::vector<lite::Tensor*> multi_rois_num{};
  lite::Tensor* rois_num{};
  lite::Tensor* fpn_rois{};
  int post_nms_topN{};
};

struct DistributeFpnProposalsParam : ParamBase {
  const lite::Tensor* fpn_rois{};
  const lite::Tensor* rois_num{};
  std::vector<lite::Tensor*> multi_fpn_rois{};
  std::vector<lite::Tensor*> multi_rois_num{};
  lite::Tensor* restore_index{};
  int min_level{};
  int max_level{};
  int refer_level{};
  int refer_scale{};
  bool pixel_offset{true};
};

/// --------------------- instance_norm operators --------------------
/// --------------------- group_norm operators --------------------

/// --------------------- grid sampler operators --------------------























// For DeformableConvolution op

struct PixelShuffleParam : ParamBase {
  lite::Tensor* x{nullptr};
  lite::Tensor* output{nullptr};
  int upscale_factor{1};
};


struct WhereIndexParam : ParamBase {
  const lite::Tensor* input{nullptr};
  lite::Tensor* output{nullptr};
};

struct WhereParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const lite::Tensor* y{nullptr};
  const lite::Tensor* condition{nullptr};
  lite::Tensor* out{nullptr};
};

struct ClipParam : ParamBase {
  Tensor* x{};
  Tensor* min_tensor{};
  Tensor* max_tensor{};
  Tensor* out{};
  float min{};
  float max{};
};

struct PrintParam : ParamBase {
  const lite::Tensor* in{};
  lite::Tensor* out{};
  std::string name;
  int first_n{-1};
  std::string message;
  int summarize{20};
  bool print_tensor_name{true};
  bool print_tensor_type{true};
  bool print_tensor_shape{true};
  bool print_tensor_lod{true};
  bool print_tensor_layout{true};
  std::string print_phase;
  bool is_forward{true};
};

struct OneHotParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* depth_tensor{nullptr};
  lite::Tensor* Out{};
  int depth;
  int dtype;
  bool allow_out_of_range;
};

struct TrigonometricParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* out{};
};

using SinParam = TrigonometricParam;
using CosParam = TrigonometricParam;

struct FlattenContiguousRangeParam : ParamBase {
  const lite::Tensor* x{nullptr};
  lite::Tensor* out{nullptr};
  lite::Tensor* xshape{nullptr};
  int start_axis{1};
  int stop_axis{1};
};

struct LoDArrayLengthParam : ParamBase {
  std::vector<lite::Tensor>* x{};
  lite::Tensor* out{};
};

struct SelectInputParam : ParamBase {
  std::vector<lite::Tensor*> X{};
  lite::Tensor* Mask{};
  lite::Tensor* Out{};
};

struct TensorArrayToTensorParam : ParamBase {
  std::vector<lite::Tensor>* X{};
  lite::Tensor* Out{};
  lite::Tensor* OutIndex{};
  int axis{0};
  bool use_stack{false};
};




struct ScatterNdAddParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* indexs{};
  lite::Tensor* updates{};
  lite::Tensor* output{};
};

struct CumsumParam : ParamBase {
  const lite::Tensor* X{nullptr};
  lite::Tensor* Out{nullptr};

  int axis{-1};
  bool flatten{false};
  bool exclusive{false};
  bool reverse{false};
};

struct SamplingIdParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* out{};

  float min{0.f};
  float max{1.f};
  int seed{0};
};

struct PolygonBoxTransformParam : ParamBase {
  const lite::Tensor* input{nullptr};
  lite::Tensor* output{nullptr};
};




struct RoiPerspectiveTransformParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const lite::Tensor* rois{nullptr};
  lite::Tensor* out{nullptr};
  lite::Tensor* mask{nullptr};
  lite::Tensor* transfor_matrix{nullptr};
  lite::Tensor* out2in_idx{nullptr};
  lite::Tensor* out2in_weight{nullptr};
  float spatial_scale{1.f};
  int transformed_height{1};
  int transformed_width{1};
};

struct CorrelationParam : ParamBase {
  const lite::Tensor* input1{nullptr};
  const lite::Tensor* input2{nullptr};
  lite::Tensor* output{nullptr};
  int pad_size;
  int kernel_size;
  int max_displacement;
  int stride1;
  int stride2;
  int corr_type_multiply{1};
};

struct ArgsortParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* Indices{};

  int axis{-1};
  bool descending{false};
};

struct FlipParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  std::vector<int> axis;
};

struct CosSimParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const lite::Tensor* y{nullptr};
  lite::Tensor* out{nullptr};
  lite::Tensor* x_norm{nullptr};
  lite::Tensor* y_norm{nullptr};
};





}  // namespace operators
}  // namespace lite
}  // namespace paddle
