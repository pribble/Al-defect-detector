// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/operators/conv_op.h"
#include<intelfpga.h>
#include <cstdio>
#include <cmath>
#include "lite/kernels/intel_fpga/graph.h"
#include "core/subgraph_bridge_registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace intel_fpga {

int ConvConverter(void *ctx, OpLite *op, KernelBase *kernel) {
    VLOG(4) << "Converting conv2d op for intelfpga.";
    CHECK(ctx != nullptr);
    CHECK(op != nullptr);
    auto graph = static_cast<Graph*>(ctx);

    operators::ConvParam& param = kernel->Param<operators::ConvParam>();
    std::string op_type = op->op_info()->Type();
    auto op_info = op->op_info();
    auto scope = op->scope();

    auto input_name = op_info->Input("Input").front();
    auto filter_name = op_info->Input("Filter").front();
    auto output_name = op_info->Output("Output").front();
    auto output = scope->FindMutableTensor(output_name);
    auto w_dims = param.filter->dims();
    auto i_dims = param.x->dims();
    auto o_dims = param.output->dims();

    Node* node = new Node();
    node->name_ = output_name;
    // Create node's device param.
    FpgaConvParam* device_param = new FpgaConvParam();
    node->node_param_ = dynamic_cast<NodeParam*>(device_param);
    node->is_output = graph->IsOutput(output_name);
    node->is_input = graph->IsInput(input_name);
    node->op_type_ = (param.groups==1)?INTELFPGA_Conv2D:INTELFPGA_DW_Conv2D;

    // Find this node's parent according to input tensor.
    if(graph->GetNodeByTensorName(input_name)) {
        node->parent_vec_.push_back(graph->GetNodeByTensorName(input_name));
        int byte_offset =
          FpgaWord2ByteOffset(node->parent_vec_[0]->op_type_,
              FpgaGetOutputOffset(node->parent_vec_[0]));
      device_param->param.input_offset = FpgaByte2WordOffset(node->op_type_, byte_offset);
    } else {
      node->parent_vec_.push_back(nullptr);
      device_output_config config = FpgaMemMalloc(node->op_type_,
          device_param->d_ia, i_dims[1], i_dims[2], i_dims[3]);
      device_param->param.input_offset = config.output_offset;
    }
    if (graph->getGraphRootNode() == nullptr) {
      graph->setGraphRootNode(node);
    }

    // Malloc output and set offset.
    device_output_config config = FpgaMemMalloc(node->op_type_,
        device_param->d_oa, o_dims[1], o_dims[2], o_dims[3]);
    device_param->param.output_offset = config.output_offset;
    device_param->param.output_size = config.output_size;
    VLOG(4) << "output_offset: " << device_param->param.output_offset;
    VLOG(4) << "output_siez: " << device_param->param.output_size;

    // Put this node's output tensor in map.
    graph->setTensor2Node(output_name, node);

    // Let predecessor node in topological order link to this node.
    auto pre_node = graph->getGraphTailNode();
    if(pre_node) {
      pre_node->next_ = node;
    }

    graph->setGraphTailNode(node);
    node->next_= nullptr;

    device_param->ia = param.x->mutable_data<int8_t>();
    device_param->oa = param.output->mutable_data<int8_t>();
    device_param->ka = param.filter->mutable_data<int8_t>();
    float *ba = param.bias ? param.bias->mutable_data<float>() : nullptr;
    float *scale=param.weight_scale.data() ? param.weight_scale.data() : nullptr;

    // Fill fpga_param.
    int group = param.groups;
    auto paddings = *param.paddings;
    auto dilations = *param.dilations;
    // CHECK_EQ(dilations[0], 1);
    uint32_t at_;

    switch (param.activation_param.active_type) {
        case lite_api::ActivationType::kRelu:
        at_ = INTELFPGA_ACT_RELU;
        break;
        case lite_api::ActivationType::kRelu6:
        at_ = INTELFPGA_ACT_RELU6;
        break;
        case lite_api::ActivationType::kLeakyRelu:
        at_ = INTELFPGA_ACT_LEAKYRELU;
        device_param->param.lr = param.activation_param.Leaky_relu_alpha;
        break;
        default:
        at_ = INTELFPGA_ACT_NONE;
        break;
    }
    // init scale：每输出通道 4 个 int32 定点 requant 参数（FPGA 侧量化优先，无浮点）
    //   乘后域（CPU cvt_kernel 语义，与 float 公式 round(acc·ws·is/os + bias/os) 对齐）：
    //   scale[0..out_c)          = mult      = round_half_away((ws*is/os)*2^shift)
    //   scale[out_c..2*out_c)    = bias_mul  = round_half_away(bias/os * 2^(shift-8)) ← q(shift-8)（int32 安全）
    //   scale[2*out_c..3*out_c)  = shift（每层自适应，见下）
    //   scale[3*out_c..4*out_c)  = rcl6      = round_half_away(6/os * 2^(shift-8))    ← q(shift-8)（relu6 上限，int8 域）
    // 硬件侧（cnn_core requant）把 bias_mul/rcl6 左移 8 位对齐 qshift 乘后域
    //（v = acc·mult + bias_mul<<8；relu6 钳 v ≤ rcl6<<8）。
    //
    // shift 自适应（2026-08 box 头 score 修复）：旧实现硬编码 shift=30，
    // box 头 output_scale 极小 → ws·is/os 大，mult=round(scale·2^30) 超 int32
    // 回绕成负值 → 该通道 logits 系统性错误（score 全偏）。现按层内最大
    // |ws·is/os| 选 shift：30 起逐次 -1，保证 |mult| < 2^31；下界 26
    //（2^26 > 全模型最大 |acc| ≈ 3.7e7，仍可精确舍入）。bias/rcl6 同步用
    // q(shift-8)，RTL 的 <<8 对齐不依赖具体 shift 值。
    device_param->scale = new int32_t[4*o_dims[1]];
    device_param->param.input_scale = param.input_scale;
    device_param->param.output_scale = param.output_scale;

    VLOG(4) << "input scale: " << param.input_scale;
    VLOG(4) << "output scale: " << param.output_scale;
    {
        auto rnd = [](double x) -> int32_t {
            return (int32_t)(x >= 0 ? std::floor(x + 0.5) : std::ceil(x - 0.5));
        };
        const double is_ = (double)param.input_scale;
        const double os_ = (double)param.output_scale;
        const int oc = o_dims[1];
        // 选层 shift：使 max|ws·is/os|·2^shift < 2^31（mult 不超 int32）
        int shift = 30;
        double max_scale = 0.0;
        for (int i = 0; i < oc; i++) {
            const double ws = scale ? (double)scale[i] : 1.0;
            const double f = std::fabs(ws * is_ / os_);
            if (f > max_scale) max_scale = f;
        }
        while (shift > 26 && max_scale * (double)(1LL << shift) >= 2147483648.0)
            shift--;
        if (max_scale * (double)(1LL << shift) >= 2147483648.0) {
            // scale ≥ 32 的极端层：mult 无法用 int32 表示，clamp 并告警
            fprintf(stderr, "[SCALE] %s: |ws*is/os| max=%.6g 超出 int32 可表示"
                    " 范围（shift=%d 仍溢出），mult 将被 clamp——score 不可信\n",
                    node->name_.c_str(), max_scale, shift);
        }
        for (int i = 0; i < oc; i++) {
            const double ws = scale ? (double)scale[i] : 1.0;
            const double b  = ba    ? (double)ba[i]    : 0.0;
            const double f  = ws * is_ / os_;
            const double denom = ws * is_;
            if (fabs(denom) < 1e-12) {
                // ws 极小（量化权重≈0 的通道）：输出≈0（bias 主导后 relu 钳 0）。
                // mult/bias_mul 保底 0、rcl6 保底 INT_MAX（硬件比较恒不成立=无上限）。
                device_param->scale[i] = 0;
                device_param->scale[oc + i] = 0;
                device_param->scale[3 * oc + i] = INT_MAX;
            } else {
                double mv = f * (double)(1LL << shift);
                device_param->scale[i] =
                    (mv >= 2147483647.0) ? INT_MAX
                    : (mv <= -2147483648.0) ? INT_MIN
                    : rnd(mv);
                // bias_mul q(shift-8)：rnd 前 clamp 防 int32 回绕
                double bm = b / os_ * (double)(1LL << (shift - 8));
                device_param->scale[oc + i] =
                    (bm >= 2147483647.0) ? INT_MAX
                    : (bm <= -2147483648.0) ? INT_MIN
                    : rnd(bm);
                // rcl6 溢出保底：6/os·2^(shift-8) 在 os 很小时超 int32
                //（回绕成负值会让硬件 relu6 比较恒成立 → 该层输出全错）。
                // 超限给 INT_MAX = 硬件比较恒不成立（无上限）；
                // 此时 6/os > 127，饱和本身已充当上限，clamp 冗余。
                const double rcl6 = 6.0 / os_ * (double)(1LL << (shift - 8));
                device_param->scale[3 * oc + i] =
                    (rcl6 >= 2147483647.0) ? INT_MAX : rnd(rcl6);
            }
            device_param->scale[2 * oc + i] = shift;
        }
    }

    //ignore batch dimension TODO
    device_param->param.scale_offset = 0;
    device_param->d_ka = nullptr;
    device_param->param.in_c=i_dims[1];
    device_param->param.in_h=i_dims[2];
    device_param->param.in_w=i_dims[3];
    device_param->param.output_c=o_dims[1];
    device_param->param.output_h=o_dims[2];
    device_param->param.output_w=o_dims[3];
    device_param->param.in_pad=paddings[0];
    device_param->param.kernel=w_dims[2];
    device_param->param.stride=param.strides[0];
    device_param->param.relu=at_;
    device_param->param.dilation=dilations[0];

    device_param->param.type=(param.groups==1)?INTELFPGA_Conv2D:INTELFPGA_DW_Conv2D;
    if(param.groups==1){
        struct device_weight_config config= conv2d_weight_reorganize(
            device_param->ka,
            (int8_t**)(&(device_param->d_ka)),
            w_dims[0],
            w_dims[1],
            w_dims[2],
            w_dims[3],
            filter_name.c_str());
        device_param->param.weight_size = config.weight_size;
        device_param->param.weight_offset = config.weight_offset;
    }
    else{
        struct device_weight_config config = dw_conv2d_weight_reorganize(device_param->ka,(int8_t**)(&(device_param->d_ka)),w_dims[0],w_dims[2],w_dims[3]);
        device_param->param.weight_size = config.weight_size;
        device_param->param.weight_offset = config.weight_offset;
    }
    VLOG(4) << "Converting conv2d op for intelfpga end.";
  return SUCCESS;
}

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    conv2d,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::ConvConverter);

REGISTER_SUBGRAPH_BRIDGE(
    depthwise_conv2d,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::ConvConverter);
