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

#include "lite/kernels/arm/conv_depthwise_common.h"
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void DepthwiseConvCommon<PRECISION(kFloat),
                         PRECISION(kFloat)>::ReInitWhenNeeded() {}

template <>
void DepthwiseConvCommon<PRECISION(kFloat),
                         PRECISION(kFloat)>::PrepareForRun() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8),
                         PRECISION(kFloat)>::ReInitWhenNeeded() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
}

template <>
void DepthwiseConvCommon<PRECISION(kInt8),
                         PRECISION(kInt8)>::ReInitWhenNeeded() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {}

template <>
void DepthwiseConvCommon<PRECISION(kFloat), PRECISION(kFloat)>::Run() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8), PRECISION(kFloat)>::Run() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8), PRECISION(kInt8)>::Run() {}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
