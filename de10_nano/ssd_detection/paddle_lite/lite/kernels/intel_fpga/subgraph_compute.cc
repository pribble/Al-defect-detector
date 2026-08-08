#include "lite/kernels/intel_fpga/subgraph_compute.h"
#include <dlfcn.h>
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "core/op_registry.h"
#include "lite/kernels/intel_fpga/paddle_use_bridges.h"

#include "core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace intel_fpga {

}  // namespace apu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kIntelFPGA,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::intel_fpga::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
