if(LITE_WITH_INTEL_FPGA)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
  # subgraph bridge libs (moved up from bridges/ subdir)
  message(STATUS "++++++++INTEL_FPGA_SDK Library: ${intel_fpga_runtime_libs}")
  lite_cc_library(subgraph_bridge_graph_intel_fpga SRCS graph.cc DEPS ${intel_fpga_runtime_libs})
  lite_cc_library(subgraph_bridge_conv_op_intel_fpga SRCS conv_op.cc DEPS core subgraph_bridge_graph_intel_fpga)
  set(intel_fpga_subgraph_bridges
          core
          subgraph_bridge_graph_intel_fpga
          subgraph_bridge_conv_op_intel_fpga
          CACHE INTERNAL "intel_fpga_subgraph_bridges")
  set(lite_kernel_deps ${lite_kernel_deps} ${intel_fpga_subgraph_bridges} ${intel_fpga_target_wrapper} CACHE INTERNAL "")
  set(intel_fpga_deps ${lite_kernel_deps} ${intel_fpga_runtime_libs} CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()

#add_kernel(calib_compute_intel_fpga INTEL_FPGA basic SRCS calib_compute.cc)
#add_kernel(conv_compute_intel_fpga INTEL_FPGA basic SRCS conv_compute.cc)
add_kernel(subgraph_compute_intel_fpga INTEL_FPGA basic SRCS subgraph_compute.cc)
#add_kernel(pool_compute_intel_fpga INTEL_FPGA basic SRCS pool_compute.cc)
message(STATUS "intel library ${intel_fpga_runtime_libs}")
