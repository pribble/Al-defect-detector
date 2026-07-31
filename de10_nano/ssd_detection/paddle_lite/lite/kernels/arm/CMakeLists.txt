# NOTE we leave the add_kernel not protected by LITE_WITH_ARM so that all the kernels will be registered
# to the model_optimize_tool.

if(LITE_WITH_ARM AND NOT LITE_ON_MODEL_OPTIMIZE_TOOL)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
  set(lite_kernel_deps ${lite_kernel_deps} math_arm CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()

message(STATUS "compile with lite ARM kernels")

# 1. basic kernels for basic models
add_kernel(activation_compute_arm ARM basic SRCS activation_compute.cc)
add_kernel(scale_compute_arm ARM basic SRCS scale_compute.cc)
add_kernel(softmax_compute_arm ARM basic SRCS softmax_compute.cc)
add_kernel(elementwise_compute_arm ARM basic SRCS elementwise_compute.cc)
add_kernel(concat_compute_arm ARM basic SRCS concat_compute.cc)
add_kernel(calib_compute_arm ARM basic SRCS calib_compute.cc)
add_kernel(transpose_compute_arm ARM basic SRCS transpose_compute.cc)
add_kernel(slice_compute_arm ARM basic SRCS slice_compute.cc)
add_kernel(layout_compute_arm ARM basic SRCS layout_compute.cc)
# conv kernels for ssd_mobilenet_v1
add_kernel(conv_depthwise ARM basic SRCS conv_depthwise.cc)
add_kernel(conv_depthwise_common ARM basic SRCS conv_depthwise_common.cc)
add_kernel(conv_direct ARM basic SRCS conv_direct.cc)
add_kernel(conv_gemmlike ARM basic SRCS conv_gemmlike.cc)
add_kernel(conv_winograd ARM basic SRCS conv_winograd.cc)
add_kernel(conv_compute_arm ARM basic SRCS conv_compute.cc)
## extra ops needed for ssd_mobilenet_v1 (exp activation)
add_kernel(activation_extra_compute_arm ARM extra SRCS activation_extra_compute.cc)
