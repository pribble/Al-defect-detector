if(LITE_ON_MODEL_OPTIMIZE_TOOL)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  set(lite_kernel_deps ${lite_kernel_deps} math_host CACHE INTERNAL "")
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
endif()

message(STATUS "compile with lite host kernels")

# basic kernels
add_kernel(feed_compute_host Host basic SRCS feed_compute.cc)
add_kernel(fetch_compute_host Host basic SRCS fetch_compute.cc)
add_kernel(reshape_compute_host Host basic SRCS reshape_compute.cc)
add_kernel(unsqueeze_compute_host Host basic SRCS unsqueeze_compute.cc)
add_kernel(multiclass_nms_compute_host Host basic SRCS multiclass_nms_compute.cc)
add_kernel(stack_compute_host Host basic SRCS stack_compute.cc)
add_kernel(prior_box_compute_host Host basic SRCS prior_box_compute.cc)
