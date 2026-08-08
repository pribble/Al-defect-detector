lite_cc_library(target_wrapper_host SRCS target_wrapper.cc)

# host math libs (moved up from math/ subdir)
lite_cc_library(math_host SRCS
    prior_box.cc
    gpc.cc
    DEPS core)