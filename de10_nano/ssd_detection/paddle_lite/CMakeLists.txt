cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PADDLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PADDLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 11)
set(ARM_TARGET_OS armlinux)
set(ARM_TARGET_ARCH_ABI armv7hf)
set(ARM_TARGET_LANG gcc)

# Paddle-Lite feature flags (always ON for this project)
set(LITE_ON_TINY_PUBLISH ON)
set(LITE_WITH_ARM ON)
set(LITE_WITH_INTEL_FPGA ON)
set(LITE_WITH_OPENMP ON)
set(LITE_WITH_LOG ON)
set(WITH_ARM_DOTPROD ON)
set(LITE_BUILD_EXTRA ON)

# 这些虽然 OFF，但被 cmake execute_process 作为参数传给 Python 脚本
set(LITE_BUILD_TAILOR OFF)
set(LITE_WITH_ARM82_FP16 OFF)

set(LITE_OPTMODEL_DIR "")
set(THIRD_PARTY_PATH "${PADDLE_BINARY_DIR}/third_party" CACHE STRING "")

include(system)
include(functions)
include(os/common)
project(paddle CXX C)

include(simd)
# flatbuffers runtime library (merged from third-party)
include_directories("${PADDLE_SOURCE_DIR}/model_parser/flatbuffers/third_party")
# fbs_headers target kept for dependency compatibility (files already in place)
add_custom_target(fbs_headers COMMAND ${CMAKE_COMMAND} -E echo "fbs headers ready")
include_directories("${PADDLE_SOURCE_DIR}")
include_directories("${PADDLE_BINARY_DIR}")

include(lite)
include(backends/intel_fpga)
include(generic)
include(configure)
include(postproject)

add_subdirectory(core)
add_subdirectory(model_parser)
add_subdirectory(lite)
add_subdirectory(api)
