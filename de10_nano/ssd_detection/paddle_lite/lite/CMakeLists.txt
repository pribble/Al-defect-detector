message(STATUS "LITE_WITH_INTEL_FPGA:\t${LITE_WITH_INTEL_FPGA}")
message(STATUS "LITE_WITH_ARM:\t${LITE_WITH_ARM}")

add_subdirectory(backends)
add_subdirectory(utils)
add_subdirectory(operators)
add_subdirectory(kernels)

# ----------------------------- PUBLISH -----------------------------
add_custom_target(publish_inference)

set(INFER_LITE_PUBLISH_ROOT "${PADDLE_BINARY_DIR}/inference_lite")
message(STATUS "publish inference lib to ${INFER_LITE_PUBLISH_ROOT}")

add_custom_target(tiny_publish_cxx_lib
  COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/include"
  COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/lib"
  COMMAND cp "${PADDLE_SOURCE_DIR}/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/include"
  COMMAND cp "${PADDLE_BINARY_DIR}/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/include"
  COMMAND cp "${PADDLE_BINARY_DIR}/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/include"
  COMMAND cp "${PADDLE_BINARY_DIR}/api/libpaddle_light_api_shared.so" "${INFER_LITE_PUBLISH_ROOT}/lib"
)
add_dependencies(tiny_publish_cxx_lib paddle_light_api_shared)
add_dependencies(publish_inference tiny_publish_cxx_lib)

if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_custom_command(TARGET tiny_publish_cxx_lib POST_BUILD
    COMMAND ${CMAKE_STRIP} "-s" ${INFER_LITE_PUBLISH_ROOT}/lib/libpaddle_light_api_shared.so)
endif()
