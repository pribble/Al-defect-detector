find_package(PythonInterp REQUIRED)
file(MAKE_DIRECTORY ${PADDLE_BINARY_DIR}/api)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE_SOURCE_DIR}/api/paddle_lite_factory_helper.h ${PADDLE_BINARY_DIR}/api/)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/lite/tools/cmake_tools/parse_kernel_registry.py
    ${kernels_src_list} ${fake_kernels_src_list}
    ${PADDLE_BINARY_DIR}/api/paddle_use_kernels.h
    "${LITE_OPTMODEL_DIR}/.tailored_kernels_list"
    ${LITE_BUILD_TAILOR} ${LITE_BUILD_EXTRA} ${LITE_WITH_ARM82_FP16}
    RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT "${error}" STREQUAL "")
    message(FATAL_ERROR ${error})
endif()
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/lite/tools/cmake_tools/parse_op_registry.py
    ${ops_src_list} ${PADDLE_BINARY_DIR}/api/paddle_use_ops.h
    "${LITE_OPTMODEL_DIR}/.tailored_ops_list"
    ${LITE_BUILD_TAILOR} ${LITE_BUILD_EXTRA}
    RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT "${error}" STREQUAL "")
    message(FATAL_ERROR ${error})
endif()

set(LIGHT_API_SRC  paddle_api.cc paddle_place.cc)
set(light_lib_DEPS utils core kernels model_parser ops CACHE INTERNAL "")
set(external_libs_DEPS "" CACHE INTERNAL "")

set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")

if(LITE_WITH_INTEL_FPGA)
    set(external_libs_DEPS ${external_libs_DEPS} ${intel_fpga_deps})
endif()

lite_cc_library(paddle_api_light SRCS ${LIGHT_API_SRC} DEPS ${light_lib_DEPS} ${external_libs_DEPS})

add_library(PADDLELITE_OBJS OBJECT ${__lite_cc_files})
add_dependencies(PADDLELITE_OBJS fbs_headers)
set_target_properties(PADDLELITE_OBJS PROPERTIES COMPILE_FLAGS "-fdata-sections -flto")
add_library(paddle_light_api_shared SHARED $<TARGET_OBJECTS:PADDLELITE_OBJS>)
