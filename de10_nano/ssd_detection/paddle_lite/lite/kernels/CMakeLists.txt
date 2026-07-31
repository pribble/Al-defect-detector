message(STATUS "add lite kernels")

set(lite_kernel_deps "" CACHE INTERNAL "" FORCE)

add_subdirectory(host)
add_subdirectory(arm)
add_subdirectory(intel_fpga)

# Generate kernel registry headers
file(MAKE_DIRECTORY ${PADDLE_BINARY_DIR}/all_kernel_faked_dir)
find_package(PythonInterp REQUIRED)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/lite/tools/cmake_tools/create_fake_kernel_registry.py
    ${kernels_src_list}
    ${fake_kernels_src_list}
    ${PADDLE_BINARY_DIR}/all_kernel_faked_dir
    ${PADDLE_BINARY_DIR}/all_kernel_faked.h
    ${PADDLE_BINARY_DIR}/all_kernel_faked.cc
    ${PADDLE_BINARY_DIR}/kernel_src_map.h
    RESULT_VARIABLE result
    ERROR_VARIABLE error)
if(NOT "${error}" STREQUAL "")
  message(FATAL_ERROR ${error})
endif()

file(GLOB all_kernel_faked_src ${PADDLE_BINARY_DIR}/all_kernel_faked_dir/*.cc)
lite_cc_library(kernels SRCS ${KERNELS_SRC} ${all_kernel_faked_src} DEPS ${lite_kernel_deps})
add_dependencies(kernels utils)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/lite/tools/cmake_tools/record_supported_kernel_op.py
    ${kernels_src_list}
    ${fake_kernels_src_list}
    ${ops_src_list}
    ${PADDLE_BINARY_DIR}/supported_kernel_op_info.h
    ${LITE_BUILD_EXTRA}
    RESULT_VARIABLE result
    ERROR_VARIABLE error)
if(NOT "${error}" STREQUAL "")
    message(FATAL_ERROR ${error})
endif()
