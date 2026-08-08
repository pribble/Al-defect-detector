# model/ 与 subgraph/ 子目录已上提到 core/ 根，顶层 GLOB 即覆盖全部
FILE(GLOB_RECURSE UNIT_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*test*)
FILE(GLOB CORE_BASE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

if(UNIT_TEST_SRC)
  LIST(REMOVE_ITEM CORE_BASE_SRC ${UNIT_TEST_SRC})
endif()

set(CORE_SRC ${CORE_BASE_SRC})
set(CORE_DEPS "")

configure_file(version.h.in version.h)

lite_cc_library(core SRCS ${CORE_SRC}
  DEPS target_wrapper_host fbs_headers ${CORE_DEPS}
  INTEL_FPGA_DEPS intel_fpga_target_wrapper)
