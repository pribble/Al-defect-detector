# 这些宏在 CMakeLists.txt 中固定为 ON，无条件定义
add_definitions("-DLITE_WITH_ARM")
add_definitions("-DLITE_WITH_INTEL_FPGA")
add_definitions("-DLITE_WITH_LOG")
add_definitions("-DLITE_WITH_FLATBUFFERS_DESC")
add_definitions("-DLITE_ON_TINY_PUBLISH")
add_definitions("-DLITE_ON_FLATBUFFERS_DESC_VIEW")

if (WITH_ARM_DOTPROD)
    add_definitions("-DWITH_ARM_DOTPROD")
endif()

if (LITE_BUILD_EXTRA)
    add_definitions("-DLITE_BUILD_EXTRA")
endif()

if (LITE_WITH_ARM82_FP16)
    add_definitions("-DLITE_WITH_ARM82_FP16")
endif()
