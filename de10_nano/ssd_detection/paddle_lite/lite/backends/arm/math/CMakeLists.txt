if(NOT LITE_WITH_ARM)
  return()
endif()


# arm math source code in current directory
FILE(GLOB ARM_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
# fp16 arm math source code in fp16/ directory
FILE(GLOB FP16_ARM_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp16/*.cc)
# sve2 arm math source code in sve2/ directory
FILE(GLOB SVE_ARM_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sve/*.cc)

if(LITE_WITH_ARM82_FP16)
  set(ARM_MATH_SRC ${ARM_MATH_SRC} ${FP16_ARM_MATH_SRC})
endif()

if(LITE_WITH_ARM8_SVE2)
   set(ARM_MATH_SRC ${ARM_MATH_SRC} ${SVE_ARM_MATH_SRC})
endif()

lite_cc_library(math_arm SRCS ${ARM_MATH_SRC})
