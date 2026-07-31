cmake_minimum_required(VERSION 3.1)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")

project(object_detection_demo)

include_directories(include)
include_directories(paddle_lite/api)
include_directories(paddle_lite)
include_directories(paddle_lite/model_parser/flatbuffers/third_party)
include_directories(paddle_lite/build/inference_lite/include)
include_directories(paddle_lite/build)
link_directories(lib)

add_definitions("-DLITE_WITH_FLATBUFFERS_DESC")
add_definitions("-DLITE_ON_FLATBUFFERS_DESC_VIEW")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=armv7-a -mfloat-abi=hard -mfpu=neon")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv7-a -mfloat-abi=hard -mfpu=neon")

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(OpenCV_DIR lib/opencv/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ssd_detection ssd_detection.cc)
target_link_libraries(ssd_detection paddle_light_api_shared vnna ${OpenCV_LIBS})
