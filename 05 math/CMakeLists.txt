cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
# 设定编译版本 默认为52 20系是76 30系是86
set(CMAKE_CUDA_ARCHITECTURES 86)
# 指定多版本
# set(CMAKE_CUDA_ARCHITECTURES 52;70;75;86)

project(hellocuda LANGUAGES CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(main main.cu)
add_executable(saxpy saxpy.cu)
# 允许__host__和__device__定义为lambda表达式
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(saxpy PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)