cmake_minimum_required(VERSION 3.18)
project(LZ77Compression LANGUAGES CXX CUDA)

# Explicitly use Clang
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Use modern CUDA support
enable_language(CUDA)

# Get CUDA version
message(STATUS "Found CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "60")

# Define targets
add_executable(sencoder sencoder.cpp)
add_executable(sdecoder sdecoder.cpp)
add_executable(pencoder pencoder.cu)
add_executable(pdecoder pdecoder.cu)

# Simple approach - let CMake handle CUDA linking automatically
# CUDA targets will automatically link against necessary CUDA libraries

# Common settings for all targets
set(COMMON_TARGETS sencoder sdecoder pencoder pdecoder)

# Compiler info
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Using CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
