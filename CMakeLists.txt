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

# Always use C++17
set(CXX_STANDARD 17)
set(CUDA_STANDARD 17)
message(STATUS "Using C++17")

# Set C++ standard
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA settings
set(CMAKE_CUDA_STANDARD ${CUDA_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

# Windows-specific settings applied to all targets
if(WIN32)
    target_compile_definitions(sencoder PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(sdecoder PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(pencoder PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(pdecoder PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Compiler info
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Using CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "Building for Windows")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
