# Project configurations
cmake_minimum_required(VERSION 3.18)
project(Pi-Benchmark LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# Check GCC version
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
        message(FATAL_ERROR "GCC version must be at least 13.0")
    endif()
else()
    message(FATAL_ERROR "Only GCC is supported")
endif()

# CUDA configurations
include(CheckLanguage)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CUDA_STANDARD 11)
check_language(CUDA)

# Set compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --use_fast_math")

# Set binaries - main
set(SOURCE
    src/main.cpp
    src/config.cpp
    src/input.cpp)
add_executable(cuckoo_hash ${SOURCE})

if(ENABLE_TESTS)
    # Set binaries - test_config
    set(SOURCE_TEST_CONFIG
        src/config.cpp
        tests/test-config.cpp)
    add_executable(test_config ${SOURCE_TEST_CONFIG})

    # Set binaries - test_input_file
    set(SOURCE_TEST_INPUT_FILE
        src/input.cpp
        tests/test-input-file.cpp)
    add_executable(test_input_file ${SOURCE_TEST_INPUT_FILE})
endif()

# Set include path
include_directories(${PROJECT_SOURCE_DIR}/include)