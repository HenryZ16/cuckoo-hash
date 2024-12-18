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
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CUDA_STANDARD 11)
check_language(CUDA)

# Set compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fopenmp -O2 --use_fast_math")

# Set macros
if(BENCHMARK)
    add_definitions(-DBENCHMARK)
endif()
if(DEBUG)
    add_definitions(-DDEBUG)
endif()

# Set binaries - main
set(SOURCE
    src/main.cpp
    src/config.cpp
    src/input.cpp)
add_executable(cuckoo_hash ${SOURCE})

# Set binaries - baseline
set(SOURCE_BASELINE
    src/baseline.cpp
    src/config.cpp
    src/input.cpp)
add_executable(baseline ${SOURCE_BASELINE})

# Set binaries - cuckoo_hash_lib
set(SOURCE_CUCKOO_HASH_LIB
    src/cuckooHashDefault.cpp
    src/cuckooHashCUDA.cu
    src/config.cpp
    src/input.cpp
    src/primeFilter.cpp)
add_library(cuckoo_hash_lib ${SOURCE_CUCKOO_HASH_LIB})

# Set binaries - data_generator
set(SOURCE_DATA_GENERATOR
    src/dataGenerator.cpp)
add_executable(data_generator ${SOURCE_DATA_GENERATOR})

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
target_include_directories(cuckoo_hash_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Set link libraries
target_link_libraries(cuckoo_hash cuckoo_hash_lib)
target_link_libraries(baseline cuckoo_hash_lib)