cmake_minimum_required(VERSION 3.20)
project(label_aware_hnsw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release so that a plain `cmake -B build` produces benchmark-grade code.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LAH_NATIVE_ARCH "Compile with -march=native" ON)

add_library(label_aware_hnsw INTERFACE)
target_include_directories(label_aware_hnsw INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_compile_features(label_aware_hnsw INTERFACE cxx_std_20)
target_compile_options(label_aware_hnsw INTERFACE -Wall -Wextra -Wpedantic)

# Separate configurations for tests and benchmarks:
#   cmake -B build-test  -DCMAKE_BUILD_TYPE=Debug     (asserts enabled)
#   cmake -B build-bench -DCMAKE_BUILD_TYPE=Release   (-O3, NDEBUG)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(label_aware_hnsw INTERFACE -O3 -DNDEBUG)
else()
  target_compile_options(label_aware_hnsw INTERFACE -O1 -g)
endif()

if(LAH_NATIVE_ARCH)
  target_compile_options(label_aware_hnsw INTERFACE -march=native)
endif()

enable_testing()
file(GLOB LAH_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/test_*.cpp)
foreach(test_source ${LAH_TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME_WE)
  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} PRIVATE label_aware_hnsw)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()