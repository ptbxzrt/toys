cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(toys)

set(CMAKE_CXX_STANDARD 23)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/3rd/include
)

add_subdirectory(tests)
add_subdirectory(variant)
add_subdirectory(producer_consumer)
add_subdirectory(test_coro_rpc)
