cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(toys)

set(CMAKE_CXX_STANDARD 20)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/3rd/yalantinglibs/include
  ${CMAKE_CURRENT_SOURCE_DIR}/3rd/fmt/include
)

link_directories(3rd/fmt/lib)

add_subdirectory(tests)
add_subdirectory(variant)
add_subdirectory(producer_consumer)
add_subdirectory(test_coro_rpc)
add_subdirectory(design_pattern)
add_subdirectory(rate_limit)
add_subdirectory(ec_set)
add_subdirectory(data_placement)
