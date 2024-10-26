cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(toys)

set(CMAKE_CXX_STANDARD 23)

add_subdirectory(variant)
