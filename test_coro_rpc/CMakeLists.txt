add_executable(proxy proxy.cpp)
add_executable(coordinator coordinator.cpp)

target_link_libraries(proxy PRIVATE libfmt.a)
target_link_libraries(coordinator PRIVATE libfmt.a)