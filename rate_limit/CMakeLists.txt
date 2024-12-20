add_executable(client client.cpp)
add_executable(server server.cpp)

target_link_libraries(client PRIVATE libfmt.a pthread)
target_link_libraries(server PRIVATE libfmt.a pthread)