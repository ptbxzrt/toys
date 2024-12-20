# add_compile_options(-Wall -Wextra -Wconversion)
add_executable(test1 test1.cpp)
add_executable(test2 test2.cpp)
add_executable(test3 test3.cpp)

target_link_libraries(test1 PRIVATE libfmt.a)
target_link_libraries(test2 PRIVATE libfmt.a)
target_link_libraries(test3 PRIVATE libfmt.a)