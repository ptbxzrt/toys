add_executable(data_placement main.cpp)
add_executable(final_design final_design.cpp)

target_link_libraries(data_placement PRIVATE libfmt.a pthread)
target_link_libraries(final_design PRIVATE libfmt.a pthread)