cmake_minimum_required(VERSION 3.10)
project(lab01)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(lab01 src/main.cpp)
target_link_libraries(lab01 ${OpenCV_LIBS})
target_compile_definitions(lab01 PRIVATE PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_threshold src/test_threshold.cpp)
target_link_libraries(test_threshold ${OpenCV_LIBS})
target_compile_definitions(test_threshold PRIVATE PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(main_v2 src/main_v2.cpp)
target_link_libraries(main_v2 ${OpenCV_LIBS})
target_compile_definitions(main_v2 PRIVATE PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
