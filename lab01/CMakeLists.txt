cmake_minimum_required(VERSION 3.10)
project(lab01)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(lab01 src/main.cpp)
target_link_libraries(lab01 ${OpenCV_LIBS})

add_executable(test_threshold src/test_threshold.cpp)
target_link_libraries(test_threshold ${OpenCV_LIBS})
