cmake_minimum_required(VERSION 3.10)
project(Lab02_CoinDetection)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lab02 src/main.cpp)
target_link_libraries(lab02 ${OpenCV_LIBS})