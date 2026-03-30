cmake_minimum_required(VERSION 3.10)
project(Lab03_PhotoMerge)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lab03 src/main.cpp)
target_link_libraries(lab03 ${OpenCV_LIBS})
target_compile_definitions(lab03 PRIVATE PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}") 