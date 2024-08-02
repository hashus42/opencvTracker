cmake_minimum_required(VERSION 3.28)
project(opencvTracker)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

add_executable(opencvTracker main.cpp)

target_link_libraries(opencvTracker ${OpenCV_LIBS})