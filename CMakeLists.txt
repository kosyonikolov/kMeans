cmake_minimum_required(VERSION 3.0.0)
project(kMeans VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_DIRS})

add_executable(kMeans main.cpp)

target_link_libraries(kMeans ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
