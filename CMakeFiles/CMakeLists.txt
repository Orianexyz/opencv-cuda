cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0146 OLD)

project(my_gpu_app LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED COMPONENTS core cudaimgproc cudacodec)

add_executable(my_gpu_app main.cpp)
target_link_libraries(my_gpu_app PRIVATE ${OpenCV_LIBS})
