#!/usr/bin/env bash
set -e

rm -rf CMakeCache.txt CMakeFiles

# point CMake at both OpenCV and CUDA-toolkit
cmake -G Ninja .

# build
ninja

#test
./my_gpu_app
