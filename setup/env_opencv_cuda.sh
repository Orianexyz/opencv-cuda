# Let pkg-config see /usr/local before the system
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Let CMake find your /usr/local packages (OpenCV, CUDA)
export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH

# Point to your CUDA toolkit
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export CUDA_HOME=$CUDA_TOOLKIT_ROOT_DIR
