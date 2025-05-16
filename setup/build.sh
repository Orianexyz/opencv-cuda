#!/usr/bin/env bash
set -euo pipefail

# make pkg-config fall back to empty if unset
PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}"
export VIDEO_CODEC_SDK_ROOT=~/nvcodec/Video_Codec_SDK

# clean build dir
rm -rf build && mkdir build && cd build

echo "=== Building OpenCV-CUDA (logging to ../build.log) ==="
cmake -G Ninja \
  -DWITH_CUDA=ON \
  -DWITH_NVCUVID=ON \
  -DWITH_NVCUVENC=ON \
  -DBUILD_opencv_cudacodec=ON \
  -DVIDEO_CODEC_SDK_ROOT="$VIDEO_CODEC_SDK_ROOT" \
  -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
  -DCUDA_ARCH_BIN=75 \
  -DPYTHON3_EXECUTABLE="$(which python3)" \
  -DOPENCV_PYTHON3_INSTALL_PATH="$(python3 -c 'import site;print(site.getsitepackages()[0])')" \
  -DCMAKE_BUILD_TYPE=Release \
  ../opencv 2>&1 | tee ../build.log

ninja -j"$(nproc)"        2>&1 | tee -a ../build.log
sudo ninja install        2>&1 | tee -a ../build.log
sudo ldconfig

echo "=== Build complete. Logs in ../build.log ==="
