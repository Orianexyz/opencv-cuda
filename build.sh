#!/usr/bin/env bash
set -euo pipefail
LOG_FILE="build.log"
rm -f build.log

{
# — Copy the Video Codec SDK headers into your CUDA include directory —
# (Adjust the source path if yours differs)
cp -r nvidia/Video_Codec_SDK_12.2.72/ ~/nvcodec/Video_Codec_SDK
sudo cp ~/nvcodec/Video_Codec_SDK/Interface/{cuviddec.h,nvcuvid.h,nvEncodeAPI.h} \
     "$CUDA_HOME/include/"

# — 1) Fresh build directory —
rm -rf build
mkdir build
cd build

# — 2) Configure OpenCV with CUDA & video-codec support —
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DWITH_CUDA=ON \
  -DWITH_NVCUVID=ON \
  -DBUILD_opencv_cudacodec=ON \
  -DVIDEO_CODEC_SDK_ROOT="$CUDA_HOME" \
  -DBUILD_opencv_rgbd=OFF \
  -DBUILD_opencv_xfeatures2d=OFF \
  -DBUILD_opencv_ximgproc=OFF \
  -DCUDA_FAST_MATH=ON \
  -DCUDA_ARCH_BIN=7.5 \
  -DOPENCV_EXTRA_MODULES_PATH="../opencv_contrib/modules" \
  -DPYTHON3_EXECUTABLE="$(which python3)" \
  -DOPENCV_PYTHON3_INSTALL_PATH="$(python3 -c 'import site;print(site.getsitepackages()[0])')" \
  ../opencv

# — 3) Build & install —
ninja -j"$(nproc)"
sudo ninja install
sudo ldconfig

} 2>&1 | tee -a "$LOG_FILE"
