#!/usr/bin/env bash
set -e

# install.sh — install all OS-level deps for OpenCV-CUDA build on Ubuntu

sudo apt-get update

sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    ca-certificates \
    wget \
    unzip \
    yasm \
\
    # GUI / image I/O
    libgtk-3-dev \
    libcanberra-gtk3-module \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenexr-dev \
\
    # Video I/O (FFmpeg, V4L2)
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    libv4l-dev \
    v4l-utils \
    libxvidcore-dev \
    libx264-dev \
\
    # linear algebra & performance
    libtbb2 \
    libtbb-dev \
    libatlas-base-dev \
    gfortran \
\
    # Python support
    python3-dev \
    python3-numpy \
\
    # NVDEC/NVENC headers for FFmpeg & OpenCV cudacodec
    nv-codec-headers

sudo apt-get install -y nvidia-cuda-toolkit

echo
echo "=== Dependencies installed. Ready to run ./build.sh ==="
