# OpenCV-CUDA Video Processing Tools

A collection of tools for GPU-accelerated video processing using OpenCV with CUDA support. This repository includes scripts for building OpenCV with CUDA support, environment setup, and example applications for video processing.

## Features

- GPU-accelerated video processing using OpenCV-CUDA
- Zero-copy frame processing with NVDEC
- Batched Sobel magnitude computation on GPU
- CUDA streams and events for optimized performance
- Optional Cupy kernels for median-border heuristic
- Two-stage pipeline: analyze first, encode later

## Prerequisites

- Ubuntu Linux
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- Python 3.x
- CMake
- Git

## Installation

1. Clone this repository:

```bash
git clone htts://github.com/Orianexyz/opencv-cuda
cd opencv-cuda
```

2. Run the installation script to install system dependencies:

```bash
./install.sh
```

3. Build opencv-cuda with NVIDIA GPU support:

```bash
./build.sh
```

4. Set up the environment:

```bash
source env_open_cuda.sh
```

5. (Optional) Create a Conda virtual environment:

```bash
./conda_venv.sh
```

## Usage

### Testing CUDA Support

To verify CUDA support and basic functionality:

```bash
python test_cuda.py
```

This script demonstrates basic CUDA operations including:

- GPU matrix operations
- Box filtering
- Gaussian filtering

### Video Cropping Tool

The `cropper.py` script provides GPU-accelerated video cropping capabilities:

```bash
python cropper.py -i <input_directory> -o <output_directory> [options]
```

Options:

- `-i, --input`: Input directory containing MP4 files (default: "../videos")
- `-o, --output`: Output directory for cropped videos (default: "../output/cropped_gpu")
- `--fps`: Sample rate for analysis (default: 1)
- `--cpu`: Force CPU processing even if CUDA is available

## Environment Variables

The `env_open_cuda.sh` script sets up the following environment variables:

- `OpenCV_DIR`: Path to OpenCV CMake config files
- `PKG_CONFIG_PATH`: Path for pkg-config
- `PYTHONPATH`: Python binding location

## Building from Source

To build OpenCV with CUDA support:

```bash
./build.sh
```

## Dependencies

This project relies on several key technologies:

- [OpenCV](https://opencv.org/) - Computer vision library
- [OpenCV Contrib](https://github.com/opencv/opencv_contrib) - Additional OpenCV modules
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk) - Hardware-accelerated video encoding/decoding
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - NVIDIA's parallel computing platform
- [FFmpeg](https://ffmpeg.org/) - Multimedia framework
- [Cupy](https://cupy.dev/) - GPU-accelerated NumPy implementation

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## References

1. OpenCV Documentation: <https://docs.opencv.org/>
2. NVIDIA CUDA Documentation: <https://docs.nvidia.com/cuda/>
3. NVIDIA Video Codec SDK Documentation: <https://developer.nvidia.com/nvidia-video-codec-sdk/documentation>
4. FFmpeg Documentation: <https://ffmpeg.org/documentation.html>
5. Cupy Documentation: <https://docs.cupy.dev/>

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenCV team for the excellent computer vision library
- NVIDIA for CUDA and Video Codec SDK
- FFmpeg project for video processing capabilities
