#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << "\n";
    int n = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA-enabled GPUs: " << n << "\n";
    return 0;
}
