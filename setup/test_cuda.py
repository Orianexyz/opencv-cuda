import cv2
import numpy as np

cpu_mat = np.random.randint(0,256,(480,640),dtype=np.uint8)
gpu_mat = cv2.cuda_GpuMat()
gpu_mat.upload(cpu_mat)

# box filter (as before)
box = cv2.cuda.createBoxFilter(cv2.CV_8UC1, cv2.CV_8UC1, (15,15))
gpu_box = box.apply(gpu_mat)
res_box = gpu_box.download()

# gaussian filter — note the two zeros for sigmaX and sigmaY
# positional only:
gauss = cv2.cuda.createGaussianFilter(
    cv2.CV_8UC1,        # srcType
    cv2.CV_8UC1,        # dstType
    (15,15),            # ksize
    0,                  # sigmaX  (0→auto from ksize)
    0,                  # sigmaY  (0→auto from ksize)
    cv2.BORDER_DEFAULT  # borderMode – no keyword here!
)

gpu_gauss = gauss.apply(gpu_mat)
res_gauss = gpu_gauss.download()

print("Box:",  res_box.shape,  "Gaussian:", res_gauss.shape)
