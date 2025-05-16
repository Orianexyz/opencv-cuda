python3 - <<EOF
import cv2
print("OpenCV version:", cv2.__version__)
print("CUDA GPUs detected:", cv2.cuda.getCudaEnabledDeviceCount())
# Optional: print full build info to verify WITH_NVCUVID and CUDA flags
info = cv2.getBuildInformation().splitlines()
for line in info:
    if "CUDA" in line or "NVCUVID" in line or "cudacodec" in line:
        print(line)
EOF
