python3 - <<EOF
import cv2
# Simple version print
print("cv2 version:", cv2.__version__)

# Full build info (optional)
# print(cv2.getBuildInformation())

print("CUDA-enabled GPUs:", cv2.cuda.getCudaEnabledDeviceCount())
EOF
