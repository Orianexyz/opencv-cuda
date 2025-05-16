python3 - <<EOF
import cv2

video_path = "assets/C_P0qe2pzqZ.mp4"

# 0) Probe the resolution once on the CPU
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video_path}")
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"Video resolution: {width}×{height}")

# 1) Open the CUDA VideoReader
reader = cv2.cudacodec.createVideoReader(video_path)

# 2) Your desired ROI
x, y, w, h = 100, 50, 640, 360
# Clamp it so it never runs past the frame edges:
x_end = min(x + w, width)
y_end = min(y + h, height)
if x >= width or y >= height:
    raise ValueError("Your ROI start is outside the frame")

frame_idx = 0
while True:
    ret, gpu_frame = reader.nextFrame()
    if not ret:
        break

    # 3) Crop on GPU
    gpu_roi = gpu_frame.rowRange(y, y_end).colRange(x, x_end)

    # 4) Download to CPU only when needed
    cpu_roi = gpu_roi.download()

    frame_idx += 1
    # 5) Save every 100th cropped frame
    if frame_idx % 100 == 0:
        cv2.imwrite(f"crop_{frame_idx:06d}.png", cpu_roi)
EOF
