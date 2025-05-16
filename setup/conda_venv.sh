# create a venv
python3 -m venv opencv-cuda-env
source opencv-cuda-env/bin/activate

# install numpy (cv2 needs it), then point PYTHONPATH at your build
pip install numpy
export PYTHONPATH=/usr/local/python/cv2/python-3.*/:$PYTHONPATH

# test
python -c "import cv2; print(cv2.getBuildInformation())"
