#!/usr/bin/env bash
set -euo pipefail
LOG_FILE="install.log"
rm -f install.log

{
echo "=== Step 1: Install NVIDIA GPG Key ==="
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-47045A0D-keyring.gpg /usr/share/keyrings/
sudo apt-key add /var/cuda-repo-ubuntu2404-12-8-local/cuda-47045A0D-keyring.gpg

echo "=== Step 2: Add CUDA Repo Pin ==="
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

echo "=== Step 3: Download and Install CUDA 12.8 Local Repo ==="
wget --progress=dot:giga -q \
  https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb \
  -O cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb

echo "=== Step 4: Update APT and Install CUDA Toolkit ==="
sudo apt update
sudo apt install -y cuda-toolkit-12-8

echo "=== Step 5: Add Environment Variables to ~/.bashrc ==="
if ! grep -q "export CUDA_HOME=/usr/local/cuda-12.8" ~/.bashrc; then
  echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
  echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

echo "=== Step 6: Source ~/.bashrc ==="
source ~/.bashrc

echo "=== Installation Complete ==="
nvidia-smi
nvcc --version

# After downloading cuDNN.tar
tar -xzf cudnn-linux-x86_64-*-cuda12.x.tgz
sudo cp cuda/include/cudnn*.h $CUDA_HOME/include
sudo cp cuda/lib/libcudnn* $CUDA_HOME/lib64
sudo chmod a+r $CUDA_HOME/include/cudnn*.h $CUDA_HOME/lib64/libcudnn*

} 2>&1 | tee -a "$LOG_FILE"
