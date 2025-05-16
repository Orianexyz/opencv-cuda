#!/usr/bin/env bash
set -euo pipefail

sudo add-apt-repository ppa:graphics-drivers/ppa 
sudo apt update
sudo ubuntu-drivers autoinstall   # installs the recommended NVIDIA driver
# or specify a version:
# sudo apt install nvidia-driver-535
reboot
