#!/bin/sh

sudo apt update
sudo apt install python3-pip -y
sudo apt install software-properties-common -y
sudo apt install nvidia-cuda-toolkit -y
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
