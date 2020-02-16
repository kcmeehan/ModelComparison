#!/bin/bash

# Install PyTorch-YOLOv3 model
cd PyTorch-YOLOv3/
pip3 install -r requirements.txt

# Download weights
cd weights
bash download_weights.sh

# Install RepPoints
cd ../../RepPoints/
bash ./init.sh

