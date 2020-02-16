# SmartDetect
TODO:
-write rest of installation instructions and test on independent system

A user-friendly tool for performing object detection and comparing model performance on an image data set.  
This repo uses YOLOv3 and RepPoints models, implemented in pytorch.
The YOLOv3 implementation uses code from https://github.com/eriklindernoren/PyTorch-YOLOv3, a minimal PyTorch implementation of YOLOv3, a model developed by Joseph Redmon and Ali Farhadi (https://pjreddie.com/darknet/yolo/). The RepPoints model uses code from  https://github.com/microsoft/RepPoints developed by Ze Yang, Shaohui Liu, Han Hu, Liwei Wang, and Stephen Lin from Microsoft (https://arxiv.org/abs/1904.11490).
 

## 1. Installation

### A. Prerequisites: 
- At least one gpu
- Setup anaconda environment with python 3:
```bash
conda create -n smartdetectenv python=3.7.4
conda activate smartdetectenv
```
- [Install PyTorch 1.0+](https://pytorch.org/get-started/locally/)
- CUDA 9.0+

### B. Setup:

Clone the repository to your working space:
```bash
git clone https://github.com/kcmeehan/SmartDetect.git
```

**Install requirements**
```bash
cd PyTorch-YOLOv3/
pip3 install -r requirements.txt
```

**Download pre-trained weights**
```bash
cd weights
bash download_weights.sh
```

### B. Setup the RepPoints code

```bash
cd RepPoints
sh ./init.sh
```

## 2. Usage

To run the streamlit app, go into the RepPoints directory, and run the streamlit command:

```bash
cd Reppoints
streamlit run mmdetection/smart_detect.py --server.port 5000
```

Then pull up the ip address at port 5000 in a web browser. The address should be something like: 
**http://<your_ip_address>:5000**
