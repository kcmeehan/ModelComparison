# SmartDetect
TODO:
-write rest of installation instructions and test on independent system
-add links/credits to respective repos

A user-friendly tool for performing object detection and comparing model performance on an image data set.

## 1. Installation

Clone the repository to your working space.

### A. Setup the Yolov3 code

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
bash download_weights.sh
```

## 2. Usage

To run the streamlit app, go into the RepPoints directory, and run the streamlit command:

```bash
cd Reppoints
streamlit run mmdetection/smart_detect.py --server.port 5000
```

Then pull up the ip address at port 5000 in a web browser. The address should be something like: 
**http://<your_ip_address>:5000**


