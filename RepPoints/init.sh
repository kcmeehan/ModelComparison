#!/bin/bash

# download data
bash download_coco2017.sh

# download pre-trained reppoints model
bash download_model.sh

# copy files
cp -r src/reppoints_head/* mmdetection/mmdet/models/anchor_heads/
cp -r src/reppoints_detector/* mmdetection/mmdet/models/detectors/
cp -r src/reppoints_generator/* mmdetection/mmdet/core/anchor/
cp -r src/reppoints_assigner/* mmdetection/mmdet/core/bbox/assigners/

# install mmdetection
cd mmdetection
pip install git+https://github.com/open-mmlab/mmcv@v0.2.10 --user
pip install -v -e .
cd ..
