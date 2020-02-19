#!/bin/bash

curl -c /tmp/cookies "https://drive.google.com/uc?export=download&confirm=SiR0&id=1nr9gcVWxzeakbfPC6ON9yvKOuLzj_RrJ" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > reppoints_moment_x101_dcn_fpn_2x_mt.pth 
mkdir -p checkpoints/
mv reppoints_moment_x101_dcn_fpn_2x_mt.pth checkpoints/
