#! /usr/bin/bash env

cd ../
python build_rawframes.py /media/data/vuba/tsn/mmaction/data/ucf101/videos/ /home/asilla/vuba/data/ucf101_extracted/ --level 2  --ext avi
echo "Raw frames (RGB only) generated for train and val set"

cd ucf101/