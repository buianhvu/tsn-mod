#! /usr/bin/bash env

cd ../
python build_rawframes.py /media/data/vuba/tsn/mmaction/data/nucla/videos/ /home/asilla/vuba/data/nucla_extracted/ --level 2  --ext avi
echo "Raw frames (RGB only) generated for train and val set"

cd nucla/