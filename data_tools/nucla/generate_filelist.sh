#! /usr/bin/bash env

cd ../../
PYTHONPATH=. python data_tools/build_file_list.py nucla /home/asilla/vuba/data/nucla_extracted/ --level 2 --format rawframes --shuffle
echo "Filelist for rawframes generated."

PYTHONPATH=. python data_tools/build_file_list.py nucla data/nucla/videos/ --level 2 --format videos --shuffle
echo "Filelist for videos generated."

cd data_tools/nucla/