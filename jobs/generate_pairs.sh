#!/bin/bash
[[ $# -eq 0 ]] && echo "$0 src, save_path, window_size" && exit 0
set -e
set -x

src=$1
save_path=$2
window_size=$3

find $src/ -name *jpg |sort > ./img_list.txt

python generate_pairs.py img_list.txt $save_path $window_size


