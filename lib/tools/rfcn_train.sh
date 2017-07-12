#!/bin/bash

[[ $# -eq 0 ]] && echo "$0 img_label_size cache_name[imgwithallbox_pd, voc,...] gpu_id" && exit 0

set -x
set -e

root_dir=../../..
export PYTHONUNBUFFERED="True"
export PYTHONPATH=$HOME/tools:$root_dir/lib:$root_dir/lib/tools:$root_dir/caffe/python:$PYTHONPATH

img_label_size=$1
cache_name=$2
GPU_ID=$3

ITERS=2000000
LOG="log_`date +'%Y-%m-%d_%H-%M-%S'`"

python $root_dir/lib/tools/train_net.py --gpu ${GPU_ID} \
  --solver solver.prototxt \
  --cache $root_dir/data/cache/${cache_name}_roidb.pkl \
  --img_label_size ${img_label_size} \
  --iters ${ITERS} \
  --cfg config.yaml  \
