#!/bin/bash

[[ $# -eq 0 ]] && echo "$0 img_path caffemodel gpu_id [--vis] [--save_dir]" && exit 0

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHONPATH=$HOME/tools:../../../lib:../../../caffe/python:$PYTHONPATH

img_label=$1
caffemodel=$2
GPU_ID=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

python ../../../lib/tools/test_net.py --gpu ${GPU_ID} \
  --deploy deploy.prototxt \
  --model ${caffemodel} \
  --img_label $img_label \
  --cfg config.yaml  \
  ${EXTRA_ARGS}
