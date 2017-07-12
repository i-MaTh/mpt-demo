#!/bin/bash

[[ $# -eq 0 ]] && echo "$0 img_label cache_name[imgwithallbox_pd, voc,...] resume_model model_solverstate_flag[m/s] gpu_id" && exit 0

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHONPATH=$HOME/tools:../../../lib:../../../caffe/python:$PYTHONPATH

img_label=$1
cache_name=$2
resume_model=$3
model_solverstate_flag=$4
GPU_ID=$5

ITERS=6000000
LOG="log_`date +'%Y-%m-%d_%H-%M-%S'`"


if [ $model_solverstate_flag = "m" ];then
	python ../../../lib/tools/train_net.py --gpu ${GPU_ID} \
	  --solver solver.prototxt \
	  --weights  $resume_model \
	  --cache ../../../data/cache/${cache_name}_roidb.pkl \
	  --img_label ${img_label} \
	  --iters ${ITERS} \
	  --cfg config.yaml  
else
	python ../../../lib/tools/train_net.py --gpu ${GPU_ID} \
	  --solver solver.prototxt \
	  --snapshot $resume_model \
	  --cache ../../../data/cache/${cache_name}_roidb.pkl \
	  --img_label ${img_label} \
	  --iters ${ITERS} \
	  --cfg config.yaml  
fi


