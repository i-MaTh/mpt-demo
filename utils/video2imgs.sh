#!/bin/bash

video=$1
dst=$2
extract_num_per_second=$3
prefix=$4

[[ $# -eq 0 ]]&& echo "./$0 video dst extract_num_per_second prefix" && exit 0

mkdir $dst

if [ "$prefix" = "" ];then
    avconv -i $video -r $extract_num_per_second -q:v 2 -f image2  $dst/%6d.jpg
else
    avconv -i $video -r $extract_num_per_second -q:v 2 -f image2  $dst/${prefix}_%6d.jpg
fi
