#!/bin/bash

[[ $# -eq 0 ]] && echo "./$0 img_label resize_height  resize_width" && exit 0

img_label=$1
resize_height=$2
resize_width=$3

cat $img_label |awk '{print $1, 1}' > filelist

python ~/tools/get_train_test.py filelist 0.1

convert_imageset "" train.txt train_lmdb -resize_height=$resize_height  -resize_width=$resize_width -shuffle
convert_imageset "" val.txt val_lmdb -resize_height=$resize_height  -resize_width=$resize_width -shuffle
mkdir lmdb
mv train_lmdb lmdb
mv val_lmdb lmdb

mkdir info
mv train.txt info
mv val.txt info
mv info lmdb

rm filelist



