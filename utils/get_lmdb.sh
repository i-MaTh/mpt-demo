#!/bin/bash

[ $# -eq 0 ]&& echo "$0 img_label dst gt_type [ratio=0.1]" && exit 0

img_label=$1
dst=$2
gt_type=$3

if [ $# -eq 3 ];then
	ratio=0.1
else
	ratio=$4
fi

[ -e $dst ] && echo "dst dir exist!" && exit 0
mkdir $dst
mkdir $dst/info

shuf $img_label >${img_label}_tmp
mv ${img_label}_tmp $img_label

python ~/tools/get_train_test.py $img_label $ratio

mkdir $dst/img_label_lmdb

for data in train  val;do
{
	cat ${data}.txt |awk '{print $1, -1}' >${data}_img_tmp
	cat ${data}.txt |awk '{print $2, -1}' >${data}_label_tmp

	~/caffe/build/tools/convert_imageset "" ${data}_img_tmp  $dst/img_label_lmdb/${data}_img_lmdb

	python ~/tools/lmdb_lib.py ${data}_label_tmp $dst/img_label_lmdb/${data}_label_lmdb $gt_type
	mv ${data}.txt  $dst/info
	mv ${data}_img_tmp  $dst/info
	mv ${data}_label_tmp  $dst/info
}
done
wait 
echo "done!"




