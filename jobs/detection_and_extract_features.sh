#/bin/bash
[[ $# -eq 0 ]] && echo "$0 src gpu_id" && exit 0
set -e
set -x

src=$1
gpu_id=$2

export PYTHONPATH=$PYTHONPATH:$PWD

find `pwd`/$src/img1/ -name *jpg |sort > $src/imgs_list.txt

#echo `pwd`/$src/imgs/
../../../lib/tools/rfcn_test.sh  $src/imgs_list.txt unnormalized_iter_1030000.caffemodel $gpu_id --save_path=$src/dets.txt
#../../../lib/tools/rfcn_test.sh  $src/imgs_list.txt unnormalized_iter_1030000.caffemodel $gpu_id --vis

#mot-16-challenge/train/MOT16_09
#cd $src

#python ../convert_img_list_to_img_label.py img_path.txt > img_path_with_label.txt
#python ../smooth_trace.py img_path_with_label.txt
#python ../crop_person.py  smoothed_bbs.pkl crop_imgs 
#python ../extract_features.py crop_imgs persons_feature.pkl $gpu_id

