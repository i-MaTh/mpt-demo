resnet_type=$1
data_path=$2
[[ $# -eq 0 ]] && echo "$0 resnet_type[50/101/152] data_path" && exit 0

if [ $resnet_type -eq 50 ];then
	python ../../../lib/tools/create_net.py  $data_path all resnet$resnet_type  res4f  res5c
	python ../../../lib/tools/resnet_rfcn_prototxt_process.py config.yaml train.prototxt 50 train
	python ../../../lib/tools/resnet_rfcn_prototxt_process.py config.yaml deploy.prototxt 50 test
fi

if [ $resnet_type -eq 101 ];then
	python ../../../lib/tools/create_net.py  $data_path all resnet$resnet_type  res4b22  res5c
	python ../../../lib/tools/resnet_rfcn_prototxt_process.py config.yaml train.prototxt 101 train
	python ../../../lib/tools/resnet_rfcn_prototxt_process.py config.yaml deploy.prototxt  101 test
fi

if [ $resnet_type -eq 152 ];then
	echo "not implemented!"
fi
