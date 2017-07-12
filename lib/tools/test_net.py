from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
import caffe
import argparse
import pprint
import sys


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
						default=0, type=int)
	parser.add_argument('--deploy', dest='deploy',
						help='prototxt file defining the network',
						default=None, type=str)
	parser.add_argument('--model', dest='model',
						help='model to test',
						default=None, type=str)
	parser.add_argument('--cfg', dest='cfg',
						help='optional config file', default=None, type=str)
	parser.add_argument('--img_label', dest='img_label',
						help='dataset to test', type=str)
	parser.add_argument('--vis', dest='vis', help='visualize detections',
						action='store_true')
	parser.add_argument('--save_dir', dest='save_dir', help='preds save dir',
						default=None, type=str)
	parser.add_argument('--save_path', dest='save_path', default=None, type=str, help='save dets in to file')
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()

	print('Called with args:')
	print(args)

	if args.cfg is not None:
		cfg_from_file(args.cfg)

	cfg.GPU_ID = args.gpu_id

	print('Using config:')
	pprint.pprint(cfg)

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)
	net = caffe.Net(args.deploy, args.model, caffe.TEST)

	import os

	# if args.save_dir:
	# 	if os.path.exists(args.save_dir):
	# 		print "save_dir already exists!"
	# 		sys.exit(0)
	# 	os.mkdir(args.save_dir)

	test_net(net, args.img_label, vis=args.vis, save_dir=args.save_dir, save_path=args.save_path)
