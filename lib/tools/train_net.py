from fast_rcnn.train import train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.dataset import dataset
import caffe
import argparse
import pprint
import numpy as np
import sys


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
	parser.add_argument('--gpu', dest='gpu_id',
						help='GPU device id to use [0]',
						default=0, type=int)
	parser.add_argument('--solver', dest='solver',
						help='solver prototxt',
						default=None, type=str)
	parser.add_argument('--iters', dest='max_iters',
						help='number of iterations to train',
						default=40000, type=int)
	parser.add_argument('--weights', dest='pretrained_model',
						help='initialize with pretrained model weights',
						default=None, type=str)
	parser.add_argument('--snapshot', dest='snapshot',
						help='initialize with solverstate',
						default=None, type=str)
	parser.add_argument('--cfg', dest='cfg_file',
						help='optional config file',
						default=None, type=str)
	parser.add_argument('--cache', dest='cache',
						help='dataset cache path', type=str)
	parser.add_argument('--img_label_size', dest='img_label_size',
						help='img label path', type=str)
	parser.add_argument('--classes', dest='classes',
						help='classes name joined by ,', type=str)
	parser.add_argument('--rand', dest='randomize',
						help='randomize (do not use a fixed seed)',
						action='store_true')
	parser.add_argument('--set', dest='set_cfgs',
						help='set config keys', default=None,
						nargs=argparse.REMAINDER)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args


def get_roidb(img_label_size, cache):
	ds = dataset(img_label_size, cache)
	roidb = ds.get_roidb()

	return roidb


if __name__ == '__main__':
	args = parse_args()

	print('Called with args:')
	print(args)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	cfg.GPU_ID = args.gpu_id

	print('Using config:')
	pprint.pprint(cfg)

	if not args.randomize:
		np.random.seed(cfg.RNG_SEED)
		caffe.set_random_seed(cfg.RNG_SEED)

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)

	ds = dataset(args.img_label_size, args.cache)
	roidb = ds.get_roidb()
	print '{:d} roidb entries'.format(len(roidb))

	train_net(args.solver, roidb, solverstate=args.snapshot, pretrained_model=args.pretrained_model,
			  max_iters=args.max_iters)
