# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import os
import sys
import pickle
import errno
import argparse

sys.path.insert(0, '/home/huyangyang/dgd_person_reid/external/caffe/python')
import caffe
import cv2


def extractor(video_dir, detection_file, output_file, gpu_id, model, model_weights):
	# deploy pre-trained caffe model
	caffe.set_device(gpu_id)
	caffe.set_mode_gpu()
	net = caffe.Net(model, model_weights, caffe.TEST)

	'''
	try:
		os.makedirs(output_dir)
	except OSError as exception:
		if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
			pass
		else:
			raise ValueError('Failed to created output directory %s' % output_dir)
	'''
	#img_dir = os.path.join(video_dir, 'img1')
	img_dir = video_dir
	print img_dir
	img_filenames = {int(os.path.splitext(f)[0]): os.path.join(img_dir, f) 
					for f in os.listdir(img_dir)
					}
	#det_file = os.path.join(detection_dir, 'dets0704.txt')
	dets_in = np.loadtxt(detection_file, delimiter=',').astype(np.float)
	dets_out = []

	frame_indices = dets_in[:, 0].astype(np.int)
	min_frame_idx = frame_indices.min()
	max_frame_idx = frame_indices.max()

	print 'min_idx: %d, max_idx: %d' % (min_frame_idx, max_frame_idx)
	
	for idx in range(min_frame_idx, max_frame_idx + 1):
		mask = (frame_indices == idx)
		sub_dets = dets_in[mask]
		
		if idx not in img_filenames:
			print('WARNING could not find image for frame %d' % idx)
			continue
		
		batch_patch = np.zeros((len(sub_dets), 3, 144, 56), dtype=np.float32)
		bgr_img = cv2.imread(img_filenames[idx])
		for i in range(len(sub_dets)):
			# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
			patch = bgr_img.copy()[abs(int(sub_dets[i][3])) : abs(int(sub_dets[i][3] + sub_dets[i][5])) - 1, 
									abs(int(sub_dets[i][2])) : abs(int(sub_dets[i][2] + sub_dets[i][4])) - 1]
			patch = cv2.resize(patch, (56, 144))
			patch = np.transpose(patch, (2, 0, 1))
			norm_patch = np.array(patch, dtype=np.float32)
			norm_patch[0, :] = norm_patch[0, :] - 102.0
			norm_patch[1, :] = norm_patch[1, :] - 102.0
			norm_patch[2, :] = norm_patch[2, :] - 101.0
			batch_patch[i, :, :, :] = norm_patch

		net.blobs['data'].reshape(*(batch_patch.shape))
		net.blobs['data'].data[:, :, :, :] = batch_patch
		output = net.forward()
		try:
			feature = net.blobs['fc7_bn'].data[:]
		except:
			feature = net.blobs['fc7'].data[:]
	
		#merge dets and features
		dets_out += [np.r_[(d, f)] for d, f in zip(sub_dets, feature)]
		if idx % 100 == 0:
				print 'processed {}'.format(idx)

	#out_path = os.path.join(output_dir, 'demo0707.npy')
	np.save(output_file, np.asarray(dets_out), allow_pickle=False)
		

def parse_args():
	"""
	Parse command line arguments.
	"""
	parser = argparse.ArgumentParser(description="Re-ID feature extractor")
	parser.add_argument("-video_dir", help="Path to MOTChallenge directory (train or test)",required=True)
	parser.add_argument("-detection_file", help="Path to custom detections.", default=None)
	parser.add_argument("-output_file", help="Output file. Will be created if it does not exist.", default="detections")
	parser.add_argument("-gpu_id", help="gpu id", default=0, type=int)

	return parser.parse_args()



if __name__ == '__main__':
	#assert len(sys.argv) > 1, 'please use imgs_path save_path gpu_id'

	model = "/home/huyangyang/dgd_person_reid/external/exp/jstl/debing_with_search_all_long_iter/deploy.prototxt"
	model_weights = "/home/huyangyang/dgd_person_reid/external/exp/jstl/debing_with_search_all_long_iter/jstl_iter_150000.caffemodel"
	args = parse_args()
	
	extractor(args.video_dir, args.detection_file, args.output_file, int(args.gpu_id), model, model_weights)




