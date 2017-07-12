import numpy as np
import os
import sys
import pickle as pl

sys.path.insert(0, '/home/huyangyang/dgd_person_reid/external/caffe/python')
import caffe
import cv2


if len(sys.argv) == 1:
	print "python .. crop_imgs_dir save_features_map_pkl gpu_id"
	sys.exit(0)

crop_imgs_dir = sys.argv[1]
save_features_map_pkl = sys.argv[2]
gpu_id= int(sys.argv[3])

model_def = "/home/huyangyang/dgd_person_reid/external/exp/jstl/debing_with_search_all_long_iter/deploy.prototxt"
model_weights = "/home/huyangyang/dgd_person_reid/external/exp/jstl/debing_with_search_all_long_iter/jstl_iter_150000.caffemodel"

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

net = caffe.Net(model_def, model_weights, caffe.TEST)

features_map = {}
with open('img_path_with_label', 'r') as f:
	lines = f.readlines()

img_id = 0
for line in lines:
	ori_img_path, _ = line.split()
	img_name = ori_img_path.strip().split('/')[-1].strip()[:-4]
	img_path = crop_imgs_dir+"/" + img_name
	lst = os.listdir(img_path)
	if len(lst)==0:
		continue
	batch_data = np.zeros((len(lst), 3, 144, 56), dtype=np.float32)
	for idx in range(len(lst)):
		crop_img_name = "{}.jpg".format(idx)
		crop_img_path = "{}/{}".format(img_path, crop_img_name)
		img = cv2.imread(crop_img_path)
		img = cv2.resize(img, (56, 144))
		img = np.transpose(img, (2, 0, 1))
		norm_img = np.array(img, dtype=np.float32)
		norm_img[0, :] = norm_img[0, :] - 102.0
		norm_img[1, :] = norm_img[1, :] - 102.0
		norm_img[2, :] = norm_img[2, :] - 101.0
		batch_data[idx, :, :, :] = norm_img

	net.blobs['data'].reshape(*(batch_data.shape))
	net.blobs['data'].data[:, :, :, :] = batch_data
	output = net.forward()
	try:
		feature = net.blobs['fc7_bn'].data[:]
	except:
		feature = net.blobs['fc7'].data[:]
	features_map[img_id] = {'feature': feature.copy()}
	img_id += 1
	if img_id % 100 == 0:
		print 'processed {}'.format(img_id)

with open(save_features_map_pkl, 'w') as f:
	pl.dump(features_map, f)
