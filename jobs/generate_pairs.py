import os
import sys
import numpy as np
import csv

def create_train_pairs(path, save_path, window_size = 5):
	imgs = []
	with open(path, 'r') as infile:
		for idx, line in enumerate(infile):
			imgs.append('/'.join(line.strip('\n').split('/')[-2:]))
			#if idx > 20: break # testing
	#print imgs
	
	num_imgs = len(imgs)
	pairs = []
	for i in range(num_imgs - 1):
		s_frame_id = imgs[i].split('_')[0].split('/')[-1] # string type
		s_det_id = imgs[i].split('_')[1].split('.')[0] # string type
		for j in range(i + 1, num_imgs):
			t_frame_id = imgs[j].split('_')[0].split('/')[-1] # string type
			t_det_id = imgs[j].split('_')[1].split('.')[0] # string type
			if abs(int(t_frame_id) - int(s_frame_id)) > window_size: break
			tmp = []
			tmp.append(imgs[i])
			tmp.append(imgs[j])
			if int(t_det_id) == int(s_det_id):
				tmp.append(1)
			#else:
			#	tmp.append(0)
			elif abs(int(t_frame_id) - int(s_frame_id)) > 2:
				tmp.append(0)
			else:
				continue
			pairs.append(tmp)

	with open(save_path, 'wb') as outfile:
		writer = csv.writer(outfile)
		writer.writerows(pairs)
	
def create_test_pairs(path, save_path, window_size = 5):
	imgs = []
	with open(path, 'r') as infile:
		for idx, line in enumerate(infile):
			imgs.append('/'.join(line.strip('\n').split('/')[-2:]))
			
	num_imgs = len(imgs)
	pairs = []
	for i in range(num_imgs - 1):
		s_frame_id = imgs[i].split('/')[-1].split('_')[1] # data format: demo_crop-1/reid_000001_0.jpg 
		for j in range(i + 1, num_imgs):
			t_frame_id = imgs[j].split('/')[-1].split('_')[1]
			if abs(int(t_frame_id) - int(s_frame_id)) > window_size: break
			tmp = []
			tmp.append(imgs[i])
			tmp.append(imgs[j])
			pairs.append(tmp)

	with open(save_path, 'wb') as outfile:
		writer = csv.writer(outfile)
		writer.writerows(pairs)


def create_test_pairs2(path, save_path, window_size=5):
	pass


if __name__ == '__main__':
	assert len(sys.argv) > 1, 'please input (path, save_path, window_size)'
	path = sys.argv[1]
	save_path = sys.argv[2]
	window_size = int(sys.argv[3])
	create_train_pairs(path, save_path, window_size)
	#create_test_pairs(path, save_path, window_size)


