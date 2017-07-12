import os
import sys
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import csv
import pickle

def create_dataset(pairs, confs, reids, save_path):
	"""
	Inputs:
	- pairs: ndarray
	- confs: dictionary(key:img name, value: confidence of the detection)
	- reid_features: dictionary
	- save_path: write data to the file

	Outputs:
	- data: train set or test set
	"""
	data = []
	is_train = True
	#if len(pairs[0]) == 2: is_train = False

	for idx,p in enumerate(pairs):
		print "process: %d" % idx
		if len(p) == 2: is_train = False
		tmp = []
#print reids[p[0]], reids[p[1]]
#print type(reids[p[0]]), type(reids[p[1]])
#print confs[p[0]], confs[p[1]]
#print type(confs[p[0]]), type(confs[p[1]])

		min_conf = min(float(confs[p[0]]), float(confs[p[1]]))
		cosine_dis = cosine(reids[p[0]], reids[p[1]])
		tmp.append(min_conf)
		tmp.append(cosine_dis)
		tmp.append(min_conf * cosine_dis)
		tmp.append(min_conf ** 2)
		tmp.append(cosine_dis ** 2)
		if is_train:
			tmp.append(p[2])
		data.append(tmp)

	print "saving..."
	with open(save_path, 'wb') as outfile:
		writer = csv.writer(outfile)
		writer.writerows(data)
	

def main():
	pairs_path = sys.argv[1]
	confs_path = sys.argv[2]
	reids_path = sys.argv[3]
	save_path = sys.argv[4]

	# load data from the file respectively
	infile = open(pairs_path, 'r')
	pairs = csv.reader(infile) #
	confs = pickle.load(open(confs_path, 'r')) # dictionary
	reids = pickle.load(open(reids_path, 'r')) # dictionary
	
	create_dataset(pairs, confs, reids, save_path)



if __name__ == '__main__':
	assert len(sys.argv) > 1, 'please input (pairs_path, confs_path, reids_path, save_path)'
	main()







