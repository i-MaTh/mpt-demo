import numpy as np
from scipy.spatial.distance import cosine, euclidean, cdist
import os, sys
import argparse
from datetime import datetime
import pickle
from sklearn.preprocessing import normalize
from collections import OrderedDict


def reassign_id(trk_reid_file1, trk_reid_file2):
	trk_reid1 = np.load(trk_reid_file1)
	#print 'trk_reid shape: {}'.format(trk_reid1.shape)
	trk_reid2 = np.load(trk_reid_file2)
	#print 'trk_reid shape: {}'.format(trk_reid2.shape)
	indices_trk1 = trk_reid1[:,1].astype(np.int)
	indices_trk2 = trk_reid2[:,1].astype(np.int)
	
	unique_id_trk1 = set()
	for idx in indices_trk1:
		unique_id_trk1.add(idx)

	unique_id_trk2 = set()
	for idx in indices_trk2:
		unique_id_trk2.add(idx)

	print 'number of unique id in trk1: %d' % len(unique_id_trk1)
	print 'number of unique id in trk2: %d' % len(unique_id_trk2)

	id_set1 = {k:(v+1) for v, k in enumerate(unique_id_trk1)}
	id_set2 = {k:(v+1+len(unique_id_trk1)) for v, k in enumerate(unique_id_trk2)}
	
	print id_set1
	print id_set2
	
	indices_trk1 = [id_set1[idx] for idx in indices_trk1]
	trk_reid1[:, 1] = indices_trk1
	indices_trk2 = [id_set2[idx] for idx in indices_trk2]
	trk_reid2[:, 1] = indices_trk2

	np.save(trk_reid_file1, trk_reid1)
	np.save(trk_reid_file2, trk_reid2)


def merge_camera(trk_reid_file1, trk_reid_file2, threshold=0.14):
	trk_reid1 = np.load(trk_reid_file1)
	trk_reid2 = np.load(trk_reid_file2)
	indices_trk1 = trk_reid1[:,1].astype(np.int)
	indices_trk2 = trk_reid2[:,1].astype(np.int)

	unique_id_trk1 = set()
	for idx in indices_trk1:
		unique_id_trk1.add(idx)
	print unique_id_trk1

	unique_id_trk2 = set()
	for idx in indices_trk2:
		unique_id_trk2.add(idx)
	print unique_id_trk2

	#print unique_id_trk1
	print 'number of unique id in trk1: %d' % len(unique_id_trk1)
	print 'number of unique id in trk2: %d' % len(unique_id_trk2)
	

	reid_features1 = trk_reid1[:, 10:]
	reid_features2 = trk_reid2[:, 10:]
	distances = OrderedDict()

	for i in unique_id_trk1:
		mask_i = (indices_trk1 == i)
		reid_i = reid_features1[mask_i]
		for j in unique_id_trk2:
			mask_j = (indices_trk2 == j)
			reid_j = reid_features2[mask_j]
			dis = cdist(reid_i, reid_j, metric = 'cosine')

			key = '1-%d_2-%d' % (i, j)
			distances[key] = dis.min()

	#pickle.dump(distances, open('two_camera_distance.pkl', 'wb'))
	#print distances	
	cands = [it for it in distances.items() if it[1] < threshold]
	print cands
	
	for c in cands:
		tmp = c[0].split('_')
		i = map(int, tmp[0].split('-'))[1]
		j = map(int, tmp[1].split('-'))[1]
		print 'i: %d, j: %d' % (i, j)
		indices_trk2 = [i if idx == j else idx for idx in indices_trk2]
	
	trk_reid2[:, 1] = indices_trk2
	
	np.save(trk_reid_file2, trk_reid2)
	

def merge_component(trks_file, dis_file, output_file=None, threshold=0.1):
	dis = pickle.load(open(dis_file, 'r'))
	#trks = np.loadtxt(trks_file, delimiter=',').astype(np.int)
	trk_reid = np.load(trks_file)
	trks = trk_reid[:, :10].astype(np.int)
	
	max_trk_id = trks[:,1].max()
	min_trk_id = trks[:,1].min()
	max_frame_id = trks[:,0].max()
	min_frame_id = trks[:,0].min()
	print 'max_frame_id: %d, min_frame_id: %d' % (max_frame_id, min_frame_id)
	print 'max_tracker_id: %d, min_tracker_id: %d' % (max_trk_id, min_trk_id)
	trk_indices = trks[:,1]
	
	print 'number of dis < {}: {}'.format(threshold , sum(np.asarray(dis.values()) < threshold))
	cands = [it for it in dis.items() if it[1] < threshold]
	#print cands		    
	for c in cands:
		tmp = map(int, c[0].split('_'))
		trk_indices = [tmp[0] if idx == tmp[1] else idx for idx in trk_indices]
									        
	#trks[:,1] = trk_indices
	#np.savetxt(output_file, trks, fmt='%.2f', delimiter=',')
	trk_reid[:,1] = trk_indices
	np.save(trks_file, trk_reid)


def compute_similarity(trk_reid_file, norm=False):
	'''
	trks = np.loadtxt(trk_file, delimiter=',')
	num_per_trk = {i:len(trks[trks[:,1].astype(np.int) == i]) for i in range(min_trk_id, max_trk_id+1)}
	#print num_per_trk
	'''
	trk_reid = np.load(trk_reid_file) # [frame_id, tracker_id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1, reid_features]
	trks = trk_reid[:,:10].astype(np.int)
	#print trks.shape
	reid_features = trk_reid[:,10:]
	if norm:
		reid_features = normalize(reid_features) # normalize

	#print reid_features.shape
	max_trk_id = trks[:,1].max()
	min_trk_id = trks[:,1].min()
	max_frame_id = trk_reid[:,0].max()
	min_frame_id = trk_reid[:,0].min()
	print 'max_frame_id: %d, min_frame_id: %d' % (max_frame_id, min_frame_id)
	print 'max_tracker_id: %d, min_tracker_id: %d' % (max_trk_id, min_trk_id)
	
	trk_indices = trks[:,1]
	distances = OrderedDict()

	start_time = datetime.now()

	for i in range(min_trk_id, max_trk_id):
		mask_i = (trk_indices == i)
		reid_i = reid_features[mask_i]
		if reid_i.shape[0] == 0: continue
		for j in range(i+1, max_trk_id+1):
			mask_j = (trk_indices == j)
			reid_j = reid_features[mask_j]
			if reid_j.shape[0] == 0: continue
			dis = cdist(reid_i, reid_j, metric = 'cosine')
			key = '%d_%d' % (i, j)
			distances[key] = dis.min()

		'''
		values = []
		for j in range(min_trk_id, max_trk_id+1):
			mask_j = (trk_indices == j)
			reid_j = reid_features[mask_j]
			if reid_j.shape[0] ==0: continue
			
			if i != j:
				dis = cdist(reid_i, reid_j, metric = 'cosine')
				#v = dis.max()
				v = dis.min()
			else:
				v = 0
			values.append(v)
		distances[i] = values
		'''
	
	outpath = os.path.splitext(trk_reid_file)[0] + '_distance_lookup.pkl'
	pickle.dump(distances, open(outpath, 'wb'))

	end_time = datetime.now()
	duration = end_time - start_time
	print 'duration time: %s' % duration


def clean_detection(det_reid_file):
	det_reid = np.load(det_reid_file)
	dets_in = det_reid[:, :10].astype(np.int)
	dets_out = []
	for i in range(dets_in.shape[0]):
		bb_left = dets_in[i][2]
		bb_top = dets_in[i][3]
		if bb_left < 30 or bb_left > 1500 or bb_top < 170 or bb_top > 810: continue
		dets_out.append(det_reid[i])

	np.save(det_reid_file, dets_out)



def clean_tracker(trk_reid_file, threshold=20):
	trk_reid = np.load(trk_reid_file)
	trks = trk_reid[:,:10].astype(np.int)
	min_trk_id = trks[:,1].min()
	max_trk_id = trks[:,1].max()
	s = set()
	for i in trks[:,1]:
		s.add(i)
	print s
	num_per_trk = {i:len(trks[trks[:,1] == i]) for i in range(min_trk_id, max_trk_id+1)}
	print num_per_trk
	print sorted(num_per_trk.values())
	
			
	new_trks = []
	for i in range(trks.shape[0]):
		trk_id = trks[i][1]
		bb_left = trks[i][2]
		bb_top = trks[i][3]
		if num_per_trk[trk_id] < threshold: continue
		#if bb_left > 1700 or bb_top > 800: continue
		new_trks.append(trk_reid[i])

	#_tuples = os.path.splitext(trk_reid_file)
	#outpath = _tuples[0] + '_new' + _tuples[1]
	outpath = trk_reid_file
	np.save(outpath, new_trks)


def parse_args():
	""" 
	Parse command line arguments.
	"""
	parser = argparse.ArgumentParser(description="Post-Processing Tracker Result")
	parser.add_argument("-dis_file", help="Path to tracker file.", default=None)
	parser.add_argument("-trk_reid_file", help="Path to reid of tracker file.", default=None)
	parser.add_argument("-output_file", help="Path to output file.", default=None)
	return parser.parse_args()

def test(path, threshold=0.1):
	#path = './demo_test_data/distances.pkl'
	
	dis = pickle.load(open(path, 'r'))
	# dis.items() type is list	
	'''
	trk_indices = {i:dis.keys()[i] for i in range(len(dis))}	
	matches = {}
	for k in dis:
		#min_dis = sorted(dis[k])
		min_idx = np.argsort(dis[k])[1]
		kt = trk_indices[min_idx]
		key = '%d_%d' % (k, kt)
		value = dis[k][min_idx]
		matches[key] = value
	'''
	matches = dis

	max_dis = max(matches.values())
	min_dis = min(matches.values())
	mean_dis = np.mean(matches.values())
	median_dis = np.median(matches.values())
	print 'max_dis: {}, min_dis: {}'.format(max_dis, min_dis)
	print 'mean_dis: {}, median_dis: {}'.format(mean_dis, median_dis)
	print 'number of dis < {}: {}'.format(threshold , sum(np.asarray(matches.values()) < threshold))
	print dis	
	candidates = [it for it in matches.items() if it[1] < threshold]
	
	print candidates
	#trks_path = './demo_test_data/tracks0707_2.txt'
	#trks = np.loadtxt(trks_path, delimiter=',')


def convert_format(inpath, outpath):
	trk_reid = np.load(path)
	trks = trk_reid[:,:10].astype(np.int)
	#print trks.shape
	reid_features = trk_reid[:,10:]
	



if __name__ == '__main__':
	args = parse_args()
	#test(sys.argv[1])	
	#clean_tracker(args.trk_reid_file)
	#compute_similarity(args.trk_reid_file, True)
	test(args.dis_file)
	#merge_component(args.trk_reid_file, args.dis_file)
	#reassign_id(sys.argv[1], sys.argv[2])
	#merge_camera(sys.argv[1], sys.argv[2])
	#clean_detection(sys.argv[1])



