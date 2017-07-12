import numpy as np
from scipy.spatial.distance import cosine, euclidean, cdist
import os, sys
import argparse
from datetime import datetime
import pickle
from sklearn.preprocessing import normalize
from collections import OrderedDict



def merge_component(dis_file, trks_file, threshold=0.5):
	dis = pickle.load(open(dis_file, 'r'))
	trks = np.loadtxt(trks_file, delimiter=',').astype(np.int)
	
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
									        
	trks[:,1] = trk_indices
	np.savetxt('refine_res2.txt', trks, fmt='%.2f', delimiter=',')


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
	distances = {}

	start_time = datetime.now()
	for i in range(min_trk_id, max_trk_id+1):
		mask_i = (trk_indices == i)
		reid_i = reid_features[mask_i]
		if reid_i.shape[0] == 0: continue
		values = []
		for j in range(min_trk_id, max_trk_id+1):
			mask_j = (trk_indices == j)
			reid_j = reid_features[mask_j]
			if reid_j.shape[0] ==0: continue
			
			if i != j:
				dis = cdist(reid_i, reid_j, metric = 'cosine')
				v = dis.max()
			else:
				v = 0
			values.append(v)
		distances[i] = values

	pickle.dump(distances, open('distance_lookup.pkl', 'wb'))

	end_time = datetime.now()
	duration = end_time - start_time
	print 'duration time: %s' % duration


def clean_tracker(trk_reid_file, threshold=5):
	trk_reid = np.load(trk_reid_file)
	trks = trk_reid[:,:10].astype(np.int)
	min_trk_id = trks[:,1].min()
	max_trk_id = trks[:,1].max()

	num_per_trk = {i:len(trks[trks[:,1] == i]) for i in range(min_trk_id, max_trk_id+1)}
	print sorted(num_per_trk.values())
	
	new_trks = []
	for i in range(trks.shape[0]):
		trk_id = trks[i][1]
		if num_per_trk[trk_id] < threshold: continue
		new_trks.append(trk_reid[i])

	_tuples = os.path.splitext(trk_reid_file)
	outpath = _tuples[0] + '_new' + _tuples[1]
	np.save(outpath, new_trks)
	

def parse_args():
	""" 
	Parse command line arguments.
	"""
	parser = argparse.ArgumentParser(description="Post-Processing Tracker Result")
	parser.add_argument("-infile", help="Path to tracker file.", default=None)
	parser.add_argument("-trk_reid_file", help="Path to reid of tracker file.", default=None)
	parser.add_argument("-output_file", help="Path to output file.", default=None)
	return parser.parse_args()

def test(path, threshold=0.6):
	#path = './demo_test_data/distances.pkl'
	
	dis = pickle.load(open(path, 'r'))
	# dis.items() type is list	
	trk_indices = {i:dis.keys()[i] for i in range(len(dis))}
	
	matches = {}
	for k in dis:
		#min_dis = sorted(dis[k])
		min_idx = np.argsort(dis[k])[1]
		kt = trk_indices[min_idx]
		key = '%d_%d' % (k, kt)
		value = dis[k][min_idx]
		matches[key] = value


	max_dis = max(matches.values())
	min_dis = min(matches.values())
	mean_dis = np.mean(matches.values())
	median_dis = np.median(matches.values())
	print 'max_dis: {}, min_dis: {}'.format(max_dis, min_dis)
	print 'mean_dis: {}, median_dis: {}'.format(mean_dis, median_dis)
	print 'number of dis < {}: {}'.format(threshold , sum(np.asarray(matches.values()) < threshold))
	
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
	compute_similarity(args.trk_reid_file)
	#test(args.infile)
	#merge_component(args.trk_reid_file)


