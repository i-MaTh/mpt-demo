import numpy as np
from scipy.spatial.distance import cosine, euclidean, cdist
import os, sys
import argparse
from datetime import datetime
import pickle
from sklearn.preprocessing import normalize

def merge_component():
    dis = pickle.load(open('./norm_distances2.pkl', 'r'))
    trks = np.loadtxt('./tracks07091309_2.txt', delimiter=',').astype(np.int)
    print type(trks)
    
    max_trk_id = trks[:,1].max()
    min_trk_id = trks[:,1].min()
    max_frame_id = trks[:,0].max()
    min_frame_id = trks[:,0].min()
    print 'max_frame_id: %d, min_frame_id: %d' % (max_frame_id, min_frame_id)
    print 'max_tracker_id: %d, min_tracker_id: %d' % (max_trk_id, min_trk_id)
    trk_indices = trks[:,1]
    print type(trk_indices)
    threshold = 0.11
    print 'number of dis < {}: {}'.format(threshold , sum(np.asarray(dis.values()) < threshold))
    cands = [it for it in dis.items() if it[1] < threshold]
    #print cands
    
    for c in cands:
        tmp = map(int, c[0].split('_'))
        trk_indices = [tmp[0] if idx == tmp[1] else idx for idx in trk_indices]
        
    trks[:,1] = trk_indices
    np.savetxt('refine_res2.txt', trks, fmt='%.2f', delimiter=',')
    
        
def compute_similarity(trk_reid_file):
   
    trk_reid = np.load(trk_reid_file) # [frame_id, tracker_id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1, reid_features]
    trks = trk_reid[:,:10].astype(np.int)
    #print trks.shape
    reid_features = trk_reid[:,10:]
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
    k = 3
    count = 0
    start_time = datetime.now()
    for i in range(min_trk_id, max_trk_id):
        mask_i = (trk_indices == i)
        reid_i = reid_features[mask_i]
        if reid_i.shape[0] == 0: continue
        for j in range(i+1, max_trk_id+1):
            mask_j = (trk_indices == j)
            reid_j = reid_features[mask_j]
            if reid_j.shape[0] ==0: continue
            #print reid_j.shape
            dis = cdist(reid_i, reid_j, metric = 'cosine')
            key = '%d_%d' % (i, j)
            distances[key] = np.mean(sorted(dis.ravel())[:k])
            #distances[key] = dis.min()
            count += 1
            if count % 100 == 0:
                print 'process: %d' % count
            
	pickle.dump(distances, open('norm_distances2_3.pkl', 'wb'))

	end_time = datetime.now()
	duration = end_time - start_time
	print 'duration time: %s' % duration

	'''		
	new_trks = []
	# filter
	
	for i in range(len(trks)):
		#print num_per_trk[int(trks[i][1])]
		key = int(trks[i][1])
		if num_per_trk[key] < 5:
			continue
		new_trks.append(trks[i])
	
	np.savetxt('new_trackers0707_2.txt', new_trks,delimiter=',', fmt='%.2f')
	'''

def parse_args():
	""" 
	Parse command line arguments.
	"""
	parser = argparse.ArgumentParser(description="Post-Processing Tracker Result")
	#parser.add_argument("--tracker_file", help="Path to tracker file.", default=None, required=True)
	parser.add_argument("--trk_reid_file", help="Path to reid of tracker file.", default=None)
	parser.add_argument("--output_file", help="Path to output file.", default=None)
	return parser.parse_args()

def test(path):
	#path = './demo_test_data/distances.pkl'
	
	dis = pickle.load(open(path, 'r'))
	max_dis = max(dis.values())
	min_dis = min(dis.values())
	mean_dis = np.mean(dis.values())
	median_dis = np.median(dis.values())
	print 'max_dis: {}, min_dis: {}'.format(max_dis, min_dis)
	print 'mean_dis: {}, median_dis: {}'.format(mean_dis, median_dis)
	threshold = 0.11
	print 'number of dis < {}: {}'.format(threshold , sum(np.asarray(dis.values()) < threshold))
	print dis['43_89']
	candidates = [it for it in dis.items() if it[1] < threshold]
	
	print candidates
	#trks_path = './demo_test_data/tracks0707_2.txt'
	#trks = np.loadtxt(trks_path, delimiter=',')
	

def convert_format(inpath):
    #import OrderDict
    trk_reid = np.load(inpath)
    trks = trk_reid[:,:10].astype(np.int)
    #print trks.shape
    #reid_features = trk_reid[:,10:]
    #print reid_features.shape
    
    max_trk_id = trks[:,1].max()
    min_trk_id = trks[:,1].min()
    max_frame_id = trk_reid[:,0].max()
    min_frame_id = trk_reid[:,0].min()
    print 'max_frame_id: %d, min_frame_id: %d' % (max_frame_id, min_frame_id)
    print 'max_tracker_id: %d, min_tracker_id: %d' % (max_trk_id, min_trk_id)
	
    trk_indices = trks[:,1]
    res = {}
    for i in range(min_trk_id, max_trk_id+1):
        mask = (trk_indices == i)
        if sum(mask) == 0: continue
        #print trk_reid[mask].shape
        values = []
        for row in trk_reid[mask]:
            tmp = []
            tmp.append(int(row[0]))
            tmp.append(row[2:6].tolist())
            tmp.append(row[6:].tolist())
            values.append(tmp)

        res[i] = values

    pickle.dump(res, open('result1.pkl', 'wb'))


if __name__ == '__main__':
    #args = parse_args()
    compute_similarity('./trk_reid_07091309_2.npy')
    #test('norm_distances1.pkl')	
    #path = './trk_reid_07091309.npy'
    #convert_format(path)
    #merge_component()




