import numpy as np
import h5py
import os
import sys
import matplotlib.pyplot as plt 
from random import random as rand

seed = 1024
np.random.seed(seed)

def show_trackers(img_path, dets_path, scale = 1.0):
	plt.cla()
	plt.axis('off')
	img = plt.imread(img_path)
	#print np.shape(img)
	#plt.imshow(img)

	dets = h5py.File(dets_path, 'r')
	dets = dets['person'][...]
	#img_name = img_path.split('/')[-1]
	#out_path = ''.join(img_path.split('/')[:-1])
	num_dets = len(dets)
	#color_buffs = [(rand(), rand(), rand()) for _ in range(num_tracker)]
	for i in range(num_dets):
		bbox = dets[i][:4] * scale
		color = (rand(), rand(), rand())
		# rectangle params: lower left(x,y), weight, height
		rect = plt.Rectangle((bbox[0], bbox[1]), 
							  bbox[2] - bbox[0], 
					          bbox[3] - bbox[1], fill = False, 
							  edgecolor = color_buffs[bbox[-1]], linewidth = 2.5)
		plt.gca().add_patch(rect)
		
		if dets.shape[1] == 5:
			tracker_id = dets[i][-1]
			plt.gca().text(bbox[0], bbox[1], '{}'.format(tracker_id), 
					bbox=dict(facecolor=color, alpha=0.5), fontsize= 0.09*(bbox[2]-bbox[0]), color='white')
	plt.show()

if __name__ == '__main__':
	img_path = sys.argv[1]	
	dets_path = sys.argv[2]
	show_trackers(img_path, dets_path)
