import numpy as np
#import h5py
import os
import sys
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from random import random as rand
import cv2

def show_boxes():
	img_dir = sys.argv[1]
	dets_path = sys.argv[2]
	idx = int(sys.argv[3])
	scale = 1.0

	#dets_in = np.loadtxt(dets_path, delimiter=',').astype(np.int)
	dets_in = np.load(dets_path)[:,:10].astype(np.int)
	print sum(dets_in[:,1] == idx)
	det = dets_in[dets_in[:,1] == idx][0]
	img_path = os.path.join(img_dir, str(det[0]).zfill(6) + '.jpg')
	print img_path

	plt.cla()
	plt.axis('off')
	img = plt.imread(img_path)
	print np.shape(img)
	plt.imshow(img)
	

	bbox = det[2:6] * scale
	color = (rand(), rand(), rand())
	# rectangle params: lower left(x,y), weight, height
	rect = plt.Rectangle((bbox[0], bbox[1]), 
						  bbox[2], bbox[3],
						  fill = False, edgecolor = color, 
						  linewidth = 2.5)
	plt.gca().add_patch(rect)
		

	idx = det[1]
	plt.gca().text(bbox[0], bbox[1], 'ID: {}'.format(idx), 
				bbox=dict(facecolor=color, alpha=0.5), fontsize= 7, color='white')
	plt.show()


def show_tracks(img_dir, det_path, out_dir, scale=1.0):
	
	dets_in = np.load(det_path)
	#dets_in = np.loadtxt(det_path, delimiter=',')
	img_paths = {int(os.path.splitext(f)[0]): os.path.join(img_dir, f) for f in os.listdir(img_dir)}
	img_names = {int(os.path.splitext(f)[0]): f for f in os.listdir(img_dir)}
	frame_indices = dets_in[:, 0].astype(np.int)

	max_frame_idx = max(frame_indices)
	min_frame_idx = min(frame_indices)
	
	#colours = (100 * np.random.rand(25, 3)).astype(np.int)
	colours = [(100,149,237), (78,238,148), (127,255,0), (238,238,0), (255,106,106), 
				(255,127,36), (255,20,147), (255,52,179), (155,48,255), (144,238,144),
				(0,191,255), (0,0,205), (105,89,205), (255,228,225), (255,222,173),
				(139,137,137), (160,32,240), (255,0,255), (199,21,133), (255,69,0),
				(250,128,114),(50,205,50), (0,0,255)
				]
	print 'colours size: %d' % len(colours)
	print 'max_id: %d, min_id: %d' % (max(dets_in[:, 1].astype(np.int)), min(dets_in[:, 1].astype(np.int)))
	for i in range(min_frame_idx, max_frame_idx):
		img = cv2.imread(img_paths[i])
		mask = (frame_indices == i)
		sub_dets = dets_in[mask, 1:7]

		#print sub_dets.shape
		for d in sub_dets:
			d = d.astype(np.int)
			cv2.rectangle(img, (d[1], d[2]),(d[1] + d[3],d[2] + d[4]),
							  color = colours[d[0]], thickness = 3)
			'''
			cv2.rectangle(img, (d[1] - 7, d[2] - 7), 
				    (d[1] + 5, d[2] + 5),
				    color = colours[d[0]], thickness = -1)
			'''
			if d[0] in [5, 6, 7]:
			#if d[0] in [2, 3]:
				cv2.putText(img, 'From C1', (d[1]+d[3]+5, d[2]+17), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
						color = (255,255,255),
						thickness = 3
						)
			cv2.putText(img, 'ID:{}'.format(d[0]), (d[1]+5, d[2]-8),
					cv2.FONT_HERSHEY_SIMPLEX, 0.9, color=(255, 255, 255),
					thickness = 2)
			
		out_path = os.path.join(out_dir, img_names[i])
		cv2.imwrite(out_path, img)
		#break
		if i % 100 == 0:
			print 'process: %d' % i


def show_video(video_dir, dets_path, scale=1.0):
	"""
	det format: frame_id, -1, bbox_left, bbox_right, width, height, score
	"""
	from skimage import io
	import time
	import os

	#dets_in = np.loadtxt(dets_path, delimiter=',') #load detections, format: ndarray
	dets_in = np.load(dets_path)
	#img_dir = os.path.join(video_dir, 'img2')
	img_dir = video_dir
	img_paths = {int(os.path.splitext(f)[0]): os.path.join(img_dir, f) for f in os.listdir(img_dir)}
	frame_indices = dets_in[:, 0].astype(np.int)

	max_frame_idx = max(frame_indices)
	min_frame_idx = min(frame_indices)
	print 'min_frame_idx: %d' % min_frame_idx
	seed = 1024
	np.random.seed(seed)
	plt.ion()
	fig = plt.figure()
	#colours = np.random.rand(int(dets_in[:,1].max()), 3)
	colours = np.random.rand(100,3)
	for i in range(min_frame_idx, max_frame_idx + 1):
		#detection and frame numbers begin at 1
		sub_dets = dets_in[dets_in[:,0] == i, 1:7]

		ax1 = fig.add_subplot(111, aspect='equal')
		img_path = img_paths[i]
		img =plt.imread(img_path)
		plt.imshow(img)
		plt.title('Frame ID: {}'.format(i))
		for d in sub_dets:
			trk_id = d[0]
			ax1.add_patch(patches.Rectangle((d[1], d[2]), 
											 d[3], d[4], 
											 fill = False,
											 lw = 3,
											 ec = colours[0]))
			ax1.set_adjustable('box-forced')
			#score = d[-1]
			plt.gca().text(d[1], d[2], 'ID:{}'.format(int(trk_id)), 
								bbox=dict(facecolor=colours[1], alpha=0.5), 
								#fontsize= 0.08*(d[3]), 
								fontsize = 8,
								color='white')
		fig.canvas.flush_events()
		plt.draw()
		ax1.cla()

def post_processing(img1_dir, img2_dir, output_dir):
	img1_paths = {int(os.path.splitext(f)[0]): os.path.join(img1_dir, f) for f in os.listdir(img1_dir)}
	img2_paths = {int(os.path.splitext(f)[0]): os.path.join(img2_dir, f) for f in os.listdir(img2_dir)}
	img_names = {int(os.path.splitext(f)[0]): f for f in os.listdir(img1_dir)}

	for idx, key in enumerate(img1_paths):
		img1 = cv2.imread(img1_paths[key])
		img2 = cv2.imread(img2_paths[key])
				
		img1 = img1[:1000, :, :]
		img1 =cv2.resize(img1, (1920, 1320))
			
		img2 = img2[:1000, :, :]
		img2 =cv2.resize(img2, (1920, 1320))
		img = np.concatenate((img1, img2), axis=1)
		
		alpha = 0.5
		position_x = 500
		overlay = img.copy()
		output = img.copy()
		cv2.rectangle(overlay, (position_x + 20, 20),(position_x + 620, 130),
			              color = (250, 235,215), thickness = -1)

		cv2.putText(overlay, 'Camera ID: C1', (position_x + 30, 100),
			            cv2.FONT_HERSHEY_SIMPLEX, 2.5, color=(255,0,0),
						            thickness = 5)

		cv2.rectangle(overlay, (1920 + position_x + 20, 20),(1920 + position_x + 620, 130),
			              color = (250, 235,215), thickness = -1)

		cv2.putText(overlay, 'Camera ID: C2', (1920 + position_x + 30, 100),
			            cv2.FONT_HERSHEY_SIMPLEX, 2.5, color=(255,0,0),
						            thickness = 5)
		# apply the overlay
		cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

		output_path = os.path.join(output_dir, img_names[key])
		#print output_path
		cv2.imwrite(output_path, output)
		
		if idx % 100 == 0: print 'process: %d' % idx
		
		#break


def refine(img_dir, output_dir):
	img_paths = {int(os.path.splitext(f)[0]): os.path.join(img_dir, f) for f in os.listdir(img_dir)}
	img_names = {int(os.path.splitext(f)[0]): f for f in os.listdir(img_dir)}
	
	alpha = 0.7
	position_x = 1330

	for idx, key in enumerate(img_paths):
		img = cv2.imread(img_paths[key])
		#print img.shape
		overlay = img.copy()
		output = img.copy()
		cv2.rectangle(overlay, (position_x + 20, 120), (position_x + 960, 230),
			              color = (250, 235,215), thickness = -1)

		cv2.putText(overlay, 'Cross Camera ReID', (position_x + 30, 200),
			            cv2.FONT_HERSHEY_SIMPLEX, 3, color=(0,0,255),
						            thickness = 8)

		cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
		
		#output = cv2.resize(output, (3840, 1220))
		#print output.shape
		output_path = os.path.join(output_dir, img_names[key])
		cv2.imwrite(output_path, output)

		if idx % 100 == 0: 
			print 'process: %d' % idx


if __name__ == '__main__':
	#img_path = sys.argv[1]	
	#dets_path = sys.argv[2]
	#show_boxes()
	'''	
	video_dir = sys.argv[1]
	det_path = sys.argv[2]
	show_video(video_dir, det_path)
	'''
	'''	
	#assert len(sys.argv) > 1, 'please input (video_dir, det_path, out_dir)'
	video_dir = sys.argv[1]
	det_path = sys.argv[2]
	out_dir = sys.argv[3]
	show_tracks(video_dir, det_path, out_dir)
	'''

	#post_processing(sys.argv[1], sys.argv[2], sys.argv[3])
	refine(sys.argv[1], sys.argv[2])





