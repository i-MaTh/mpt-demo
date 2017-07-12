import numpy as np
<<<<<<< HEAD
#import h5py
=======
>>>>>>> 826c5be7aab7f79158cc1c26a1bfd42b9cd1bf60
import os
import sys
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from random import random as rand


def show_boxes():
<<<<<<< HEAD
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
=======
    img_dir = './img1'
    dets_path = './tracks07091309.txt'
    #dets_path = './refine_res1.txt'
    idx = int(sys.argv[1])
    scale = 1.0

    dets_in = np.loadtxt(dets_path, delimiter=',').astype(np.int)
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
>>>>>>> 826c5be7aab7f79158cc1c26a1bfd42b9cd1bf60


def show_tracks(img_dir, det_path, out_dir, scale=1.0):
	import cv2
<<<<<<< HEAD
	
	#dets_in = np.load(det_path)
=======

>>>>>>> 826c5be7aab7f79158cc1c26a1bfd42b9cd1bf60
	dets_in = np.loadtxt(det_path, delimiter=',')
	img_paths = {int(os.path.splitext(f)[0]): os.path.join(img_dir, f) for f in os.listdir(img_dir)}
	img_names = {int(os.path.splitext(f)[0]): f for f in os.listdir(img_dir)}
	frame_indices = dets_in[:, 0].astype(np.int)

	max_frame_idx = max(frame_indices)
	min_frame_idx = min(frame_indices)
	
	colours = [(int(100*rand()), int(100*rand()), int(100*rand())) for _ in range(int(max(dets_in[:,1])))]
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
							  color = colours[d[0]-1], thickness = 3)
			'''
			cv2.rectangle(img, (d[1] - 7, d[2] - 7), 
				    (d[1] + 5, d[2] + 5),
				    color = colours[d[0]], thickness = -1)
			'''
<<<<<<< HEAD
			cv2.putText(img, 'ID:{}'.format(d[0]), (d[1], d[2]),
=======
			cv2.putText(img, 'Person ID:{}'.format(d[0]), (d[1], d[2]),
>>>>>>> 826c5be7aab7f79158cc1c26a1bfd42b9cd1bf60
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(255, 255, 255),
					thickness = 2)
			
		out_path = os.path.join(out_dir, img_names[i])
		cv2.imwrite(out_path, img)
		#break
		if i % 100 == 0:
			print 'process: %d' % i

<<<<<<< HEAD
def show_video(video_dir, dets_path, scale=1.0):
=======
def show_boxes3(video_dir, dets_path, scale=1.0):
>>>>>>> 826c5be7aab7f79158cc1c26a1bfd42b9cd1bf60
	"""
	det format: frame_id, -1, bbox_left, bbox_right, width, height, score
	"""
	from skimage import io
	import time
	import os

	dets_in = np.loadtxt(dets_path, delimiter=',') #load detections, format: ndarray
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


if __name__ == '__main__':
	#img_path = sys.argv[1]	
	#dets_path = sys.argv[2]
	#show_boxes()
<<<<<<< HEAD
	'''	
	video_dir = sys.argv[1]
	det_path = sys.argv[2]
	show_video(video_dir, det_path)
=======
	
	#video_dir = sys.argv[1]
	#det_path = sys.argv[2]
	#show_boxes3('./img1', './refine_res1.txt')
>>>>>>> 826c5be7aab7f79158cc1c26a1bfd42b9cd1bf60
	
	'''
	assert len(sys.argv) > 1, 'please input (video_dir, det_path, out_dir)'
	video_dir = sys.argv[1]
	det_path = sys.argv[2]
	out_dir = sys.argv[3]
<<<<<<< HEAD
	show_tracks(video_dir, det_path, out_dir)
=======
    '''
	show_tracks('./img', det_path, out_dir)
	
>>>>>>> 826c5be7aab7f79158cc1c26a1bfd42b9cd1bf60
	






