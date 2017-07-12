import os
import cv2

def get_video(imgs_dir, videopath, fps):
	fourcc = cv2.cv.CV_FOURCC(*'XVID')
	out = cv2.VideoWriter(videopath, fourcc, fps, (1920, 1080))
	print os.listdir(imgs_dir)
	img_list = sorted(os.listdir(imgs_dir), key=lambda x: x.split('.')[0])
	print img_list
	for p in img_list:
		img = cv2.imread(imgs_dir + '/' + p)
		img = cv2.resize(img, (1920, 1080))
		out.write(img)
	out.release()
	return

import sys
print "haha"
print sys.argv[1], sys.argv[2], sys.argv[3]
get_video(sys.argv[1], sys.argv[2], int(sys.argv[3]))
