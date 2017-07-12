import pickle as pl
import sys
import os
import cv2



def crop_person_by_hyy():
	if len(sys.argv) == 1:
		print "python .. smoothed_bbs.pkl save_dir"
		sys.exit(0)

	pl_path = sys.argv[1]
	save_dir = sys.argv[2]
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	assert os.path.exists(pl_path)
	if os.path.exists(pl_path):
		with open(pl_path, 'r') as f:
			smoothed_info_map = pl.load(f)

	for info in smoothed_info_map.values():
		img_path, bbs = info
		im = cv2.imread(img_path)
		new_crop_dir = "{}/{}".format(save_dir, img_path.split('/')[-1].strip()[:-4])
		if not os.path.exists(new_crop_dir):
			os.system("mkdir -p {}".format(new_crop_dir))
		for idx, bb in enumerate(bbs):
			crop_img = im[bb[1]:bb[1] + bb[3] - 1, bb[0]:bb[0] + bb[2] - 1, :]
			crop_img = cv2.resize(crop_img, (56, 144))
			new_img_path = "{}/{}.jpg".format(new_crop_dir, idx)
			cv2.imwrite(new_img_path, crop_img)

def crop_person():
	assert len(sys.argv) > 1, 'please input hdf5 paths and save dir'
	
	inpath = sys.argv[1]
	save_dir = sys.argv[2]

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	import h5py 
	with open(inpath, 'r') as infile:
		for line in infile:
			img_path = line.strip('\n').replace('.h5', '.jpg')
			img_id = line.strip('\n').split('/')[-1].split('.')[0] # such as 000001, 000002
			img = cv2.imread(img_path)
			dets = h5py.File(line.strip('\n'), 'r')
			bboxes = dets['person'][...]
			for idx, bbox in enumerate(bboxes):
				# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]	
				crop_img = img.copy()[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
				crop_path = '{0}/{1}_{2}.jpg'.format(save_dir, img_id, idx)
				cv2.imwrite(crop_path, crop_img)
			dets.close()
			#break


if __name__ == '__main__':
	crop_person()





