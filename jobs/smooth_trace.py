import sys
import os
import h5py
import numpy as np
import cv2
import pickle as pl


def smooth(bbs):
	bbs[:, 2] = bbs[:, 0] + bbs[:, 2] - 1
	bbs[:, 3] = bbs[:, 1] + bbs[:, 3] - 1
	mean_bb = np.mean(bbs, axis=0)
	mean_bb[2] = mean_bb[2] - mean_bb[0] + 1
	mean_bb[3] = mean_bb[3] - mean_bb[1] + 1
	mean_bb = [int(x) for x in mean_bb]
	return mean_bb


def get_bbs_info(img_label):
	info_map = {}
	with open(img_label, 'r') as f:
		lines = f.readlines()
	cnt = 0
	for idx, line in enumerate(lines):
		img_path, anno_path = line.strip().split()
		with h5py.File(anno_path, 'r') as f:
			bbs = f["person"][...]
			if len(bbs) > 0:
				info_map[cnt] = [img_path, bbs]
				cnt += 1
	return info_map


def intersection(a, b):
	x = max(a[0], b[0])
	y = max(a[1], b[1])
	w = min(a[0] + a[2], b[0] + b[2]) - x
	h = min(a[1] + a[3], b[1] + b[3]) - y
	if w < 0 or h < 0: return None
	return (x, y, w, h)


def area(a):
	if a is None:
		return 0.0
	else:
		return a[2] * a[3]


def iou(a, b):
	c = intersection(a, b)
	iou = area(c) * 1.0 / (area(a) + area(b) - area(c))
	return iou


def get_bb_arr_to_smooth(info_map, id, bbox, frame_sum):
	start_id = max(id - 2, 0)
	end_id = min(id + 3, frame_sum)
	bb_arr_to_smooth = []
	if end_id - start_id < 2:
		bb_arr_to_smooth.append(bbox)
	else:
		for frame_id in range(start_id, end_id):
			if frame_id == id:
				bb_arr_to_smooth.append(bbox)
			else:
				bbs = info_map[frame_id][1]
				iou_list = []
				for bb in bbs:
					iou_list.append(iou(bb, bbox))
				if len(iou_list) > 0:
					max_idx = np.argmax(np.array(iou_list))
					max_iou = iou_list[max_idx]
					max_bb = bbs[max_idx]
					if max_iou > 0.7:
						bb_arr_to_smooth.append(max_bb)
					else:
						bb_arr_to_smooth.append(bbox)
				else:
					bb_arr_to_smooth.append(bbox)

	bb_arr_to_smooth = np.array(bb_arr_to_smooth)
	return bb_arr_to_smooth


def show(smoothed_info_map):
	for id, info in smoothed_info_map.items():
		img_path, bbs = info
		img = cv2.imread(img_path)
		color = (0, 0, 255)
		for bb in bbs:
			cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2] - 1, bb[1] + bb[3] - 1), color, 2)
		cv2.imshow("smoothed", img)
		if cv2.waitKey(-1) == 27:
			sys.exit(0)


def main(img_label):
	path = "smoothed_bbs.pkl"
	if os.path.exists(path):
		with open(path, 'r') as f:
			smoothed_info_map = pl.load(f)
	else:
		info_map = get_bbs_info(img_label)
		smoothed_info_map = {}
		frame_sum = len(info_map.keys())
		for id, info in info_map.items():
			img_path, bbs = info
			smoothed_info_map[id] = [img_path, []]
			for bb in bbs:
				bb_arr = get_bb_arr_to_smooth(info_map, id, bb[:4], frame_sum)
				smoothed_bb = smooth(bb_arr)
				#smoothed_bb = np.append(smoothed, bb[-1]) # with score of bounding box
				smoothed_info_map[id][1].append(smoothed_bb)
			smoothed_info_map[id][1] = np.array(smoothed_info_map[id][1])
		with open(path, 'w') as f:
			pl.dump(smoothed_info_map, f)
		# show(smoothed_info_map)


if __name__ == '__main__':
	img_label = sys.argv[1]
	main(img_label)
