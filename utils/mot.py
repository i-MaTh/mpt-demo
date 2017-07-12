import numpy as np
import sys
import cPickle as pl
import cv2
import os

with open("colors.pkl", 'r') as f:
	colors_list = pl.load(f)

def get_videos_info(video1, video2):
	videos_info_list = []
	for video in [video1, video2]:
		with open("{}/info_map_post.pkl".format(video), 'r') as f:
			info_map = pl.load(f)
		with open("{}/id_info_map.pkl".format(video), 'r') as f:
			id_info_map = pl.load(f)
		with open("{}/persons_feature_post.pkl".format(video), 'r') as f:
			features_map = pl.load(f)
		videos_info_list.append([info_map, id_info_map, features_map])
	return videos_info_list


# def convert_video_info_to_feat_id_arr(info_map, features_map):
# 	features_with_id_arr = []
# 	for img_id in features_map.keys():
# 		features = features_map[img_id]['feature']
# 		person_id_list = []
# 		for feat_idx, feat in enumerate(features):
# 			person_id = info_map[img_id][1][feat_idx][-1]
# 			person_id_list.append(person_id)
# 		features_with_id = np.hstack((features.copy(), np.array(person_id_list)[:, np.newaxis]))
# 		features_with_id_arr.append(features_with_id)
# 	features_with_id_arr = np.vstack(features_with_id_arr)
# 	return features_with_id_arr

def convert_video_info_to_feat_id_arr(id_info_map, features_map):
	features_with_id_arr = []
	for person_id, id_info in id_info_map.items():
		feat_arr = []
		for episode in id_info:
			for img_id, bb_index, bb in episode:
				feat = features_map[img_id]['feature'][bb_index]
				feat_arr.append(feat)
		feat_arr = np.hstack((np.array(feat_arr), np.ones(len(feat_arr))[:, np.newaxis] * person_id))
		max_num = min(len(feat_arr), 100)
		index_list = range(0, len(feat_arr), len(feat_arr) / max_num + 1)
		feat_arr = feat_arr[index_list, :]
		features_with_id_arr.append(feat_arr)
	features_with_id_arr = np.vstack(features_with_id_arr)
	return features_with_id_arr


def convert_id_info_to_feat_arr(id_info, features_map):
	feat_arr = []
	for episode in id_info:
		for img_id, bb_index, bb in episode:
			feat = features_map[img_id]['feature'][bb_index]
			feat_arr.append(feat)
	return np.array(feat_arr)


def compute_feat_dist(feat1, feat2):
	dist = np.sum((feat1 - feat2) ** 2) ** 0.5
	return dist


# def compute_set_feat_dist(feat_gallary, feat_probes):
# 	dist_list = []
# 	for feat_probe in feat_probes:
# 		gallary_dist_list = np.sum((feat_gallary[:, :-1] - feat_probe) ** 2, axis=1) ** 0.5
# 		min_gallary_index = np.argmin(gallary_dist_list)
# 		min_gallary_dist = gallary_dist_list[min_gallary_index]
# 		dist_list.append([min_gallary_dist, min_gallary_index])
# 	min_dist_index = np.argmin(np.array(dist_list)[:, 0])
# 	min_dist_id = feat_gallary[dist_list[min_dist_index][1], -1]
# 	min_dist = dist_list[min_dist_index][0]
# 	return min_dist_id, min_dist

def compute_set_feat_dist(feat_gallary, feat_probes):
	dist_id_list = []
	for feat_probe in feat_probes:
		gallary_dist_list = np.sum((feat_gallary[:, :-1] - feat_probe) ** 2, axis=1) ** 0.5
		dist_id_list.extend(zip(gallary_dist_list, feat_gallary[:, -1]))
	dist_id_list = sorted(dist_id_list, key=lambda x: x[0])
	return dist_id_list


def match_id_across_videos(videos_info_list):
	info_map1, id_info_map1, features_map1 = videos_info_list[0]
	info_map2, id_info_map2, features_map2 = videos_info_list[1]

	features_with_id_arr1 = convert_video_info_to_feat_id_arr(id_info_map1, features_map1)
	video1_max_id = max(id_info_map1.keys())

	id_re_map = {}
	used_id_list = []
	for person_id2, id_info2 in id_info_map2.items():
		feat_arr2 = convert_id_info_to_feat_arr(id_info2, features_map2)
		num = len(feat_arr2)
		if num < 10:
			index_list = range(0, num)
		elif num < 40:
			index_list = range(0, num, 2)
		elif num < 100:
			index_list = range(0, num, 3)
		else:
			index_list = range(0, num, num / 100)
		feat_arr2 = feat_arr2[index_list, :]
		dist_id_list = compute_set_feat_dist(features_with_id_arr1, feat_arr2)
		for min_dist, pid in dist_id_list:
			pid = int(pid)
			if min_dist < 70:
				if pid not in used_id_list:
					id_re_map[person_id2] = pid
					used_id_list.append(pid)
					break
				else:
					continue
			else:
				id_re_map[person_id2] = video1_max_id
				used_id_list.append(video1_max_id)
				video1_max_id += 1
				break

	id_info_map_new = {}
	for person_id, infolst in id_info_map2.items():
		id_info_map_new[id_re_map[person_id]] = infolst

	new_info_map2 = convert_to_info_map(id_info_map_new, info_map2)
	return new_info_map2


def convert_to_info_map(bbs_id_info, info_map):
	new_info_map = {}
	for person_id, infolst in bbs_id_info.items():
		for info_episode in infolst:
			for img_id, bb_index, bb in info_episode:
				bb = list(bb)
				bb.append(person_id)
				if new_info_map.has_key(img_id):
					new_info_map[img_id][1].append(bb)
				else:
					new_info_map[img_id] = [info_map[img_id][0], [bb]]
	for img_id in range(len(new_info_map.keys())):
		new_info_map[img_id][1] = np.array(new_info_map[img_id][1])

	return new_info_map


def show(info_map):
	for id in range(len(info_map.keys())):
		img_path, bbs = info_map[id]
		img = cv2.imread(img_path)
		for bb in bbs:
			bb = [int(x) for x in bb]
			cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2] - 1, bb[1] + bb[3] - 1), colors_list[bb[-1]], 2)
			s = str(bb[-1])
			cv2.putText(img, s, (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
		cv2.putText(img, str(id), (img.shape[1] / 10, img.shape[0] / 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
		cv2.imshow("ids", img)
		if cv2.waitKey(-1) == 27:
			sys.exit(0)

def save_reid_imgs(new_info_map, save_dir):
	for id in range(len(new_info_map.keys())):
		img_path, bbs = new_info_map[id]
		img = cv2.imread(img_path)
		for bb in bbs:
			cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2] - 1, bb[1] + bb[3] - 1), colors_list[bb[-1]], 2)
			s = str(bb[-1])
			cv2.putText(img, s, (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
		# cv2.putText(img, str(id), (img.shape[1] / 10, img.shape[0] / 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
		cv2.imwrite("{}/{}.jpg".format(save_dir, id), img)

def main(video1, video2):
	videos_info_list = get_videos_info(video1, video2)
	new_info_map = match_id_across_videos(videos_info_list)
	video2_new_info_map_save_dir = video2 + "/mot_new_info_map.pkl"
	video2_save_imgs_dir= video2 + "/mot_id_re_map"
	if not os.path.exists(video2_save_imgs_dir):
		os.mkdir(video2_save_imgs_dir)
	with open(video2_new_info_map_save_dir, 'w')as f:
		pl.dump(new_info_map, f)
	save_reid_imgs(new_info_map, video2_save_imgs_dir)

	# with open(video2_new_info_map_save_dir, 'r')as f:
	# 	new_info_map = pl.load(f)
	# show(new_info_map)


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print "python .. video1, video2"
		sys.exit(0)

	video1, video2 = sys.argv[1:3]
	main(video1, video2)
