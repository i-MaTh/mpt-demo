import numpy as np
from fast_rcnn.config import cfg
import h5py as h5
import os.path as osp
import cPickle as pkl
import copy
import xmltodict


class dataset:
	def __init__(self, img_label_size, cache):
		self._classes = cfg.CLASSES
		self.num_classes = cfg.NUM_CLASSES
		assert len(self._classes) == len(self._classes)
		self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
		self._img_label_size = img_label_size
		self._cache_path = cache
		self._roidb = []
		self.set_roidb()

	def set_roidb(self):
		cache_path = self._cache_path
		if osp.exists(cache_path):
			with open(cache_path, 'r') as cache_f:
				self._roidb = pkl.load(cache_f)
				self.num_images = len(self._roidb)
		else:
			print "begin set roidb"
			cnt = 0
			with open(self._img_label_size, 'r') as f:
				try:
					while True:
						img_path, label_path, height, width = f.next().strip().split()
						height = int(height)
						width = int(width)
						bbs, gt_classes = self._load_gt(label_path)
						roidb = {'image': img_path,
								 'width': width,
								 'height': height,
								 'bboxes': bbs,
								 'gt_classes': gt_classes,
								 'flipped': False}
						if self.is_valid(roidb):
							self._roidb.append(roidb)
							cnt += 1
						if cnt % 1000 == 0:
							print "processed {}".format(cnt)
				except StopIteration:
					pass
			self.num_images = cnt

			if cfg.TRAIN.USE_FLIPPED:
				print 'Appending horizontally-flipped training examples...'
				self.append_flipped_images()
				print 'done'

			print "set roidb done!"
			with open(cache_path, 'wb', pkl.HIGHEST_PROTOCOL) as cache_f:
				pkl.dump(self._roidb, cache_f)

	def is_valid(self, roidb):
		flag = True
		if roidb["bboxes"].size == 0:
			flag = False
		return flag

	def get_roidb(self):
		return self._roidb

	def _load_gt(self, label_path):

		bbs_all = np.zeros((0, 4), dtype=np.float32)
		gt_classes = np.zeros((0), dtype=np.float32)
		label_path = label_path.strip()
		if label_path.endswith(".h5"):
			with h5.File(label_path, 'r') as f:
				for cls in f.keys():
					if cls not in self._classes:
						continue
					bbs = f[cls][...]
					if bbs.size == 0:
						continue
					bbs[:, 2] = bbs[:, 0] + bbs[:, 2] - 1
					bbs[:, 3] = bbs[:, 1] + bbs[:, 3] - 1
					bbs_all = np.vstack((bbs_all, bbs))
					gt_classes = np.hstack(
						(gt_classes, np.ones(len(bbs)) * self._class_to_ind[cls]))

		if label_path.endswith(".xml"):
			with open(label_path, 'r') as f:
				d = xmltodict.parse(f.read())
				anno = d['annotation']
				objs = anno['object']
				m = {}
				if not isinstance(objs, list):
					objs = [objs]
				for obj in objs:
					label = obj['name']
					box = obj['bndbox']
					x1 = box['xmin']
					y1 = box['ymin']
					x2 = box['xmax']
					y2 = box['ymax']
					bb = [x1, y1, x2, y2]
					bb = [int(x) for x in bb]
					if m.has_key(label):
						m[label].append(bb)
					else:
						m[label] = [bb]
				for cls in m.keys():
					if cls not in self._classes:
						continue
					bbs = np.array(m[cls])
					if bbs.size == 0:
						continue
					bbs_all = np.vstack((bbs_all, bbs))
					gt_classes = np.hstack(
						(gt_classes, np.ones(len(bbs)) * self._class_to_ind[cls]))

		assert len(bbs_all) == len(gt_classes), "bbs, gt_classes len differ!"

		return bbs_all, gt_classes

	def append_flipped_images(self):
		for i in xrange(self.num_images):
			entry = copy.deepcopy(self._roidb[i])
			bboxes = entry['bboxes']
			if bboxes.size != 0:
				width = entry['width']
				oldx1 = bboxes[:, 0].copy()
				oldx2 = bboxes[:, 2].copy()
				bboxes[:, 0] = width - oldx2 - 1
				bboxes[:, 2] = width - oldx1 - 1
				assert (bboxes[:, 2] + 1 >= bboxes[:, 0]).all()
				entry['bboxes'] = bboxes
			entry['flipped'] = True
			self._roidb.append(entry)
		self.num_images = self.num_images * 2
