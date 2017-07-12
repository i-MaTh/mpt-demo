import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg


def get_minibatch(roidb):
	img_blob, gt_boxes_blob, im_info = preprocess(roidb)
	if gt_boxes_blob.size == 0:
		return None
	return img_blob, gt_boxes_blob, im_info


def choice_crop_rect(h, w):
	r = np.random.uniform(cfg.TRAIN.DATA_AUG.CROP_RATIO_MIN_BOUND, 1)
	s = int(r * min(w, h))
	x1 = np.random.randint(0, w - s + 1)
	y1 = np.random.randint(0, h - s + 1)
	croped_rect = [x1, y1, s, s]
	return croped_rect


def preprocess(roidbs):
	target_size = cfg.TRAIN.SCALE
	img_blob = np.empty((cfg.TRAIN.IMS_PER_BATCH, 3,
	                    target_size, target_size), dtype=np.float32)
	gt_boxes_blob = np.empty((0, 6), dtype=np.float32)
	for batch_index, roidb in enumerate(roidbs):
		im = cv2.imread(roidb['image'])
                if im is None:
                        print roidb['image']
		if roidb['flipped']:
			im = im[:, ::-1, :]
		height, width = im.shape[:2]
		x1, y1, croped_scale, croped_scale = choice_crop_rect(height, width)
		im_scale = float(target_size) / float(croped_scale)
		# gt boxes: (x1, y1, x2, y2, cls, batch_index)
		gt_boxes = np.empty((len(roidb['gt_classes']), 6), dtype=np.float32)
		gt_boxes[:, 0:4] = roidb['bboxes']
		gt_boxes[:, 4] = roidb['gt_classes']
		gt_boxes[:, 5] = batch_index
		gt_boxes = convert_gt_boxes(gt_boxes, [x1, y1, croped_scale, croped_scale])
		if len(gt_boxes) != 0 and gt_boxes.size != 0:
			gt_boxes[:, 0:4] = gt_boxes[:, 0:4] * im_scale
			gt_boxes = filter_gt(gt_boxes)
		gt_boxes_blob = np.vstack((gt_boxes_blob, gt_boxes))

		im = im[y1:y1 + croped_scale, x1:x1 + croped_scale, :]
		im = cv2.resize(im, (target_size, target_size),
		                interpolation=cv2.INTER_LINEAR)
		im = im.astype(np.float32, copy=False)
		im = (im - cfg.TRAIN.DATA_AUG.MEAN) * 1.0 / cfg.TRAIN.DATA_AUG.NORMALIZER
		img_blob[batch_index, ...] = im.transpose((2, 0, 1))
	im_info = np.array([[target_size, target_size]], dtype=np.float32)

	return img_blob, gt_boxes_blob, im_info


def filter_gt(gt_boxes):
	size_filter_inds = (gt_boxes[:, 2] - gt_boxes[:, 0] +
	                    1 >= cfg.TRAIN.DATA_AUG.MIN_SIZE)
	gt_boxes = gt_boxes[size_filter_inds, :]
	return gt_boxes


def convert_gt_boxes(gt_boxes, crop_rect):
	x1, y1, croped_scale, _ = crop_rect
	ori_gt_boxes = gt_boxes.copy()
	gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] - x1
	gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] - y1
	gt_boxes[:, :4] = np.maximum(np.minimum(gt_boxes[:, :4], croped_scale - 1), 0)
	gt_boxes_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
	                 (gt_boxes[:, 3] - gt_boxes[:, 1])
	ori_gt_boxes_area = (ori_gt_boxes[
	                     :, 2] - ori_gt_boxes[:, 0]) * (ori_gt_boxes[:, 3] - ori_gt_boxes[:, 1])
	base_inds = ori_gt_boxes_area > 1
	if len(base_inds) == 0 or base_inds.size == 0:
		gt_boxes = []
	else:
		inds = gt_boxes_area[base_inds] * 1.0 / ori_gt_boxes_area[base_inds] > cfg.TRAIN.DATA_AUG.GT_CROP_MIN_RATIO
		gt_boxes = gt_boxes[inds, :]

	return gt_boxes
