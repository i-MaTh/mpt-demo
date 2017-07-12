import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

DEBUG = False


class ProposalTargetLayer(caffe.Layer):
	"""
	Assign object detection proposals to ground-truth targets. Produces proposal
	classification labels and bounding-box regression targets.
	"""

	def setup(self, bottom, top):
		self._num_classes = 2 if cfg.AGNOSTIC else cfg.NUM_CLASSES
		if DEBUG:
			self._fg_num = 0
			self._bg_num = 0
			self._count = 0

		# sampled rois (batch_ind, x1, y1, x2, y2)
		top[0].reshape(1, 5, 1, 1)
		# labels: 0 or 1, no others
		top[1].reshape(1, 1, 1, 1)
		# bbox_targets
		top[2].reshape(1, self._num_classes * 4, 1, 1)
		# bbox_inside_weights
		top[3].reshape(1, self._num_classes * 4, 1, 1)
		# bbox_outside_weights
		top[4].reshape(1, self._num_classes * 4, 1, 1)
		if len(top) > 5:
			top[5].reshape(1, 1, 1, 1)

	def forward(self, bottom, top):
		# Proposal ROIs (batch_ind, x1, y1, x2, y2) coming from RPN
		# (i.e., rpn.proposal_layer.ProposalLayer), or any other source
		all_rois_batch = bottom[0].data
		# GT boxes (x1, y1, x2, y2, label, batch_ind)
		gt_boxes_batch = bottom[1].data

		if cfg.DEBUG.RPN_PROPOSAL_SHOW:
			assert len(bottom) >= 3, "no data bottom!"
			labels = np.zeros(bottom[0].data.shape[0], dtype=np.float32)
			_vis(bottom[2].data, gt_boxes_batch, all_rois_batch, labels, "proposal", "proposal")

		rois_batch = np.empty((0, 5), dtype=np.float32)
		labels_batch = np.empty((0,), dtype=np.int)
		bbox_targets_batch = np.empty((0, 4 * self._num_classes), dtype=np.float32)
		bbox_inside_weights_batch = np.empty((0, 4 * self._num_classes), dtype=np.float32)
		for i in range(cfg.TRAIN.IMS_PER_BATCH):
			all_rois_inds = np.where(all_rois_batch[:, 0] == i)[0]
			gt_boxes_inds = np.where(gt_boxes_batch[:, -1] == i)[0]
			all_rois = all_rois_batch[all_rois_inds, :]
			gt_boxes = gt_boxes_batch[gt_boxes_inds, :]
			# Include ground-truth boxes in the set of candidate rois
			all_rois = np.vstack(
				(all_rois, np.hstack((gt_boxes[:, -1][:, np.newaxis], gt_boxes[:, :4])))
			)

			"""if BATCH_SIZE==-1: take all bg samples"""
			rois_per_image = np.inf if cfg.TRAIN.BATCH_SIZE == -1 else cfg.TRAIN.BATCH_SIZE
			fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

			# Sample rois with classification labels and bounding box regression
			# targets
			has_gt = True
			if len(gt_boxes) == 0:
				gt_boxes = np.array([[0, 0, 1, 1, 1, i]], dtype=np.float32)
				has_gt = False
			labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
				all_rois, gt_boxes, fg_rois_per_image,
				rois_per_image, self._num_classes, has_gt)

			if DEBUG:
				print 'each num fg: {}'.format((labels > 0).sum())
				print 'each num bg: {}'.format((labels == 0).sum())
				self._count += 1
				self._fg_num += (labels > 0).sum()
				self._bg_num += (labels == 0).sum()
				print 'each num fg avg: {}'.format(self._fg_num / self._count)
				print 'each num bg avg: {}'.format(self._bg_num / self._count)
			rois_batch = np.vstack((rois_batch, rois))
			labels_batch = np.hstack((labels_batch, labels))
			bbox_targets_batch = np.vstack((bbox_targets_batch, bbox_targets))
			bbox_inside_weights_batch = np.vstack((bbox_inside_weights_batch, bbox_inside_weights))

		# rois shape: (N, 5, 1,1), each row: (batch_ind,x1,y1,x2,y2)
		rois_batch = rois_batch.reshape((rois_batch.shape[0], rois_batch.shape[1], 1, 1))
		top[0].reshape(*rois_batch.shape)
		top[0].data[...] = rois_batch

                if DEBUG:
                        print "batch_fg_num: {}  batch_bg_num: {}".format(np.sum(labels_batch>0), np.sum(labels_batch==0))
		# labels shape: (N, 1,1,1), label: 0 or 1
		labels_batch = labels_batch.reshape((labels_batch.shape[0], 1, 1, 1))
		top[1].reshape(*labels_batch.shape)
		top[1].data[...] = labels_batch

		# bbox_targets shape: (N,num_classes*4,1,1)
		# num_classes=2 if AGNOSTIC else true num_classes
		# bbox_targets's each row's index: [corresponding class index*4: corresponding class index*4+4]= bbox_target info, which may include class 0
		bbox_targets_batch = bbox_targets_batch.reshape(
			(bbox_targets_batch.shape[0], bbox_targets_batch.shape[1], 1, 1))
		top[2].reshape(*bbox_targets_batch.shape)
		top[2].data[...] = bbox_targets_batch

		# bbox_inside_weights as same as bbox_targets
		# has value only in cls>0
		bbox_inside_weights_batch = bbox_inside_weights_batch.reshape(
			(bbox_inside_weights_batch.shape[0], bbox_inside_weights_batch.shape[1], 1, 1))
		top[3].reshape(*bbox_inside_weights_batch.shape)
		top[3].data[...] = bbox_inside_weights_batch

		# bbox_outside_weights
		# modified by ywxiong
		bbox_inside_weights_batch = bbox_inside_weights_batch.reshape(
			(bbox_inside_weights_batch.shape[0], bbox_inside_weights_batch.shape[1], 1, 1))
		top[4].reshape(*bbox_inside_weights_batch.shape)
		top[4].data[...] = np.array(bbox_inside_weights_batch > 0).astype(np.float32)

		if len(top) > 5:
			pos_num = np.sum(labels_batch > 0)
			pos_tmp = np.ones((pos_num, 1, 1, 1), dtype=np.float32)
			top[5].reshape(*pos_tmp.shape)
			top[5].data[...] = pos_tmp

		if cfg.DEBUG.RPN_PROPOSAL_CHOSEN_SHOW:
			assert len(bottom) >= 3, "no data bottom!"
			_vis(bottom[2].data, gt_boxes_batch, rois_batch, labels_batch, "proposal_chosen", "proposal chosen")

	def backward(self, top, propagate_down, bottom):
		"""This layer does not propagate gradients."""
		pass

	def reshape(self, bottom, top):
		"""Reshaping happens during the call to forward."""
		pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
	"""Bounding-box regression targets (bbox_target_data) are stored in a
	compact form N x (class, tx, ty, tw, th)

	This function expands those targets into the 4-of-4*K representation used
	by the network (i.e. only one class has non-zero targets).

	Returns:
		K: num_classes
		bbox_target (ndarray): N x 4K blob of regression targets
		bbox_inside_weights (ndarray): N x 4K blob of loss weights
	"""

	clss = bbox_target_data[:, 0]
	bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
	bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
	inds = np.where(clss > 0)[0]
	if cfg.AGNOSTIC:
		for ind in inds:
			cls = clss[ind]
			start = 4 * (1 if cls > 0 else 0)
			end = start + 4
			bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
			# has value only in cls>0
			bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
	else:
		for ind in inds:
			cls = int(clss[ind])
			start = 4 * cls
			end = start + 4
			bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
			bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
	return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
	"""Compute bounding-box regression targets for an image."""

	assert ex_rois.shape[0] == gt_rois.shape[0]
	assert ex_rois.shape[1] == 4
	assert gt_rois.shape[1] == 4

	# targets:  transformed [dx,dy,dw,dh]
	targets = bbox_transform(ex_rois, gt_rois)
	if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
		# Optionally normalize targets by a precomputed mean and stdev
		targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
				   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
	return np.hstack(
		(labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, has_gt):
	"""Generate a random sample of RoIs comprising foreground and background
	examples.
	if BATCH_SIZE==-1: take all bg samples
	"""
	# overlaps: (rois x gt_boxes)
	overlaps = bbox_overlaps(
		np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
		np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
	gt_assignment = overlaps.argmax(axis=1)
	max_overlaps = overlaps.max(axis=1)
	labels = gt_boxes[gt_assignment, 4]

	# Select foreground RoIs as those with >= FG_THRESH overlap
	fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
	# Guard against the case when an image has fewer than fg_rois_per_image
	# foreground RoIs
	fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
	# Sample foreground regions without replacement
	if fg_inds.size > 0:
		fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)

	# Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
	bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
					   (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
	# Compute number of background RoIs to take from this image (guarding
	# against there being fewer than desired)
	bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
	bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
	# Sample background regions without replacement
	if bg_inds.size > 0:
		bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)

	# The indices that we're selecting (both fg and bg)
	if has_gt:
		keep_inds = np.append(fg_inds, bg_inds)
	else:
		keep_inds = bg_inds

	# Select sampled values from various arrays:
	labels = labels[keep_inds]
	# Clamp labels for the background RoIs to 0
	labels[int(fg_rois_per_this_image):] = 0
	rois = all_rois[keep_inds]

	# bbox_target_data: [cls tx ty tw th]
	bbox_target_data = _compute_targets(
		rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

	bbox_targets, bbox_inside_weights = \
		_get_bbox_regression_labels(bbox_target_data, num_classes)

	return labels, rois, bbox_targets, bbox_inside_weights


def _vis(img_blob_batch, gt_boxes_batch, rois_batch, labels_batch, show_name, flag):
	for batch_ind in range(cfg.TRAIN.IMS_PER_BATCH):
		img_blob = img_blob_batch[batch_ind, ...]
		rois_inds = np.where(rois_batch[:, 0] == batch_ind)[0]
		gt_boxes_inds = np.where(gt_boxes_batch[:, -1] == batch_ind)[0]
		rois = rois_batch[rois_inds, :]
		gt_boxes = gt_boxes_batch[gt_boxes_inds, :]
		labels = labels_batch[rois_inds]

		if len(gt_boxes) == 0 or gt_boxes.size == 0:
			gt_boxes = np.array([[0, 0, 1, 1, 1, batch_ind]], dtype=np.float32)

		gt_boxes = gt_boxes[:, :4]
		N, c = rois.shape[:2]
		rois = rois.reshape((N, c))[:, 1:]
		labels = labels.reshape((labels.size,))
		assert N == labels.size, "rois shape and labels shape don't match"

		img = img_blob.transpose((1, 2, 0)) * cfg.TRAIN.DATA_AUG.NORMALIZER + cfg.TRAIN.DATA_AUG.MEAN
		img = img.astype(np.uint8, copy=False)

		gt_boxes = gt_boxes.astype(np.int, copy=False)
		import cv2
		img = img.copy()

		for i in range(labels.size):
			roi = rois[i, :]
			label = labels[i]
			if flag == "proposal" or (
								flag == "proposal chosen" and cfg.DEBUG.RPN_PROPOSAL_NEG_CHOSEN_SHOW and label == 0):
				color = (255, 0, 0)
				cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), color, 1)
			if label > 0:
				color = (0, 0, 255)
				cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), color, 1)

		for gt_box in gt_boxes:
			color = (0, 255, 0)
			if gt_box[2] - gt_box[0] <= 1 or gt_box[3] - gt_box[1] <= 1:
				continue
			cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color, 1)

		cv2.imshow(show_name, img)
		key = cv2.waitKey()
		if (key == 27 or key == -1):
			import sys
			sys.exit(0)
