import caffe
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv

DEBUG = False


class AnchorTargetLayer(caffe.Layer):
	def setup(self, bottom, top):
		anchor_generator = cfg.ANCHOR_GENERATOR
		self._anchors = generate_anchors(anchor_generator)
		self._num_anchors = self._anchors.shape[0]
		self._feat_stride = cfg.FEAT_STRIDE
		# allow boxes to sit over the edge by a small amount
		self._allowed_border = cfg.ALLOWED_BORDER
		self.anchors = []

		height, width = bottom[0].data.shape[-2:]
		A = self._num_anchors
		# labels: -1,0,1
		batch_size = cfg.TRAIN.IMS_PER_BATCH
		top[0].reshape(batch_size, 1, A * height, width)
		# bbox_targets
		top[1].reshape(batch_size, A * 4, height, width)
		# bbox_inside_weights:  has value only when labels==1
		top[2].reshape(batch_size, A * 4, height, width)
		# bbox_outside_weights
		top[3].reshape(batch_size, A * 4, height, width)

	def get_proper_anchors(self, bottom, im_info):
		height, width = bottom[0].data.shape[-2:]
		shift_x = np.arange(0, width) * self._feat_stride
		shift_y = np.arange(0, height) * self._feat_stride
		shift_x, shift_y = np.meshgrid(shift_x, shift_y)
		shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
							shift_x.ravel(), shift_y.ravel())).transpose()
		A = self._num_anchors
		K = shifts.shape[0]
		all_anchors = (self._anchors.reshape((1, A, 4)) +
					   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
		all_anchors = all_anchors.reshape((K * A, 4))
		total_anchors = int(K * A)

		inds_inside = np.where(
			(all_anchors[:, 0] >= -self._allowed_border) &
			(all_anchors[:, 1] >= -self._allowed_border) &
			(all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
			(all_anchors[:, 3] < im_info[0] + self._allowed_border)  # height
		)[0]

		anchors = all_anchors[inds_inside, :]

		# label: 1 is positive, 0 is negative, -1 is dont care
		labels = np.empty((len(inds_inside),), dtype=np.float32)
		labels.fill(-1)
		return all_anchors, anchors, total_anchors, inds_inside, labels

	def forward(self, bottom, top):
		batch_size, _, height, width = bottom[0].data.shape
		# GT boxes (x1, y1, x2, y2, label, batch_ind)
		gt_boxes_batch = bottom[1].data
		im_info = bottom[2].data[0, :]
		A = self._num_anchors

		if self.anchors == []:
			self.all_anchors, self.anchors, self.total_anchors, self.inds_inside, self.labels = self.get_proper_anchors(
				bottom, im_info)
		labels_batch = np.zeros((batch_size, 1, A * height, width), dtype=np.float32)
		bbox_targets_batch = np.zeros((batch_size, A * 4, height, width), dtype=np.float32)
		bbox_inside_weights_batch = np.zeros((batch_size, A * 4, height, width), dtype=np.float32)
		bbox_outside_weights_batch = np.zeros((batch_size, A * 4, height, width), dtype=np.float32)

		for batch_ind in range(batch_size):
			# overlaps between the anchors and the gt boxes
			# overlaps (ex, gt)
			labels = self.labels.copy()
			gt_boxes = gt_boxes_batch[np.where(gt_boxes_batch[:, -1] == batch_ind)[0], :]
			has_gt = True
			if len(gt_boxes) == 0:
				gt_boxes = np.array([[0, 0, 1, 1, 1, batch_ind]], dtype=np.float32)
				has_gt = False

			overlaps = bbox_overlaps(
				np.ascontiguousarray(self.anchors, dtype=np.float),
				np.ascontiguousarray(gt_boxes, dtype=np.float))
			argmax_overlaps = overlaps.argmax(axis=1)
			# max_overlaps: each anchor assigned to gt which has max overlap
			max_overlaps = overlaps[np.arange(len(self.inds_inside)), argmax_overlaps]
			# get most anchors whose overlap = max_overlap between the anchors and the gt boxes
			# gt_argmax_overlaps may include same anchor index
			gt_argmax_overlaps = overlaps.argmax(axis=0)
			gt_max_overlaps = overlaps[gt_argmax_overlaps,
									   np.arange(overlaps.shape[1])]
			gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

			if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
				# assign bg labels first so that positive labels can clobber them
				labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

			# fg label: for each gt, anchor with highest overlap
			if has_gt:
				labels[gt_argmax_overlaps] = 1

			# fg label: above threshold IOU
			labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

			if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
				# assign bg labels last so that negative labels can clobber positives
				labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

			# subsample positive labels if we have too many
			num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
			fg_inds = np.where(labels == 1)[0]
			if len(fg_inds) > num_fg:
				disable_inds = npr.choice(
					fg_inds, size=(len(fg_inds) - num_fg), replace=False)
				labels[disable_inds] = -1

			# subsample negative labels if we have too many
			num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
			bg_inds = np.where(labels == 0)[0]
			if len(bg_inds) > num_bg:
				disable_inds = npr.choice(
					bg_inds, size=(len(bg_inds) - num_bg), replace=False)
				labels[disable_inds] = -1

			bbox_targets = _compute_targets(self.anchors, gt_boxes[argmax_overlaps, :])

			bbox_inside_weights = np.zeros((len(self.inds_inside), 4), dtype=np.float32)
			bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

			bbox_outside_weights = np.zeros((len(self.inds_inside), 4), dtype=np.float32)
			if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
				# uniform weighting of examples (given non-uniform sampling)
				num_examples = np.sum(labels >= 0)
				positive_weights = np.ones((1, 4)) * 1.0 / num_examples
				negative_weights = np.ones((1, 4)) * 1.0 / num_examples
			else:
				assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
						(cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
				positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
									np.sum(labels == 1))
				negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
									np.sum(labels == 0))
			bbox_outside_weights[labels == 1, :] = positive_weights
			bbox_outside_weights[labels == 0, :] = negative_weights

			# map up to original set of anchors
			labels = _unmap(labels, self.total_anchors, self.inds_inside, fill=-1)
			bbox_targets = _unmap(bbox_targets, self.total_anchors, self.inds_inside, fill=0)
			bbox_inside_weights = _unmap(bbox_inside_weights, self.total_anchors, self.inds_inside, fill=0)
			bbox_outside_weights = _unmap(bbox_outside_weights, self.total_anchors, self.inds_inside, fill=0)

			# labels
			labels = labels.reshape((height, width, A)).transpose(2, 0, 1)
			labels = labels.reshape((1, A * height, width))
			labels_batch[batch_ind, ...] = labels

			# bbox_targets
			# top[1] channel: (A1_x1, A1_y1, A1_x2, A1_y2, A2_x1, A2_y1,..., AN_x1, AN_y1,AN_x2, AN_y2)
			bbox_targets = bbox_targets \
				.reshape((height, width, A * 4)).transpose(2, 0, 1)
			bbox_targets_batch[batch_ind, ...] = bbox_targets

			# bbox_inside_weights
			bbox_inside_weights = bbox_inside_weights \
				.reshape((height, width, A * 4)).transpose(2, 0, 1)
			assert bbox_inside_weights.shape[1] == height
			assert bbox_inside_weights.shape[2] == width
			bbox_inside_weights_batch[batch_ind, ...] = bbox_inside_weights

			# bbox_outside_weights
			bbox_outside_weights = bbox_outside_weights \
				.reshape((height, width, A * 4)).transpose(2, 0, 1)
			assert bbox_outside_weights.shape[1] == height
			assert bbox_outside_weights.shape[2] == width
			bbox_outside_weights_batch[batch_ind, ...] = bbox_outside_weights

		if DEBUG:
			print "batch_pos_num: {}  batch_neg_num: {}".format(np.sum(labels_batch == 1), np.sum(labels_batch == 0))
		top[0].reshape(*labels_batch.shape)
		top[0].data[...] = labels_batch

		top[1].reshape(*bbox_targets_batch.shape)
		top[1].data[...] = bbox_targets_batch

		top[2].reshape(*bbox_inside_weights_batch.shape)
		top[2].data[...] = bbox_inside_weights_batch

		top[3].reshape(*bbox_outside_weights_batch.shape)
		top[3].data[...] = bbox_outside_weights_batch

		if cfg.DEBUG.RPN_ANCHOR_TARGET_SHOW:
			_vis(bottom[3].data, self.all_anchors, labels_batch, bbox_targets_batch)

	def backward(self, top, propagate_down, bottom):
		"""This layer does not propagate gradients."""
		pass

	def reshape(self, bottom, top):
		"""Reshaping happens during the call to forward."""
		pass


def _unmap(data, count, inds, fill=0):
	""" Unmap a subset of item (data) back to the original set of items (of
	size count)
	former data shape: (M,)+shapeA, processed data's shape:(N,)+shapeA, N>=M"""
	if len(data.shape) == 1:
		ret = np.empty((count,), dtype=np.float32)
		ret.fill(fill)
		ret[inds] = data
	else:
		ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
		ret.fill(fill)
		ret[inds, :] = data
	return ret


def _compute_targets(ex_rois, gt_rois):
	"""Compute bounding-box regression targets for an image."""

	assert ex_rois.shape[0] == gt_rois.shape[0]
	assert ex_rois.shape[1] == 4
	assert gt_rois.shape[1] == 6

	return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def _vis(img_blob_batch, all_anchors, labels_batch, bbox_targets_batch):
	height, width = bbox_targets_batch.shape[2:]
	A = bbox_targets_batch.shape[1] / 4
	for batch_ind in range(img_blob_batch.shape[0]):
		img_blob = img_blob_batch[batch_ind, ...]
		labels = labels_batch[batch_ind, ...]
		bbox_targets = bbox_targets_batch[batch_ind, ...]
		labels = labels.reshape((A, height, width)).transpose((1, 2, 0)).ravel()
		bbox_targets = bbox_targets.transpose((1, 2, 0)).reshape((height * width * A, 4))
		inds_0 = np.where(labels == 0)[0]
		inds_1 = np.where(labels == 1)[0]
		inds_n1 = np.where(labels == -1)[0]
		assert len(inds_0) + len(inds_1) == cfg.TRAIN.RPN_BATCHSIZE
		assert len(inds_0) + len(inds_1) + len(inds_n1) == len(labels)
		img = img_blob.transpose((1, 2, 0)) * cfg.TRAIN.DATA_AUG.NORMALIZER + cfg.TRAIN.DATA_AUG.MEAN
		img = img.astype(np.uint8, copy=False)
		gt_boxes = bbox_transform_inv(all_anchors, bbox_targets)

		all_anchors = all_anchors.astype(np.int, copy=False)
		gt_boxes = gt_boxes.astype(np.int, copy=False)
		import cv2
		img = img.copy()
		for i in range(len(inds_0)):
			ind = inds_0[i]
			anchor = all_anchors[ind, :]
			color = (255, 0, 0)
			if cfg.DEBUG.RPN_ANCHOR_TARGET_NEG_SHOW:
				cv2.rectangle(img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), color,
							  1)
			# cv2.putText(img, "neg", (anchor[0], anchor[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

		for i in range(len(inds_1)):
			ind = inds_1[i]
			anchor = all_anchors[ind, :]
			gt_box = gt_boxes[ind, :]
			color = (0, 0, 255)
			cv2.rectangle(img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), color,
						  1)
			color = (0, 255, 0)
			cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color,
						  1)
			info_str="{}x{}".format(gt_box[3]-gt_box[1], gt_box[2]-gt_box[0])
			cv2.putText(img, info_str, (gt_box[0], gt_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 1)
		cv2.imshow("rpn anchor target", img)
		key = cv2.waitKey()
		if (key == 27 or key == -1):
			import sys
			sys.exit(0)
