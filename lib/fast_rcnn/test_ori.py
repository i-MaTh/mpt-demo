from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from utils.timer import Timer
import numpy as np
import cv2
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg
import sys
import os


def _get_blobs(im):
	height = im.shape[0]
	target_size = cfg.TEST.SCALE
	im_scale = float(target_size) / float(height)
	im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
					interpolation=cv2.INTER_LINEAR)
	im = im.astype(np.float32, copy=False)
	im = (im - cfg.TRAIN.DATA_AUG.MEAN) * 1.0 / cfg.TRAIN.DATA_AUG.NORMALIZER
	h, w = im.shape[:2]
	im_info = np.array([[h, w]], dtype=np.float32)
	blob = np.array([im], dtype=np.float32)
	blob = blob.transpose((0, 3, 1, 2))
	blobs = {'data': blob, 'im_info': im_info}, im_scale
	return blobs


def im_detect(net, im):
	"""Detect object classes in an image given object proposals.

	Arguments:
		net (caffe.Net): Fast R-CNN network to use
		im (ndarray): color image to test (in BGR order)
		boxes (ndarray): R x 4 array of object proposals or None (for RPN)

	Returns:
		scores (ndarray): R x K array of object class scores (K includes
			background as object category 0)
		boxes (ndarray): R x (4*K) array of predicted bounding boxes
	"""
	blobs, im_scale = _get_blobs(im)
	resized_shape = (int(im.shape[0] * im_scale), int(im.shape[1] * im_scale), im.shape[2])

	net.blobs['data'].reshape(*(blobs['data'].shape))
	net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

	forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
	forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
	blobs_out = net.forward(**forward_kwargs)

	rois = net.blobs['rois'].data.copy()
	boxes = rois[:, 1:5]
	scores = blobs_out['cls_prob']
	box_deltas = blobs_out['bbox_pred']
	pred_boxes = bbox_transform_inv(boxes, box_deltas)
	pred_boxes = clip_boxes(pred_boxes, resized_shape)
	pred_boxes = pred_boxes / im_scale

	return scores, pred_boxes


def vis_detections(im, cls_class_all, dets, thresh):
	classes = cfg.CLASSES
	window = "test"
	cv2.namedWindow(window)
	# im = im[:, :, (2, 1, 0)]
	im = im.copy()
	num = len(cls_class_all)
	class_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
	for i in xrange(num):
		bbox = dets[i, :4]
		bbox = [int(x) for x in bbox]
		score = dets[i, -1]
		class_name = classes[int(cls_class_all[i])]
		color = class_color[int(cls_class_all[i])]
		if score > thresh:
			cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
			# info="{}_{}".format(class_name, str(score)[:3])
			# info="{}x{} {:.3f}".format(str((bbox[3]-bbox[1])*resize_scale/im.shape[0]), str((bbox[2]-bbox[0])*resize_scale/im.shape[0]), score)
			info = "{:.3f}".format(score)
			cv2.putText(im, info, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
	cv2.imshow(window, im)
	if cv2.waitKey(-1) == 27:
		sys.exit(0)


def apply_nms(all_boxes, thresh):
	num_classes = len(all_boxes)
	num_images = len(all_boxes[0])
	nms_boxes = [[[] for _ in xrange(num_images)]
				 for _ in xrange(num_classes)]
	for cls_ind in xrange(num_classes):
		for im_ind in xrange(num_images):
			dets = all_boxes[cls_ind][im_ind]
			if dets == []:
				continue
			# CPU NMS is much faster than GPU NMS when the number of boxes
			# is relative small (e.g., < 10k)
			# TODO(rbg): autotune NMS dispatch
			keep = nms(dets, thresh, force_cpu=True)
			if len(keep) == 0:
				continue
			nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
	return nms_boxes


def save_det(img_path, save_dir, cls_dets_all, cls_classes_all):
	lst = img_path.split('/')
	pred_dir = "{}/{}".format(save_dir, "preds")
	if not os.path.exists(pred_dir):
		os.mkdir(pred_dir)
	pred_path = pred_dir + "/" + lst[-1].strip()[:-3] + "txt"
	with open(pred_path, 'w') as f:
		for i in xrange(len(cls_classes_all)):
			cls = cls_classes_all[i]
			det = cls_dets_all[i]
			bbox = det[:4]
			score = det[-1]
			bbox = [int(x) for x in bbox]
			bbox[2] = bbox[2] - bbox[0] + 1
			bbox[3] = bbox[3] - bbox[1] + 1
			bbox = [str(x) for x in bbox]
			s = "{} {} {} {} {} {}\n".format(str(cls), bbox[0], bbox[1], bbox[2], bbox[3], score)
			f.write(s)


def test_net(net, img_label, thresh=0.05, vis=False, save_dir=None):
	num_classes = cfg.NUM_CLASSES
	with open(img_label, 'r') as f:
		lines = f.readlines()
	imgs = [line.split()[0] for line in lines]

	# timers
	_t = {'im_detect': Timer(), 'misc': Timer()}

	cnt = 0
	for i in xrange(len(imgs)):
		im = cv2.imread(imgs[i])
		_t['im_detect'].tic()
		scores, boxes = im_detect(net, im)
		_t['im_detect'].toc()

		_t['misc'].tic()
		cls_dets_all = np.empty((0, 5), dtype=np.float)
		cls_classes_all = np.empty((0,), dtype=np.int)
		for j in xrange(1, num_classes):
			inds = np.where(scores[:, j] > thresh)[0]
			if inds.size == 0:
				continue
			cls_scores = scores[inds, j]
			if cfg.AGNOSTIC:
				cls_boxes = boxes[inds, 4:8]
			else:
				cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
			cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
				.astype(np.float32, copy=False)
			keep = nms(cls_dets, cfg.TEST.NMS)
			cls_dets = cls_dets[keep, :]
			cls_dets_all = np.vstack((cls_dets_all, cls_dets))
			cls_classes_all = np.hstack((cls_classes_all, np.ones(len(keep), dtype=np.int) * j))
		if vis:
			vis_detections(im, cls_classes_all, cls_dets_all, cfg.TEST.THRESH)
		if save_dir:
			save_det(imgs[i], save_dir, cls_dets_all, cls_classes_all)
		cnt += 1
		if cnt % 100 == 0:
			print "processed {}".format(cnt)

		_t['misc'].toc()
