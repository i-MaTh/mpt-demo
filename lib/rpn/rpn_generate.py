from utils.timer import Timer
import numpy as np
import cv2
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
import sys, os


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
	blobs, im_scale = _get_blobs(im)

	net.blobs['data'].reshape(*(blobs['data'].shape))
	net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

	forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
	forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
	blobs_out = net.forward(**forward_kwargs)

	boxes = net.blobs['rois'].data.copy()[:, 1:] * 1.0 / im_scale
	scores = blobs_out['rpn_scores']
	return scores, boxes


def apply_nms(dets, thresh):
	if dets == []:
		return []
	keep = nms(dets, thresh, force_cpu=True)
	if len(keep) == 0:
		return []
	return dets[keep, :]


def vis_detections(im, dets):
	window = "test"
	if dets != []:
		boxes = dets[:, :4]
		scores = dets[:, -1]
		cv2.namedWindow(window, cv2.WINDOW_NORMAL)
		im = im[:, :, (2, 1, 0)]
		im = im.copy()
		num = len(scores)
		color = (0, 255, 0)
		for i in xrange(num):
			box = boxes[i, :4]
			box = [int(x) for x in box]
			score = scores[i]
			cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color, int(im.shape[0] / 150.0))
		# cv2.putText(im, str(score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
	cv2.imshow(window, im)
	if cv2.waitKey(-1) == 27:
		sys.exit(0)


def save_det(img_path, save_dir, dets):
	lst = img_path.split('/')
	pred_dir = "{}/{}".format(save_dir, "preds")
	if not os.path.exists(pred_dir):
		os.mkdir(pred_dir)
	pred_path = pred_dir + "/" + lst[-1].strip("jpg") + "txt"
	with open(pred_path, 'w') as f:
		for i in xrange(len(dets)):
			bbox = dets[i, :4]
			score = dets[i, -1]
			bbox = [int(x) for x in bbox]
			bbox[2] = bbox[2] - bbox[0] + 1
			bbox[3] = bbox[3] - bbox[1] + 1
			bbox = [str(x) for x in bbox]
			s = "{} {} {} {} {}\n".format(bbox[0], bbox[1], bbox[2], bbox[3], str(score))
			f.write(s)


def rpn_test_net(net, img_label, score_thresh, nms_thresh, vis=False, save_dir=None):
	with open(img_label, 'r') as f:
		lines = f.readlines()
	imgs = [line.split()[0] for line in lines]

	_t = {'im_detect': Timer(), 'misc': Timer()}

	cnt = 0
	for i in xrange(len(imgs)):
		im = cv2.imread(imgs[i])
		_t['im_detect'].tic()
		scores, boxes = im_detect(net, im)
		_t['im_detect'].toc()

		_t['misc'].tic()
		inds = np.where(scores > score_thresh)[0]
		if inds.size == 0:
			dets = []
		else:
			scores = scores[inds]
			boxes = boxes[inds, :]
			dets = np.hstack((boxes, scores[:, np.newaxis]))
		dets = apply_nms(dets, nms_thresh)
		if vis:
			vis_detections(im, dets)
		if save_dir:
			save_det(imgs[i], save_dir, dets)
		cnt += 1
		if cnt % 100 == 0:
			print "processed {}".format(cnt)

		_t['misc'].toc()

	print 'im_detect time: {:.3f}s  misc time: {:.3f}s' \
		.format(_t['im_detect'].average_time, _t['misc'].average_time)
	print
