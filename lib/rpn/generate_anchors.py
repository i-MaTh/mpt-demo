import numpy as np
from easydict import EasyDict as edict
from fast_rcnn.config import cfg


def generate_anchors(anchor_generator):
	"""
	Generate anchor (reference) windows by enumerating aspect ratios X
	scales wrt a reference (0, 0, 15, 15) window.
	"""
	base_size = cfg.FEAT_STRIDE
	if anchor_generator.POLICY == "SCALE_RATIO":
		# ratio= h/w
		ratios = anchor_generator.RATIOS
		scales = anchor_generator.SCALES
		scales = np.array([x * 1.0 / base_size for x in scales], dtype=np.float32)
		base_anchor = np.array([1, 1, base_size, base_size]) - 1
		ratio_anchors = _ratio_enum(base_anchor, ratios)
		anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
							 for i in xrange(ratio_anchors.shape[0])])

	if anchor_generator.POLICY == "WIDTH_HEIGHT":
		widths = anchor_generator.SLIDING_WINDOW_WIDTH
		heights = anchor_generator.SLIDING_WINDOW_HEIGHT
		anchors = np.zeros((4, len(widths)), dtype=np.float32)
		bss = np.ones(len(widths), dtype=np.float32) * base_size
		anchors[0, :] = (bss - widths) * 0.5
		anchors[1, :] = (bss - heights) * 0.5
		anchors[2, :] = (bss + widths - 2) * 0.5
		anchors[3, :] = (bss + heights - 2) * 0.5
		anchors = anchors.transpose()

	return anchors


def generate_anchors_ori(base_size=16, ratios=[0.5, 1, 2],
						 scales=2 ** np.arange(3, 6)):
	"""
	Generate anchor (reference) windows by enumerating aspect ratios X
	scales wrt a reference (0, 0, 15, 15) window.
	"""
	base_anchor = np.array([1, 1, base_size, base_size]) - 1
	ratio_anchors = _ratio_enum(base_anchor, ratios)
	anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
						 for i in xrange(ratio_anchors.shape[0])])
	return anchors


def _whctrs(anchor):
	"""
	Return width, height, x center, and y center for an anchor (window).
	"""

	w = anchor[2] - anchor[0] + 1
	h = anchor[3] - anchor[1] + 1
	x_ctr = anchor[0] + 0.5 * (w - 1)
	y_ctr = anchor[1] + 0.5 * (h - 1)
	return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
	"""
	Given a vector of widths (ws) and heights (hs) around a center
	(x_ctr, y_ctr), output a set of anchors (windows).
	"""

	ws = ws[:, np.newaxis]
	hs = hs[:, np.newaxis]
	anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
						 y_ctr - 0.5 * (hs - 1),
						 x_ctr + 0.5 * (ws - 1),
						 y_ctr + 0.5 * (hs - 1)))
	return anchors


def _ratio_enum(anchor, ratios):
	"""
	Enumerate a set of anchors for each aspect ratio wrt an anchor.
	"""

	w, h, x_ctr, y_ctr = _whctrs(anchor)
	size = w * h
	size_ratios = size / ratios
	ws = np.round(np.sqrt(size_ratios))
	hs = np.round(ws * ratios)
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors


def _scale_enum(anchor, scales):
	"""
	Enumerate a set of anchors for each scale wrt an anchor.
	"""

	w, h, x_ctr, y_ctr = _whctrs(anchor)
	ws = w * scales
	hs = h * scales
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors


if __name__ == '__main__':
	anchor_generator = edict()
	# anchor_generator.POLICY= "SCALES_RATIOS"
	anchor_generator.POLICY = "WIDTH_HEIGHT"
	anchor_generator.RATIOS = (0.5, 1, 2)
	anchor_generator.SCALES = (16 * 8, 16 * 16, 16 * 32)
	anchor_generator.SLIDING_WINDOW_WIDTH = [181.0, 362.0, 724.0, 128.0, 256.0, 512.0, 91.0, 181.0, 362.0]
	anchor_generator.SLIDING_WINDOW_HEIGHT = [90.0, 181.0, 362.0, 128.0, 256.0, 512.0, 182.0, 362.0, 724.0]
	# wlst=[]
	# hlst=[]
	# for ratio in anchor_generator.RATIOS:
	# 	for scale in anchor_generator.SCALES:
	# 		w=np.round(np.sqrt(scale*scale/ratio))
	# 		h= np.round(w*ratio)
	# 		wlst.append(w)
	# 		hlst.append(h)
	# print wlst
	# print hlst

	import time

	t = time.time()
	generate_anchors(anchor_generator)
	t1 = time.time()
	generate_anchors_ori()
	t2 = time.time()
	print t1 - t
	print t2 - t1
