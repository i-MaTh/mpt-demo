import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

# in yaml, only has list, no tuple!
__C = edict()
cfg = __C

__C.TRAIN = edict()

__C.TRAIN.UNNORMALIZED_SNAPSHOT_ITERS = 5000
__C.TRAIN.SCALE = 240
__C.TRAIN.IMS_PER_BATCH = 1
# Minibatch size (number of regions of interest [ROIs], which sends to ROI pooling layer)
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25
__C.TRAIN.FG_THRESH = 0.5
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

__C.TRAIN.USE_FLIPPED = True
__C.TRAIN.SNAPSHOT_ITERS = 5000

__C.TRAIN.USE_PREFETCH = False

__C.TRAIN.DATA_AUG = edict()
__C.TRAIN.DATA_AUG.MIN_SIZE = 8
__C.TRAIN.DATA_AUG.NORMALIZER = 1.0
__C.TRAIN.DATA_AUG.MEAN_POLICY = "CONST"
__C.TRAIN.DATA_AUG.MEAN = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.TRAIN.DATA_AUG.CROP_RATIO_MIN_BOUND = 1.0
__C.TRAIN.DATA_AUG.GT_CROP_MIN_RATIO = 0.7

__C.TRAIN.ROI_PER_IMG = 128

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
# only used in proposal target layer, so rfcn_bbox should be decode when testing
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
# only used in sampling Rois to compute rpn_loss
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Deprecated (outside weights), role as mask
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
# #positive and negative bbox weight
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

#
# Testing options
#

__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALE = 600
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3
## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
__C.TEST.THRESH = 0.9


# anchor generator
__C.ANCHOR_GENERATOR = edict()
__C.ANCHOR_GENERATOR.POLICY = "SCALE_RATIO"
# __C.ANCHOR_GENERATOR.SCALES = (32, 64, 128)
__C.ANCHOR_GENERATOR.SCALES = (128, 256, 512)
__C.ANCHOR_GENERATOR.RATIOS = (0.5, 1, 2)
__C.ANCHOR_GENERATOR.SLIDING_WINDOW_WIDTH = (32,)
__C.ANCHOR_GENERATOR.SLIDING_WINDOW_HEIGHT = (32,)

__C.DEBUG = edict()
__C.DEBUG.RPN_ANCHOR_TARGET_SHOW = False
__C.DEBUG.RPN_ANCHOR_TARGET_NEG_SHOW = False
__C.DEBUG.RPN_PROPOSAL_SHOW = False
__C.DEBUG.RPN_PROPOSAL_CHOSEN_SHOW = False
__C.DEBUG.RPN_PROPOSAL_NEG_CHOSEN_SHOW = False
#
# MISC
#
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with

__C.CLASSES = ('__background__',)
__C.MULTICLASS_SPLIT = False
__C.NUM_CLASSES = 2
__C.POSITION_NUM = 3
__C.FEAT_STRIDE = 16
__C.ALLOWED_BORDER = 0
__C.AGNOSTIC = True

__C.SOLVER = edict()
__C.SOLVER.base_lr = 0.001
__C.SOLVER.max_epoch = 50
__C.SOLVER.step_epoch= 5
__C.SOLVER.snapshot= 5000

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0


def get_output_dir(imdb, net=None):
	"""Return the directory where experimental artifacts are placed.
	If the directory does not exist, it is created.

	A canonical path is built using the name from an imdb and a network
	(if not None).
	"""
	outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
	if net is not None:
		outdir = osp.join(outdir, net.name)
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	return outdir


def _merge_a_into_b(a, b):
	"""Merge config dictionary a into config dictionary b, clobbering the
	options in b whenever they are also specified in a.
	"""
	if type(a) is not edict:
		return

	for k, v in a.iteritems():
		# a must specify keys that are in b
		if not b.has_key(k):
			raise KeyError('{} is not a valid config key'.format(k))

		# the types must match, too
		old_type = type(b[k])
		if old_type is not type(v):
			if isinstance(b[k], np.ndarray):
				v = np.array(v, dtype=b[k].dtype)
			else:
				raise ValueError(('Type mismatch ({} vs. {}) '
								  'for config key: {}').format(type(b[k]),
															   type(v), k))

		# recursively merge dicts
		if type(v) is edict:
			try:
				_merge_a_into_b(a[k], b[k])
			except:
				print('Error under config key: {}'.format(k))
				raise
		else:
			b[k] = v


def cfg_from_file(filename):
	"""Load a config file and merge it into the default options."""
	import yaml
	with open(filename, 'r') as f:
		yaml_cfg = edict(yaml.load(f))

	_merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
	"""Set config keys via list (e.g., from command line)."""
	from ast import literal_eval
	assert len(cfg_list) % 2 == 0
	for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
		key_list = k.split('.')
		d = __C
		for subkey in key_list[:-1]:
			assert d.has_key(subkey)
			d = d[subkey]
		subkey = key_list[-1]
		assert d.has_key(subkey)
		try:
			value = literal_eval(v)
		except:
			# handle the case when v is a string literal
			value = v
		assert type(value) == type(d[subkey]), \
			'type {} does not match original type {}'.format(
				type(value), type(d[subkey]))
		d[subkey] = value
