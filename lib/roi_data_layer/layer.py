import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
from multiprocessing import Process, Queue


class RoIDataLayer(caffe.Layer):
	def _shuffle_roidb_inds(self):
		if cfg.TRAIN.ASPECT_GROUPING:
			widths = np.array([r['width'] for r in self._roidb])
			heights = np.array([r['height'] for r in self._roidb])
			horz = (widths >= heights)
			vert = np.logical_not(horz)
			horz_inds = np.where(horz)[0]
			vert_inds = np.where(vert)[0]
			inds = np.hstack((
				np.random.permutation(horz_inds),
				np.random.permutation(vert_inds)))
			inds = np.reshape(inds, (-1, 2))
			row_perm = np.random.permutation(np.arange(inds.shape[0]))
			inds = np.reshape(inds[row_perm, :], (-1,))
			self._perm = inds
		else:
			self._perm = np.random.permutation(np.arange(len(self._roidb)))
		self._cur = 0

	def _get_next_minibatch_inds(self):
		if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
			self._shuffle_roidb_inds()

		db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
		self._cur += cfg.TRAIN.IMS_PER_BATCH
		return db_inds

	def _get_next_minibatch(self):
		if cfg.TRAIN.USE_PREFETCH:
			while 1:
				info = self._blob_queue.get()
				if info is not None:
					return info
		else:
			while 1:
				db_inds = self._get_next_minibatch_inds()
				minibatch_db = [self._roidb[i] for i in db_inds]
				info = get_minibatch(minibatch_db)
				if info is not None:
					return info

	def set_roidb(self, roidb):
		self._roidb = roidb
		self._shuffle_roidb_inds()
		if cfg.TRAIN.USE_PREFETCH:
			self._blob_queue = Queue(10)
			self._prefetch_process = BlobFetcher(self._blob_queue,
												 self._roidb,
												 self._num_classes)
			self._prefetch_process.start()

			# Terminate the child process when the parent exists
			def cleanup():
				print 'Terminating BlobFetcher'
				self._prefetch_process.terminate()
				self._prefetch_process.join()

			import atexit
			atexit.register(cleanup)

	def setup(self, bottom, top):
		self._num_classes = eval(self.param_str)["num_classes"]

		top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, cfg.TRAIN.SCALE, cfg.TRAIN.SCALE)
		top[1].reshape(1, 2)
		top[2].reshape(1, 6)

	def forward(self, bottom, top):
		im_blob, gt_boxes_blob, im_info = self._get_next_minibatch()
		# self.vis(im_blob, gt_boxes_blob)

		top[0].reshape(*(im_blob.shape))
		top[0].data[...] = im_blob.astype(np.float32, copy=False)
		top[1].reshape(*(im_info.shape))
		top[1].data[...] = im_info.astype(np.float32, copy=False)
		top[2].reshape(*(gt_boxes_blob.shape))
		top[2].data[...] = gt_boxes_blob.astype(np.float32, copy=False)

	def vis(self, im_blob, gt_boxes_blob):

		import cv2
		for i in xrange(im_blob.shape[0]):
			im = im_blob[i, :, :, :].transpose((1, 2, 0)).copy()
			im = im * cfg.TRAIN.DATA_AUG.NORMALIZER + cfg.TRAIN.DATA_AUG.MEAN
			im = im.astype(np.uint8)
			for bbox_info in gt_boxes_blob:
				batch_ind = bbox_info[-1]
				if batch_ind != i:
					continue
				cls = bbox_info[-2]
				bbox = bbox_info[:4]
				cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), int(im.shape[0] / 200.0))
				cv2.putText(im, str(cls), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			cv2.imshow("debug", im)
			if cv2.waitKey(-1) == -1:
				break

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass


class BlobFetcher(Process):
	def __init__(self, queue, roidb, num_classes):
		super(BlobFetcher, self).__init__()
		self._queue = queue
		self._roidb = roidb
		self._num_classes = num_classes
		self._perm = None
		self._cur = 0
		self._shuffle_roidb_inds()
		# fix the random seed for reproducibility
		np.random.seed(cfg.RNG_SEED)

	def _shuffle_roidb_inds(self):
		"""Randomly permute the training roidb."""
		# TODO(rbg): remove duplicated code
		self._perm = np.random.permutation(np.arange(len(self._roidb)))
		self._cur = 0

	def _get_next_minibatch_inds(self):
		"""Return the roidb indices for the next minibatch."""
		# TODO(rbg): remove duplicated code
		if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
			self._shuffle_roidb_inds()

		db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
		self._cur += cfg.TRAIN.IMS_PER_BATCH
		return db_inds

	def run(self):
		print 'BlobFetcher started'
		while True:
			db_inds = self._get_next_minibatch_inds()
			minibatch_db = [self._roidb[i] for i in db_inds]
			blobs = get_minibatch(minibatch_db)
			self._queue.put(blobs)
