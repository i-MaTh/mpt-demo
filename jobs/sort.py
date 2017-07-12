"""
    SORT: A Simple, Online and Realtime Tracker
"""

from __future__ import print_function

from numba import jit
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import math


@jit
def iou(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1, y1, x2, y2]
    """
    x1 = max(bb_test[0], bb_gt[0])
    y1 = max(bb_test[1], bb_gt[1])
    x2 = min(bb_test[2], bb_gt[2])
    y2 = min(bb_test[3], bb_gt[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    ai = w * h
    au = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + \
        (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - ai

    return ai / float(au)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1, y1, x2, y2] and returns z 
    in the form [x, y, s, r] where x, y is the centre of box and s
    is the scale(area) and r is the aspect ratio.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)

    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre from [x, y, s, r] and returns 
    it in the form [x1, y1, x2, y2] where x1, y1 is the top left and
    x2, y2 is the bottom right.
    """
    w = math.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score]).reshape((1, 5))


class KalmanBoxTracker():
    """
    This class represents the internel state of individual tracked
    objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x = 7, dim_z = 4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],\
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox 
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_state(self):
        """
        Return the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding box)
    
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """ 
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort():
    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        Params:
        - dets: a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        
        NOTE: The number of objects returned may differ from the number of detections.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        # create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 5))

def parse_args():
	"""
    Parse input arguments.
	"""
	parser = argparse.ArgumentParser(description='SORT demo')
	parser.add_argument('--seqs', dest='seqs_dir', help='input sequences file', type=str)
	parser.add_argument('--dets', dest='dets_path', help='input detections file',type=str)
	parser.add_argument('--out', dest='out_path', help='output result file', type=str)
	parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
	args = parser.parse_args()    
	return args


def main():
    # all 
    seed = 1024
    np.random.seed(seed)
    args = parse_args()
    seqs_dir = args.seqs_dir
    dets_path = args.dets_path
    out_path = args.out_path
    display = args.display
	

    phase = 'train'
    duration_time = 0.
    total_frames = 0
    colours = np.random.rand(32, 3)
    if display:
        '''
        if not os.path.exists('mot_benchmark')
            print('\nERROR: mot_benchmark not found!\n\n')
            exit()
        '''
        plt.ion()
        fig = plt.figure()
	
	# if not os.path.exists('output'):
		#os.makedirs('output')
    
    mot_tracker = Sort() # create instance of the SORT tracker
    seq_dets = np.loadtxt(dets_path, delimiter=',') # load detections
    with open(out_path, 'wb') as outfile:
        for frame in range(int(seq_dets[:, 0].max())):
            frame += 1
            dets = seq_dets[seq_dets[:, 0] == frame-1, 2:7] # to do modify
            dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            total_frames += 1            
            if display:
				ax1 = fig.add_subplot(111, aspect='equal')
				img_path = seqs_dir + '/%06d.jpg' % frame
				img = io.imread(img_path)
				ax1.imshow(img)
				plt.title('Tracked Targets.')

            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            duration_time += cycle_time

            for d in trackers:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]), file=outfile)
                if display:
                    d = d.astype(np.int32)
                    ax1.add_patch(patches.Rectangle((d[0], d[1]),
                                                d[2] - d[0],
                                                d[3] - d[1],
                                                fill=False,lw=3,
                                                ec=colours[d[4]%32,:]
                                                )
                                )
                    ax1.set_adjustable('box-forced')

            if display:
                fig.canvas.flush_events()
                plt.draw()
                ax1.cla()


if __name__ == '__main__':
    main()






