import os
import sys
from collections import OrderedDict
import pickle
import h5py
import pandas as pd

def txt2pkl():
	inpath = sys.argv[1]
	outpath = sys.argv[2]

	confs = OrderedDict()
	with open(inpath, 'r') as infile:
		for line in infile:
			tmp = line.strip('\n').split(',')
			#key = tmp[0]
			key = tmp[0].replace('-', '_crop-')
			value = tmp[5]
			confs[key] = value
	
	pickle.dump(confs, open(outpath, 'wb'))

def hdf2txt():
	inpath = sys.argv[1]
	outpath = sys.argv[2]

	outfile = open(outpath, 'wb')
	with open(inpath, 'r') as infile:
		for line in infile:
			#print line.strip('\n')		
			dets = h5py.File(line.strip('\n'), 'r')
			img_id = '/'.join(line.strip('\n').split('/')[-2:]).split('.')[0]
			for idx, det in enumerate(dets['person'][...]):
				outfile.write('{0},{1},{2},{3},{4},{5}\n'.format(
							img_id + '_{0}.jpg'.format(idx), int(det[0]), 
							int(det[1]), int(det[2]), int(det[3]), det[4]))
			dets.close()
			#break

def create_nodes_lookup():
	'''
	str_path = sys.argv[1]
	save_path = sys.argv[2]

	nodes = OrderedDict()
	with open(str_path, 'r') as infile:
		for idx, line in enumerate(infile):
			key = '/'.join(line.strip('\n').split('/')[-2:])
			nodes[key] = idx

	pickle.dump(nodes, open(save_path, 'wb'))
	'''
	path = sys.argv[1]
	nodes = OrderedDict()
	with open(path, 'r') as infile:
		for idx, line in enumerate(infile):
			key1 = line.split(',')[0]
			if key1 not in nodes:
				nodes[key1] = len(nodes)
	
			key2 = line.split(',')[1]
			if key2 not in nodes:
				nodes[key2] = len(nodes)
	print 'nodes size: %d' % len(nodes)
	pickle.dump(nodes, open('mot_09_nodes.pkl', 'wb'))

def labeling():
	dets_path = sys.argv[1]
	labels_path = sys.argv[2]
	
	dets = pd.read_csv(dets_path, header=None, names = ['bbox_id', 'x1', 'y1', 'x2', 'y2', 'confs'])
	ml_res = h5py.File(labels_path, 'r')
	labels = ml_res['labels'][...]
	dets['labels'] = labels

	dets.to_csv('result.csv', index=None)


def filter_data():
	path = sys.argv[1]
	infile = open(path, 'r')
	outfile = open('mot16_09_pairs.txt', 'wb')
	for idx, line in enumerate(infile):
		#print line.split('/')[0]
		if line.split('/')[0] == 'MOT16-09':
			outfile.write(line)

def create_edgelist():
	pair_path = sys.argv[1]
	nodes_dic = pickle.load(open(sys.argv[2], 'r'))
	infile = open(pair_path, 'r')
	outfile = open('mot16_09_pairs.csv', 'wb')
	edgelist = []
	for line in infile:
		line_parts = line.strip('\n').strip('\r').split(',')
		tmp = []
		tmp.append(nodes_dic[line_parts[0]])
		tmp.append(nodes_dic[line_parts[1]])
		tmp.append(line_parts[2])
		edgelist.append(tmp)
	
	import csv
	writer = csv.writer(outfile)
	writer.writerows(edgelist)


def main():
	#txt2pkl()
	#hdf2txt()
	#create_nodes_lookup()
	#labeling()
	#filter_data()
	create_edgelist()

if __name__ == '__main__':
	main()



