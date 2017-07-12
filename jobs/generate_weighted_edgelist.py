import os, sys
import numpy as np
import pickle
import csv


def create_edgelists(pairs, nodes_dic, weights, save_path):
	edgelists = []
	for idx, p in enumerate(pairs):
		tmp = []
		tmp.append(nodes_dic[p[0]])
		tmp.append(nodes_dic[p[1]])
		tmp.append(weights[idx])
		edgelists.append(tmp)

	with open(save_path, 'wb') as outfile:
		writer = csv.writer(outfile)
		writer.writerows(edgelists)

def main():
	assert len(sys.argv) > 1, 'please input (pair_path, node_path, weight_path, save_path)'
	pair_path = sys.argv[1]
	node_path = sys.argv[2]
	weight_path = sys.argv[3]
	save_path = sys.argv[4]

	pairs = []
	with open(pair_path, 'r') as infile:
		for line in infile:
			line_parts = line.strip('\n').strip('\r').split(',')
			tmp = []
			tmp.append(line_parts[0])
			tmp.append(line_parts[1])
			pairs.append(tmp)
	
	nodes = pickle.load(open(node_path, 'r'))
	weights = []
	with open(weight_path, 'r') as infile:
		for p in infile:
			w = np.log(float(p) / (1 - float(p)) + 1e-6)
			weights.append(w)

	create_edgelists(pairs, nodes, weights, save_path)
	

if __name__ == '__main__':
	main()


	




