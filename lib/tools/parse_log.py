import re
import sys
import collections
import subprocess as sp
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
from easydict import EasyDict as edict
import time
import scipy

field_list = ["iter", "loss", "accuracy", "loss_bbox", "loss_cls", "rpn_loss_bbox", "rpn_loss_cls"]
field_tobe_list = ["accuracy", "loss_bbox", "loss_cls", "rpn_loss_bbox", "rpn_loss_cls"]
show_field_list = ["loss", "accuracy", "loss_bbox", "loss_cls", "rpn_loss_bbox", "rpn_loss_cls"]


def smooth(x, smooth_len=256):
	box = np.ones(smooth_len) / smooth_len
	x_smooth = np.convolve(x, box, mode='same')
	for i in range(smooth_len / 2 + 1):
		start = 0
		end = i + smooth_len / 2
		x_smooth[i] = np.sum(x[start:end]) * 1.0 / (end - start)
	for i in range(-1, -smooth_len / 2, -1):
		start = i - smooth_len / 2
		x_smooth[i] = np.sum(x[start:]) * 1.0 / (-start)
	return x_smooth


# def smooth_array(x,smooth_len=256):
# 	x_smooth= scipy.ndimage.filters.convolve1d(x,np.ones(smooth_len,dtype=np.float32),0, mode='constant')
#



def cut_log(log):
	indexlist = []
	with open(log, 'r') as f:
		index = 0
		for line in f:
			if "Iteration" in line and "loss" in line:
				indexlist.append(index)
			index += 1
	tmp_list = indexlist[1:]
	tmp_list.append(index)
	cut_indexlist = zip(indexlist, tmp_list)
	return cut_indexlist


def dowith_rpn_partlog(partlog):
	info = np.zeros((len(field_tobe_list), 30), dtype=np.float)

	field_cnt = np.zeros(len(field_tobe_list), dtype=np.int32)
	for line in partlog:
		if "Iteration" in line and "loss" in line:
			lst = line.split()
			iter = int(lst[-4][:-1])
			loss = float(lst[-1])

		for index, field in enumerate(field_tobe_list):
			if field + " =" in line:
				info[index, field_cnt[index]] = float(line.split(field+ ' =')[1].split('(')[0])
				field_cnt[index] += 1

	info_sum = np.sum(info, 1)
	info_mean = info_sum * 1.0 / field_cnt
	info_list = [iter, loss]
	info_list.extend(info_mean)
	return info_list


def is_filter_partlog(partlog):
	flag = False

	tmpstr = "".join(partlog)
	if "Testing net" in tmpstr:
		flag = True

	return flag


def parse_log(log):
	lines = open(log, 'r').readlines()
	part_log_indexlist = cut_log(log)
	all_info = np.zeros((len(part_log_indexlist), len(field_list)))
	cnt = 0
	for start, end in part_log_indexlist:
		partlog = lines[start:end]
		if is_filter_partlog(partlog):
			continue
		info = dowith_rpn_partlog(partlog)
		all_info[cnt, :] = info
		cnt += 1
	return all_info[:cnt, :]


def show_log(all_info, is_show):
	iter_index = field_list.index("iter")
	iter_list = all_info[:, iter_index]
	plt.figure()
	for field in show_field_list:
		index = field_list.index(field)
		lst = all_info[:, index]
		lst = smooth(lst, 512)
		n = min(len(iter_list), len(lst))
		plt.plot(iter_list[:n], lst[:n])
		plt.hold(True)
	plt.hold(False)
	plt.grid(True)
	plt.xlim([min(iter_list), max(iter_list) + (max(iter_list) - min(iter_list)) * 0.5])
	plt.ylim([0, 1.5])
	plt.xlabel("iter")

	plt.legend(show_field_list)
	plt.savefig("logs/run_info.png")
	if is_show:
		plt.show()


log = sys.argv[1]
is_show= True
if len(sys.argv)==3 and sys.argv[2]=="noshow":
	is_show= False

d = parse_log(log)
show_log(d, is_show)




# log_loss_file = "logs/log_loss"
# cmd_string = "cat logs/log | grep Iteration |grep -a loss > {}".format(log_loss_file)
# # sp.Popen(cmd_string, shell=True)
# os.system(cmd_string)
#
# loss_regx = r"Iteration (\d+), loss = (\d+\.\d+)"
# iter_list = []
# loss_list = []
# f = open(log_loss_file, 'r')
# lines = f.readlines()
# for line in lines:
# 	match = re.search(loss_regx, line.strip())
# 	if match:
# 		iter = int(match.group(1))
# 		loss = float(match.group(2))
# 		iter_list.append(iter)
# 		loss_list.append(loss)
#
# f.close()
#
# if len(sys.argv) == 1:
# 	ignore_num = 0
# else:
# 	ignore_num = int(sys.argv[1])
#
# iter_list = np.array(iter_list)
# loss_list = np.array(loss_list)
# I = iter_list > ignore_num
# iter_list = iter_list[I]
# loss_list = loss_list[I]
#
# loss_list = smooth(loss_list, 512)
# plt.figure(1)
# plt.subplot(111)
# plt.plot(iter_list, loss_list)
# plt.xlabel("iter")
# plt.ylabel("loss")
# plt.title("loss/iter")
# plt.savefig("results/loss_iter.png")
# plt.show()
# def dowith_rpn_partlog(partlog):
# 	acc_list = []
# 	cls_loss_list = []
# 	reg_loss_list = []
# 	precision_list = []
# 	recall_list = []
#
# 	iter_regx = r"Iteration (\d+), loss = (\d+\.\d+)"
# 	acc_regx = r"acc = (\d+\.\d+)"
# 	cls_loss_regx = r"cls_loss = (\d+\.\d+)"
# 	reg_loss_regx = r"reg_loss = (\d+\.\d+)"
# 	precision_regx = r"precision = (\d+\.\d+)"
# 	recall_regx = r"recall = (\d+\.\d+)"
#
# 	field_tobe_list = ["acc", "cls_loss", "reg_loss", "precision", "recall"]
# 	for line in partlog:
# 		line = line.strip()
# 		train_loss_match = re.search(iter_regx, line)
# 		if train_loss_match:
# 			iter = int(train_loss_match.group(1))
# 			loss = float(train_loss_match.group(2))
#
# 		for expr in field_tobe_list:
# 			exec ("{}_match = re.search({}_regx, \"{}\")".format(expr, expr, line))
# 			if eval("{}_match".format(expr)):
# 				x = "{:.3f}".format(float(eval("{}_match.group(1)".format(expr))))
# 				exec ("{}_list.append({})".format(expr, x))
#
# 	d = edict()
#
# 	for field in field_tobe_list:
# 		if eval("{}_list==[]".format(field)):
# 			return d
#
# 	for expr in field_tobe_list:
# 		exec ("{}_list=np.array({}_list)".format(expr, expr))
# 		exec ("{}= np.mean({}_list)".format(expr, expr))
# 		x = eval(expr)
# 		exec ("{}= {:.3f}".format(expr, x))
# 		exec ("d.{}={}".format(expr, expr))
# 	d.iter = iter
# 	d.loss = loss
#
# 	return d
