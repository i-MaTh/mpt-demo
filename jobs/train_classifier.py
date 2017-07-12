import numpy as np
from sklearn import model_selection as ms
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.svm import SVC, LinearSVC
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pandas as pd
from plot_roc import plot_roc_curve, plot_pr_curve
import os, sys
from datetime import datetime

seed = 32
np.random.seed(seed)

def svm_clf(train, test):
	train_y = train.label
	train_x = train.drop('label', axis = 1)

	#test_x = test
	train_x, test_x, train_y, test_y = ms.train_test_split(train_x, train_y, test_size = 0.3, random_state = seed)
	print 'train size: %d' % len(train_y)
	print 'test size: %d' % len(test_x)

	clf = LogisticRegression(penalty='l2', C=0.01, max_iter=300, n_jobs=-1)
	clf.fit(train_x, train_y)
	#preds = clf.predict_proba(test_x)
	preds = clf.predict(test_x)
	pos = preds
	#print preds
	#pos = [p[1] for p in preds]
	
	#df = pd.DataFrame()
	#df['pred'] = pos # probabilities of 1
	#df.to_csv('result.csv', index = None)
	print(classification_report(test_y, pos))
	#auc = roc_auc_score(test_y, pos)
	#print("AUC is %f" % auc)
	#plot_roc_curve(test_y, pos)
	#plot_pr_curve(test_y, pos)

def main():
	assert len(sys.argv) > 1, 'please input (train_path, test_path)'
	train_path = sys.argv[1]
	train = pd.read_csv(train_path, header = None, names = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'label'])
	#test_path = sys.argv[2]
	#test = pd.read_csv(test_path, header = None, names = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5'])
	test = []
	start = datetime.now()
	svm_clf(train, test)
	end = datetime.now()
	duration = end - start
	print 'time cost: %s' % duration


if __name__ == '__main__':
	main()
