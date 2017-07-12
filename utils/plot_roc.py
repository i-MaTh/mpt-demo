import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_roc_curve(gt, pred):
	print 'gt size: %d' % len(gt)
	fpr, tpr, threshold = metrics.roc_curve(gt, pred)
	roc_auc = metrics.auc(fpr, tpr)

	# method I: plt
	plt.title('Receiver Operating Characteristic')
	plt.plot(np.log10(fpr + 1e-5), tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.grid()
	plt.legend(loc = 'upper left')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('log(False Positive Rate)')
	plt.savefig('auc_curve2.jpg')

def plot_pr_curve(gt, pred):
	p, r, _ = metrics.precision_recall_curve(gt, pred)
	
	plt.title('Precision Recall Curve')
	plt.plot(r, p, 'g')
	plt.grid()
	plt.legend(loc = 'upper left')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.savefig('pr_curve.jpg')



