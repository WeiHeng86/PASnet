import numpy as np
import torch
from sklearn.model_selection import KFold

from DataLoader import load_data, load_pathway
from EvalFunc import auc, f1
#from Train import trainPASNet
from Train2 import trainPASNet

dtype = torch.FloatTensor
''' Net Settings'''
In_Nodes = 113438 ###number of SNPs
Gene_Nodes = 11145 ###number of genes
Pathway_Nodes = 662 ###number of pathways
Hidden_Nodes = 100 ###number of hidden nodes
Out_Nodes = 2 ###number of hidden nodes in the last hidden layer
''' Initialize '''
nEpochs = 10000 ###for training 10000
Dropout_Rates = [0.8, 0.8, 0.7] ###sub-network setup
''' load data  and pathway '''
pathway_mask = load_pathway("/home/outlier/u43/huan1388/Documents/Git/data_snp/pathway-gene-mask.csv", dtype)
gene_mask = load_pathway("/home/outlier/u43/huan1388/Documents/Git/data_snp/gene-snp-mask.csv", dtype)

N = 1 # number of repeated times
K = 5 # number of folds
opt_lr = 1e-4
opt_l2 = 3e-4
test_auc = []
test_f1 = []
X_train, Y_train = load_data("/home/outlier/u43/huan1388/Documents/Git/data_snp/train.csv", dtype)
X_test, Y_test = load_data("/home/outlier/u43/huan1388/Documents/Git/data_snp/validation.csv", dtype)

for replicate in range(N):
	i = 0
	kf = KFold(n_splits=K)
	for train_index, val_index in kf.split(X_train):
		print("replicate: ", replicate, "fold: ", i)
		x_train, x_val = X_train[train_index,], X_train[val_index,]
		y_train, y_val = Y_train[train_index,], Y_train[val_index,]
		# x_train, y_train = load_data("data/train_"+str(replicate)+"_"+str(fold)+".csv", dtype)
		# x_test, y_test = load_data("data/std_test_"+str(replicate)+"_"+str(fold)+".csv", dtype)

		pred_train, pred_test, loss_train, loss_test = trainPASNet(x_train, y_train, x_val, y_val, gene_mask, pathway_mask, \
															In_Nodes, Gene_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
															opt_lr, opt_l2, nEpochs, Dropout_Rates, optimizer = "Adam")
		###if gpu is being used, transferring back to cpu
		if torch.cuda.is_available():
			pred_test = pred_test.cpu().detach()
		###
		pred_test = pred_test.cpu().detach()
		np.savetxt("output/PASNet_p                                                                                                                                                              red_"+str(replicate)+"_"+str(i)+".txt", pred_test.numpy(), delimiter = ",")
		auc_te = auc(y_val, pred_test)
		f1_te = f1(y_val, pred_test)
		print("AUC in Test: ", auc_te, "F1 in Test: ", f1_te)
		test_auc.append(auc_te)
		test_f1.append(f1_te)
		i = i+1
		
np.savetxt("PASNet_AUC.txt", test_auc, delimiter = ",")
np.savetxt("PASNet_F1.txt", test_f1, delimiter = ",")
