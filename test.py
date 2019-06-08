
import numpy as np
from sklearn.model_selection import KFold

X = np.array([[1,2],[3,4],[5,6],[7,8]])
y = np.array([1,2,3,4,5])

kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)
for train_index, test_index in kf.split(X):
    print("Train:",train_index,"Test:",test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


################ Pytorch ##################################################
import torch

'''torch.where(condition, x, y)
  When condition is true, yield x, otherwise yield y
'''
x = torch.randn(3,2)
y = torch.ones(3,2)
torch.where((x>0)&(y>0.1),x,y)


'''torch.argmax
 input:
   input(Tensor): the input tensor
   dim(int): the dimension to reduce. if 'None', the argmax of the flattened input is returned
   keepdim(bool): whether the output tensors have dim retained or not. Ignored if dim=None
 output:
   return the indices of the maximum values of a tensor across a dimension
'''
a = torch.randn(4,4)
torch.argmax(a)
torch.argmax(a,dim=1)





