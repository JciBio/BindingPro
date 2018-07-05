import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold
import tensorflow as tf

X_N = []
Y_N = []
X_P = []
Y_P = []

data = sio.loadmat('PDNA-224-ONEHOT-11.mat')
X = data['data']
data_size=len(X)
Y = data['target']

#print(type(X))


for sss in range(data_size):
    if Y[sss][1] == 1: #判定是否为正样本
        #print(X[sss],Y[sss])
        X_P.append(X[sss])
        Y_P.append(Y[sss])
        
    else:
        X_N.append(X[sss])
        Y_N.append(Y[sss])
    
#将正样本保存到mat文件中
P_dataset={}
P_dataset['data']=X_P
P_dataset['target'] = Y_P
sio.savemat('PDNA-224-ONEHOT-11-P.mat',P_dataset)

#将负样本保存到mat文件中
N_dataset={}
N_dataset['data']=X_N
N_dataset['target'] = Y_N
sio.savemat('PDNA-224-ONEHOT-11-N.mat',N_dataset)
print('yes!')
  