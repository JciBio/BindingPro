import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import confusion_matrix

def rf(x_train, x_test, y_train):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    return y_pred
    
#load benchmark dataset
data = sio.loadmat('../data/PDNA-224-PSSM-Norm-11.mat')

X = data['data']
Y = data['target']

row = 57348
col = 460
X = X.reshape(row,-1)

Y_pred = np.zeros([row,2],dtype=np.int32)

kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
   
    Y_pred[test_index] = rf(X_train,X_test,Y_train)

correct = 0
for i in range(row):
    if (Y[i] == Y_pred[i]).all():
        correct += 1
print("correct accuracy: {}".format(correct/row))

Y1 = np.ndarray([row])
PY = np.ndarray([row])
for i in range(row):
    if Y[i][0] == 1:
        Y1[i] = 0
    else:
        Y1[i] = 1
    
    if Y_pred[i][0] == 1:
        PY[i] = 0
    else:
        PY[i] = 1
cnf_matrix = confusion_matrix(Y1, PY)

print(cnf_matrix)
recall = cnf_matrix[1][1] / (cnf_matrix[1][0] + cnf_matrix[1][1])
print('recall: ', recall)
#plot_confusion_matrix(cnf_matrix, [0, 1], cmap=plt.cm.Reds)




