# -*- coding: utf-8 -*-
"""
Created on Tue May  8 08:34:46 2018

@author: weizhong lin
"""

import tensorflow as tf
import scipy.io as sio
from sklearn.model_selection import KFold
import numpy as np
import random
       
def nn(x_train, x_test,y_train,  y_test, batch_size):
    N = len(x_train)
    n_batch = N//batch_size
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None,460])
    y = tf.placeholder(tf.float32,[None,2])

    #创建一个简单的神经网络
    W = tf.Variable(tf.random_normal( [460,2],stddev=1))
    b = tf.Variable(tf.random_normal([2],stddev=1))
    prediction = tf.nn.softmax(tf.matmul(x,W)+b)

    #二次代价函数
    loss = tf.reduce_mean(tf.square(y-prediction))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
    #使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    #初始化变量
    init = tf.global_variables_initializer()

    #结果存放在一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
    #求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(3):
            for batch in range(n_batch):
                if (batch+1)*batch_size > N:
                    batch_xs = x_train[batch*batch_size:]
                    batch_ys = y_train[batch*batch_size:]
                else:
                    batch_xs = x_train[batch*batch_size:(batch+1)*batch_size]
                    batch_ys = y_train[batch*batch_size:(batch+1)*batch_size]
                
                sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys})
                
            acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test})
            print("Iter " + str(epoch) + "Testing Accuracy=" + str(acc))


#load benchmark dataset
data = sio.loadmat('../data/PDNA-224-PSSM-Norm-11.mat')

#X = data['data']
X = np.ndarray((57348,460))
for i in range(57348):
    for j in range(460):
        X[i][j] = random.random()
Y = data['target']
#X = X.reshape(57348,-1)
#loo = LeaveOneOut()
kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    #print("len(X_train)={},len(Y_test)={}".format(len(X_train),len(Y_test)))
    nn(X_train,X_test,Y_train,Y_test,100)