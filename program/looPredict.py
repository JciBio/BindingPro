#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 11:40:07 2018

@author: weizhong
"""

import scipy.io as sio
#from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import tensorflow as tf

#def cnnByTensor(x_train, x_test, y_train, y_test,batch_size=100):
    #define two placeholder
#    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    #每个批次的大小
    #batch_size=100
    #计算一共多少个批次
   # n_batch=len(x_train)//batch_size
#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=1) #生成一个截断的正态分布
    return tf.Variable(initial)
    #return tf.Variable(tf.zeros(shape))

#初始化偏置
def bias_variable(shape):
    initial = tf.truncated_normal(shape,stddev=1)
    return tf.Variable(initial)
    #return tf.Variable(tf.zeros(shape))

#卷积层
def conv2d(x,W):
    #input tensor of shape [batch, in_height, in_width, in_channels]
    #filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #strides[0]=strides[3]=1. strides[1]代表ｘ方向的步长，strids[2]代表ｙ方向的步长
    #padding: A string from "SAME", "VALID"
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

#池化层
def max_pool(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def cnn(x_train,x_test,y_train,y_test,n_batch):
    N = len(x_train)
    batch_size = N//n_batch
    #定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 460])#23*20
    y = tf.placeholder(tf.float32, [None, 2])

    #改变x的格式转为４Ｄ的向量【batch, in_height, in_width, in_channels]
    x_image = tf.reshape(x,[-1, 23, 20 ,1])

    #初始化第一个卷积层的权值和偏量
    W_conv1 = weight_variable([5,5,1,16])#5*5的采样窗口，３２个卷积核从4个平面抽取特征
    b_conv1 = bias_variable([16])#每一个卷积核一个偏置值

    #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)#进行max-pooling,12-by-40

    #初始化第二个卷积层的权值和偏置
    W_conv2 = weight_variable([5,5,16,32]) #5*5的采样窗口，64个卷积核从32个平面抽取特征
    b_conv2 = bias_variable([32]) #每一个卷积核一个偏置值

    #把H_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)#6-by-10

    #23*160的图片第一次卷积后还是23*20,第一次池化后变为12*10
    #第二次卷积后为12*10,第二次池化后变为6*5
    #进过上面操作后得到64张6*5的平面

    #初始化第一全链接层的权值
    W_fc1 = weight_variable([6*5*32,500]) #上一层有6*10*64个神经元,全连接层有1024个神经元
    b_fc1 = bias_variable([500])

    #把池化层2的输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2,[-1,6*5*32])
    #求第一个全连接层的输出
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #keep_prob用了表示神经元的输出概率
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #初始化第二个全连接层
    W_fc2 = weight_variable([500,2])
    b_fc2 = bias_variable([2])

    #计算输出
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    #交叉熵代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
    #使用AdamOptimizer进行优化
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #结果存放在一个布尔列表中
    correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    #求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(3):
            for batch in range(n_batch):
                if (batch+1)*batch_size > N:
                    batch_xs = x_train[batch*batch_size:]
                    batch_ys = y_train[batch*batch_size:]
                else:
                    batch_xs = x_train[batch*batch_size:(batch+1)*batch_size]
                    batch_ys = y_train[batch*batch_size:(batch+1)*batch_size]
                #batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
            
            acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob:1.0})
            print("Iter " + str(epoch) + "Testing Accuracy=" + str(acc))
            
#load benchmark dataset
data = sio.loadmat('../data/PDNA-224-PSSM-Norm-11.mat')

X = data['data']
Y = data['target']
X = X.reshape(57348,-1)
#loo = LeaveOneOut()
kf = KFold(n_splits=5)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    #print("len(X_train)={},len(Y_test)={}".format(len(X_train),len(Y_test)))
    cnn(X_train,X_test,Y_train,Y_test,100)

