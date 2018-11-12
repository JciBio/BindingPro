#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDNA-迁移学习-逐个训练集传递

"""

import scipy.io as sio
#from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import random



#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1) #生成一个截断的正态分布
    return tf.Variable(initial)

#初始化偏置
def bias_variable(shape):
    #initial = tf.truncated_normal(shape,stddev=0.1)
    initia2 = tf.constant(0.1,shape=shape)
    return tf.Variable(initia2)

#卷积层
def conv2d(x,W):
    #input tensor of shape [batch, in_height, in_width, in_channels]
    #filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #strides[0]=strides[3]=1. strides[1]代表ｘ方向的步长，strids[2]代表ｙ方向的步长
    #padding: A string from "SAME", "VALID"
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def cnn(x_train,x_test,y_train,y_test,n_batch,is_train,num,a):
     
    N = len(x_train)
    batch_size = N//n_batch
    
    #定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 460])#28*28
    y = tf.placeholder(tf.float32, [None, 2])

    #改变x的格式转为４Ｄ的向量【batch, in_height, in_width, in_channels]
    x_image = tf.reshape(x,[-1, 23, 20 ,1])

    #初始化第一个卷积层的权值和偏量
    W_conv1 = weight_variable([5,5,1,32])#5*5的采样窗口，３２个卷积核从１个平面抽取特征
    b_conv1 = bias_variable([32])#每一个卷积核一个偏置值

    #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)#进行max-pooling

    #初始化第二个卷积层的权值和偏置
    W_conv2 = weight_variable([5,5,32,64]) #5*5的采样窗口，64个卷积核从32个平面抽取特征
    b_conv2 = bias_variable([64]) #每一个卷积核一个偏置值

    #把H_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #23*20的图片第一次卷积后还是23*20,第一次池化后变为12*10
    #第二次卷积后为12*10,第二次池化后变为6*5
    #进过上面操作后得到64张6*5的平面

    #初始化第一全链接层的权值
    W_fc1 = weight_variable([6*5*64,1024]) #上一层有7*7*64个神经元,全连接层有1024个神经元
    b_fc1 = bias_variable([1024])

    #把池化层2的输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2,[-1,6*5*64])
    #求第一个全连接层的输出
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #keep_prob用了表示神经元的输出概率
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #初始化第二个全连接层
    W_fc2 = weight_variable([1024,2])
    b_fc2 = bias_variable([2])

    #计算输出
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    #交叉熵代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))\
#     +tf.contrib.layers.l1_regularizer(0.0001)(W_fc2)
    
    #使用AdamOptimizer进行优化
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    
    #结果存放在一个布尔列表中
    correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    #求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        info="模型第一次训练"
        saver=tf.train.Saver(max_to_keep=3)#保存模型 保存最新的模型
        
        if not is_train :  #判断是否为第一次训练，不是则加载最新的模型
            
           
            ckpt = tf.train.get_checkpoint_state('one_hot_logs/')                          # 通过检查点文件锁定最新的模型
            
            print(ckpt.model_checkpoint_path)
            saver1 = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')   # 载入图结构，保存在.meta文件中
            model_file=tf.train.latest_checkpoint('one_hot_logs/')
            saver1.restore(sess,model_file)
            info='加载最新的模型'
            
        print(info)
        for epoch in range(50): 
            for batch in range(n_batch):
                #训练模型

                if (batch+1)*batch_size > N:
                        batch_xs = x_train[batch*batch_size:N]
                        batch_ys = y_train[batch*batch_size:N]
                else:
                        batch_xs = x_train[batch*batch_size:(batch+1)*batch_size]
                        batch_ys = y_train[batch*batch_size:(batch+1)*batch_size]
                        
                #print(batch_ys[1:10])
                #print("")
                sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
             
            acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob:1.0})
            print("Iter " + str(epoch) + "，Testing Accuracy=" + str(acc))
            #print('循环'+str(epoch)+"，Iter " + str(i) + "，Testing Accuracy=" + str(acc))
            
            
            
            
            
        saver.save(sess,'one_hot_logs/onehot',global_step=num*5+a)  
        sess.close()
            

            
            
            
            
#对数据进行5折交叉验证，并传递数据集序号
def KF(x,y,num):

    X_t = x.reshape(len(x),-1)
    Y_t = y.reshape(len(y),-1) 
    #print(X)
    #loo = LeaveOneOut()
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_t)
    a=0
    for train_index, test_index in kf.split(X_t):
        X_train, X_test = X_t[train_index], X_t[test_index]
        Y_train, Y_test = Y_t[train_index], Y_t[test_index]
        print("len(X_train)={},len(X_test)={}".format(len(X_train),len(X_test)))
        if num==a==0:
            is_train= True
        else:
            is_train= False
            
        cnn(X_train,X_test,Y_train,Y_test,200,is_train,num,a)
        a+=1
        

        
        print("")
        print("")


        


#load benchmark dataset
N_data = sio.loadmat('PDNA-224-ONEHOT-11-N.mat')
P_data = sio.loadmat('PDNA-224-ONEHOT-11-P.mat')
n_X = N_data['n_data']
n_Y = N_data['n_target']
p_X = P_data['p_data']
p_Y = P_data['p_target']

#随机打乱正样本数据并返回
N_p = p_X.shape[0]
print("N_p \t",N_p)
indx = list(range(N_p))
random.shuffle(indx)
p_x=p_X[indx]
p_y=p_Y[indx]

#随机打乱负样本数据并返回
N_n = n_X.shape[0]
print("N_n \t",N_n)
indx = list(range(N_n))
random.shuffle(indx)
n_x=n_X[indx]
n_y=n_Y[indx]

#处理训练集，正样本数量不变，在打乱的负样本中抽取3822个不放回，组成7600条数据为一组的训练集
NN = len(n_x)
#print(NN)
#负样本批次大小
N_size = 3822
#计算负样本多少个批次 53570/3822=14.01
NN_batch = NN // N_size
#print(NN_batch)
#得到14个数据集，正样本不变，负样本从'PDNA-224-ONEHOT-11-N.mat中抽出3822个不放回

for i in range(NN_batch):
               
        if (i+2)*N_size > NN: #  
                X = np.vstack((n_X[i*N_size:NN],p_X))
                Y = np.vstack((n_Y[i*N_size:NN],p_Y))
        else:
                X = np.vstack((n_X[i*N_size:(i+1)*N_size],p_X))
                Y = np.vstack((n_Y[i*N_size:(i+1)*N_size],p_Y))
                #print(i*N_size) 
        
        #随机打乱数据，避免过拟合
        N_X = X.shape[0]
        print("N_X \t",N_X)
        indx = list(range(N_X))
        random.shuffle(indx)
        X=X[indx]
        Y=Y[indx]    
        print("数据集%d"%i,'长度\t',len(X),len(Y))
        
        #对数据集进行五折交叉验证,并传递数据集序号
        KF(X,Y,i)
        
        





