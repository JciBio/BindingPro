#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


得到PDNA-224-ONEHOT-11.mat数据集


"""

import re
import scipy.io as sio
import numpy as np
# get pssm
#ncbi.getPSSMMatFileFromFastafile( 'PDNA-224-PSSM', 'PDNA-224.fasta', 'PDNA-224-PSSM.mat')

# param ws: slip windown size
def establishBenchmarkDataset(ws, savefile):
    X = []
    Y = []
    k = 0
    t = 0
    # load PDNA-224-PSSM.mat
    datafile = 'PDNA-224-onehot.mat'
    pssm = sio.loadmat(datafile)

    # read fasta file
    #fastafile = 'PDNA-224.fasta'
    #seq_records = SeqIO.parse(fastafile)

    # build slip window with size 11*2+1=23
    
    # read 'PDNA-224-binding-sites.txt'
    with open('PDNA-224-binding-sites.txt', 'r') as pbsreader:
        for line in pbsreader:
            line = line.strip()
            
            if '>' in line:         
                sid = line[1:]   #sid  蛋白质序列ID
                p = pssm[sid]
                seqlen = len(p)
                #print(seqlen)
                line_num=ws*2+1 #行数
                
                for j in range(seqlen):
                    #create a array
                    d = np.ndarray(shape=(line_num,20),dtype=np.int16)
                    
                    if j < ws:
                        d[0:ws-j] = p[j-ws:]
                        d[ws-j:2*ws+1] = p[0: ws+j+1]
                    elif j > seqlen - ws -1:
                        d[0:ws] = p[j-ws:j]
                        d[ws:ws + seqlen -j] = p[j:]
                        d[ws+seqlen-j:] = p[0:ws-seqlen+j+1]
                    else:
                        d[::]=p[j-ws:j+ws+1]
                        
#                     for dl in range(line_num):
#                         for dr in range(20):
#                             if d[dl,dr]<0:
                                
#                                 d[dl,dr]=d[dl,dr]+256
                                
#                     print('')
#                     print(d)
#                     print('')
                    X.insert(k,d)
                    Y.append([1,0])
                    #Y.append([1])
                    k += 1
                    #print('k={},t={}'.format(k,t))
            else:
                sites = line.split() #结合位点位置
                #print(sites)
                for s in sites:
                    idx = eval(s)
                    #print(idx)
                    #print('t={},idx={}'.format(t,idx))
                    #print(t+idx-1)
                    Y[t + idx-1] = [0,1]
                    #print(t + idx-1 )
                    
                t=k        
    #save benchmark data set
    
    dataset={}
    dataset['data']=X
    dataset['target'] = Y
    sio.savemat(savefile,dataset)

establishBenchmarkDataset(11,'PDNA-224-ONEHOT-11.mat')  
print('yes')