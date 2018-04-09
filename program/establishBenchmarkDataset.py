#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:27:54 2018

@author: weizhong
"""


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
    datafile = 'PDNA-224-PSSM.mat'
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
                sid = line[1:]
                p = pssm[sid]
                seqlen = len(p)
                for j in range(seqlen):
                    #create a array
                    d = np.ndarray(shape=(ws*2+1,20),dtype=np.int16)
                    
                    if j < ws:
                        d[0:ws-j] = p[j-ws:]
                        d[ws-j:2*ws+1] = p[0: ws+j+1]
                    elif j > seqlen - ws -1:
                        d[0:ws] = p[j-ws:j]
                        d[ws:ws + seqlen -j] = p[j:]
                        d[ws+seqlen-j:] = p[0:ws-seqlen+j+1]
                    else:
                        d[::]=p[j-ws:j+ws+1]
                
                    X.insert(k,d)
                    Y.append([1,0])
                    k += 1
                    #print('k={},t={}'.format(k,t))
            else:
                sites = line.split()
                for s in sites:
                    idx = eval(s)
                    #print('t={},idx={}'.format(t,idx))
                    #print(t+idx-1)
                    Y[t + idx -1] = [0,1]
                t=k        
    #save benchmark data set
    dataset={}
    dataset['data']=X
    dataset['target'] = Y
    sio.savemat(savefile,dataset)

establishBenchmarkDataset(11,'PDNA-224-PSSM-11.mat')                