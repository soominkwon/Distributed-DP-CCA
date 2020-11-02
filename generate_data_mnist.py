#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:50:00 2019

@author: hafizimtiaz
"""
import h5py
import numpy as np
from scipy.io import loadmat, savemat
import pdb
    
# function for data generation
def generate_data():
#    pdb.set_trace()
    filepath = '/Users/KennyKwon/Desktop/'
    filename = 'MNIST_preprocessed_d100_N30k_new.mat'

    tmp = {}
    f = h5py.File(filepath + filename)
    for k, v in f.items():
        tmp[k] = np.array(v)
    
    tmpX = tmp['x']
    dx = int(tmp['d'])
    X = tmpX[:, :dx]
    Y = tmpX[:, dx:]
    labels = tmp['src_id'].T
    m = X.shape[0]
    
    ids = np.arange(m)
    np.random.shuffle(ids)
    X = X[ids, :]
    Y = Y[ids, :]
    labels = labels[ids, :]
    
    return X, Y, labels
    

# This part of the code defines the following variables:
# pdb.set_trace()
S = 3
C = 0
X, Y, labels = generate_data()
N = X.shape[0]
Ns = N // S
Z = np.array([])
st = 0
en = Ns
for number in range(S):
    X_s = X[st:en, :].T  
    Y_s = Y[st:en, :].T
    labels_s = labels[st:en, :]
    st += Ns
    en += Ns
    
    Z_s = np.concatenate([X_s, Y_s])
    
    C_s = 1/N * (np.matmul(Z_s, Z_s.transpose()))
    
    d = {}
    d['Xs'] = X_s
    d['Ys'] = Y_s
    d['Cs'] = C_s
    d['labels'] = labels_s
    filepath = 'test/local' + str(number) + '/simulatorRun/'
    savemat(filepath + 'value' + str(number), d)