#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:50:00 2019

@author: hafizimtiaz
"""
import numpy as np
from scipy.io import loadmat, savemat
    
# function for data generation
def generate_data():
    
    filepath = '/Users/KennyKwon/Desktop/'
    filename = 'XRMB_preprocessed_d50_p30_new.mat'
    tmp = loadmat(filepath + filename)
    
    tmpX = tmp['x']
    dx = int(tmp['dx'])
    X = tmpX[:dx, :].T
    Y = tmpX[dx:, :].T
    labels = tmp['src_id']
    m = X.shape[0]
    ids = np.arange(m)
    np.random.shuffle(ids)
    X = X[ids, :]
    Y = Y[ids, :]
    labels = labels[ids, :]
    
    # different sample size algorithm
    N = 10000
    X = X[:N, :]
    Y = Y[:N, :]
    labels = labels[:N, :]
    return X, Y, labels
    

# This part of the code defines the following variables:
S = 3
C = 0
X, Y, labels = generate_data()
N = X.shape[0]
Ns = N // S
Z = np.array([])
st = 0
en = Ns
for number in range(S):
    X_s = X[st:en, ].T  
    Y_s = Y[st:en, ].T
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
    