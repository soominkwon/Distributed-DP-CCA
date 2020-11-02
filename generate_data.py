#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:50:00 2019

@author: hafizimtiaz
"""
import numpy as np
from scipy.io import loadmat, savemat
    
# function for data generation
def generate_data(m = 999, n = 2):
    mu0 = -1
    mu1 = 0
    mu2 = 1
    
    X0 = mu0 + np.random.randn(m//3, n)
    Y0 = mu0 + np.random.randn(m//3, n+5)
    labels0 = 0 * np.ones([m//3, 1])
    X1 = mu1 + np.random.randn(m//3, n)
    Y1 = mu1 + np.random.randn(m//3, n+5)
    labels1 = 1 * np.ones([m//3, 1])
    X2 = mu2 + np.random.randn(m//3, n)
    Y2 = mu2 + np.random.randn(m//3, n+5)
    labels2 = 2 * np.ones([m//3, 1])
    
    X = np.concatenate((X0, X1, X2), axis = 0)
    Y = np.concatenate((Y0, Y1, Y2), axis = 0)

    labels = np.concatenate((labels0, labels1, labels2), axis = 0)
    ids = np.arange(m)
    np.random.shuffle(ids)
    X = X[ids, :]
    Y = Y[ids, :]
    labels = labels[ids, :]
    return X, Y, labels
    

# This part of the code defines the following variables:
D_x = 20
N = 999
S = 3
C = 0
Ns = N // S

X, Y, labels = generate_data(N, D_x)
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