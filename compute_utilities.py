#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt


###############################################################################
# function for CH index calculation
def CHindex(X, labels):
    N = X.shape[0]
    C = len(np.unique(labels))
    num = 0
    den = 0
    z = np.mean(X, axis=0, keepdims = True)
    
    for c in range(C):
        ids = (labels == c)
        Xc = X[ids, :]
        Nc = Xc.shape[0]
        zc = np.mean(Xc, axis=0, keepdims = True)
        den += np.linalg.norm(Xc - zc, 'fro')**2
        num += Nc * np.linalg.norm(zc - z, 'fro')**2
    num = num / (C - 1)
    den = den / (N - C)
    return num / den

# calculate the utility indices
def CCAindex(X, Y, U, V):
    Xro = U.T @ X
    Yro = V.T @ Y

    N = X.shape[1]
    s1o = (1/N) * np.linalg.norm(Xro - Yro, 'fro')
    s2o = (1/N) * np.trace(np.dot(Xro, Yro.T))
    return s1o, s2o

# handle imaginary features
def myFeatures(X):
    if np.iscomplexobj(X):
        Xre = np.real(X)
        Xim = np.imag(X)
        X = np.concatenate((Xre, Xim), axis = 0)
    return X
        
        
###############################################################################
# load data
S = 3
X = np.array([])
Y = np.array([])
labels = np.array([])

C = 0
for number in range(S):
    filepath = 'test/local' + str(number) + '/simulatorRun/'
    tmp = loadmat(filepath + 'value' + str(number) + '.mat')
    X_s = tmp['Xs']
    Y_s = tmp['Ys']
    C_s = tmp['Cs']
    labels_s = tmp['labels']
    
    X = np.concatenate((X, X_s), axis = 1) if X.size else X_s    
    Y = np.concatenate((Y, Y_s), axis = 1) if Y.size else Y_s
    labels = np.concatenate((labels, labels_s), axis = 0) if labels.size else labels_s
    C += C_s
    
C = C / S    
K = min(X.shape[0], Y.shape[0])   

# load results
num_class = 2
maxItr = 10
epsilon_all = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
s1o_all = np.zeros([maxItr, len(epsilon_all)])
s2o_all = np.zeros([maxItr, len(epsilon_all)])
s1h_all = np.zeros([maxItr, len(epsilon_all)])
s2h_all = np.zeros([maxItr, len(epsilon_all)])
chindex_h_all = np.zeros([maxItr, len(epsilon_all)])
chindex_o_all = np.zeros([maxItr, len(epsilon_all)])

for itr in range(maxItr):
    for eps_id in range(len(epsilon_all)):
        filepath = 'test/output/remote/simulatorRun/'
        tmp = np.load(filepath + 'result_' + str(itr) + '_' + str(eps_id) + '.npz')
        Uh = tmp['arr_0'][:, :K]
        Vh = tmp['arr_1'][:, :K]
        Us = tmp['arr_2'][:, :K]
        Vs = tmp['arr_3'][:, :K]
        
#        pdb.set_trace()
        # compute the CCA indices
        s1o, s2o = CCAindex(X, Y, Us, Vs)
        s1h, s2h = CCAindex(X, Y, Uh, Vh)
        
        s1o_all[itr, eps_id], s2o_all[itr, eps_id] = s1o, s2o
        s1h_all[itr, eps_id], s2h_all[itr, eps_id] = s1h, s2h
        
        # compute the clustering indices
        Xrh = myFeatures(Uh.T @ X)
        km = KMeans(n_clusters = num_class, random_state = 0)
        km.fit(Xrh.T)
        labels_h = km.labels_
        chindex_h = CHindex(X.T, labels_h)
        
        Xro = myFeatures(Us.T @ X)
        km = KMeans(n_clusters = num_class, random_state = 0)
        km.fit(Xro.T)
        labels_o = km.labels_
        
        chindex_o = CHindex(X.T, labels_o)
        chindex_h_all[itr, eps_id], chindex_o_all[itr, eps_id] = chindex_h, chindex_o

myFile = 'test/output/remote/all_indices'
np.savez(myFile, s1o_all, s2o_all, s1h_all, s2h_all, chindex_o_all, chindex_h_all)

#%% plot results
s1o_avg = s1o_all.mean(axis = 0)
s2o_avg = s2o_all.mean(axis = 0)

s1h_avg = s1h_all.mean(axis = 0)
s2h_avg = s2h_all.mean(axis = 0)

chindex_o_avg = chindex_o_all.mean(axis = 0)
chindex_h_avg = chindex_h_all.mean(axis = 0)

plt.semilogx(epsilon_all, s1o_avg, 'r', epsilon_all, s1h_avg, 'b')
plt.title('s1 index'); plt.xlabel('Epsilon')
plt.show()

plt.semilogx(epsilon_all, s2o_avg, 'r', epsilon_all, s2h_avg, 'b')
plt.title('s2 index'); plt.xlabel('Epsilon')
plt.show()

plt.semilogx(epsilon_all, chindex_o_avg, 'r', epsilon_all, chindex_h_avg, 'b')
plt.title('ch index'); plt.xlabel('Epsilon')
plt.show()

