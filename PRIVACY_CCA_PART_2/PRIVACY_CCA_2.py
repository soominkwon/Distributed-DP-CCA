import numpy as np
import scipy.io

matrix = scipy.io.loadmat('value0.mat')
X_0 = matrix['Xs']
Y_0 = matrix['Ys']

matrix = scipy.io.loadmat('value1.mat')
X_1 = matrix['Xs']
Y_1 = matrix['Ys']

matrix = scipy.io.loadmat('value2.mat')
X_2 = matrix['Xs']
Y_2 = matrix['Ys']

X = np.concatenate([X_0, X_1, X_2], axis=1)
Y = np.concatenate([Y_0, Y_1, Y_2], axis=1)

K = min(X_0.shape[0], Y_0.shape[0])

with np.load('test.npz') as data:
    U_hat = data['U'][:, :K]
    V_hat = data['V'][:, :K]
    U = data['Us'][:, :K]
    V = data['Vs'][:, :K]

N = X.shape[1]


# Calculating s1_hat:
s1_matrices = np.dot(U_hat.T, X)
s1_matrices2 = np.dot(V_hat.T, Y)

s1_hat = 1/N * (np.linalg.norm(s1_matrices - s1_matrices2))

a = np.dot(U.T, X)
b = np.dot(V.T, Y)
s1 = 1/N * (np.linalg.norm(a - b))

# Calculating s2_hat
s2_matrices = np.dot(U_hat.T, X)
s2_matrices2 = np.dot(Y.T, V_hat)

s2_hat = 1/N * (np.trace(np.dot(s2_matrices, s2_matrices2)))

c = np.dot(U.T, X)
d = np.dot(Y.T, Y)

s2 = 1/N * (np.trace(np.dot(c, d)))

print(s1)
print(s1_hat)
print(s2)
print(s2_hat)
