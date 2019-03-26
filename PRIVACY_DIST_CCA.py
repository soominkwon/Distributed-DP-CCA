import numpy as np

print("This program will demonstrate the use of privacy distributed CCA. \n")

base_path = "/path/to/base"

# This part of the code defines the following variables:
D_x = 2
D_y = 3
N_s = D_x + D_y
S = 10
C = 0


# This makes an S amount of matrices and makes the covariance matrix, C:
for number in range(S):
    X_s = np.random.rand(D_x, N_s)
    Y_s = np.random.rand(D_y, N_s)
    Z_s = np.concatenate([X_s, Y_s])

    C_s = 1/N_s * (np.matmul(Z_s, Z_s.transpose()))
    C += C_s

C = 1/S * C

# Constants for Gaussian noise:
delta = 0.01
epsilon = 1
sigma = (1 / (N_s * number)) * np.sqrt(2 * np.log(1.25 / delta))
temp = np.random.normal(0, sigma, size=[N_s, N_s])
temp2 = np.triu(temp)
temp3 = temp2.T
temp4 = np.tril(temp3, -1)

E = temp2 + temp4

# Addition of noise to C matrix:
C_hat = C + E

# Extracting C_xx, C_yy, C_xy, and C_yx from the concatenated matrix:
C_xx = C_hat[:D_x, :D_x]
C_yy = C_hat[D_x:N_s, D_x:N_s]
C_yx = C_hat[D_x:N_s, :D_x]
C_xy = C_yx.transpose()


# This computes the two covariance matrices, in which we will take the eigenvalues of:
C_x1 = np.matmul(np.linalg.inv(C_xx), C_xy)
C_x2 = np.matmul(np.linalg.inv(C_yy), C_yx)
C_u = np.matmul(C_x1, C_x2)


C_y1 = np.matmul(np.linalg.inv(C_yy), C_yx)
C_y2 = np.matmul(np.linalg.inv(C_xx), C_xy)
C_v = np.matmul(C_y1, C_y2)


# Computing final values of U and V:
tmp, U = np.linalg.eig(C_u)
tmp, V = np.linalg.eig(C_v)

print(U)
print(V)
