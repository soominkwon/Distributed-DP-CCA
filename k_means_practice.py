import numpy as np
from sklearn.cluster import KMeans

m = 999
n = 2

mu_0 = -2 * np.ones([1, n])
mu_1 = 0 * np.ones([1, n])
mu_2 = 2 * np.ones([1, n])

x_0 = mu_0 + np.random.randn(m // 3, n)

x_1 = mu_1 + np.random.randn(m // 3, n)

x_2 = mu_2 + np.random.randn(m // 3, n)

X = np.concatenate((x_0, x_1, x_2), axis=0)

ids = np.arange(m)
np.random.shuffle(ids)
X = X[ids, :]

# K-means clustering:
k = 3
N_k = X.shape[0]
k_means = KMeans(k, random_state=0).fit(X)

labels = k_means.labels_
z = np.mean(X, axis=0)
numerator = 0
denominator = 0

for number in range(k):
    ids = labels == (k-1)
    X_k = X[ids, :]
    Z_k = np.mean(X_k, axis=0)
    numerator += np.linalg.norm(Z_k - z)
    for x, y in zip(X, labels):
        if y == (k-1):
            Z_nk = x
    denominator += np.linalg.norm(Z_nk - Z_k)


numerator_1 = (1/(k - 1)) * N_k * np.square(numerator)
denominator_1 = (1/(N_k - k)) * np.square(denominator)

CH = numerator_1 / denominator_1
print(CH)
