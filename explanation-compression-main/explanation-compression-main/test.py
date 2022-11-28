from sklearn.datasets import load_diabetes
import numpy as np
X, _ = load_diabetes(return_X_y=True)
print(X.shape)
X = X[0:128, :]
print(X.shape)
from goodpoints import kt, compress
def kernel_gaussian(y, X, gamma=1):
    k_vals = np.sum((X-y)**2, axis=1)
    return(np.exp(-gamma*k_vals/2))
f_halve = lambda x: kt.thin(X=x, m=1, split_kernel=kernel_gaussian, swap_kernel=kernel_gaussian)
id_compressed = compress.compress(X, halve=f_halve, g=0)
print(id_compressed)