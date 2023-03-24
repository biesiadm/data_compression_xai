"""Utility functions
"""
import numpy as np
from sklearn import metrics


def kernel_gaussian(y, X, gamma=1):
    k_vals = np.sum((X-y)**2, axis=1)
    return(np.exp(-gamma*k_vals/2))


def kernel_polynomial(y, X, degree=2):
    k_vals = np.sum(X*y, axis=1)
    return((k_vals + 1)**degree)



#:# https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_gaussian(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()



#:# https://www.geeksforgeeks.org/highest-power-2-less-equal-given-number
def highestPowerof2(N):
    if (not (N & (N - 1))):
        return N;
    return 0x8000000000000000 >>  (64 - N.bit_length())


