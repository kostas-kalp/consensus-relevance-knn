import logging
logger = logging.getLogger(__package__)

import numpy as np

import itertools

def cartesian_product(*arrays, allownd=True):
    # compute the cartesian product of the rows of a sequence of numpy arrays either as a 2d or nd array
    # eg
    # q = cartesian_product(np.asarray([[1,2], [3,4]]), np.asarray([[10,20,30], [50, 60, 70], [0, 1, 2], [9, 9, 8]]))
    # returns X with shape [2,4,5] and values
    # X = [[[ 1  2 10 20 30]
    #   [ 1  2 50 60 70]
    #   [ 1  2  0  1  2]
    #   [ 1  2  9  9  8]]
    #  [[ 3  4 10 20 30]
    #   [ 3  4 50 60 70]
    #   [ 3  4  0  1  2]
    #   [ 3  4  9  9  8]]]
    # while if allownd=False, it returns X as a [8,5] array
    # [[ 1  2 10 20 30]
    #  [ 1  2 50 60 70]
    #  [ 1  2  0  1  2]
    #  [ 1  2  9  9  8]
    #  [ 3  4 10 20 30]
    #  [ 3  4 50 60 70]
    #  [ 3  4  0  1  2]
    #  [ 3  4  9  9  8]]

    n_arrays = len(arrays)
    arrays = [np.asarray(arr) for arr in arrays]
    shp = np.asarray([arr.shape for arr in arrays])
    dtype = np.result_type(*arrays)
    n_rows, n_cols = np.prod(shp[:, 0]), np.sum(shp[:, 1])

    f = lambda x: list(itertools.chain.from_iterable(x))
    X = np.asarray(list(map(f, itertools.product(*arrays))), dtype=dtype)
    if allownd:
        X = X.reshape((*shp[:,0], n_cols))
    return X

# X1 = np.asarray([[1,2,3], [3,4,5]])
# X2 = np.asarray([[10,20], [30,40], [50, 60]])
# X3 = cartesian_product(X1, X2)
# X3 = X3.reshape((X1.shape[0]*X2.shape[0], -1))
# X4 = cartesian_product(X1)
