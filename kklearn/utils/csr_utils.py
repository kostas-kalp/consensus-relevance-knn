import logging
logger = logging.getLogger(__package__)

from scipy.sparse import csr_matrix, isspmatrix_csr

"""
suppose that A is a scipy.sparse.csr_matrix
then the indices of the non-zero elements on the ith row are
[ (i,j) for j in A.indices[A.indptr[i]:A.indptr[i+1]] ]
and their values are 
[A.data[A.indptr[i]:A.indptr[i+1]]

the number of non-zero elements in each row are np.diff(A.indptr) == A.getnnz(axis=1)
note that A can be made into a shape A.shape[0] x A.getnnz(axis=1).max()
eg csr_matrix((A.data, A.indices, A.indptr), shape=(A.shape[0], A.getnnz(axis=1).max()))

A[i] and A.getrow(i) both return the ith row as a 1d csr matrix
when A is a 1d csr matrix A.data and A.indices return the value and indices of its non-zero elements

"""

def csr_viewrow(X, i):
    # getting a row view allows to make assignments to the elements of the view that are performed against the X's data
    i = int(i)
    if X is None:
        raise ValueError(f'arg X should not be empty or None')
    elif not isspmatrix_csr(X):
        raise ValueError(f'X should be a csr_matrix')
    elif i < 0 or i > X.shape[0]:
        raise IndexError(f'arg i={i} is out of bounds')
    arr = X.data[X.indptr[i]:X.indptr[i+1]], X.indices[X.indptr[i]:X.indptr[i+1]]
    return arr
