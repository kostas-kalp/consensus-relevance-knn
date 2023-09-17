import logging
logger = logging.getLogger(__package__)

import numpy as np
import sklearn

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.neighbors import NearestNeighbors as _NearestNeighbors

def _check_match_to_fitted(X, fit_X):
    """
    Check whether X and fit_X match and return a tuple (X_query, X_arg) with two arrays
    If they match then return (None, fit_X)
    If X is None then return (None, fit_X)
    If X is not None and X and fit_X do not match then return (X, fit_X)
    Usually called by the predict()/transform() methods of NN-estimators to handle the situations where
    the estimator is called with no X or with X used in the last fit() call.
    See KNeighborsClassifier() in this package for an example.
    Args:
        X: 2d array-like
        fit_X: 2-d array-like
    Returns:
        two 2d arrays (X_query, X_arg)
    """
    # check whether X matches the fitted X (or is None) return (X_query, X)
    # X_query is suitable for calling the sklearn.nn methods and X is the implied data (test or fitted)
    # return (None, fit_X) if there  is a match or X is None, else return (X, X)
    if fit_X is None:
        raise ValueError('fit_X should be given (not None)')
    if X is None:
        X_query, X_arg = None, fit_X
    else:
        X_query = check_array(X, accept_sparse='csr', ensure_2d=True)
        X_arg = X_query
        if X_query.shape == fit_X.shape and np.allclose(X_query, fit_X):
            X_query = None
    return X_query, X_arg


class NearestNeighbors(_NearestNeighbors):

    # overwrite methods to check that X is not self._fit_X
    # so that kneighbors etc work as if X=None in those cases (and the query points are
    # not considered their own neighbors
    #

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self, '_fit_X')
        X_query, X = _check_match_to_fitted(X, self._fit_X)
        neigh_dist, neigh_ind = super(NearestNeighbors, self).kneighbors(X=X_query, n_neighbors=n_neighbors,
                                                                         return_distance=return_distance)
        return neigh_dist, neigh_ind if return_distance else neigh_ind

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        check_is_fitted(self, '_fit_X')
        X_query, X = _check_match_to_fitted(X, self._fit_X)
        A = super(NearestNeighbors, self).kneighbors_graph(X=X_query, n_neighbors=n_neighbors, mode=mode)
        return A

    def radius_neighbors(self, X=None, radius=None, return_distance=True):
        check_is_fitted(self, '_fit_X')
        X_query, X = _check_match_to_fitted(X, self._fit_X)
        neigh_dist, neigh_ind = super(NearestNeighbors, self).radius_neighbors(X=X_query, radius=radius,
                                                                               return_distance=return_distance)
        return neigh_dist, neigh_ind if return_distance else neigh_ind

    def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity'):
        check_is_fitted(self, '_fit_X')
        X_query, X = _check_match_to_fitted(X, self._fit_X)
        A = super(NearestNeighbors, self).radius_neighbors_graph(X=X_query, radius=radius, mode=mode)
        return A

