import logging
logger = logging.getLogger(__package__)

import numpy as np
from hashlib import sha1
from sklearn.base import BaseEstimator, TransformerMixin

from ..validation.common import check_is_fitted, has_fit_parameter
from ..validation.common import check_random_state
from ..validation.common import check_array
from ..validation.common import column_or_1d, check_consistent_length

from ..utils import cartesian_product

class InvMapTransformer(BaseEstimator, TransformerMixin):

    # create an inverse map transformer for a 2d array so that it transforms a subarray of the fitted array to
    # a 1d array of row indexes in the fitted array or (nan if not among the rows of the fitted array)

    def __init__(self, hash_func=None):
        super(InvMapTransformer, self).__init__()
        self.hash_func = hash_func
        if hash_func is None or not callable(hash_func):
            self.hash_func = lambda x: int(sha1(x.view(np.uint8)).hexdigest(), 16)

    def fit(self, X, y=None):
        if X is None:
            raise ValueError('X should not be None')
        X = check_array(X, ensure_2d=False, allow_nd=False)
        if X.ndim == 1:
            X = X.reshape((1, -1))
        self.map_ = {}
        for i, x in enumerate(X):
            key = self.hash_func(x)
            if self.map_.get(key, None) is not None:
                raise ValueError('hash conflict for ', i, x)
            self.map_[key] = i
        self._X = X
        return self

    def transform(self, X):
        check_is_fitted(self, 'map_')
        X = check_array(X, ensure_2d=False, allow_nd=False)
        if X.ndim == 1:
            X = X.reshape((1, -1))
        keys = [self.hash_func(x) for x in X]
        idx = [self.map_.get(key, np.nan) for key in keys]
        if not np.isnan(idx).any():
            assert(np.allclose(X, self._X[idx]))

        return np.asarray(idx)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)


