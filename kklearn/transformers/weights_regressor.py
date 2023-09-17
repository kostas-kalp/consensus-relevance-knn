import logging
logger = logging.getLogger(__package__)

import numpy as np

from hashlib import sha1

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

from ..validation.common import check_is_fitted, has_fit_parameter
from ..validation.common import check_random_state
from ..validation.common import check_array
from ..validation.common import column_or_1d, check_consistent_length

from .inv_map_transformer import InvMapTransformer

from ..utils import cartesian_product

class WeightsRegressor(BaseEstimator, RegressorMixin):

    # Used to store and lookup precomputed weights w(a,b) for pairs of samples (a,b)=(from,to)
    # this map w(.) may depend on both a and b, only on a, or only on b
    # the fit() method is used to 'memorize' the values of the map for the whole universe/domain of the from/to samples
    # the predict() returns the projection of the map on the provided from/to data that it is depended on
    # the returned weights are either 2d (if depend on both) or 1d if it depends on one only

    def __init__(self, depends='fit', hash_func=None, id_func=None):
        super(WeightsRegressor, self).__init__()
        self.depends = depends
        self.hash_func = hash_func
        self.id_func = id_func
        pass

    def _validate_funcs(self):
        if not (isinstance(self.depends, str) and self.depends in ('both', 'query', 'fit')):
            raise ValueError(f'invalid arg depends={depends}')
        hash_func, id_func = self.hash_func, self.id_func
        if hash_func is None or not callable(hash_func):
            hash_func = lambda x: int(sha1(x.view(np.uint8)).hexdigest(), 16)
        if id_func is None:
            id_func = lambda *args: cartesian_product(*args, allownd=False)
        return hash_func, id_func

    def fit(self, *X, y=None):
        hash_func, id_func = self._validate_funcs()
        X = id_func(*X)
        y = check_array(y, allow_nd=False, ensure_2d=False)
        y = y.reshape(-1)
        check_consistent_length((X, y))

        est = InvMapTransformer(hash_func=hash_func)
        est = est.fit(X)
        self.est_ = est
        self.y_ = y
        return self

    def predict(self, *X):
        check_is_fitted(self, ['est_', 'y_'])
        hash_func, id_func = self._validate_funcs()

        X = id_func(*X)
        y = self.y_

        idx = self.est_.transform(X)
        if np.isnan(idx).any():
            raise ValueError('some idx are nan')
        val = y[idx]
        val = np.asarray(val)
        return val

    def __str__(self):
        return f"{self.__class__.__name__}({self.depends})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.depends})"
