import logging
logger = logging.getLogger(__package__)

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from ..validation.common import check_is_fitted, has_fit_parameter
from ..validation.common import check_random_state
from ..validation.common import check_array
from ..validation.common import column_or_1d, check_consistent_length

from .inv_map_transformer import InvMapTransformer

class WeightsTransformer(BaseEstimator, TransformerMixin):

    # Used to store and lookup precomputed weights w(a,b) for pairs of samples (a,b)=(from,to)
    # this map w(.) may depend on both a and b, only on a, or only on b
    # the fit() method is used to 'memorize' the values of the map for the whole universe/domain of the from/to samples
    # the transform() returns the projection of the map on the provided from/to data that it is depended on
    # the returned weights are either 2d (if depend on both) or 1d if it depends on one only

    def __init__(self, depends='to'):
        super(WeightsTransformer, self).__init__()

        if isinstance(depends, str) and depends in ('both', 'from', 'to'):
            self.depends = depends
        else:
            raise ValueError('invalid depends')

    def _parse_X_domain(self, X, depends=None):
        depends = self.depends if depends is None else depends
        if depends == 'both' and isinstance(X, (list, tuple)):
            X_from, X_to = X
            X_to = check_array(X_to, allow_nd=False, ensure_2d=True)
            X_from = check_array(X_from, allow_nd=False, ensure_2d=True)
            if X_to.shape[1] != X_from.shape[1]:
                raise ValueError('from/to arrays should have same number of columns')
        elif depends == 'to':
            X_from, X_to = None, X
            X_to = check_array(X_to, allow_nd=False, ensure_2d=True)
        elif depends == 'from':
            X_from, X_to = X, None
            X_from = check_array(X_from, allow_nd=False, ensure_2d=True)
        else:
            raise ValueError('invalid depends')
        return (X_from, X_to)

    def fit(self, X, y):
        X_from, X_to = self._parse_X_domain(X)
        if self.depends == 'both':
            y = check_array(y, ensure_2d=True, allow_nd=False)
            if y.shape[0] != X_from.shape[0] or y.shape[1] != X_to.shape[0]:
                raise ValueError('y should have shape matching the #rows of the from and to arrays')
        elif self.depends == 'to':
            y = check_array(y, allow_nd=False, ensure_2d=False)
            y = y.reshape(-1)
            check_consistent_length((X_to, y))
        elif self.depends == 'from':
            y = check_array(y, allow_nd=False, ensure_2d=False)
            y = y.reshape(-1)
            check_consistent_length((X_from, y))

        itx_from, itx_to = None, None
        if X_from is not None:
            itx_from = InvMapTransformer()
            itx_from = itx_from.fit(X_from)
        if X_to is not None:
            itx_to = InvMapTransformer()
            itx_to = itx_to.fit(X_to)
        self.from_, self.to_ = itx_from, itx_to
        self.y_ = y
        return self

    def transform(self, X):
        check_is_fitted(self, ['from_', 'to_', 'y_'])
        X_from, X_to = None, None
        if isinstance(X, (list, tuple)):
            X_from, X_to = self._parse_X_domain(X, depends='both')
        elif isinstance(X, np.ndarray):
            X_from, X_to = self._parse_X_domain(X)

        idx_from = self.from_.transform(X_from) if X_from is not None and self.from_ is not None else None
        idx_to = self.to_.transform(X_to) if X_to is not None and self.to_ is not None else None

        y = self.y_
        if idx_to is not None and idx_from is not None:
            if np.isnan(idx_from).any() or bp.isnan(idx_to).any():
                raise ValueError('some idx are nan')
            val = y[idx_from, idx_to]
        elif idx_to is not None and idx_from is None:
            if np.isnan(idx_to).any():
                raise ValueError('some idx are nan')
            val = y[idx_to]
        elif idx_to is None and idx_from is not None:
            if np.isnan(idx_from).any():
                raise ValueError('some idx are nan')
            val = y[idx_from]
        else:
            val = y
        return val

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)


# L = [0, 2, 1, 5]
# from kklearn.transformers import InvMapTransformer, WeightsTransformer
# itx = InvMapTransformer()
# itx = itx.fit(X_train)
# Li = itx.transform(X_train[L])
# print('equal? ', np.allclose(purity_weights[L], purity_weights[Li]))
#
# L1, L2 = L, [0, 5, 2]
# db = WeightsTransformer(depends='to')
# db.fit(X_train, y=purity_weights)
# z = db.transform((X_train[L2], X_train[L1]))
