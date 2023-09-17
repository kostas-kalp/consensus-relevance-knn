import logging
logger = logging.getLogger(__package__)

import numpy as np

from ..validation.types import check_random_state, check_array, column_or_1d, check_consistent_length
from ..validation.types import check_is_fitted, has_fit_parameter
from ..validation import num_samples

from .nn import _check_match_to_fitted
from .knn_classifier import KNeighborsClassifier
from .knn_weighted_helpers import _check_search_neighborhood

from ..utils import dict_from_keys

class PKnn(KNeighborsClassifier):

    # need to write special __init__ to pass on the expansion factor parameter
    # search_n_neighbors = 1.0

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None, support='uniform',
                 search_n_neighbors=1.0, **kwargs):

        super_kwargs = dict_from_keys(locals(), keep=KNeighborsClassifier._get_param_names())
        super(PKnn, self).__init__(**super_kwargs)
        self.search_n_neighbors = search_n_neighbors
        pass


    def fit(self, X, y=None):
        super(PKnn, self).fit(X, y=y)
        self._estimate_purity()
        return self

    def _estimate_purity(self):
        # the purity (radius) of an item is given by the distance to the 1st nearest neighbor
        # with different label than itself
        #
        #
        check_is_fitted(self, ['_fit_X', '_y'])
        X, y = self._fit_X, self._y

        n_samples, n_features = X.shape
        pure = {}
        n_neighbors = 1
        while len(pure.keys()) < n_samples and n_neighbors < n_samples:
            neigh_dist, neigh_ind = super(PKnn, self).kneighbors(n_neighbors=n_neighbors, return_distance=True)
            for i in range(n_samples):
                if pure.get(i, None) is not None:
                    continue
                u = y[neigh_ind[i]]
                for j, z in enumerate(u):
                    if z != y[i]:
                        assert (pure.get(i, None) is None)
                        # found nearest neighbor with different label than us - note its distance and rank
                        pure[i] = (neigh_dist[i, j], j + 1)
                        break
            # print(f'cover at {n_neighbors} is {len(pure.keys())} out of {n_samples}')
            if n_neighbors >= n_samples:
                break
            n_neighbors *= 2
            pass
        # self._purity_radius = np.array([pure[i][0] for i in range(n_samples)])
        self._purity = np.array(list(zip(*sorted(pure.items())))[1])

        pass

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        # find the kneighbors of X where their distances are scaled down by their purity radius,
        # so prototypes with large purity radius are deemed closer
        #
        check_is_fitted(self, '_fit_X')
        X_query, X = _check_match_to_fitted(X, self._fit_X)

        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        n_samples, n_samples_fit =  num_samples(X), num_samples(self._fit_X)

        frac = _check_search_neighborhood(self.search_n_neighbors)
        n_neighbors_ext = int(np.ceil(n_samples_fit * frac)) if isinstance(frac, float) else int(frac)

        n_neighbors_ext = max(1, min(n_neighbors_ext, n_samples_fit - 1 if X_query is None else n_samples_fit))
        neigh_dist, neigh_ind = super(KNeighborsClassifier, self).kneighbors(X=X_query, n_neighbors=n_neighbors_ext, return_distance=return_distance)

        # n_neighbors_ext = neigh_dist.shape[1]
        radius = self._purity[:, 0]
        for i in range(n_samples):
            d = np.zeros(n_neighbors_ext)
            for j in range(n_neighbors_ext):
                d[j] = neigh_dist[i, j] / radius[neigh_ind[i, j]]
            I = np.argsort(d)
            a, b, c = neigh_dist[i, I], neigh_ind[i, I], d[I]
            neigh_dist[i, :], neigh_ind[i, :] = c, b

        neigh_dist, neigh_ind = neigh_dist[:, :n_neighbors], neigh_ind[:, :n_neighbors]
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind

