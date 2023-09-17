import logging
logger = logging.getLogger(__package__)

import numpy as np
import sklearn
from sklearn.base import TransformerMixin

from ..validation.common import check_random_state, check_array, check_X_y, column_or_1d, check_consistent_length
from ..validation.common import check_is_fitted, has_fit_parameter

# from sklearn.neighbors import KNeighborsClassifier

from ..neighbors import KNeighborsClassifier

class KnnTransformer(KNeighborsClassifier, TransformerMixin):
    #
    # transform each sample to the juxtaposition of the features of its k-nearest neighbors
    #
    # needs KNeighborsClassifier with method signatures kneighbors(X=None),
    # predict(X=None) and predict_proba(X=None)
    #

    def transform(self, X=None):
        check_is_fitted(self, '_fit_X')
        if X is not None:
            X = check_array(X, accept_sparse="csr", ensure_2d=True)
        if X is not None and X.shape == self._fit_X.shape and np.allclose(X, self._fit_X):
            X_tmp, X = X, None
        else:
            X_tmp = None

        y_scores_tr = self.predict_proba()
        y_scores = self.predict_proba(X)
        y_pred = self.predict(X)
        y_knn_dist, y_knn_ind = self.kneighbors(X=X, return_distance=True)

        X = self._fit_X if X is None else X
        X = X_tmp if X_tmp is not None else X
        X = check_array(X, accept_sparse='csr')

        n_samples = X.shape[0]
        n_features = y_scores[0].size
        n_neighbors = self.n_neighbors
        z = np.zeros((n_samples, n_neighbors, n_features))
        for i in range(n_samples):
            for k in range(n_neighbors):
                j = y_knn_ind[i, k]
                z[i, k, :] = y_scores_tr[j, :]

        u = z.reshape((n_samples, -1))
        return u

