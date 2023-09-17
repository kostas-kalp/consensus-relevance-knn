import logging
logger = logging.getLogger(__package__)

import numpy as np

import sklearn
from sklearn.utils import check_random_state, check_array, check_X_y, column_or_1d, check_consistent_length
from sklearn.utils.validation import check_is_fitted, has_fit_parameter
from ..validation import num_samples

from scipy import stats
from sklearn.neighbors.base import _get_weights
from sklearn.utils.extmath import weighted_mode
# from .modes import weighted_mode

from .nn import _check_match_to_fitted
from .nn import NearestNeighbors


class KNeighborsClassifier(sklearn.neighbors.KNeighborsClassifier):

    # Overwrite kneighbors(X) etc methods so that they work as if X=None when X is close fot _fit_X
    #
    # overwrite predict(X) and predict_proba(X) to accept predict(X=None) and predict_proba(X=None)
    # so that we can get predictions and scores for the training data as well by calling them with X=None
    # also define decision_function() to return the last column of predict_proba()
    #

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self, '_fit_X')
        X_query, X = _check_match_to_fitted(X, self._fit_X)
        neigh_dist, neigh_ind = super(KNeighborsClassifier, self).kneighbors(X=X_query, n_neighbors=n_neighbors,
                                                                             return_distance=return_distance)
        return neigh_dist, neigh_ind if return_distance else neigh_ind


    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        check_is_fitted(self, '_fit_X')
        X_query, X = _check_match_to_fitted(X, self._fit_X)
        A = super(KNeighborsClassifier, self).kneighbors_graph(X=X_query, n_neighbors=n_neighbors, mode=mode)
        return A

    def predict(self, X=None):
        check_is_fitted(self, ['_fit_X', '_y'])
        X_query, X = _check_match_to_fitted(X, self._fit_X)
        # the rest of this method is essentially identical to that of the superclass

        neigh_dist, neigh_ind = self.kneighbors(X_query)
        n_samples = num_samples(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]
        n_outputs = len(classes_)

        weights = _get_weights(neigh_dist, self.weights)

        y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

    def predict_proba(self, X=None):
        check_is_fitted(self, '_fit_X')
        X_query, X = _check_match_to_fitted(X, self._fit_X)
        # the rest of this method is essentially identical to that of the superclass

        neigh_dist, neigh_ind = self.kneighbors(X_query)

        n_samples = num_samples(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)

        all_rows = np.arange(n_samples)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_samples, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # event_loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities

    def decision_function(self, X=None):
        return self.predict_proba(X)[:, -1]

