import logging
logger = logging.getLogger(__package__)

import numpy as np

from scipy.sparse import csr_matrix, issparse

import sklearn
import sklearn.neighbors.base as nnbase

from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin
from sklearn.neighbors.base import SupervisedIntegerMixin
from sklearn.base import is_regressor, is_classifier

from sklearn.exceptions import NotFittedError

from ..validation.types import check_random_state, check_array, column_or_1d, check_consistent_length
from ..validation.types import check_is_fitted, has_fit_parameter
from ..validation import num_samples
from ..validation.common import is_integer, is_float

from ..utils import dict_from_keys
from ..validation import check_sample_weights
from ..validation import check_hasmethod
from ..validation import is_object_array, is_numeric_array, get_array_profile
from ..validation import check_consistent_shape
from ..validation import check_super_hasmethod

from .nn import _check_match_to_fitted
from .knn_weighted_helpers import _project, enhanced_take_along_axis
from .knn_weighted_helpers import _check_weights, _get_weights_array
from .knn_weighted_helpers import _get_neigh_predictor
from .knn_weighted_helpers import _get_weighted_kneighbors
from .knn_weighted_helpers import _check_search_neighborhood

from scipy.sparse import csr_matrix, isspmatrix_csr
from .knn_weighted_helpers import csr_viewrow

####################################################################################

# base class - subclass it to define fit(), predict() etc

class WeightedNearestNeighborsBase(nnbase.NeighborsBase, nnbase.KNeighborsMixin, nnbase.RadiusNeighborsMixin):

    # there are four weights altogether
    #  weight is the classic importance weight
    #  support is the weight for computing the knn-radius, aka relevance weights
    #  gamma weights scale the knn kernel function
    #  eta weights scale the importance weights (useful for experiments)
    #
    # The function learned is sum_i w[i]*eta(x, x[i])*gamma(x, x[i]) * Phi(x, x[i], tau) Psi(y[i]) over sum_i w[i]*eta(x, x[i])
    # where tau os the kth weighted order statistic of dist(x,x[i]) with weights given by support(x, x[i])
    # These weights are often given by a pretrained regressor callable that may depend on both x and x[i] or just x[i]
    #
    #

    def __init__(self, n_neighbors=5, radius=1.0, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None, n_jobs=None,
                 weights='uniform', support='uniform', gamma='uniform', eta='uniform',
                 search_n_neighbors=1.0, **kwargs):

        self.n_neighbors = int(n_neighbors)
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = int(leaf_size)
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.n_jobs = n_jobs

        self._check_algorithm_metric()
        #
        self.weights = _check_weights(weights)
        self.support = _check_weights(support)
        self.gamma = _check_weights(gamma)
        self.eta = _check_weights(eta)
        # fraction or number of n_neighbors of the training samples to search for n_neighbors with enough support
        self.search_n_neighbors = _check_search_neighborhood(search_n_neighbors)
        # self.kwargs = kwargs
        if bool(kwargs):
            logger.warning(f'extra kwargs {kwargs}')

    def _super_hasmethod(self, method_name):
        t = check_super_hasmethod(type(self), method_name=method_name)
        # t = check_hasmethod(super(self.__class__, self), method_name=method_name)
        return t

    def _uses_support(self):
        t = True if self.support not in (None, 'uniform') else False
        return t

    def set_weights_ifdef(self, support=None, weights=None, gamma=None, eta=None):
        # return self.set_params_ifdef(support=support, weights=weights)
        if weights is not None:
            self.weights = _check_weights(weights)
        if support is not None:
            self.support = _check_weights(support)
        if gamma is not None:
            self.gamma = _check_weights(gamma)
        if eta is not None:
            self.eta = _check_weights(eta)
        return self

    def set_params_ifdef(self, **params):
        # to replace set_weights_ifdef
        def_params = {}
        for key, value in params.items():
            if value is not None:
                def_params[key] = value
        self.set_params(**def_params)
        return self

    def check_support_fitness(self):
        try:
            if self.support is not None and is_regressor(self.support):
                check_is_fitted(self.support, ["est_", "y_"])
            if self.gamma is not None and is_regressor(self.gamma):
                check_is_fitted(self.gamma, ["est_", "y_"])
            if self.eta is not None and is_regressor(self.eta):
                check_is_fitted(self.eta, ["est_", "y_"])
        except NotFittedError as e:
            logger.warning(repr(e), exc_info=True)

    def _fit(self, X, y=None, support=None, weights=None, gamma=None, eta=None):
        # called by fit() of subclasses implementing SupervisedIntegerMixin, SupervisedFloatMixin, etc
        #
        # pass only X to super._fit(), since the subclasses could be either supervised or unsupervised
        # the Mixin has already handled y
        #
        X = check_array(X, accept_sparse='csr', ensure_2d=True) if X is not None else X
        if num_samples(X) < 2:
            raise ValueError('fit data should have at least 2 samples')
        if y is not None:
            logger.warning('odd to have y that is not None here!', exc_info=True)
        self.set_weights_ifdef(support=support, weights=weights, gamma=gamma, eta=eta)
        super(WeightedNearestNeighborsBase, self)._fit(X)
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        # overwrite the method in KNeighborsMixin()
        # returns neigh_dist and neigh_ind which are 2d arrays of shape (n_samples, n_samples_fit)
        #
        # since it may need to use the regressor to get the support for (X, self._fit_X), support should have
        # been set up already prior to this call
        #
        check_is_fitted(self, ["_fit_X"])
        X_query, X = _check_match_to_fitted(X, self._fit_X)

        # if the estimator does not use support fall back to the super's kneighbors() [which should be faster]
        if not self._uses_support() and self._super_hasmethod('kneighbors'):
            ans = super(WeightedNearestNeighborsBase, self).kneighbors(X_query, n_neighbors=n_neighbors, return_distance=True)
            neigh_dist, neigh_ind = ans
            self.support_ = None
            # self.support_ = _get_weights_array(dist=neigh_dist, weights=self.support, ind=neigh_ind, X=X,
            #                                    X_fit=self._fit_X)
            self.gamma_ = _get_weights_array(dist=neigh_dist, weights=self.gamma, ind=neigh_ind, X=X, X_fit=self._fit_X)
            self.eta_ = _get_weights_array(dist=neigh_dist, weights=self.eta, ind=neigh_ind, X=X, X_fit=self._fit_X)
            self.weights_ = _get_weights_array(dist=neigh_dist, weights=self.weights, ind=neigh_ind, X=X, X_fit=self._fit_X)

            return neigh_dist, neigh_ind if return_distance else neigh_ind

        n_neighbors = int(n_neighbors) if n_neighbors else self.n_neighbors
        n_samples, n_samples_fit = num_samples(X), num_samples(self._fit_X)
        assert (1 <= n_neighbors <= n_samples_fit)

        # find extended nearest-neighbors without regard to the training sample support;
        # weighed NN will be chosen among those

        # self._search(X=X, query_is_train=query_is_train, n_neighbors=n_neighbors)
        frac = _check_search_neighborhood(self.search_n_neighbors)
        n_neighbors_ext = int(np.ceil(n_samples_fit*frac)) if isinstance(frac, float) else int(frac)
        n_neighbors_ext = max(n_neighbors_ext, n_neighbors)
        n_neighbors_ext = max(1, min(n_neighbors_ext, n_samples_fit-1 if X_query is None else n_samples_fit))

        # use our super's kneighbors() for the extented n_neighbors
        neigh_dist, neigh_ind = super(WeightedNearestNeighborsBase, self).kneighbors(X=X_query,
                                                                                     n_neighbors=n_neighbors_ext,
                                                                                     return_distance=True)

        # get the support from fitted X to the current X. Should have already set the support regressor
        # for the pair (X, self._fit_X) if applicable
        support = _get_weights_array(dist=neigh_dist, weights=self.support, ind=neigh_ind, X=X, X_fit=self._fit_X)

        neigh_dist, neigh_ind, neigh_radius = _get_weighted_kneighbors(neigh_dist, neigh_ind, support=support,
                                                                       k=n_neighbors,
                                                                       query_is_train=True if X_query is None else False)

        # calculate the support for the chosen neighbors - it is used by _predict()
        self.support_ = _get_weights_array(dist=neigh_dist, weights=self.support, ind=neigh_ind, X=X, X_fit=self._fit_X)
        self.gamma_ = _get_weights_array(dist=neigh_dist, weights=self.gamma, ind=neigh_ind, X=X, X_fit=self._fit_X)
        self.eta_ = _get_weights_array(dist=neigh_dist, weights=self.eta, ind=neigh_ind, X=X, X_fit=self._fit_X)
        self.weights_ = _get_weights_array(dist=neigh_dist, weights=self.weights,  ind=neigh_ind, X=X, X_fit=self._fit_X)

        return neigh_dist, neigh_ind if return_distance else neigh_ind

    def _predict(self, X=None, algorithm='mode', output_dim=None):
        # called by predict() and predict_proba() methods of subclasses implementing
        # SupervisedIntegerMixin, SupervisedFloatMixin, etc
        #
        # call the neighboorhood-based prediction algorithm for each row in X
        #
        # output_dim is 'no_classes', 'variable', or an int
        # the y_pred is on array of shape n_samples x output_dim or a list
        #

        check_is_fitted(self, ['_fit_X', '_y'])
        X_query, X = _check_match_to_fitted(X, self._fit_X)

        if issparse(X) and self.metric == 'precomputed':
            raise ValueError("sparse matrices not supported for prediction with precomputed kernels. "
                             "Densify your matrix.")

        n_samples_fit, n_features_fit = self._fit_X.shape
        n_samples, n_features = X.shape

        # fix _y classes_ n_outputs; classes_ is a list of ndarrays of class integer labels
        _y, classes_, n_outputs = None, None, 0
        if hasattr(self, '_y') and self._y is not None:
            _y = self._y
            classes_ = self.classes_ if is_classifier(self) else []
            if hasattr(self, 'outputs_2d_') and not self.outputs_2d_ or _y.ndim == 1:
                _y = _y.reshape((-1, 1), order='F')
                classes_ = [classes_] # if len(classes_) else []
            n_outputs = _y.shape[1]
            if len(classes_) != n_outputs and sklearn.base.is_classifier(self):
                raise ValueError(f'classes_ does not have {n_outputs} rows')
        if n_outputs == 0:
            raise ValueError(f'nothing to predict, n_outputs is {n_outputs}')

        # prepare y_pred and prediction callable
        pfunc, output_dim = _get_neigh_predictor(algorithm, output_dim=output_dim)
        if isinstance(output_dim, int) and output_dim >= 1:
            shp = (n_samples, n_outputs) if output_dim == 1 else (n_samples, output_dim, n_outputs)
            y_pred = np.empty(shp, dtype=_y.dtype, order='F')
            y_pred[:] = np.nan if is_float(y_pred[:][0]) else -1
        elif output_dim in ('no_classes', 'variable'):
            y_pred = list()

        # find nearest neighbors and their weights and support
        neigh_dist, neigh_ind = self.kneighbors(X_query)

        # self.weights_ is already set in self.kneightbors(X_query)
        self.weights_ = _get_weights_array(dist=neigh_dist, weights=self.weights,  ind=neigh_ind,
                                           X=X, X_fit=self._fit_X)

        unit_weights = enhanced_take_along_axis(np.ones(n_samples_fit), neigh_ind, axis=1)

        weights = self.weights_ if self.weights_ is not None else unit_weights

        # self.support_ is set by kneighbors(X_query)
        support = self.support_ if self.support_ is not None else unit_weights
        gamma = self.gamma_ if self.gamma_ is not None else unit_weights
        eta = self.eta_ if self.eta_ is not None else unit_weights

        all_rows = np.arange(n_samples)
        for k in range(n_outputs):
            # make a prediction for each output
            classes_k = classes_[k] if classes_ is not None and k < len(classes_) else np.asarray([])
            if output_dim == 'no_classes':
                if classes_k.size >= 1:
                    output_k = np.empty((n_samples, classes_k.size))
                    output_k[:] = np.nan
                else:
                    raise ValueError(f'classes for output should be non-empty -- {classes_k} for {k}')
            elif isinstance(output_dim, int):
                output_k = y_pred[:, k] if output_dim == 1 else y_pred[:, :, k]
            else:
                # list since the predictor's output is of variable length
                output_k = OrderedDict()

            for j in all_rows:
                if isspmatrix_csr(neigh_ind):
                    neigh_ind_j, _ = csr_viewrow(neigh_ind, j)
                    weights_j, _ = csr_viewrow(weights, j)
                    support_j, _ = csr_viewrow(support, j)
                    gamma_j, _ = csr_viewrow(gamma, j)
                    eta_j, _ = csr_viewrow(eta, j)
                else:
                    neigh_ind_j = neigh_ind[j]
                    weights_j = weights[j]
                    support_j = support[j]
                    gamma_j = gamma[j]
                    eta_j = eta[j]
                result = pfunc(_y[neigh_ind_j, k], weights=weights_j, support=support_j, gamma=gamma_j, eta=eta_j, klasses=classes_k)
                output_k[j] = result
                pass

            if isinstance(output_k, dict):
                output_k = list(output_k.values())
            if isinstance(y_pred, list):
                y_pred.append(output_k)

        if n_outputs == 1:
            y_pred = y_pred[0] if isinstance(y_pred, list) else y_pred.ravel()

        return y_pred


#######################################################################################################

class WeightedNearestNeighbors(WeightedNearestNeighborsBase):

    def fit(self, X, support=None, weights=None, gamma=None, eta=None):
        # self.set_weights_ifdef(support=support, weights=weights, gamma=gamma, eta=eta)
        self._fit(X, support=support, weights=weights, gamma=gamma, eta=eta)
        return self

#######################################################################################################


class KNeighborsWeightedClassifier(WeightedNearestNeighborsBase, SupervisedIntegerMixin, ClassifierMixin):

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None, support='uniform',
                 gamma='uniform', eta='uniform',
                 search_n_neighbors=1.0, **kwargs):

        super_kwargs = dict_from_keys(locals(), keep=WeightedNearestNeighborsBase._get_param_names(), ignore=True)
        super(KNeighborsWeightedClassifier, self).__init__(**super_kwargs)
        pass

    def fit(self, X, y, support=None, weights=None, gamma=None, eta=None):
        # calls SupervisedIntegerMixin.fit() which calls self._fit(X)
        check_consistent_length((X,y))
        self.set_weights_ifdef(support=support, weights=weights, gamma=gamma, eta=eta)
        # self.check_support_fitness()
        super(KNeighborsWeightedClassifier, self).fit(X, y=y)
        return self

    def predict(self, X=None, support=None, weights=None, gamma=None, eta=None):
        self.set_weights_ifdef(support=support, weights=weights, gamma=gamma, eta=eta)
        # self.check_support_fitness()

        # if not self._uses_support() and self._super_hasmethod('predict'):
        #     return super(KNeighborsWeightedClassifier, self).predict(X)

        y_pred = super(KNeighborsWeightedClassifier, self)._predict(X=X, algorithm='mode')
        return y_pred

    def predict_proba(self, X=None, support=None, weights=None, gamma=None, eta=None):
        self.set_weights_ifdef(support=support, weights=weights, gamma=gamma, eta=eta)

        # if not self._uses_support() and self._super_hasmethod('predict_proba'):
        #     return super(KNeighborsWeightedClassifier, self).predict_proba(X)

        y_pred = super(KNeighborsWeightedClassifier, self)._predict(X=X, algorithm='proba')
        return y_pred

