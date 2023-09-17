import logging
logger = logging.getLogger(__package__)

import numpy as np

import sklearn

import sklearn.neighbors.base as nnbase

# NeighborsBase, KNeighborsMixin, RadiusNeighborsMixin

from ..utils import dict_from_keys
from ..validation import check_sample_weights
from ..validation import check_hasmethod
from ..validation.types import is_object_array, is_numeric_array
from ..validation import check_consistent_shape, get_array_profile

from ..validation.types import check_random_state, check_array, column_or_1d, check_consistent_length
from ..validation.types import check_is_fitted, has_fit_parameter
from ..validation import num_samples
from sklearn.utils.extmath import stable_cumsum

from sklearn.utils.extmath import weighted_mode
# from .modes import weighted_mode

from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin

from sklearn.base import is_regressor

from functools import partial

from collections import OrderedDict
import itertools

from scipy.sparse import csr_matrix, isspmatrix_csr
from ..utils import csr_viewrow


def enhanced_take_along_axis(X, indices, axis=1):
    # Take a subsequence from 1-D slices along the given axis of a 2d array X
    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.apply_along_axis.html
    if X.ndim == 1:
        X = X.reshape((1, -1))
    n, m = num_samples(X), num_samples(indices)

    if is_numeric_array(indices):
        # dense numeric numpy array
        vals = np.take_along_axis(X, indices, axis=axis)
    elif isspmatrix_csr(indices):
        if axis != 1:
            raise ValueError('only axis=1 is supported for csr_matrix indices')
        x_data = np.zeros_like(indices.data, dtype=X.dtype)
        width = indices.getnnz(axis=1)
        for j in range(m):
            if width[j]:
                S, _ = csr_viewrow(indices, j)
                i = j if j < n else -1 # which row of X to pull data from
                x_data[indices.indptr[j]:indices.indptr[j+1]] = X[i][S]
        vals = csr_matrix((x_data, indices.indices, indices.indptr))
    elif is_object_array(indices):
        if n == 1:
            vals = [X[0][S] for i, S in enumerate(indices)]
        else:
            vals = [X[i][S] for i, S in enumerate(indices)]
        vals = np.asarray(vals)
    else:
        raise ValueError('invalid indices')
    return vals


def _project(data, rows=None, cols=None):
    # if data is an array then return specified rows and cols of data
    # if data is a list return elements at indices specified by rows
    if rows is None:
        return data
    if isinstance(data, list):
        return data[rows]
    elif isinstance(data, np.ndarray) or isspmatrix_csr(data):
        if cols is None:
            return data[rows] if rows is not None else data
        else:
            if rows is None:
                return data[:][cols]
            else:
                return data[rows][cols]
    elif isspmatrix_csr(data):
        # dok = data.todok()
        # n, m = data.shape
        # cols = set(cols) if cols is not None else set(range(m))
        # rows = set(rows) if rows is not None else set(range(n))
        # for key, opts in dok:
        logger.debug('this is now obsolete')
        pass
    else:
        raise ValueError(f'invalid type of data {type(data)}')


def _weighted_order_statistic(X, weight=None, k=None):
    # find the k-th weighted order statistic of an 1d array x
    #
    X = column_or_1d(X)
    weight = np.ones_like(X) if weight is None else column_or_1d(weight)
    check_consistent_length((X, weight))

    n = X.shape[0]
    k = n // 2 if k is None else k
    k = int(k)
    if k < 1 or k > n:
        logger.warning(f'k={k} is out of bounds [1,{n}] - clipping it back to boundes', exc_info=True)
        k = min(max(k,1), n)
        # raise ValueError(f'k={k} is out of bounds [1,{n}]')

    weight_threshold = np.nanmean(weight) * k
    if np.any(np.diff(X) < 0):
        # need to sort X (and weight)
        ind = np.argsort(X)
        val, weight = X[ind], weight[ind]
    else:
        val = X

    w = stable_cumsum(weight)
    # leftmost index with cumulative weight >= threshold
    # j = np.searchsorted(w, weight_threshold, side='left')
    j = np.argmax(w >= weight_threshold)
    wos = val[j] if j < n else val[-1]
    # print(f'\t WOS threshold={weight_threshold} wos={wos}')
    return wos

####################################################################################################################

# overwrite _check_weights() and  _get_weights() to allow weights and support via a regressor on the samples


def _check_weights(weights):
    # ensure that weights is None, 'uniform', 'distance', callable, or regressor
    if weights is None:
        # weights = 'uniform'
        pass
    elif is_regressor(weights):
        # print('check_weights is a regressor', weights)
        pass
    elif isinstance(weights, str) and weights in (None, 'uniform', 'distance') or callable(weights):
        weights = nnbase._check_weights(weights)
    elif isinstance(weights, (tuple, list)) and len(weights) == 2:
        # we want to phase out these tuple-weights
        w_1 = _check_weights(weights[0])
        w_2 = _check_weights(weights[1])
        # weights = (w_1, w_2)
        pass
    else:
        raise ValueError(f'invalid weights/support type {type(weights)}')
    return weights


def _get_weights_array(dist=None, weights=None, ind=None, X=None, X_fit=None):
    # Computes weights_arr for given dist, weights, and (X_a, X_b) data
    # X_a -> X, X_b -> X_fit
    # weights_arr matches the profile of dist or ind (list of column indices)
    #    if dist is ndarray then weights_arr is also an ndarray with shape dist.shape
    #    if dist is a list of arrays then weights_arr is also a list of arrays with the same length and shapes
    #
    # ind is what is the neigh_ind returned by predict()
    # if weights is 'distance' then weights_arr = 1/dist
    # if weights is callable then weights_arr = callable(dist)
    # if weights is 'uniform' or None then weights_arr = np.ones(dist.shape)
    # if weights is an estimator/regressor est then weights_arr = est.predict(X_a, X_b)
    #
    weights = _check_weights(weights)

    if weights in (None, 'uniform'):
        return None
    if is_regressor(weights):
        if X is None:
            X = X_fit
        if X is None or X_fit is None or ind is None:
            raise ValueError('X, X_fit, and ind arguments are needed for getting weights from estimator')
        vals = weights.predict(X, X_fit)
        weights_arr = enhanced_take_along_axis(vals, ind, axis=1)
        return weights_arr
    # weights is 'distance' or callable(dist)
    if weights in ('distance',) or callable(weights):
        if dist is None:
            raise ValueError('dist arg is required')
        if is_numeric_array(dist):
            weights_arr = nnbase._get_weights(dist, weights)
        elif isspmatrix_csr(dist):
            if callable(weights):
                weights_arr = weights(dist)
            else:
                # why is this case here?
                # need to process row by row since 1/csr_matrix is not supported
                weights_arr_data = np.ones_like(dist.data)
                for j, row in enumerate(dist):
                    u = row.data
                    # u = u.reshape((1,-1))
                    # w = nnbase._get_weights(u, weights)
                    w = 1./u
                    if w is not None:
                        weights_arr_data[dist.indptr[j]:dist.indptr[j+1]] = w
                weights_arr = csr_matrix((weights_arr_data, dist.indices, dist.indptr), shape=dist.shape)
        else:
            raise ValueError('invalid distances argument')
        # weights_arr matches the type and profile of dist
        return weights_arr

    elif isinstance(weights, (tuple, list)) and len(weights) == 2:
        # can we get rid of this case?
        w_0 = _get_weights_array(dist=dist, weights=weights[0], ind=ind, X=X, X_fit=X_fit)
        w_1 = _get_weights_array(dist=dist, weights=weights[1], ind=ind, X=X, X_fit=X_fit)
        if w_0 is not None and w_1 is not None:
            weights_arr = np.multiply(w_0, w_1)
            # weights_arr = w_0.multiply(w_1)
            # assert( np.allclose(weights_arr, weights_arr2))
        elif w_0 is not None:
            weights_arr = w_0
        elif w_1 is not None:
            weights_arr = w_1
        else:
            weights_arr = None
    else:
        # handle case of tuple of weights ('distance' or callable, weights regressor)
        # in which case, return the element-wise (Haddamard) product of the two weight arrays
        raise ValueError(f'invalid weights argument {weights}')
    return weights_arr
    pass


def _check_search_neighborhood(search_n_neighbors=1.0):
    # check that the argument is either an integer >=1 or a float clipped to be in [0,1]
    if isinstance(search_n_neighbors, int):
        return max(search_n_neighbors, 1)
    elif isinstance(search_n_neighbors, float):
        return min(1., max(0., search_n_neighbors))
    else:
        raise ValueError(f"invalid search_n_neighhors {search_n_neighbors}")


####################################################################################################################


def _get_neigh_predictor(algorithm='mode', output_dim=None):
    # algorithm is 'mode', 'proba', or a callable
    # output_dim is None, 'no_classes', 'variable' or a positive integer
    # if algorithm is a callable then the output_dim specifies the
    # dimension of algorithm's return value for each sample and each output
    #
    # rename output_dim => n_predictions
    #
    # returns a 2-tuple with components
    #       callable predictor(y, weights=None, sample_weights=None, klasses=None, func=np.multiply) and
    #       dimension of output of predictor
    #
    # the dimension of 'mode' and 'proba' is always 1 and 'no_classes' respectively.
    #
    # predictor's func argument is a callable func(support, weights) that combines the two weight vectors into
    # one weight vector to use for y's elements when applying the aggregation algorithm
    #
    #

    def _apply_weight_merger(n, support=None, weights=None, gamma=None, eta=None, func=None):
        e = np.ones(n)
        if callable(func):
            weights = e if weights is None else weights
            support = e if support is None else support
            gamma = e if gamma is None else gamma
            eta = e if eta is None else eta
            check_consistent_length((e, support, weights, gamma, eta))
            # w = func(support, weights)
            w = func(eta, weights)
        else:
            # why is this case needed?
            if weights is not None:
                w = weights
            elif eta is not None:
                w = eta
            elif support is not None:
                w = support
            else:
                w = e
            check_consistent_length((e, w))
        s = np.nansum(w)
        if np.allclose(s, 0.):
            w = e / np.nansum(e)
            w = e
        else:
            # w = w / s
            pass
        return w

    def mode_predictor(y, weights=None, support=None, gamma=None, eta=None, klasses=None, func=np.multiply):
        w = _apply_weight_merger(num_samples(y), support=support, weights=weights, gamma=gamma, eta=eta, func=func)
        # need to see this for gamma and eta
        w = np.multiply(w, gamma)
        mode, mode_weight = weighted_mode(y, w, axis=0)
        mode, mode_weight = mode[0], mode_weight[0]
        ans = klasses.take(int(mode)) if isinstance(klasses, (np.ndarray, list, tuple)) else mode
        return ans

    def proba_predictor(y, weights=None, support=None, gamma=None, eta=None, klasses=None, func=np.multiply):
        w = _apply_weight_merger(num_samples(y), support=support, weights=weights, gamma=gamma, eta=eta, func=func)
        m = klasses.size if isinstance(klasses, (np.ndarray, list, tuple)) else 0
        # need to see this for gamma and eta
        w = np.multiply(w, gamma)
        ans = np.bincount(y, weights=w, minlength=m)
        r = np.sum(ans)
        ans = np.ones(m)/m if np.allclose(r, 0.) else ans / r
        return ans

    output_dim = output_dim if output_dim else 'variable'
    if not(isinstance(output_dim, int) and output_dim >= 1 or output_dim in ('no_classes', 'variable')):
        raise ValueError(f'unknown option output_dim={output_dim}')

    if callable(algorithm):
        return algorithm, output_dim
    elif algorithm == 'mode':
        return mode_predictor, 1
    elif algorithm == 'proba':
        return proba_predictor, 'no_classes'
    else:
        raise ValueError(f'unknown algorithm {algorithm}')


####################################################################################################################

def _get_weighted_kneighbors(neigh_dist, neigh_ind, k=None, query_is_train=False, support=None):
    # given arrays of neighboorhood distances and indices (into a training dataset)
    # return a list of distances, indices, and radii of k-weighted-nearest-neighboors
    # query_train = False # query_train case is not needed

    if isspmatrix_csr(neigh_dist):
        logger.error('got a neigh_dist that is CSR, was not expecting it! ', exc_info=True)
    neigh_dist = check_array(neigh_dist, accept_sparse='csr', ensure_2d=True, allow_nd=False)
    neigh_ind = check_array(neigh_ind, accept_sparse='csr', ensure_2d=True, allow_nd=False)
    if support is None:
        neigh_support = np.ones_like(neigh_dist)
    else:
        neigh_support = check_array(support, accept_sparse='csr', ensure_2d=True, allow_nd=False)
    check_consistent_shape(neigh_dist, neigh_ind, neigh_support)

    n_samples = num_samples(neigh_dist)

    weighted_neigh_dist,  weighted_neigh_ind, weighted_neigh_indices = [], [], []
    weighted_neigh_num, weighted_radius = np.zeros((n_samples,), dtype=int), np.zeros((n_samples,), dtype=float)

    # need to use the neigh_ind to get only as a mask
    weighted_neigh_indptr = np.zeros((n_samples+1,), dtype=int)
    for j in range(n_samples):
        # neigh_ind_j, neigh_dist_j, neigh_support_j = _project(neigh_ind, j), _project(neigh_dist, j), _project(neigh_support, j)
        neigh_dist_j, neigh_ind_j, neigh_support_j = neigh_dist[j], neigh_ind[j], neigh_support[j]
        if query_is_train and True:
            # p = np.where(neigh_ind_j == j)
            # d, w = _sort_compress_mask_vectors(neigh_dist_j, neigh_support_j, xi=p)
            p = np.where(neigh_ind_j != j)
            neigh_dist_j, neigh_ind_j, neigh_support_j = neigh_dist_j[p], neigh_ind_j[p], neigh_support_j[p]
            wos = _weighted_order_statistic(neigh_dist_j, neigh_support_j, k=k)
        else:
            wos = _weighted_order_statistic(neigh_dist_j, neigh_support_j, k=k)

        # S = np.argwhere(neigh_dist_j <= wos).reshape(-1)
        # dist, ind = neigh_dist_j[S], neigh_ind_j[S]
        # weighted_neigh_num[j] = len(S)
        # weighted_radius[j] = wos
        # t = len(S)

        # because neigh_dist in sorted ascending order this S is equivalent to the one above
        if np.any(np.diff(neigh_dist_j)<0):
            raise ValueError(f'row {j} of neigh_dist is not sorted in ascending order')

        if False:
            t = np.argmax(neigh_dist_j > wos)
            S = np.arange(t, dtype=int)
        elif True:
            S = np.argwhere(neigh_dist_j <= wos).reshape(-1)
            t = len(S)
        else:
            raise NotImplementedError('this option of finding weighted NN is not implemented')

        dist, ind = neigh_dist_j[:t], neigh_ind_j[:t]
        weighted_neigh_num[j] = t
        weighted_radius[j] = wos
        if t > 0:
            weighted_neigh_dist.extend(dist)
            weighted_neigh_ind.extend(ind)
            # weighted_neigh_indices.extend(list(range(weighted_neigh_num[j])))
            weighted_neigh_indices.extend(np.arange(t))
            weighted_neigh_indptr[j+1] = weighted_neigh_indptr[j] + t
        else:
            raise ValueError('no neighbors selected in weighted NN - wos was too high?')
        # weighted_neigh_dist.append(list(dist))
        # weighted_neigh_ind.append(list(ind))
        # weighted_radius.append(wos)

    # weighted_neigh_indptr = np.hstack(([0], np.asarray(weighted_neigh_num.cumsum())))
    # weighted_neigh_indptr = np.asarray(weighted_neigh_num.cumsum())
    # weighted_neigh_indptr = np.concatenate(([0], weighted_neigh_indptr))

    dist = csr_matrix((weighted_neigh_dist, weighted_neigh_indices, weighted_neigh_indptr))
    ind = csr_matrix((weighted_neigh_ind, weighted_neigh_indices, weighted_neigh_indptr))
    return dist, ind, weighted_radius
    # return np.asarray(weighted_neigh_dist), np.asarray(weighted_neigh_ind), np.asarray(weighted_radius)


####################################################################################################################
# The following functions are under development
#
#

def _sort_compress_mask_vectors(x, y, xi=None, yi=None):
    # assumes that x and y are column_of_1d MaskedArrays or regular Arrays
    # returns column_or_1d numpy arrays of the unmasked data

    def mask_helper(u, idx=None):
        u = u.reshape((-1,1))
        if not isinstance(u, np.ma.MaskedArray):
            u = np.ma.masked_invalid(u)
        if idx is not None and isinstance(idx, (list, tuple)):
            for i in idx:
                u[i] = np.ma.masked
        return u

    xydtype = (x.dtype, y.dtype)
    x = mask_helper(x, idx=xi)
    y = mask_helper(y, idx=yi)
    mxdata = np.ma.hstack((x, y))
    if np.ma.is_masked(mxdata):
        data = np.ma.compress_rows(mxdata)
    else:
        data = mxdata.data
    x, y = np.hsplit(data, 2)
    x, y = x.reshape(-1), y.reshape(-1)
    ind = np.lexsort((-y, x))
    x, y = x[ind].astype(xydtype[0]), y[ind].astype(xydtype[1])

    return x, y


####################################################################################################################
#  The following code is still under development

def _neighbor_list_to_csr(neigh_ind):
    # N  = max(list(map(lambda x: len(x), weighted_neigh_ind)))
    n_samples = len(neigh_ind)
    rows, cols = [[j] * len(neigh_ind[j]) for j in range(n_samples)], neigh_ind
    # cols = weighted_neigh_ind
    rows = list(itertools.chain(*rows))
    cols = list(itertools.chain(*cols))
    vals = [1] * len(rows)
    a = csr_matrix((vals, (rows, cols)))
    return a


def _search_weighted_kneighbors(est, n_neighbors=5, query_is_train=False, X=None):
    # this method is still under development
    raise NotImplementedError('under development')
    train_size = est._fit_X.shape[0]
    n_samples = X.shape[0] if X is not None else train_size

    n_neighbors_max = train_size - 1 if query_is_train else train_size
    S = np.arange(np.floor(np.log2(n_neighbors)), np.ceil(np.log2(train_size)))
    S = np.append(np.power(2, S), [n_neighbors, n_neighbors_max])
    S = np.clip(S, n_neighbors, n_neighbors_max)
    S = np.unique(S).astype(int)

    # this works when sample_weights the same for each test sample
    sample_weights = _get_weights(np.zeros((1, train_size)), self._sample_weights).ravel()
    if sample_weights is None:
        sample_weights = np.ones(train_size)
    weight_threshold = np.mean(sample_weights) * n_neighbors

    dd = OrderedDict()
    for j in range(n_samples):
        Q = X[j].reshape(1,-1)
        for k in S:
            n_neighbors_ext = k+1 if query_is_train else k

            neigh_dist, neigh_ind = super(WeightedNearestNeighborsBase, self).kneighbors(X=Q,
                                                                                         n_neighbors=n_neighbors_ext,
                                                                                         return_distance=True)

            w = sample_weights[neigh_ind].ravel()
            cw = stable_cumsum(w)
            # leftmost index with cumulative weight >= threshold
            jj = np.searchsorted(cw, weight_threshold, side='left')
            if jj < w.shape[0]:
                logger.debug(f'k={jj}/{k} w={w} cw={cw} {weight_threshold}')
                dd[j] = jj+1
                break

    logger.debug('%s', dd)
    pass


