import logging
logger = logging.getLogger(__package__)

import numpy as np

from scipy.stats import rankdata

from scipy.sparse import csr_matrix, isspmatrix_csr

from ..validation.types import check_array

from ..validation import num_samples
from ..validation.types import is_numeric_array, is_number, is_integer

from ..utils import csr_viewrow

def samworth_optimal_weights(n_features, n_neighbors):
    # Samworth (2012) nearest neighbor weights for optimal regret wrt Bayes rule
    d = n_features
    k_star = n_neighbors
    k_star *= ((d * (d + 4)) / (2 * (d + 2))) ** (d / (d + 4))
    w = np.ones((n_neighbors,), dtype=float)
    w *= (1 / k_star)
    e = 2 / d
    for i in range(n_neighbors):
        a = (i + 1) ** (1+e) - i ** (1+e)
        b = 1 + d / 2 - (d / (2 * (k_star ** e))) * a
        w[i] *= b
    # w = w/w.sum()
    w /= w.max()
    return w


def custom_weights_ext(neigh_dist, dudani=True, a=0., r=False, s=False, g=False, n_features=2):
    """
        Compute weights for the neigh_dist matrix
    Args:
        neigh_dist: matrix of distances from instances to neighbors; can be full or isspmatrix_csr matrix
        dudani: use the Dudani weights?
        a: number for the McLeod modification to Dudani weights (defaults to 1/n_neighbors if None; use a=0 to skip the McLeod adjustment)
        r: post-scale weights by rank-transform of neigh_dist?
        s: post-scale weights by the Samworth (2012) approach?; needs n_features and n_neighbors
        g: post-scale weights with Gou et al approach?
        n_features: number of features of instances and neighbors

    Returns:
        matrix of same shape as neigh_dist
    """

    def get_weights(dist, selector, a=None):
        # compute weights for dense numeric array of neigh_dist
        dist = check_array(dist, accept_sparse='csr', ensure_2d=True)
        n_samples, n_neighbors = dist.shape

        dist_max, dist_min = dist.max(axis=1).reshape((-1, 1)), dist.min(axis=1).reshape((-1,1))
        dist_ptp = dist_max - dist_min

        a = a if is_number(a) else 1. / n_neighbors
        a = a if np.isfinite(a) else 1

        # default weights
        weights = np.ones_like(dist, dtype=float)
        scale = np.ones_like(dist, dtype=float)

        samworth_factor = samworth_optimal_weights(n_features, n_neighbors)

        ranks = np.zeros_like(dist, dtype=float)
        for j, row in enumerate(dist):
            ranks[j] = 1 / rankdata(dist[j], method='average')

        with np.errstate(divide='ignore', invalid='ignore'):

            if bool(dudani):
                # Dudani weights with the McLeod adjustment parameter a (with s still equal to n_neighbors)
                weights = ((dist_max - dist) + a * dist_ptp) / ((1 + a) * dist_ptp)
                weights = np.nan_to_num(weights, nan=1.)

            # Gou et al factor to adjust the Dudani weights
            gou_factor = (dist_max + dist_min) / (dist_max + dist)

            if any(selector.values()):
                if selector['r']:
                    scale *= ranks
                if selector['g']:
                    scale *= gou_factor
                if selector['s']:
                    scale *= samworth_factor
                # scale = scale ** (1 / selector.sum())
                weights *= scale

            inf_mask = np.isinf(weights)
            inf_row = np.any(inf_mask)
            weights[inf_row] = inf_mask[inf_row]

            row_sums = weights.max(axis=1).reshape((-1, 1))
            weights = weights / row_sums

        return weights

    # Dudani sample weights and McLeod's modification and rank rescaling
    # neigh_dist is an ndarray or a csr_matrix
    n_samples = num_samples(neigh_dist)

    selector = dict(r=bool(r), g=bool(g), s=bool(s))
    a_o = a

    if is_numeric_array(neigh_dist):
        n_neighbors = neigh_dist.shape[1]
        if not is_number(a):
            a_o = 1./n_neighbors
        neigh_weights = get_weights(neigh_dist, selector, a=a_o)

    elif isspmatrix_csr(neigh_dist):
        neigh_weights = neigh_dist.copy()
        width = neigh_dist.getnnz(axis=1)
        neigh_weights.data[:] = 1
        for j in range(n_samples):
            if width[j]:
                u, _ = csr_viewrow(neigh_dist, j)
                if not is_number(a):
                    a_o = 1. / width[j]
                w = get_weights(u.reshape((1, -1)), selector, a=a_o)
                neigh_weights.data[neigh_weights.indptr[j]:neigh_weights.indptr[j+1]] = w.reshape(-1)
    else:
        raise ValueError(f'invalid type of neigh_dist {type(neigh_dist)}')

    return neigh_weights

