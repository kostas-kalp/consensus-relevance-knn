# Example code using the consensus-relevance kNN classifier introduced in the
# paper "Consensus?Relevance kNN and covariate shift mitigation" by K. Kalpakis,
# in Machine Learning (Springer Nature Journal), 2023.
#
# It depends on scikit-learn==0.22.1 (not the latest release of scikit-learn)
# See requirements.txt and install with
# $pip install -r requirements.txt
#

import logging
logger = logging.getLogger(__package__)
from kklearn.utils.decorators import LogLevel

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

from scipy import stats
from scipy.special import softmax, logsumexp

import kklearn

from kklearn.validation.common import check_random_state
from kklearn.validation.common import check_array, column_or_1d, check_consistent_length
from kklearn.validation import num_samples
from kklearn.validation import check_targets

from kklearn.neighbors import KNeighborsClassifier
from kklearn.neighbors import KNeighborsWeightedClassifier
from kklearn.transformers import WeightsRegressor

class MixtureDistribution(object):

    def __init__(self, rv_dist=None, weights=None):
        check_consistent_length(rv_dist, weights)
        self.weights = weights/np.sum(weights)
        self.rv_dists = rv_dist
        pass

    @property
    def n_components(self):
        return num_samples(self.rv_dists)

    @property
    def dim(self):
        return self.rv_dists[0].dim

    def pdf(self, x):
        z = 0.
        for k, rv in enumerate(self.rv_dists):
            with np.errstate(under='ignore'):
                c = rv.pdf(x)
                z += self.weights[k] * c
        return z

    def cdf(self, x):
        z = 0.
        for k, rv in enumerate(self.rv_dists):
            with np.errstate(under='ignore'):
                c = rv.cdf(x)
                z += self.weights[k] * c
        return z

    def logpdf(self, x):
        n_samples = num_samples(x)
        z = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            with np.errstate(under='ignore'):
                z[:,k] = self.rv_dists[k].logpdf(x)
        u = logsumexp(z, b=self.weights)
        return u

    def logcdf(self, x):
        n_samples = num_samples(x)
        z = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            with np.errstate(under='ignore'):
                z[:, k] = self.rv_dists[k].logcdf(x)
        u = logsumexp(z, b=self.weights)
        pass

    def rvs(self, size=1, random_state=None):
        random_state_ = check_random_state(random_state)
        # draw mixture component for each sample
        comps = stats.multinomial.rvs(1, self.weights, size=size, random_state=random_state_)
        n_comps = np.sum(comps, axis=0).astype(dtype=np.int)
        n_comps = stats.multinomial.rvs(size, self.weights, size=1, random_state=random_state_).reshape((-1,))
        X = None
        for k, n in enumerate(n_comps):
            if n < 1:
                continue
            x_k = self.rv_dists[k].rvs(size=n, random_state=random_state_)
            x_k = x_k.reshape((n, -1))
            X = x_k if X is None else np.concatenate((X, x_k), axis=0)
        return X

    def test(self):
        x = self.rvs(20)
        p = self.pdf(x)
        q = self.cdf(x)
        pass

class TargetConditionalDistribution(object):

    def __init__(self, targets=None, rv_dist=None):
        if targets is not None and rv_dists is not None:
            check_consistent_length(rv_dists, targets)
        self.targets = targets
        self.rv_dists = rv_dist
        pass

    @property
    def conditional_dist(self, t=0):
        # Pr[y=t|x]
        return self.rv_dists[t]

    def pmf(self, X):
        # Pr[y=k given x]
        n_targets = 2
        n_samples = num_samples(X)
        beta = lambda x: np.array([-X, -2 * X + (X / 2) ** 3])
        beta = lambda x: np.array([(X / 2) ** 3, -(X / 2) ** 3])
        z = beta(X).reshape((-1, n_targets))
        z = np.zeros_like(z)
        J = [-1 / 4, 1 / 4]
        for i in range(n_samples):
            if X[i] < J[0] or X[i] > J[1]:
                z[i, :] = [0, 2]
            else:
                z[i, :] = [2, 0]
        p = softmax(z, axis=1)
        # k = np.argmax(p, axis=1).astype(dtype=int)
        return p

    def logpmf(self, X):
        p = self.pmf(X)
        q = np.log(p)
        return q

    def rvs(self, X, random_state=None):
        random_state_ = check_random_state(random_state)
        p = self.pmf(X)
        k = np.argmax(p, axis=1).astype(dtype=int)
        n_samples = num_samples(X)
        y = np.zeros((n_samples,), dtype=np.int)
        z = np.zeros_like(p)
        for i in np.arange(n_samples):
            z[i] = stats.multinomial.rvs(1, p[i], size=1, random_state=random_state_)
        y = np.argmax(z, axis=1).astype(dtype=int).reshape((n_samples,))
        return y

# Shimodaira examples
# N(mean=0.5, cov=0.5*0.5) and N(mean=0.2, cov=0.3*0.3), LR: -x + x*x*x

def D_dist(s=.5):
    #  training data distribution - frozen random variable distribution
    D_kwargs = dict(mean=0.75, cov=s * s)
    obj = stats.multivariate_normal(**D_kwargs)
    return obj

def Q_dist2():
    # query data distribution around the class boundaries
    mu = 1/4
    s = 2/16
    md = MixtureDistribution([
        stats.multivariate_normal(mean=-3/10, cov=s * s),
        stats.multivariate_normal(mean=-1/10, cov=s * s),
        # stats.multivariate_normal(mean=4/10, cov=s * s),
        ],
        np.array([0.2, 0.8]))
    return md

def Q_dist(s=0.25):
    #  query data distribution - frozen random variable distribution
    Q_kwargs = dict(mean=0.50, cov=s * s)
    obj = stats.multivariate_normal(**Q_kwargs)
    return Q_dist2()
    return obj


def normalize_weights(w, force_positive=True, offset=None, eps_threshold=1e-6):
    w = w.shape[0] * w / w.sum()
    if offset is None: #not is_float(offset):
        offset = w[np.nonzero(w)].min() / w.shape[0] if force_positive and w.min() < eps_threshold else 0.
    w = w + np.abs(offset)
    return w

def fit_predict_score(est, X_tr, y_tr, X_te, y_te, **fit_params):
    est.fit(X_tr, y_tr, **fit_params)
    y_pred = est.predict(X_te)
    y_score = est.predict_proba(X_te)
    y_true = y_te

    acc = sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True)
    cmat = sklearn.metrics.confusion_matrix(y_true, y_pred)
    auc = sklearn.metrics.roc_auc_score(y_true, y_score[:, -1])
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    print(f"acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}\nCM\n{cmat}\n\n")
    return

def demoKNN():

    random_state_ = 2019
    random_state_ = check_random_state(random_state_)

    D = D_dist()
    Q = Q_dist()
    T = TargetConditionalDistribution()

    n_samples = 500

    # training data from D distribution
    X_tr = D.rvs(size=n_samples, random_state=random_state_).reshape((n_samples, -1))
    y_tr = T.rvs(X_tr, random_state=random_state_)

    # test data from Q distribution
    X_te = Q.rvs(size=n_samples, random_state=random_state_).reshape((n_samples, -1))
    y_te = T.rvs(X_te, random_state=random_state_)

    # compute the density ratio of Q/D for the training data; use them as relevance weights
    w = Q.pdf(X_tr) / D.pdf(X_tr)
    w = normalize_weights(w)

    # relevance weights estimator to recall these weights
    scf = w
    scf_est = WeightsRegressor(id_func=lambda x_query, x_fit: x_fit, depends='fit')
    scf_est = scf_est.fit(X_tr, X_tr, y=scf)
    # check correctness of estimator
    # assert(np.allclose(scf_est.predict(X_tr, X_tr), scf))

    # consensus weights
    weights_tr = 'uniform'

    # hyperparamaters of kNN classifier

    n_neighbors = 3

    # consensus-relevance kNN classifier
    search_n_neighbors = 1.0
    knn_params = dict(weights=weights_tr, support=scf_est)
    est1 = KNeighborsWeightedClassifier(n_neighbors=n_neighbors, **knn_params,
                                        search_n_neighbors=search_n_neighbors)

    # classic kNN classifier
    est2 = KNeighborsClassifier(n_neighbors=n_neighbors)

    print('Consensus-relevance kNN on D-Q data')
    fit_predict_score(est1, X_tr, y_tr, X_te, y_te, **knn_params)

    print('Classic kNN on D-Q data')
    fit_predict_score(est2, X_tr, y_tr, X_te, y_te)

    print('Consensus-relevance kNN on Q-Q data')
    X_trQ = Q.rvs(size=n_samples, random_state=random_state_).reshape((n_samples, -1))
    y_trQ = T.rvs(X_te, random_state=random_state_)
    fit_predict_score(est2, X_trQ, y_trQ, X_te, y_te)

    return

if __name__ == "__main__":
    demoKNN()
    pass