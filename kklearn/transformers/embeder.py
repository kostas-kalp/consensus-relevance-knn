#Dimensionality reduction/data projection methods for visualizing classification results
import logging
logger = logging.getLogger(__package__)

import warnings
from functools import partial

from sklearn import decomposition
from sklearn import manifold
import umap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import inspect

from ..viz import scatterplot as default_scatterlot

from ..viz import mkfigure

from ..utils import dict_from_keys

class Embeder():

    def __init__(self, n_components=2, component_names=None, n_neighbors=15, kernel="rbf",
                 algo="PCA", random_state=None, **kwargs):

        if component_names is None:
            component_names = [ f"comp{i}" for i in range(n_components)]
        self.n_components = n_components
        self.component_names = component_names
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.kwargs = kwargs
        self.random_state = random_state
        self.set_reducer(algo, random_state=random_state, **kwargs)
        pass

    def get_reducer(self, algo, **kwargs):
        kwargs.setdefault('n_components', self.n_components)
        kwargs.setdefault('n_neighbors', self.n_neighbors)
        kwargs.setdefault('random_state', self.random_state)

        reducers = {'MDS': dict(obj=manifold.MDS,
                                kwargs=dict(n_components=2, max_iter=100, n_init=1)),
                    'LLE': dict(obj=manifold.LocallyLinearEmbedding,
                                kwargs=dict(n_components=2, n_neighbors=5, eigen_solver='auto')),
                    'ISO': dict(obj=manifold.Isomap,
                                kwargs=dict(n_components=2, n_neighbors=5)),
                    'SPE': dict(obj=manifold.SpectralEmbedding,
                                kwargs=dict(n_components=2, n_neighbors=5)),
                    'TSNE': dict(obj=manifold.TSNE,
                                 kwargs=dict(n_components=2, init='pca')),
                    'SPCA': dict(obj=decomposition.SparsePCA,
                                 kwargs=dict(n_components=2)),
                    'SVD': dict(obj=decomposition.TruncatedSVD,
                                kwargs=dict(n_components=2, n_iter=10)),
                    'PCA': dict(obj=decomposition.PCA,
                                kwargs=dict(n_components=2, svd_solver='auto')),
                    'NMF': dict(obj=decomposition.NMF,
                                kwargs=dict(n_components=2, init='random')),
                    'UMAP': dict(obj=umap.UMAP,
                                 kwargs=dict(n_components=2, n_neighbors=5, metric="euclidean", min_dist=.2))
                    }
        entry = reducers.get(algo, reducers['PCA'])
        obj = entry['obj']

        prms = [param.name for param in inspect.signature(obj.__init__).parameters.values() if
             param.kind == param.POSITIONAL_OR_KEYWORD]
        prms.remove('self')

        o_kwargs = entry['kwargs']
        o_kwargs.update(kwargs)
        o_kwargs = dict_from_keys(o_kwargs, prms)

        # for key in prms:
        #     v = kwargs.get(key, None)
        #     if v:
        #         o_kwargs[key] = v
        #o_kwargs.update(kwargs)
        obj = obj(**o_kwargs)
        return obj

        func = decomposition.PCA.__init__
        p = [param.name for param in inspect.signature(func).parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
        if algo == 'MDS':
            return manifold.MDS(n_components=self.n_components, max_iter=100, n_init=1, random_state=random_state, **kwargs)
        if algo == 'LLE':
            kwargs.pop('metric')
            return manifold.LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components, eigen_solver='auto', random_state=random_state, **kwargs)
        if algo == 'ISO':
            return manifold.Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components, **kwargs)
        if algo == 'SPE':
            return manifold.SpectralEmbedding(n_components=self.n_components, n_neighbors=self.n_neighbors, random_state=random_state, *kwargs)
        if algo == 'TSNE':
            return manifold.TSNE(n_components=self.n_components, init='pca', random_state=random_state, **kwargs)
        if algo == 'SPCA':
            return decomposition.SparsePCA(n_components=self.n_components, random_state=random_state,)
        if algo == 'KPCA':
            return decomposition.KernelPCA(n_components=self.n_components, kernel="rbf", gamma=10, random_state=random_state, **kwargs)
        if algo == 'SVD':
            return decomposition.TruncatedSVD(n_components=self.n_components, n_iter=10, random_state=random_state, **kwargs)
        if algo == 'PCA':
            kwargs.pop('metric')
            return decomposition.PCA(n_components=self.n_components, svd_solver='auto', random_state=random_state, **kwargs)
        if algo == "NMF":
            return decomposition.NMF(n_components=self.n_components, init='random', random_state=random_state)
        if algo == "UMAP":
            return umap.UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors, metric="euclidean", min_dist=.2, random_state=random_state)

        # PCA is default
        return decomposition.PCA(n_components=self.n_components, svd_solver='auto', random_state=random_state, **kwargs)

    def set_reducer(self, algo="PCA", **kwargs):
        self.reducer = self.get_reducer(algo, **kwargs)

    def fit(self, X, y=None):
        self.reducer.fit(X, y=y)
        return self
        pass

    def transform(self, X):
        return self.reducer.transform(X)
        pass

    def fit_transform(self, X, y=None):
        return self.reducer.fit_transform(X, y)
        if X is None:
            return None
        self.fit(X, y)
        return self.transform(X)

#####################################################################################


def test_multiple_embeddings(X, y=None, ds_name='', random_state=None,
                             xlabel='', ylabel='', scatterplot=None, **kwargs):
    # needs scatterplot(), see viz.py for an example
    if scatterplot is None:
        scatterplot = default_scatterlot

    algos = kwargs.get('algos', ["UMAP", "PCA", "SPE",  "LLE"])
    n_components = kwargs.get('n_components', 2)
    n_neighbors = kwargs.get('n_neighbors', 15)
    metric = kwargs.get('metric', 'euclidean')
    ncols = kwargs.get("ncols", 4)
    w = kwargs.get("w", 16.)

    #    algos = ("UMAP", "PCA", "SPE",  "LLE", "MDS",  "TSNE", "ISO", "NMF", "KPCA")[0:4]

    M = len(algos)
    nrows = int(np.ceil(M/ncols))
    fig = mkfigure(nrows, ncols, w=w/ncols)
    for k, algo in enumerate(algos):
        print(f"embeder {algo}")
        model = Embeder(n_components=n_components, n_neighbors=n_neighbors, algo=algo, random_state=random_state)
        embedding = model.fit_transform(X, y=y)

        ax = fig.add_subplot(nrows, ncols, k + 1, xlabel=xlabel, ylabel=ylabel)
        scatterplot(x=embedding[:,0], y=embedding[:,1], hue=y, ax=ax)
        ax.set_title(f'{algo}', fontsize=10)
        ax.set_aspect('equal', 'datalim')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Projections for the [{ds_name}] dataset")
    plt.tight_layout(pad=2.)
    plt.show()

#####################################################################################

def test_embedding_params(X, y=None, ds_name='', random_state=None,
                          xlabel='', ylabel='', scatterplot=None, **kwargs):
    # needs scatterplot(), see viz.py for an example
    algo = "UMAP"

    if scatterplot is None:
        scatterplot = default_scatterlot

    n_components = kwargs.get('n_components', [2])
    n_neighbors = kwargs.get('n_neighbors', [3, 5, 15, 30])
    metrics = kwargs.get('metrics', ['euclidean', 'manhattan', 'cosine'])

    ncols = kwargs.get("ncols", len(n_neighbors))
    w = kwargs.get("w", 16.)
    # metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev'][:2]

    M = len(metrics) * len(n_neighbors) * len(n_components)
    nrows = int(np.ceil(M/ncols))
    fig = mkfigure(nrows, ncols, w=w/ncols)

    k = 1
    for metric in metrics:
        for n_comp in n_components:
            for n_nei in n_neighbors:
                model = Embeder(n_components=n_comp, n_neighbors=n_nei, metric=metric, algo=algo, random_state=random_state)
                embedding = model.fit_transform(X, y)
                ax = fig.add_subplot(nrows, ncols, k, xlabel=xlabel, ylabel=ylabel)
                scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y, ax=ax)
                ax.set_title(f'{metric} #comps={n_comp} #neighbors={n_nei}', fontsize=10)
                ax.set_aspect('equal', 'datalim')
                #ax.set_axis_off()
                ax.set_xticks([])
                ax.set_yticks([])
                k += 1
    fig.suptitle(f"{algo} projections for [{ds_name}] dataset")
    plt.tight_layout(pad=2.)
    plt.show()

#####################################################################################

if __name__ == "__main__":
    import sys
    rng = np.random.RandomState(2019)

    sns.set()
    sns.set_style("whitegrid")
    sns.set_style("white")
    cmap = ("tab10", "muted", "deep", "dark", "Set1")[0]
    # cmap = sns.color_palette(["#3498db", "#e74c3c"])

    scatterplot = partial(sns.scatterplot, alpha=.9, marker='.', legend=False, s=75, palette=cmap)

    warnings.filterwarnings("ignore")

    n_samples = 1000
    n_features = 5

    #from sklearn import datasets as sk_datasets
    from my_datasets import sk_datasets, sklearn_dataset_to_dataframe, ndarray_to_dataframe

    if True:
        X, y = sk_datasets.make_blobs(n_samples=n_samples, centers=[[2]*n_features, [-2]*n_features], cluster_std=[0.5, 0.5], random_state=rng)
        X, y = sk_datasets.make_moons(n_samples=n_samples, noise=.12, random_state=rng)
        X = np.abs(X)
        ds, metadata = ndarray_to_dataframe(X, y, name="blob", source="synthetic")
        ds._metadata = metadata

    if True:
        #ds, metadata = sklearn_dataset_to_dataframe(sk_datasets.load_boston())
        #y = ds.apply(lambda row: row.target > 10, axis=1)
        #ds = ds[['RM', 'LSTAT', 'RAD', 'PTRATIO']]
        #ds = ds.assign(target=y)
        #features = list(set(ds) - set(['target']))
        ds, metadata = sklearn_dataset_to_dataframe(sk_datasets.load_breast_cancer(), name='breast_cancer')
        ds, metadata = sklearn_dataset_to_dataframe(sk_datasets.load_iris(), name='iris')
        ds, metadata = sklearn_dataset_to_dataframe(sk_datasets.load_wine(), name='wine')
        ds._metadata = metadata
        #ds[metadata.response_name] = pd.Series(metadata.response_labels[ds[metadata.response_name]])

    metadata = ds._metadata
    ds_name = metadata.session_id
    X, y = ds[metadata.feature_names].values, ds[metadata.response_name]

    print(f"{ds_name} {metadata}\n", ds.head(5))

    test_multiple_embeddings(X, y, ds_name=ds_name, random_state=rng, scatterplot=scatterplot)

    test_embedding_params(X, y, ds_name=ds_name, random_state=rng, xlabel='comp0', ylabel='comp1', scatterplot=scatterplot)

    input('press any key')
    pass