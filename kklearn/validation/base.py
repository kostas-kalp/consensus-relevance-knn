import logging
logger = logging.getLogger(__package__)

from .types import *
from .common import *

from sklearn.utils.validation import _num_samples

def num_samples(thing):
    # find the number of samples in an array-like thing
    return _num_samples(thing)

def get_dataframe_feature_types(df):
    # get the names of features of dataframe by type
    if not isinstance(df, pd.DataFrame):
        raise ValueError('arg should be a DataFrame')
    typemap = dict()
    for key, dtype in zip(['float', 'categorical', 'int', 'bool', 'numeric'],
                          [np.float_, 'category', np.int_, np.bool_, np.number]):
        # names = list(TypeSelector(dtype).fit_transform(df))
        names = list(df.select_dtypes(include=dtype))
        typemap[key] = names
    return typemap


def unique_labels(*args):
    # given a list of label lists, find the unique labels in the union of all the label lists
    # (ignore those args that are None)
    args = [x for x in args if x is not None]
    labels = sklearn.utils.multiclass.unique_labels(*args)
    return labels


def check_consistent_shape(*arrays, axis=None, force=True):
    """
    Check that the argument arrays have all the same shape (on an axis or all axes)
    Args:
        *arrays: array-like arguments
        axis: (int) axis along which aggreement is sought (all axes if None)
    """
    shapes = [np.asarray(X).shape for X in arrays if X is not None]
    if axis is None:
        values = np.unique(shapes, axis=0)
    else:
        n = min([np.asarray(X).ndim for X in arrays if X is not None])
        axis = int(axis)
        if axis < -1 or axis >= n:
            raise ValueError(f'argument axis={axis} is out of bounds [0,{n}) or -1')
        dims = [np.asarray(X).shape[axis] for X in arrays if X is not None]
        values = np.unique(dims)
    t = len(values) < 2
    if not t and force:
        raise ValueError("found argument arrays with inconsistent shapes %r" % shapes)
    return t
    pass


def check_train_test_data(data=None, **kwargs):
    # validate data is X_train, y_train, X_test, y_test
    kwargs.setdefault('accept_sparse', 'csr')
    if isinstance(data, (list, tuple)) and len(data) == 4:
        X_train, y_train, X_test, y_test = data
        X_train, y_train = check_X_y(X_train, y_train, **kwargs)
        X_test, y_test = check_X_y(X_test, y_test, **kwargs)
        check_consistent_shape(X_train, X_test, axis=1)
        data = (X_train, y_train, X_test, y_test)
        return data
    else:
        raise ValueError('invalid type or shape training or testing data')


def check_targets(data=None, targets=None, labels=None, categories=None, pos_label=None):
    """
    Check a targets, labels, and pos_label are consistent
    Args:
        data: an array or a tuple of arrays with responses (y)
        targets: (pd.Series) class categories indexed by labels
        labels: (list) list of class labels
            if provided and targets is None, then initialize a targets Series.
            Verify that each label is in the target's index
        categories: (list) of class categories [intended for human consumption]
            if targets is not given, then use the categories as values for the targets Series
            (defaults to labels if None)
        pos_label: label of positive class
            verify that pos_label is in target's index; if None, set it to the last index value
    Returns:
        targets (pd.Series of categories indexed by labels) and pos_label
    """
    labels_, categories_ = None, None
    if data is not None and isinstance(data, (list, tuple, np.ndarray, pd.DataFrame, pd.Series)):
        labels_ = unique_labels(*data) if isinstance(data, (list, tuple)) else unique_labels(data)
    elif targets is not None and isinstance(targets, pd.Series):
        labels_ = list(targets.index.unique())
        categories_ = [targets[i] for i in labels_]
    if categories_ is None:
        categories_ = [f'class {s}' for s in labels_] if categories is None else list(categories)
    labels_ = list(labels) if labels is not None else labels_
    if labels_ is None:
        labels_ = categories_
    if labels_ is None:
        raise ValueError('insufficient data to construct a targets Series')

    check_consistent_length((labels_, categories_))
    if targets is None or not isinstance(targets, pd.Series):
        targets = pd.Series(categories_, index=labels_)
    for v in labels_:
        if v not in targets.index:
            raise ValueError(f"label {v} is not in the targets {targets}")
    if pos_label is None or pos_label not in labels_:
        pos_label = labels_[-1]
    return targets, pos_label


def check_feature_names(feature_names, n_features=2, prefix="x"):
    # returns list of n_features feature names if feature_names is empty or None, else returns itself
    return feature_names if feature_names else [f"{prefix}{i:1d}" for i in range(n_features)]


def check_onehot_encoded(y, labels=None, targets=None, encoder=None, **kwargs):
    """
    Check that y is one-hot encoded
    Args:
        y: column or 1d of target labels, or a 2d array of 0s and 1s indicating a label
        labels: (list) if provided, extend the unique labels present in y with those in labels
        encoder: (OneHotEncoder) if provided (not None), use this encoder to 1-hot-encode y
                else construct and fit an encoder to y
        **kwargs: kwargs to pass on to the OneHotEncoder constructor

    Returns:
        a 2d indicator matrix representing a one-hot encoding of y (or y itself if its already one-hot-encoded)
        the fitted encoder (OneHotEncoder) used
    """
    # y is 1d or 2d finite array
    y = check_array(y, allow_nd=False, ensure_2d=False, accept_sparse="csr")
    # determine targets and labels
    if labels is not None:
        labels = unique_labels(labels)
    if targets is not None:
        labels = list(targets.index.unique())
    if encoder is not None and labels is None:
        labels = encoder.categories_[0]
    if labels is None:
        labels = unique_labels(y)
    # if targets is None:
    #     targets = pd.Series(labels, index=labels)

    if encoder is None and (y.ndim == 1 or targets is not None):
        kwargs.setdefault('sparse', False)
        kwargs.setdefault('handle_unknown', 'ignore')
        kwargs.setdefault('dtype', int)
        encoder = sklearn.preprocessing.OneHotEncoder(categories=[labels], **kwargs)
        encoder.fit(y.reshape(-1, 1))

    if encoder is not None and not set(labels).issubset(set(encoder.categories_[0])):
        raise ValueError('labels should be a subset of encoder categories')

    if y.ndim == 1:
        # encode in labels in one-hot encoding
        y = encoder.transform(y.reshape(-1, 1))
    else:
        vals = unique_labels(y.reshape(-1))
        if y.shape[-1] != len(labels) or not set(vals).issubset(set([0, 1])):
            raise ValueError('not one hot encoded')
        # y is already one-hot-encoded
    return y, encoder


def check_sample_weights(sample_weight, n_samples):
    """
    Check that sample_weight is a column or 1d array of weights
    Args:
        sample_weight: scalar, column, or 1d array of numbers
        n_samples: if sample_weight is scalar, the length column to create

    Returns: 1d array of numbers
    """
    if sample_weight is None:
        sample_weight = np.full(n_samples, fill_value=1.)
    elif np.isscalar(sample_weight):
        sample_weight = np.full(n_samples, fill_value=sample_weight)
    sample_weight = column_or_1d(sample_weight, warn=True)
    return sample_weight


def check_probabilities(probas=None):
    """
    Check that data is a valid 1d or 2d array of probability numbers
    Args:
        probas: column, 1d, or 2d array of numbers between 0 and 1.
            if it is 2d, then require each row to sum to 1
    Returns:

    """
    # there is a similar CONFLICTING routine for 1d probas in kklearn.sampling

    def scale(x):
        if x.ndim != 2:
            return x
        if min(x.ravel()) < 0:
            x = 1. / (1. + np.exp(-x))
        # z = x / np.sum(x, axis=1)[:, None]
        # z = x / x.sum(axis=1)
        z = sklearn.preprocessing.normalize(x, axis=1, norm='l1')
        return z

    probas = check_array(probas, allow_nd=False, ensure_2d=False, accept_sparse="csr")
    if probas.ndim == 1:
        probas = probas.reshape((-1,1))

    probas = check_array(probas, allow_nd=False, ensure_2d=True, accept_sparse="csr")

    # probas = check_array(probas)
    probas = scale(probas)
    if probas.ravel().min() < 0.:
        raise ValueError(f'min of probas={probas.ravel().min()} is less than 0.')
    elif probas.ravel().max() > 1.:
        raise ValueError(f'max probas={probas.ravel().max()} is more than 1.')
    if probas.ndim == 2:
        s = np.sum(probas, axis=1)
        if not np.allclose(s, 1.):
            logger.warning(f'probas rowsums={s} not equal to 1 {probas}')
    return probas
    pass


def check_include_exclude(include=None, exclude=None, domain=None, disjoint=False, diff=False, project=False):
    """
    Validate two list/set-like objects of values include and exclude ad return validated sets
    Args:
        include: (list-like) set of values
        exclude: (list-like) set of values
        domain: (list-like) set of values specifying the universe of values
        disjoint: enforce that the include/exclude value sets are disjoint
        diff: discard from include the values in exclude
        project: project include and exclude to only have values in domain

    Returns:
        (include, exclude): a tuple of two sets of values
    """
    if include is not None and not is_list_like(include):
        include = (include,)
    if exclude is not None and not is_list_like(exclude):
        exclude = (exclude,)
    if include is not None and not is_list_like(include):
        raise ValueError('arg include should be list-like or None')
    if exclude is not None and not is_list_like(exclude):
        raise ValueError('arg exclude should be list-like or None')
    if domain is not None and not is_list_like(domain):
        raise ValueError('arg domain should be list-like or None')

    domain_ = set(domain) if is_list_like(domain) else set()
    exclude_ = set(exclude) if is_list_like(exclude) else set()

    if include is None:
        include_ = domain_ if bool(domain) else set()
    elif not is_list_like(include):
        include_ = set()
    else:
        include_ = set(include)

    if project:
        include_ = domain_.intersection(include_)
        exclude_ = domain_.intersection(exclude_)
    if diff:
        include_ = include_.difference(exclude_)
    if disjoint and bool(include_.intersection(exclude_)):
        raise ValueError(f'include/exclude args have common elements {include_.intersection(exclude_)}')

    # force non-empty union
    # if not (bool(include) or bool(exclude)):
    #     raise ValueError(f'at least one of include or exclude should be non-empty -- '
    #                      f'include={include} and exclude={exclude}')

    return include_, exclude_


def get_array_profile(X):
    if X is None:
        return None
    elif is_numeric_array(X):
        profile = X.shape
    elif is_object_array(X):
        n = num_samples(X)
        w = np.zeros(n, dtype=int)
        for i, x in enumerate(X):
            if _is_arraylike(x):
                w[i] = num_samples(x)
        profile = (n, np.asarray(w))
    else:
        raise ValueError(f'invalid type {type(X)}')
    return profile

