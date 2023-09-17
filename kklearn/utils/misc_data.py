import logging
logger = logging.getLogger(__package__)

import numpy as np
import pandas as pd

import sklearn

from ..validation.common import check_random_state
from ..validation.common import check_array, column_or_1d, check_consistent_length
from ..validation.common import is_list_like, is_dict_like, is_array_like
from ..validation import num_samples


def row_normalize(X):
    # normalize the rows of 2d array X to sum to 1
    if is_array_like(X):
        X = X.shape[1] * X / np.nansum(X, axis=1)[:, None]
    else:
        logger.warning(f'argument is not array-like -- {type(X)}')
    return X


def to_dataframe(X, **kwargs):
    """
    Convert its argument 2d array-like to a pandas DataFrame if it's not already a DataFrame
    Args:
        X:
    Returns:
        X (if it is a dataframe) else a dataframe with a column for each column for X (in order)
    """
    if not isinstance(X, pd.DataFrame):
        try:
            if X is not None:
                X_ = check_array(X, ensure_2d=True, accept_sparse='csr')
                kwargs['data'] = X_
                X = pd.DataFrame(**kwargs)
                # X = pd.DataFrame(data=X_)
        except Exception as ex:
            logger.error(' %s', ex, exc_info=True)
            raise ex
    return X


def truncate_arrays(*arrays, n_samples_limit=1000, random_state=None, randomize=True):
    """
    truncate the argument arrays to have no more than n_samples_limit rows
    Args:
        arrays: list-like of arrays to truncate
        n_samples_limit: the limit to the number of rows to limit to
        random_state:
        randomize: (bool) if False, select first n_samples_limit samples,
                   else select samples uniformly at random
    Returns:
        list of truncated arrays
    """
    random_state_ = check_random_state(random_state)
    check_consistent_length(arrays)
    n_samples = num_samples(arrays[0])
    result = arrays
    if n_samples_limit is not None:
        n_samples_limit = int(n_samples_limit)
        if n_samples_limit > 0 and n_samples > n_samples_limit:
            logger.debug(f'truncating arrays of {n_samples} to {n_samples_limit} instances')
            if randomize:
                S = sklearn.utils.random.sample_without_replacement(n_population=n_samples, n_samples=n_samples_limit,
                                                                    random_state=random_state_)
            else:
                S = np.arange(n_samples_limit)
                # S = [i for i in range(n_samples_limit)]
            result = []
            for arr in arrays:
                z = arr[S]
                result.append(z)
    if len(result) == 1:
        return result[0]
    return result


def get_inferred_dtypes(df, include=None, exclude=None):
    """
    Series of dtypes of selected (unique columns in the difference include minus exclude) columns in dataframe
    See pandas.api.types.infer_dtype() for more details about the dtypes
    Args:
        df: (DataFrame) dataframe to find the infered dtypes of its attributes (columns)
        include: list-like(str) of columns to consider or None (all columns if it is None)
        exclude: list-like(str) of columns to ignore (or None)
    Returns:
        pd.Series: sequence of column dtypes indexed by the selected (unique in the difference of include
                minus exclude) columns. The order of the selected columns in the returned Series is consistent
                with their order in the dataframe
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f'a dataframe argument is expected')
    include = set(include) if is_list_like(include) else set(df.columns)
    exclude = set(exclude) if is_list_like(exclude) else ()
    if not include.isdisjoint(exclude):
        logger.warning(f'arguments include and exclude not disjoint, "'
                       f'include=[{include}] and exclude=[{exclude}] ==> intersection={include.intersection(exclude)}')
    keep = set(include).difference(set(exclude))
    selection = pd.Series(False, index=df.columns)
    selection[keep] = True

    col_dtypes = pd.Series((pd.api.types.infer_dtype(df[x], skipna=True) for x in df.columns if selection[x]),
                           index=(x for x in df.columns if selection[x]))
    return col_dtypes

