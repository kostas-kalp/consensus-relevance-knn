import logging
logger = logging.getLogger(__package__)

import numpy as np
import pandas as pd

from collections import Counter, OrderedDict

from ..validation.common import is_number, is_integer, is_bool
from ..validation.common import is_list_like, is_dict_like, is_numeric_dtype, is_integer_dtype
from ..validation.common import check_random_state
from ..validation.common import check_array, column_or_1d, check_X_y, check_consistent_length
from ..validation import num_samples

EPS = np.finfo(np.float64).eps

def clip(data, low=None, high=None, inplace=False):
    """
    Clip the values of aan 1d array to be in a given range (inplace or in a copy of the input)
    Args:
        data: numeric 1d array-like
        low: low value or None
        high: high value or None
        inplace: (bool) modify array in place (True) or a copy (False)

    Returns:
    array with its values clipped to be in the range [low, high]. If either low or high is None, no change
    """
    if data is None or (low is None and high is None):
        return data
    if is_number(data):
        data = low if is_number(low) and data < low else data
        data = high if is_number(high) and data > high else data
        return data
    elif is_list_like(data) and is_numeric_dtype(data):
        if not is_bool(inplace):
            raise ValueError('inplace should be a bool')
        if not inplace:
            data = np.copy(data)
        data = column_or_1d(data)
        if is_number(low):
            data[data < low] = low
        if is_number(high):
            data[data > high] = high
    else:
        raise ValueError('data should be a scalar or numeric 1d array-like')
    return data


def scale(data, sum_to=None, inplace=False):
    """
    Scale the values of a numeric 1d array to have sum sum_to
    Args:
        data: numeric 1d array
        sum_to: value for the desired sum of the input array
        inplace: (bool) if True modify the array in place or its copy if False

    Returns:
        scaled input array so that its sum is the desired value
    """
    if sum_to is None or data is None:
        return data
    if not is_numeric_dtype(data):
        raise ValueError('data should be a numeric 1d array-like')
    if is_number(sum_to):
        if not is_bool(inplace):
            raise ValueError('inplace should be a bool')
        if not inplace:
            data = np.copy(data)
        data = column_or_1d(data)
        t = np.nansum(data)
        if not np.allclose([t], [0.]):
            data = sum_to * data / t
        elif np.allclose([sum_to], [0.]):
            data = np.zeros_like(data)
        else:
            logger.warning('division by zero')
            data = np.full(data.shape, fill_value=np.nan, dtype=data.dtype)
    else:
        raise ValueError('argument sum_to should be a number or None')
    return data


def counts_to_counter(counts):
    """
    convert a list/dict/array-like counts to a Counter object
    Args:
        counts: a dict-like of counts, or list-like or array-like sequence of values to count their frequencies
    Returns:
        Counter object with an non-negative int associated with each key value
    """
    #
    if isinstance(counts, Counter):
        return counts
    if isinstance(counts, pd.Series):
        obj = counts.to_dict()
    elif is_dict_like(counts):
        obj = counts
    elif is_list_like(counts):
        counts = column_or_1d(counts)
        if not is_integer_dtype(counts):
            raise ValueError('counts should be an integer 1d array')
        obj = OrderedDict([(i, counts[i]) for i in range(counts.shape[0])])
    else:
        raise ValueError('counts should be a 1d array-like or dict-like instance of integers')
    counter = Counter(obj)
    return counter


def counter_to_counts(counter, n_instances=None, contiguous=True):
    """
    Convert a (Counter) counter to a an integer 1d numpy array indexed by the keys of the counter.
    When the counter is indexed by integers in range(), then n_instances and contiguous introduce zero counts
    for all integers in range(n_instances) from the keys of counter()
    If contiguous is True then missing index values in range(max(n_instances, max(counter.keys)) are set to zero
    Args:
        counter: (Counter) dict-like of counts
        n_instances: number of assumed integer indices for the 1d array counts
        contiguous: whether gaps in counts are to be filled with 0s

    Returns:
        1d numpy array of integer counts
    """
    if not isinstance(counter, Counter):
        raise ValueError('counter should be an instance of Counter')
    if not(n_instances is None or is_integer(n_instances) and n_instances > 0):
        raise ValueError('n_instances should be a positive integer or None')
    if not is_bool(contiguous):
        raise ValueError('contiguous should be a bool')

    # arr = pd.Series(data=list(counter.values()), index=list(counter.keys()))
    arr = pd.Series(counter)
    arr.sort_index(ascending=True, inplace=True)
    if is_integer(n_instances) and n_instances >= 0:
        contiguous = True
    else:
        n_instances = 0
    if contiguous:
        index, values = arr.index, arr.values
        if not is_integer_dtype(index) or any(index < 0):
            raise ValueError('counter should have non-negative integers as keys')
        n_instances = max(np.max(index) + 1, n_instances)
        counts = np.full(n_instances, fill_value=0, dtype=int)
        counts[index] = values
    else:
        counts = column_or_1d(arr)
    return counts
