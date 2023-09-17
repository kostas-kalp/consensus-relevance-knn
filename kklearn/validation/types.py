import logging
logger = logging.getLogger(__package__)

import numpy as np
import pandas as pd
import inspect
import sklearn

from .common import *

from sklearn.utils.validation import _is_arraylike

__all2__ = ["is_list_like",
           "is_dict_like",
           "is_number",
           "is_scalar",
           "is_integer",
           "is_bool",
           "is_float",
           "is_array_dtype",
           "is_object_array",
           "is_numeric_array",
           "check_column_or_matrix",
           "check_super_hasmethod",
           "check_hasmethod",
           "check_is_transformer",
           "is_classification_targets"]

def is_string(s):
    return isinstance(s, str)

def is_array_dtype(X, kind=None):
    # Check the type of elements in an array
    if kind is None:
        kind = 'uifc'
    dtype_kinds = set(kind)
    if X is None or not is_array_like(X):
        return False
    if np.asarray(X).dtype.kind in dtype_kinds:
        return True
    return False


def is_object_array(X):
    return is_array_dtype(X, kind='O')


def is_numeric_array(X, kind=None):
    kind = 'uifcb' if kind is None else kind
    return is_array_dtype(X, kind=kind)


def check_column_or_matrix(data, ndim=None):
    """
    Check that data is either 1d or 2d
    Args:
        data: column or 1d or 2d array
        ndim: number (1 or 2) of required dimensions or None (either 1 or 2)

    Returns:
        1d or 2d array
    """
    #data = check_array(data, ensure_2d=False, allow_nd=False)
    if ndim is None:
        data = check_array(data, ensure_2d=False, allow_nd=False, accept_sparse="csr")
    if ndim == 1:
        data = column_or_1d(data, warn=True)
    elif ndim == 2:
        data = check_array(data, ensure_2d=True, allow_nd=False, accept_sparse="csr")
    else:
        raise ValueError(f'argument ndim={ndim} should be 1 or 2')
    return data

def check_super_hasmethod(thing, method_name, force=False, exclusive=True):
    """
    Check that the superclass of thing (or thing if exclusiv is Falase) has a method of given name
    Args:
        thing: instance of a class
        method_name: (str) name of method
        force: (bool) if True raise an exception
        exclusive: (bool) if True method should be in the super-class (else it be of thing's class itself)

    Returns:
        (bool) indicating whether required method exists
    """
    cls = thing if inspect.isclass(thing) else type(thing)
    klasses = list(inspect.getmro(cls))
    if exclusive:
        klasses.remove(cls)
    ans = {x: check_hasmethod(x, method_name=method_name) for x in klasses}
    t = any(ans.values())
    if not t and force:
        raise ValueError(f'super of {cls} does not have required method {method_name}')
    return t


def check_hasmethod(thing, method_name, force=False):
    """
    Check that thing has a method with given name
    Args:
        thing: instance of a class
        method_name: (str) method name
        force: (bool) if True raise Exception if method does not exist

    Returns:
        (bool) indicating existence or not the required method
    """
    if inspect.isclass(thing):
        t = callable(getattr(thing, method_name, None))
    else:
        t = callable(getattr(thing.__class__, method_name, None))
    if not t and force:
        raise ValueError(f'thing does not have required method:: {method_name} {thing}')
    return t


def check_is_transformer(est, force=False):
    if est is not None and issubclass(type(est), sklearn.base.TransformerMixin):
        return est
    if force:
        raise ValueError(f'argument is not a subclass of TransformerMixin')
    return None


def is_classification_targets(y, force=False):
    # modeled after sklearn.utils.multiclass.check_classification_targets method
    t = sklearn.utils.multiclass.type_of_target(y)
    if t in ['binary', 'multiclass', 'multiclass-multioutput', 'multilabel-indicator', 'multilabel-sequences']:
        return True
    if force:
        raise ValueError(f'argument should be a classification target type, not {t}')
    return False
