import logging
logger = logging.getLogger(__package__)

import pandas as pd

from pandas.api.types import is_list_like, is_dict_like

# Define a masked dict subclass of dict
# .set_mask(key or list of keys, on/off)
# .get_mask(key, list of keys)
# .compress() return copy of dict without the masked keys
#

def dict_from_keys(thing, keep="all", default=None, init=False, ignore=False):
    """
    Get a sub-dictionary (copy) of a dictionary with only select keys
    Args:

        thing: (dict-like) input dictionary
        keep: "all" or list of keys
        default: default value for the key in the result if it is not in the input dict
        init:  (bool) if True insert default value for keys in keep but not in the input dict
    Returns:
        new (dict)
    """
    if not is_dict_like(thing):
        raise ValueError(f'argument thing should be dict-like')
    if not (keep in ('all', ) or is_list_like(keep)):
        raise ValueError(f'argument keep should be "all" or list-like of keys {keep}')
    if keep in ('all',):
        return thing.copy()
    obj = dict()
    for key in keep:
        if init:
            obj[key] = thing.get(key, default)
            continue
        elif key in thing:
            obj[key] = thing[key]
        elif not ignore:
            logger.warning(f'skipping key {key} in the output dict since it is not found in input dict and init={init}')
    # dz = thing.from_keys(keep)
    return obj


def dict_to_dataframe(data, onerow=False, **kwargs):
    """
     Make a dataframe from dict data

    Args:
        data: (dict) or (dict of dict = a dict whose key-values are dict)
        onerow: (bool)
            if True then the result is a single-row dataframe with one column for each key in data
            if False then the result has one row for each key of data; the columns are determined by the keys
            of the dict associated with the key's value (inner dict)
        **kwargs:
            kwargs for the pd.DataFrame()
            if the default kwargs["index"] is set to "row" or data.keys() depending on the value of onerow
    Returns: (pd.DataFrame)
        a dataframe with one row for each key in data and one column for each key in the inner-dict (see onerow above)

    Should look to v =  sklearn.feature_extraction.DictVectorizer()
        >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        >>> X = v.fit_transform(D)
        >>> v.get_feature_names()
    """
    if onerow:
        data_new = dict()
        for k, v in data.items():
            if type(v) == list and len(v) > 1 or type(v) != list:
                v = [v]
            data_new[k] = v
        kwargs.setdefault("index", "row")
        df = pd.DataFrame(data_new, **kwargs)
    else:
        kwargs.setdefault("index", data.keys())
        df = pd.DataFrame(list(data.values()), **kwargs)
    return df
