# commonly available validation checks in sklearn AND pandas packages -- simplifies imports
import logging
logger = logging.getLogger(__package__)

import numpy as np
import pandas as pd
import sklearn

from sklearn.utils import check_random_state
from sklearn.utils import check_array, column_or_1d
from sklearn.utils import check_X_y, check_consistent_length

from sklearn.utils.validation import check_is_fitted, has_fit_parameter
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _is_arraylike

from pandas.api.types import is_list_like, is_array_like, is_dict_like
from pandas.api.types import is_number, is_scalar, is_integer, is_bool, is_float
from pandas.api.types import is_integer_dtype, is_numeric_dtype, is_float_dtype
from pandas.api.types import is_categorical_dtype, is_string_dtype
from pandas.api.types import infer_dtype

__all2__ = ["check_array",
           "column_or_1d",
           "check_random_state",
           "check_X_y",
           "check_consistent_length",
           "check_is_fitted",
           "has_fit_parameter",
           "num_samples",
           "num_samples",
           "_is_arraylike"
           ]

