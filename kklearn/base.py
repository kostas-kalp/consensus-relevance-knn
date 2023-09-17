import logging
logger = logging.getLogger(__package__)

import sklearn
from sklearn.base import clone

from sklearn.utils import check_random_state, check_array, column_or_1d, check_X_y
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

from .utils import dict_from_keys
from .validation import check_sample_weights, check_hasmethod

from collections import defaultdict

def _make_estimator(base_estimator, default_estimator=None, **estimator_params):
    # helper function to call or clone an estimator
    if base_estimator is None or base_estimator == 'auto':
        base_estimator = default_estimator
    if base_estimator is None:
        raise ValueError(f'inferred base_estimator can not be None -- provide base_estimator or default_estimator')
    if callable(base_estimator):
        estimator = base_estimator(**estimator_params)
    else:
        estimator = clone(base_estimator)
        estimator.set_params(**estimator_params)
    return estimator


def get_params_actual(estimator, local_vars):
    """
    Build a dict of the actual values of the formal parameters of the __init__() of an estimator instance
    with respecy to the dict of local_vars (or variables and their values)
    Args:
        estimator: a sklearn estimator that implements get_params()
        local_vars: dict of local variable names and their values

    Returns:
        dict of the actual values of the instance's __init__() parameters
    """
    bound = {}
    if local_vars is None or not isinstance(local_vars, dict):
        local_vars = {}
    for key, val in estimator.get_params().items():
        bound[key] = local_vars.get(key, val)
    bound = dict_from_keys(local_vars, keep=estimator.get_params().keys(), init=True)
    return bound


class BaseEstimator(sklearn.base.BaseEstimator):
    # overwrite set_params() so that setting of nested parameters is "skipped" if nested object is None
    # this method is copied and edited from sklearn.BaseEstimator

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                if nested_params[key] is None or not isinstance(nested_params[key], dict):
                    nested_params[key] = {}
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            if valid_params[key] is not None and isinstance(valid_params[key], sklearn.base.BaseEstimator):
                valid_params[key].set_params(**sub_params)
            else:
                # should we do something?
                raise ValueError(f'nested estimator {key} has nested parameters {sub_params} but it is not valid')
                pass

        return self


class BasePredictor(BaseEstimator):

    def predict_scores(self, X=None, what='proba'):
        if check_hasmethod(self, 'predict_proba'):
            y_scores = self.predict_proba(X)
        elif check_hasmethod(self, 'decision_function'):
            y_scores = self.decision_function(X)
        else:
            raise NotImplementedError('predict_scores')
        return y_scores


class PredictScoresMixin:
    def predict_scores(self, X=None):
        if check_hasmethod(estimator, 'predict_proba'):
            y_scores = self.predict_proba(X)
        elif check_hasmethod(self, 'decision_function'):
            y_scores = self.decision_function(X)
        else:
            raise NotImplementedError('predict_scores')
        return y_scores
