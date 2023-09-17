import logging
logger = logging.getLogger(__package__)

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

# for defining custom transformers/estimators
from sklearn.base import BaseEstimator, TransformerMixin

from ..validation.common import is_list_like, is_dict_like
from ..validation.common import check_is_fitted, has_fit_parameter
from ..validation.common import check_array
from ..validation import num_samples
from ..validation import check_include_exclude

from ..validation.common import is_numeric_dtype, is_string_dtype

from ..utils import to_dataframe

# from ..metrics import histogram

from collections import Counter, OrderedDict

def histogram(x, domain=None):
    hist = Counter(x)
    if domain is not None:
        for v in domain:
            if v not in hist:
                hist[v] = 0
    return hist
    # computer a Counter for x and initialize counts 0 for all elements in domain
    pass

#########################################################################################################

class IdentityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super(IdentityTransformer, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    # modeled after https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/

    def __init__(self, include=None, exclude=None):
        # list of columns to include and/or exclude
        check_include_exclude(include=include, exclude=exclude, disjoint=False)
        self.include = include
        self.exclude = exclude
        pass

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('arg X should be a DataFrame')
        domain = set(list(X))
        include, exclude = check_include_exclude(include=self.include, exclude=self.exclude, domain=domain,
                                                 diff=False, project=True)
        self.include_, self.exclude_ = include, exclude
        return self

    def transform(self, X):
        check_is_fitted(self, ['include_', 'exclude_'])
        X = to_dataframe(X)
        if X is None:
            raise ValueError('argument should be array-like')
        columns = self.include_.difference(self.exclude_)
        try:
            return X[list(columns)]
        except KeyError:
            domain = set(list(X))
            raise KeyError(f"the argument is missing the columns: {list(columns - domain)}")
        pass

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)


class TypeSelector(BaseEstimator, TransformerMixin):
    # return the projection of a dataframe on columns of certain dtype
    # typical dtypes of interest are np.float_, np.int_, np.number, np.bool_, 'category', 'datetime', object
    # modeled after https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/

    def __init__(self, include=None, exclude=None):
        # see pandas.DataFrame.select_dtypes for valid dtypes
        # typical dtypes of interest are np.float_, np.int_, np.bool_, 'category', 'datetime', object
        check_include_exclude(include=include, exclude=exclude)
        self.include = include
        self.exclude = exclude
        pass

    def fit(self, X, y=None):
        include, exclude = check_include_exclude(include=self.include, exclude=self.exclude, domain=None,
                                                 diff=True, project=False)
        self.include_, self.exclude_ = include, exclude
        return self
        self.include_, self.exclude_ = set(), set()
        if self.include is not None:
            self.include_ = set(self.include) if is_list_like(self.include) else set((self.include,))
        if self.exclude is not None:
            self.exclude_ = set(self.exclude) if is_list_like(self.exclude) else set((self.exclude,))
        if bool(self.include_) and bool(self.exclude_):
                self.include_ = self.include_.difference(self.exclude_)
        if not(bool(self.include_) or bool(self.exclude_)):
            raise ValueError(f'at least one of include or exclude should be non-empty -- '
                             f'include={self.include_} and exclude={self.exclude_}')
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['include_', 'exclude_'])
        if not isinstance(X, pd.DataFrame):
            raise ValueError('argument X should be a DataFrame')
        Z = X.select_dtypes(include=self.include_, exclude=self.exclude_)
        return Z

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)


class ValuesDiscretizer(BaseEstimator, TransformerMixin):

    strategies = {'ceil': np.ceil,
                  'floor': np.floor,
                  'round': np.rint,
                  'ident': lambda X: X}

    def __init__(self, strategy='ceil'):
        self.strategy = strategy
        pass

    def fit(self, X, y=None):
        cls = self.__class__
        if self.strategy not in cls.strategies:
            raise ValueError(f"invalid strategy -- {self.strategy} is not in {cls.strategies}")
        self.strategy_ = cls.strategies.get(self.strategy)
        return self

    def transform(self, X):
        check_is_fitted(self, ['strategy_'])
        if X is None:
            raise ValueError('argument should not be None')
        X = check_array(X, dtype='numeric', ensure_2d=False, accept_sparse='csr')
        f = self.strategy_
        Z = f(X)
        Z = Z.astype(int)
        return Z

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)


class ValuesReducer(BaseEstimator, TransformerMixin):

    replace_with_options = ('group_index', 'group_value')
    strategies = ('simple',)

    def __init__(self, strategy='simple', min_freq=10, replace_with='group_index', sep='|', prefix='__', suffix='__'):
        cls = self.__class__
        self.strategy = strategy
        self.min_freq = min_freq
        self.sep = sep
        self.prefix = prefix
        self.suffix = suffix
        self.replace_with = replace_with
        if self.replace_with not in cls.replace_with_options:
            raise ValueError(f"invalid replace_with argument -- {self.replace_with} not in {cls.replace_with_options}")
        if self.strategy not in cls.strategies:
            raise ValueError(f"invalid strategy argument -- {self.strategy} not in {cls.strategies}")
        if self.min_freq < 0:
            raise ValueError(f'min_freq={self.min_freq} should be non-negative')
        pass

    def _find_groupings(self, x, domain=None):
        # group all low-freq values together into one group
        groups = []
        if self.min_freq_ < 1:
            return groups
        h = pd.Series(histogram(x, domain=domain))
        h = h.sort_values(ascending=True)
        group = None
        cum_freq = lambda G: 0 if G is None else sum(h[G].values)
        for k in h.index:
            if h[k] < self.min_freq_:
                if group is None:
                    group = []
                group.append(k)
            elif h[k] >= self.min_freq_:
                if group is not None and cum_freq(group) < self.min_freq_:
                    group.append(k)
                break
            else:
                continue
        if group is not None and len(group):
            groups.append(sorted(group))
        if False:
            logger.info('groups -- initial histogram\n', h)
            for L in groups:
                c = Counter({k: h[k] for k in L})
                logger.info(f'these classes will be fused: [{L}]')
                logger.info(c)
        return groups

    def _group_labels(self, groups):
        # given a list of groups of values return a dict that maps a value to its group representative
        # eg if input is [['a', 'b'], ['c',]] the output is dict('a': '__ab__', 'b': '__ab__', 'c':'c')
        #
        labels = {}
        for group in groups:
            if not(is_list_like(group) and len(group)):
                continue
            if len(group) == 1:
                i = group[0]
                labels[i] = i
            value = f"{str(self.prefix)}{str(self.sep).join(map(str, sorted(group)))}{str(self.suffix)}"
            for i in group:
                labels[i] = value
        labels = pd.Series(labels)
        return labels

    def fit(self, X, y=None, groupings=None, domains=None):
        # Compute regroup_map as a 2d DataFrame with two columns (value and id)
        # and rows indexed by the values in each column of X
        #
        #  groups and domains can be a dicts keyed on columns (col) of X
        #  each groups[col] and domain[col] are: (1) list of column values to be grouped together, and
        #  (2) is a Series of values of col
        #
        # self.min_freq_ is the actual minimum integer frequency count required
        if isinstance(self.min_freq, np.float_) and self.min_freq <= 1.:
            self.min_freq_ = int(np.ceil(self.min_freq * num_samples(X)))
        else:
            self.min_freq_ = int(self.min_freq)
        if not(domains is None or is_dict_like(domains)):
            raise ValueError(f'fit parameter domains should be dict-like')
        if not(groupings is None or is_dict_like(groupings)):
            raise ValueError(f'fit parameter groups should be dict-like')
        check_array(X, allow_nd=True, ensure_2d=True, accept_sparse='csr', dtype=None)

        X = to_dataframe(X)
        fusion = {}
        for col in X.columns:
            x = X[col]
            domain = domains.get(col, None) if is_dict_like(domains) else None
            groups = groupings.get(col, None) if is_dict_like(groupings) else None
            if domain is None:
                domain = x.drop_duplicates().sort_values().reset_index(drop=True)
            if groups is None:
                groups = self._find_groupings(x, domain=domain)
            elif not is_list_like(groups):
                logger.warning(f'ignoring column {col} since its grouping is not list-like -- {groups}')
                continue
            group_labels = self._group_labels(groups)
            regroup_values = pd.Series({v: group_labels.get(v, v) for v in domain})

            enc = LabelEncoder()
            enc_data = regroup_values
            if not (is_string_dtype(regroup_values) or is_numeric_dtype(regroup_values)):
                # regroup values should be either numeric or string in order to fit() a LabelEncoder() with them
                enc_data = pd.Series(data=list(map(str, regroup_values)), index=regroup_values.index)
            z = enc.fit_transform(enc_data)
            regroup_indices = pd.Series(data=z, index=regroup_values.index)
            regroup_map = pd.concat((regroup_values, regroup_indices), axis=1).rename(columns={0: 'group_value', 1: 'group_index'})
            # the map is indexed by the col's domain values and has two columns: group label (value) and a group index (id)
            fusion[col] = (groups, regroup_map)

        self.fusion_ = fusion
        return self

    def transform(self, X):
        # transform the values of each column of X based on the regroup maps (DataFrame) computed by fit()
        # use the column self.replace_with of the regroup map for each column of X
        #
        check_is_fitted(self, ['min_freq_', 'fusion_'])
        check_array(X, allow_nd=True, ensure_2d=True, accept_sparse='csr', dtype=None)
        X = to_dataframe(X)
        result = []
        for col in X.columns:
            if self.fusion_.get(col, None) is None:
                logger.debug(f'no fusion information is available for column {col}')
                continue
            x = X[col].copy()
            grouping, regroup_map = self.fusion_[col]
            if self.replace_with == 'group_value':
                x.replace(regroup_map.group_value, inplace=True)
            elif self.replace_with == 'group_index':
                x.replace(regroup_map.group_index, inplace=True)
            result.append(x)
        Z = pd.concat(result, axis=1)
        return Z

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y, **fit_params).transform(X)



from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split


class SupervisedRandomProjection(GaussianRandomProjection):

    def __init__(self, n_components='auto', eps=0.1, random_state=None):
        super(SupervisedRandomProjection, self).__init__(n_components=n_components, eps=eps, random_state=random_state)

    def fit(self, X, y=None):
        super().fit(X, y)
        n = self.n_components
        X1, X2 = train_test_split(X, test_size=0.50, random_state=self.random_state, stratify=y)
        n = min(n, X1.shape[0], X2.shape[0])
        Z1 = X.sample(n=n, replace=False, random_state=self.random_state)
        Z2 = X2.sample(n=n, replace=False, random_state=self.random_state)
        Z = Z1.values - Z2.values
        self.components_ = Z
        return self

    def transform(self, X):
        check_is_fitted(self, ['components_'])
        U = sklearn.metrics.pairwise.sigmoid_kernel(X, self.components_)
        return U


from sklearn.impute import SimpleImputer

# class CustomSimpleImputer(BaseEstimator, TransformerMixin):

class CustomSimpleImputer(SimpleImputer):

    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None, verbose=0, copy=True, add_indicator=False):
        super(CustomSimpleImputer, self).__init__(missing_values=missing_values, strategy=strategy, fill_value=fill_value,
                                                  verbose=verbose, copy=copy, add_indicator=add_indicator)

    def fit(self, X, y=None):
       if self.strategy in ('mean', 'median', 'constant', 'most_frequent'):
           super(CustomSimpleImputer, self).fit(X, y=y)
           return self
       if self.strategy == 'most_frequent':
           self.statistics_ = X.mode().iloc[0]
       else:
           raise ValueError(f'{self.strategy} is not supported')
       return self

    def transform(self, X):
       return super(CustomSimpleImputer, self).transform(X)
       return X.fillna(self.fill_value_, inplace=not self.copy)

