import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from scipy.sparse.csr import csr_matrix
import sklearn

from sklearn.utils import Bunch
from my_datasets import from_sklean_to_dataset_bunch, fetch_dataset_bunch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from multiset import Multiset
from collections import Counter, OrderedDict

from sklearn import preprocessing
from sklearn import impute
from sklearn import compose, pipeline

###############################################################################

# for defining custom transformers/estimators
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import make_pipeline, make_union

# from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.compose import make_column_transformer, ColumnTransformer

# from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
# from sklearn.preprocessing import KBinsDiscretizer, Binarizer, QuantileTransformer
# from sklearn.preprocessing import PolynomialFeatures, Normalizer, PowerTransformer
# from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder, OrdinalEncoder
# from sklearn.preprocessing import FunctionTransformer

#########################################################################################################

def compare_lists(a, b, prefix=False):
    # compare two lists return True if a is a prefix of b (prefix=True) or equal (prefix=False)

    # find #agreements on first k elements of two lists a,b
    f = lambda a, b, k: sum([1 for i, j in zip(a[:k], b[:k]) if i == j])

    if prefix:
        m = min(len(a), len(b))
        if isinstance(prefix, int) and prefix >= 0 and prefix < k:
            m = prefix
        return f(a, b, m) == m
    m = len(a)
    return len(b) == m and f(a, b, m) == m

def dict_drop_None(X):
    return OrderedDict({k: v for k, v in X.items() if v is not None})

class TransformerInterfaceMixin(TransformerMixin):
    # override fit, transform, and fit_transform methods to capture  I/O interface parameters

    _prm_fit_input_propagation = 1
    """
        The input for fit() propagates some defaults to the output using the options below
        The columns kwarg of fit() override the default output feature names (left to right)
    """
    _valids_prm_fit_input_propagation = {
        0: "none",
        1: "default names of output features are those of the fitted input"
    }

    _prm_tranform_input_validation = (2, 1)
    """
       Transform input validation a tuple (numbers, names) using the options below
       (name levels 1+ pressume a pd.DataFrame input)
    """
    _valids_prm_tranform_input_validation = [
        {0: "none - input can have any number of features",
         1: "number of input features is <= to the number of fitted features",
         2: "number of input features is equal to number of fitted features"},
        {0: "unconstrained input feature names",
         1: "input feature names are among the names of the fitted features",
         2: "named input features in the same order as the named fitted features"}
    ]

    _prm_transform_output_format = (0, 4)
    """"
        Transform output format is a tuple (type, naming) that determines the type and feature names of the output using the 
        valid options below.
        In case the output is of pd.DataFrame type its columns will be named based the naming option. 
        If the output is ndarray the naming options are meaningless (ignored - effectively 0)
    """
    _valids_prm_transform_output_format = [
        {0: "the output is the same type (dtataframe or ndarray) as the input [if columns kwarg is given, then it's dataframe]",
         1: "output will be casted to pd.DataFrame type", },
        {0: "unnamed",
         1: "feature names of pattern using default output feature name format string and the column number",
         2: "feature names overwritten from the default output feature names set during the last fit (left to right)",
         3: "feature names overwritten from the transform input feature names (left to right if input is pd.DataFrame)",
         4: "feature names are overwritten from the columns or kwarg (left to right)", },
    ]

    # format for generating default column names using the column's index
    _prm_output_features_default_names_fmt = "attr_{}"

    # state information for the transformer's interface - updated by fit() and transform()
    _interface_input_mru_dataframe = False
    _interface_output_mru_dataframe = False
    _interface_input_features_map = OrderedDict()
    _interface_output_features_map = OrderedDict()
    _interface_output_features_default_map = OrderedDict()

    def _behaviour(self):
        print("\t_prm_fit_input_propagation ", self._prm_fit_input_propagation, " :: ",
              self._valids_prm_fit_input_propagation[self._prm_fit_input_propagation])
        print("\t_prm_tranform_input_validation",
              self._prm_tranform_input_validation, " ::\n\t\t",
              self._valids_prm_tranform_input_validation[0][self._prm_tranform_input_validation[0]], "\n\t\t",
              self._valids_prm_tranform_input_validation[1][self._prm_tranform_input_validation[1]])
        print("\t_prm_transform_output_format",
              self._prm_transform_output_format, " ::\n\t\t",
              self._valids_prm_transform_output_format[0][self._prm_transform_output_format[0]], "\n\t\t",
              self._valids_prm_transform_output_format[1][self._prm_transform_output_format[1]])
        print("\t_prm_output_features_default_names_fmt ::", self._prm_output_features_default_names_fmt)

    def log(self):
        # print('calling from ', self.__class__)
        pass

    def _validate_config_options(self):
        assert(self._prm_fit_input_propagation in type(self)._valids_prm_fit_input_propagation.keys())

        assert (len(self._prm_tranform_input_validation) == 2 and
                self._prm_tranform_input_validation[0] in type(self)._valids_prm_tranform_input_validation[0].keys() and
                self._prm_tranform_input_validation[1] in type(self)._valids_prm_tranform_input_validation[1].keys())

        assert(len(self._prm_transform_output_format) == 2 and
               self._prm_transform_output_format[0] in type(self)._valids_prm_transform_output_format[0].keys() and
               self._prm_transform_output_format[1] in type(self)._valids_prm_transform_output_format[1].keys())

        assert(isinstance(self._prm_output_features_default_names_fmt, str))
        assert(isinstance(self._interface_input_features_map, dict))
        assert(isinstance(self._interface_output_features_map, dict))
        assert(isinstance(self._interface_output_features_default_map, dict))

    def _reset_interface(self):
        self._interface_input_mru_dataframe = False
        self._interface_output_mru_dataframe = False
        self._interface_input_features_map = OrderedDict()
        self._interface_output_features_map = OrderedDict()
        self._interface_output_features_default_map = OrderedDict()

    def _update_interface_on_fit(self, X, columns=(), propagate=None):
        names = list(X) if isinstance(X, pd.DataFrame) else list(range(X.shape[-1]))
        self._interface_input_features_map = OrderedDict({k: v for (k, v) in enumerate(names)})
        self._interface_input_mru_dataframe = isinstance(X, pd.DataFrame)

        propagate = self._prm_fit_input_propagation if propagate is None else propagate
        if propagate:
            self._interface_output_features_default_map = self._interface_input_features_map.copy()

        if columns:
            if isinstance(columns, list):
                columns = OrderedDict({k: v for (k, v) in enumerate(columns)})
            self._interface_output_features_default_map.update(dict_drop_None(columns))


    def _validate_transform_input(self, X):
        # validate the input and return a list of columns names to project/align the input on
        if X is None:
            raise ValueError("invalid transform input :: None")

        opt_number, opt_names = self._prm_tranform_input_validation[0], self._prm_tranform_input_validation[1]

        m, k = X.shape[-1], len(self._interface_input_features_map)
        if opt_number and m > k:
            raise ValueError(f"transform input has more columns than fitted input :: {m} vs {k}")
        if opt_number == 2 and m < k:
            raise ValueError(f"transform input has less columns than fitted input :: {m} vs {k}")

        if not self._interface_input_mru_dataframe:
            return []

        mapper = OrderedDict()

        if opt_names and isinstance(X, pd.DataFrame):
            tnames, fnames = list(X), list(self._interface_input_features_map.values())
            dnames = set(tnames) - set(fnames)

            if dnames:
                raise ValueError(f"transform input has columns not among those of the fitted input :: {dnames}")
            if opt_names == 2 and not compare_lists(tnames, fnames, prefix=True):
                raise ValueError(f"transform input columns do not align with the fitted input :: {tnames} vs {fnames}")

            # need to find a projection of X to the fitted columns
            for i, v in self._interface_input_features_map.items():
                if v in tnames:
                    mapper[i] = v
            return list(mapper.values())

        return []


    def _cast_transform_output(self, X, Z, columns=()):
        m = Z.shape[-1]
        if not (self._prm_transform_output_format[0] or isinstance(X, pd.DataFrame) or len(columns)):
            self._interface_output_mru_dataframe = False
            self._interface_output_features_map = OrderedDict({i: i for i in range(m)})
            return Z

        # need to cast Z to a dataframe -- need to construct names for its columns
        opt_naming = self._prm_transform_output_format[1]
        # if len(columns):
        #     opt_naming = max(opt_naming, 4)

        schema = OrderedDict({i: i for i in range(m)})

        if opt_naming >= 1:
            fmt = self._prm_output_features_default_names_fmt
            if fmt is not None and isinstance(fmt, str):
                schema.update({i: fmt.format(i + 1) for i in range(m)})

        if opt_naming >= 2 and self._interface_input_mru_dataframe:
            schema.update(dict_drop_None(self._interface_output_features_default_map))

        if opt_naming >= 3 and isinstance(X, pd.DataFrame):
            xnames = OrderedDict({i: v for i, v in enumerate(list(X))})
            schema.update(xnames)

        if opt_naming >= 4 or len(columns):
            if not isinstance(columns, dict):
                columns = OrderedDict({i: v for i, v in enumerate(columns)})
            schema.update(columns)

        schema = OrderedDict({i: schema[i] for i in range(m)})

        if isinstance(Z, csr_matrix):
            if not self.sparse_output_:
                raise ValueError("dense output without .sparse_output_ set to True")
            Z = Z.todense()
        if opt_naming == 0:
            Z = pd.DataFrame(Z)
        elif not isinstance(Z, pd.DataFrame):
            Z = pd.DataFrame(Z, columns=schema.values())
        else:
            # need to rename the columns of Z which is already a dataframe
            znames = OrderedDict({i: v for i, v in enumerate(list(Z))})
            column_mapper = {old_name: new_name for old_name, new_name in
                             zip(znames.values(), schema.values())}
            Z = Z.rename(axis='columns', columns=column_mapper)

        self._interface_output_mru_dataframe = True
        self._interface_output_features_map.update(schema)
        return Z

    def fit(self, *args, **kwargs):
        self.log()
        self._validate_config_options()

        columns = kwargs.pop("columns", [])
        propagate = kwargs.pop("passon", self._prm_fit_input_propagation)

        assert(len(args) and args[0] is not None and isinstance(args[0], (np.ndarray, pd.DataFrame, pd.Series)))
        assert(isinstance(columns, (list, tuple, dict)) and propagate in (0, 1, 2))

        X = args[0]
        self._reset_interface()
        self._update_interface_on_fit(X, columns=columns, propagate=propagate)

        super().fit(*args, **kwargs)
        return self

    def transform(self, *args, **kwargs):
        self.log()
        self._validate_config_options()

        assert(len(args) and args[0] is not None and isinstance(args[0], (np.ndarray, pd.DataFrame, pd.Series)))
        if len(args) > 1:
            pass
        X = args[0]
        columns = kwargs.pop("columns", [])
        assert(isinstance(columns, (list, tuple, dict)))

        # validate and align input?
        cols = self._validate_transform_input(X)
        if cols:
            # align the input
            X = X[cols]

        #Z = super().transform(X, *args[1:], **kwargs)
        Z = super().transform(X, **kwargs)

        Z = self._cast_transform_output(X, Z, columns=columns)
        return Z

    def fit_transform(self, *args, **kwargs):
        self.log()
        self._validate_config_options()

        self.fit(*args, **kwargs)

        kwargs.pop("passon", None)

        Z = self.transform(*args, **kwargs)
        return Z

    def get_feature_names(self, output=True):
        v = []
        if output:
            mru, imap = self._interface_output_mru_dataframe, self._interface_output_features_map
        else:
            mru, imap = self._interface_input_mru_dataframe, self._interface_input_features_map
        if mru:
            v = list(imap.values())
        else:
            v = list(imap.keys())
        return v


#########################################################################################################

class MyScaler(TransformerInterfaceMixin, preprocessing.StandardScaler):
    # example scaler with custom behaviour
    _prm_fit_input_propagation = 1
    _prm_tranform_input_validation = (2, 1)
    _prm_transform_output_format = (0, 4)
    _prm_output_features_default_names_fmt = "x_{}"
    pass


# extend preprocessing and impute classes

SRC_TRANSFORMERS = [preprocessing.StandardScaler, preprocessing.MaxAbsScaler, preprocessing.MinMaxScaler,
                    preprocessing.RobustScaler,
                    preprocessing.KBinsDiscretizer, preprocessing.Binarizer,
                    preprocessing.PolynomialFeatures, preprocessing.Normalizer, preprocessing.QuantileTransformer,
                    preprocessing.PowerTransformer,
                    preprocessing.LabelBinarizer, preprocessing.LabelEncoder, preprocessing.OneHotEncoder, preprocessing.OrdinalEncoder,
                    preprocessing.FunctionTransformer,
                    impute.SimpleImputer, impute.MissingIndicator,
                    ]

# create new classes
for t in SRC_TRANSFORMERS:
    cname = t.__name__
    globals()[cname] = type(cname, (TransformerInterfaceMixin, t), dict())


#########################################################################################################

class ColumnTransformer(compose.ColumnTransformer):
    _interface_input_mru_dataframe = False
    _interface_input_features_map = OrderedDict()

    def _update_interface_on_fit(self, X):
        names = list(X) if isinstance(X, pd.DataFrame) else list(range(X.shape[-1]))
        self._interface_input_features_map = OrderedDict({k: v for (k, v) in enumerate(names)})
        self._interface_input_mru_dataframe = isinstance(X, pd.DataFrame)

    def _cast_output(self, Z):
        cols = self.get_feature_names()
        if cols and len(cols) == Z.shape[-1]:
            Z = pd.DataFrame(Z, columns=cols)
        else:
            raise ValueError(f"shape of output mismatch to column names {Z.shape} vs {len(cols)}")
        return Z

    def fit(self, *args, **kwargs):
        self._update_interface_on_fit(args[0])
        return super().fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        Z = super().transform(*args, **kwargs)
        Z = self._cast_output(Z)
        return Z

    def fit_transform(self, *args, **kwargs):
        self._update_interface_on_fit(args[0])
        Z = super().fit_transform(*args, **kwargs)
        Z = self._cast_output(Z)
        return Z

    def get_feature_names(self, output=True):
        if output:
            v = self._get_engine_feature_names()
        else:
            if self._interface_input_mru_dataframe:
                v = self._interface_input_features_map.values()
            else:
                v = self._interface_input_features_map.keys()
        return list(v)

    def _get_engine_feature_names(self):
        # ColumnTransformer does implement passtrhough option for get_feature_names - implement here
        feature_names = []
        for name, trans, cols, _ in self._iter(fitted=True):
            if trans in ('_drop_at_depth', 'passthrough'):
                continue
            elif not hasattr(trans, 'get_feature_names'):
                raise AttributeError("Transformer %s (type %s) does not provide get_feature_names."
                                     % (str(name), type(trans).__name__))
            else:
                feature_names.extend([name + "__" + f for f in trans.get_feature_names()])
        if self.remainder == '_drop_at_depth':
            return feature_names

        if not self._interface_input_mru_dataframe:
            raise ValueError("unnamed columns")
        if self._remainder[2]:
            remainder_input_cols = [self._interface_input_features_map[i] for i in self._remainder[2]]
        else:
            remainder_input_cols = []
        if self._remainder[1] == 'passthrough':
            feature_names.extend([f for f in remainder_input_cols])
        else:
            # t should be  a transformer
            t = self._remainder[1]
            if not ((hasattr(t, "fit") or hasattr(t, "fit_transform")) and hasattr(t, "transform")):
                raise ValueError("invalid remainder option")
            if not hasattr(t, "get_feature_names"):
                raise ValueError(f"renainder {type(t).__name__} does not support get_feature_names()")
            feature_names.extend(t.get_feature_names())

        return feature_names
    pass

class Pipeline(pipeline.Pipeline):
    # extend Pipeline to return the feature names of its input and output
    def get_feature_names(self, output=True):
        if output:
            v = self.steps[-1][1].get_feature_names(output)
            v = [f"{self.steps[-1][0]}__{a}" for a in v]
        else:
            v = self.steps[0][1].get_feature_names(output)
        return list(v)
    pass


class FeatureUnion(pipeline.FeatureUnion):

    def get_feature_names(self, output=True):
        if output:
            feature_names = []
            for name, t in self.transformer_list:
                v = [f"{name}_{attr}" for attr in t.get_feature_names(output)]
                feature_names.extend(v)
        else:
            t = self.transformer_list[0][1]
            feature_names = t.get_feature_names(0)
        return list(feature_names)

    def _cast_output(self, Z):
        cols = self.get_feature_names()
        if cols and len(cols) == Z.shape[-1]:
            Z = pd.DataFrame(Z, columns=cols)
        else:
            raise ValueError(f"shape of output mismatch to column names {Z.shape} vs {len(cols)}")
        return Z

    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        Z = super().transform(*args, *kwargs)
        Z = self._cast_output(Z)
        return Z

    def fit_transform(self, *args, **kwargs):
        Z = super().fit_transform(*args, *kwargs)
        Z = self._cast_output(Z)
        return Z
    pass

#########################################################################################################


class MyScaler(TransformerInterfaceMixin, preprocessing.StandardScaler):
    # example scaler with custom behaviour
    _prm_fit_input_propagation = 1
    _prm_tranform_input_validation = (2, 1)
    _prm_transform_output_format = (0, 4)
    _prm_output_features_default_names_fmt = "x_{}"
    pass


class ColumnSelector(BaseEstimator, TransformerInterfaceMixin):
    # return the projection of a dataframe to include (or exclude) select columns
    _prm_fit_input_propagation = 1
    _prm_tranform_input_validation = (2, 1)
    _prm_transform_output_format = (0, 4)
    _prm_output_features_default_names_fmt = "x_{}"

    def __init__(self, columns, include=True):
        # list of columns to include (if True) or exclude (if False)
        self.columns = columns
        self.include = include

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        columns, domain = set(self.columns), set(list(X))
        columns = columns.intersection(domain) if self.include else domain.difference(columns)
        try:
            return X[list(columns)]
        except KeyError:
            raise KeyError(f"The DataFrame does not include the columns: {list(columns - domain)}")

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)


class TypeSelector(BaseEstimator, TransformerInterfaceMixin):
    # return the projection of a dataframe on columns of certain dtype

    _prm_fit_input_propagation = 1
    _prm_tranform_input_validation = (2, 1)
    _prm_transform_output_format = (0, 4)
    _prm_output_features_default_names_fmt = "x_{}"

    def __init__(self, include=None, exclude=None):
        # see pandas.DataFrame.select_dtypes for valid dtypes
        # typical dtypes of interest are np.float_, np.int_, np.bool_, 'category', 'datetime', object
        self.include = include if isinstance(include, list) else [include] if include is not None else None
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude is not None else None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=self.include, exclude=self.exclude)

# class IdentityTransformer(BaseEstimator, TransformerMixin):
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return X
#
#     def fit_transform(self, X, y=None):
#         return X

# class IdentityTransformerA(TransformerInterfaceMixin, IdentityTransformer):
#     # return the input
#     _prm_fit_input_propagation = 1
#     _prm_tranform_input_validation = (2, 1)
#     _prm_transform_output_format = (0, 4)
#     _prm_output_features_default_names_fmt = "x_{}"
#
#     pass

def get_dataframe_feature_types(df):
    # get the names of features of dataframe by type
    fmap = dict()
    for key, dtype in zip(['float', 'categorical', 'int', 'bool', 'numeric'],
                          [np.float_, 'category', np.int_, np.bool_, np.number]):
        names = list(TypeSelector(dtype).fit_transform(df))
        fmap[key] = names
    return Bunch(**fmap)

#########################################################################################################


def test_transformers(rng=0):
    df = get_dataset()
    cols = list(df)

    show = lambda t: print(f"\n{type(t).__name__} interface post fit_transform() :: \n\t{t.get_feature_names(0)}\n\t -> {t.get_feature_names()}")

    df, cols = df[cols[:-1]], cols[:-1]
    #df = pd.DataFrame(df.values[:5,:], columns=cols)
    print(df.head(3))

    # obj = MyScaler()
    # print(obj)
    # obj = obj.fit(df)
    # Z = obj.transform(df[cols])
    # Z = obj.fit_transform(df, columns=list("AB"))

    cols = list(df)
    args = [[('a', StandardScaler(), cols[:2]), ('b', RobustScaler(), cols[:3])]]
    t = ColumnTransformer(*args, remainder='passthrough')

    Z = t.fit_transform(df)
    show(t)
    print(Z)

    pp = Pipeline(steps=[('stage0', StandardScaler()),
                          ('stage1', ColumnTransformer(*args, remainder='passthrough')),
                          ('stage2', QuantileTransformer())], memory=None)

    Z = pp.fit_transform(df)
    show(pp)
    print(Z)

    args2 = [[('a', StandardScaler()), ('b', RobustScaler())]]
    t = FeatureUnion(*args2)
    t = t.fit(df)
    Z = t.transform(df)
    show(t)
    print(Z)
    print('done')

def get_dataset():
    from my_datasets import from_sklean_to_dataset_bunch, fetch_dataset_bunch
    # fetch a dataset
    ds_bunch = from_sklean_to_dataset_bunch(sklearn.datasets.load_wine(), name='wine', archive=False)
    df = fetch_dataset_bunch(ds_bunch)
    cols = ['alcohol', 'magnesium', 'flavanoids', 'malic_acid', 'proline', 'ash', 'total_phenols', 'target']
    cols = ['alcohol', 'magnesium', 'ash', 'proline', 'target']
    df = df[cols]
    cols.remove('target')
    dfn = df[cols]
    return df.iloc[:8]

if __name__ == "__main__":
    rng = np.random.RandomState(2019)
    test_transformers(rng)
    input('press any key to exit')
