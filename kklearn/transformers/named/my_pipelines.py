import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.utils import Bunch
from my_datasets import from_sklean_to_dataset_bunch, fetch_dataset_bunch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from multiset import Multiset
from collections import Counter, OrderedDict

histogram = lambda x: Counter(x)

import itertools

from sklearn import preprocessing

from scipy import interpolate

from sklearn.utils.multiclass import type_of_target

from my_validate import dict_from_keys

###############################################################################

# for defining custom transformers/estimators
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import make_pipeline, make_union

from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.compose import make_column_transformer, ColumnTransformer

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import KBinsDiscretizer, Binarizer, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures, Normalizer, PowerTransformer
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer


# compose column transformers (different for different columns)
#   ColumnTransformer(transformers, remainder=?_drop_at_depth/passthrough?, transformer_weights=None)
#   where transformers is a list of (name, transformer, column(s)) tuples

# preprocessing scalers and encoders
#   for continuous features
#       StandardScaler(), MaxAbsScaler() MinMaxScaler(feature_range=(0, 1))
#       RobustScaler(quantile_range=(25.0, 75.0)), KBinsDiscretizer(n_bins=5, encode='ordinal')
#       PowerTransformer() Binarizer(threshold=0.0)
#       PolynomialFeatures(degree=2)
#       Normalizer()
#   for categorical features
#       CategoricalEncoder(encoding='onehot/ordinal, categories=auto/[], 'handle_unknown='ignore')
#       OneHotEncoder(handle_unknown='ignore') LabelBinarizer()
#       LabelEncoder() OrdinalEncoder()
#

# sklearn.impute missing values: FeatureUnion of
#   SimpleImputer(missing_values=nan, strategy=?mean/median/most_frequent/constant?, fill_value=)
#   MissingIndicator(missing_values=nan)

# projectors
#   sklearn.decomposition.PCA()

class ColumnSelector(BaseEstimator, TransformerMixin):
    # return the projection of a dataframe to include (or exclude) select columns
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


class TypeSelector(BaseEstimator, TransformerMixin):
    # return the projection of a dataframe on columns of certain dtype

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


class DataframeCaster(BaseEstimator, TransformerMixin):
    # transform/cast a 2d-array into a dataframe with named columns

    def __init__(self, dataframe=None, columns=None):
        super().__init__()
        self.columns = columns if columns is not None else list(df) if df is None else None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X
        if self.columns is None:
            return pd.DataFrame(X)
        if X.shape[1] == len(self.columns):
            return pd.DataFrame(X, columns=self.columns)
        else:
            raise ValueError(f"shape of X {X.shape} and {len(self.columns)} mismatch")


def get_dataframe_feature_types(df):
    # get the names of features of dataframe by type
    fmap = dict()
    for key, dtype in zip(['float', 'categorical', 'int', 'bool', 'numeric'],
                          [np.float_, 'category', np.int_, np.bool_, np.number]):
        names = list(TypeSelector(dtype).fit_transform(df))
        fmap[key] = names
    return Bunch(**fmap)


def preproces_dataset(X, y, labels, categories, feature_names):
    # labels_dict = {k: v for k, v in zip(labels, categories)}
    targets = pd.Series(categories, labels)
    features = {i: v for i, v in enumerate(feature_names)}
    K = 5
    for i, f in enumerate(feature_names):
        xcol = X[:, i]
        x_type = type_of_target(xcol)
        print(f"{f} is of type {x_type}")
        if x_type == 'continuous':
            scaler = preprocessing.StandardScaler()
            scaler = preprocessing.MaxAbsScaler()
            # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
            scaler = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal')
            # scaler = preprocessing.PowerTransformer()
            scaler.fit(xcol[:, np.newaxis])
            xnew = scaler.transform(xcol[:, np.newaxis])[:, 0]
            # print(f"\t continuous {f} before {xcol[:k]} and after {xnew[:k]}")
            x_new = xcol
        if x_type == 'multiclass':
            enc = preprocessing.LabelEncoder()
            # enc = preprocessing.Binarizer()
            enc.fit(xcol)
            xnew = enc.transform(xcol)
            print(f"\t mutliclass {f} before {xcol[:K]} and after {xnew[:K]}")  # labels={enc.classes_}")
            print(f"\t\t{enc}")
    print(f"target is of type {type_of_target(y)}")

    # utils.check_array(array[, accept_sparse, ?])
    # sklearn.utils.check_array()
    # sklearn.utils.check_consistent_length()
    # sklearn.utils.column_or_1d()
    # sklearn.utils.check_X_y()
    pass


###############################################################################
from my_datasets import from_sklean_to_dataset_bunch, fetch_dataset_bunch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def clean_df(df):
    features = ['alcalinity_of_ash', 'od280/od315_of_diluted_wines', 'hue', 'nonflavanoid_phenols',
                'total_phenols', 'magnesium', 'ash', 'flavanoids', 'color_intensity', 'malic_acid',
                'proanthocyanins', 'alcohol', 'proline']
    print(list(df))

    tp = TypeSelector(float)
    tp.fit([])
    q = tp.transform(df)
    for t, name in zip([np.float_, 'category', np.int_, np.bool_, np.number],
                       ['float', 'categorical', 'int', 'bool', 'numeric']):
        print(f'{name} features', list(TypeSelector(t).fit_transform(df)))

    columns = ['ash', 'flavanoids', 'target']
    dq = ColumnSelector(columns).fit_transform(df)
    print(dq.head(5))

    pp = Pipeline(steps=[('subset of columns', ColumnSelector(columns)),
                         ('type selector', TypeSelector(np.int_))], memory=None)

    dq = pp.fit_transform(df)
    pp.fit(df)
    dq = pp.transform(df)
    print(dq.head(5))

    impute_pipeline = make_pipeline(
        ColumnSelector(columns=features),
        FeatureUnion(transformer_list=[
            ("numeric_features", make_pipeline(
                TypeSelector(np.number),
                make_union(SimpleImputer(strategy="median"),
                           MissingIndicator(missing_values=0)),
                preprocessing.StandardScaler()
            ))
        ]))
    # ,
    # ("categorical_features", make_pipeline(
    #     TypeSelector("category"),
    #     preprocessing.Imputer(strategy="most_frequent"),
    #     preprocessing.OneHotEncoder()
    # )),
    # ("boolean_features", make_pipeline(
    #     TypeSelector("bool"),
    #     preprocessing.Imputer(strategy="most_frequent")
    # ))
    #     ])
    # )

    dq = impute_pipeline.fit_transform(df)
    print(dq.head(5))


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


class TransformerInterfaceMixin(TransformerMixin):
    # override fit, transform, and fit_transform methods to use interface

    _fit_input_propagation = 0
    """
        The input for fit() propagates some defaults to the output using
            0: none
            1: default names of output features are those of the input
            The columns kwarg of fit() override the default output feature names (left to right)
        """

    _tranform_input_validation = (0, 1)

    """
    Transform input validation a tuple (numbers, names)
    where numbers is
        0: none - input can have any number of features
        1: number of input features is <= to the number of fitted features
        2: number of input features is equal to number of fitted features
    where names is
        0: unconstrained input feature names 
        1: input feature names are among the names of the fitted features 
        2: named input features in the same order as the named fitted features
        levels 1+ pressume a pd.DataFrame input
    """

    _transform_output_format = (1, 4)

    """"
    Transform output format is a tuple (type, naming) that determines the type and feature names of the output
    where type is
        0: the output type is the same as the input (DataFrame or ndarray)
        1: output will be casted to pd.DataFrame type
    and where naming is    
        0: unnamed 
        1: feature names of pattern using default output feature name format string and the column number
        2: feature names overwritten from the default output feature names set during the last fit (left to right)
        3: feature names overwritten from the transform input feature names (left to right if input is pd.DataFrame)
        4: feature names are overwritten from the columns or kwarg (left to right)
    
    In case the output is of pd.DataFrame type its columns will be named based the naming option
    """

    # actual parameters of the interface from last call of fit() or transform()
    _interface_input_features_num = 0
    _interface_input_features_map = []
    _interface_output_features_num = 0
    _interface_output_features_map = []

    _interface_output_features_default_num = 0
    _interface_output_features_default_map = []
    _interface_output_features_default_names_fmt = "attr{}"

    def log(self):
        print('calling from ', self.__class__)

    def _reset_interface(self):
        self._interface_input_features_num = np.nan
        self._interface_input_features_map = []

        self._interface_output_features_num = np.nan
        self._interface_output_features_map = []

        self._interface_output_features_default_num = np.nan
        self._interface_output_features_default_map = []


    def _update_interface_on_fit(self, X, columns=(), propagate=None):
        assert(propagate in (None, 0, 1, 2), "invalid value for propagate kwarg")
        assert(isinstance(columns, (list, dict, tuple, np.ndarray)), "invalid type for columns kwarg")

        self._interface_input_features_num = X.shape[-1]
        if isinstance(X, pd.DataFrame):
            self._interface_input_features_map = OrderedDict({k: v for (k, v) in enumerate(list(X))})

        propagate = self._fit_input_propagation if propagate is None else propagate
        if propagate:
            self._interface_output_features_default_map = self._interface_input_features_map

        if columns:
            if isinstance(columns, list):
                columns = OrderedDict({k: v for (k, v) in enumerate(columns)})
            if self._interface_output_features_default_map:
                self._interface_output_features_default_map.update(columns)
            else:
                self._interface_output_features_default_map = columns
            self._interface_output_features_default_num = len(self._interface_output_features_default_map)


    def _validate_transform_input(self, X):
        if X is None:
            raise ValueError("invalid transform input :: None")

        opt_number, opt_names = self._tranform_input_validation[0], self._tranform_input_validation[1]
        if opt_number not in (0, 1, 2) or opt_names not in (0, 1, 2):
            raise ValueError(f"invalid transform input validation option :: {self._tranform_input_validation}")

        m, k = X.shape[-1], len(self._interface_input_features_map)
        if opt_number and  m > k:
            raise ValueError(f"transform input has more columns than fitted input :: {m} vs {k}")
        if opt_number == 2 and m < k:
            raise ValueError(f"transform input has less columns than fitted input :: {m} vs {k}")

        mapper = OrderedDict()

        if opt_names and isinstance(X, pd.DataFrame):
            tnames, fnames = list(X), self._interface_input_features_map.values()
            dnames = set(tnames) - set(fnames)

            if dnames:
                raise ValueError(f"transform input has columns not among those of the fitted input :: {dnames}")
            if opt_names == 2 and not compare_lists(tnames, fnames, prefix=True):
                raise ValueError(f"transform input columns do not align with the fitted input :: {tnames} vs {fnames}")

            # need to find a projection of X to the fitted columns
            for i, v in self._interface_input_features_map.items():
                if v in tnames:
                    mapper[i] = v
            return mapper.values()

        return []


    def _validate_transform_output(self, Z):
        m, k = self._interface_output_no_features, len(self._interface_output_feature_names)
        if m and m != Z.shape[-1]:
            raise ValueError(f"actual vs expected columns :: {Z.shape[-1]} vs {m}")
        if k and k != Z.shape[-1]:
            raise ValueError("#output columns and actual output names mismatch :: {Z.shape[-1} vs {k}")
        return True


    def _cast_transform_output(self, X, Z, columns=()):
        if not(self._transform_output_format[0] in (0, 1) and self._transform_output_format[1] in (0, 1, 2, 3, 4)):
            raise ValueError("invalid transform output format option :: {self._transform_output_format}")

        if not(self._transform_output_format[0] and isinstance(X, pd.DataFrame)):
            return Z

        m = Z.shape[-1]

        # need to cast Z to a dataframe -- need to construct names for its columns
        opt_naming = self._transform_output_format[1]
        if len(columns):
            opt_naming = 4

        schema = None

        if opt_naming >= 1:
            fmt = self._interface_output_features_default_names_fmt
            schema = OrderedDict({i: fmt.format(i+1) for i in range(m)})

        if opt_naming >= 2:
            schema.update(self._interface_output_features_default_map)

        if opt_naming >= 3:
            xnames = OrderedDict({i: v for i, v in enumerate(list(X))})
            schema.update(xnames)

        if opt_naming >= 4 and len(columns):
            if not isinstance(columns, dict):
                columns = OrderedDict({i: v for i, v in enumerate(columns)})
            schema.update(columns)

        if schema is None:
            Z = pd.DataFrame(Z)
        elif not isinstance(Z, pd.DataFrame):
            Z = pd.DataFrame(Z, columns=schema.values())
        else:
            # need to rename the columns of Z which is already a dataframe
            column_mapper = {old_name : new_name for old_name, new_name in zip(xnames.values(), schema.values())}
            Q = Z.rename(axis='columns', columns=column_mapper)
            Z = Q

        return Z


    def fit(self, *args, **kwargs):
        self.log()

        columns = kwargs.pop("columns", [])
        propagate = kwargs.pop("passon", self._fit_input_propagation)

        assert(propagate in (0,1) and isinstance(columns, (list, tuple, dict)), "invalid kwargs")

        X = args[0]
        self._reset_interface()
        self._update_interface(X, columns=columns, propagate=propagate)

        super().fit(*args, **kwargs)

        # find the expected number of output features for this fit
        Z = super().transform(X[:1, :])
        self._validate_transform_output(Z)

        # if set, transforming inputs with different size will raise ValueErrors
        self._interface_output_features_num = Z.shape[-1]

        return self

    def transform(self, *args, **kwargs):
        self.log()

        assert(len(args), "transform needs at least one argument")

        X = args[0]
        assert (isinstance(X, (np.ndarray, pd.DataFrame)), "invalid X")

        columns = kwargs.pop("columns", [])
        assert(isinstance(columns, (list, tuple, dict)), f"invalid columns kwarg {columns}")

        # validate and align input?
        cols = self._validate_transform_input(X)
        if cols:
            # align the input
            X = X[cols]

        Z = super().transform(X, *args[1:], **kwargs)

        Z = self._cast_transform_output(X, Z, columns=columns)
        return Z

    def fit_transform(self, *args, **kwargs):
        self.log()

        self.fit(*args, **kwargs)

        kwargs.pop("columns", None)
        kwargs.pop("passon", None)

        Z = self.transform(*args, **kwargs)
        return Z

    def get_input_feature_names(self):
        v = self._interface_input_features_map.values()
        return v

    def get_feature_names(self):
        return self._interface_output_features_map()


class MyScaler(TransformerInterfaceMixin, preprocessing.StandardScaler):
    #     @classmethod
    # def _matchnames(cls):
    #     return True
    _match_fit_io_feature_names = True

    def log(self):
        pass

    pass


#########################################################################################################


# do a class decorator to retain column names when transforming a dataframe
# from functools import wraps
import types


def dataframe_transformer(Cls):
    class NewCls(Cls):
        # def __init__(self,*args, **kwargs):
        #     #self.oInstance = Cls(*args,**kwargs)
        #     super().__init__(*args, **kwargs)

        def get_input_feature_names(self):
            return None

        def set_input_feature_names(self, X, columns=None):
            if X is not None:
                if columns is None and isinstance(X, pd.DataFrame):
                    columns = list(X)
            # install a dynamic method upon ourselves
            self.get_input_feature_names = types.MethodType(lambda x: columns, self)

        def get_feature_names(self):
            return None

        def set_feature_names(self, columns=None):
            if self.get_input_feature_names() is not None and columns is None:
                columns = self.get_input_feature_names()
            # install a dynamic method upon ourselves
            self.get_feature_names = types.MethodType(lambda x: columns, self)

        def cast_to_dataframe(self, Z):
            columns = self.get_feature_names()
            if columns is not None and Z.shape[1] == len(columns):
                Z = pd.DataFrame(Z, columns=columns)
            return Z

        def check_compatible_feature_names(self, Z):
            return (True, np.nan)
            columns = self.get_feature_names()
            if columns is not None and (Z is None or Z.shape[1] != len(columns)):
                return (False, Z.shape[1])
            return (True, Z.shape[1])

        def fit(self, X, *args, columns=None, **kwargs):
            super().fit(X, *args, **kwargs)
            self.set_input_feature_names(X)
            self.set_feature_names(columns=columns)
            Z = super().transform(X)
            if not self.check_compatible_feature_names(Z)[0]:
                raise ValueError("shape of transform's output and output feature names mismatch")
            return self

        def fit_transform(self, X, *args, columns=None, **kwargs):
            Z = super().fit_transform(X, *args, **kwargs)
            self.set_input_feature_names(X)
            self.set_feature_names(columns=columns)
            if not self.check_compatible_feature_names(Z)[0]:
                raise ValueError("shape of transform's output and column names mismatch")
            return self.cast_to_dataframe(Z)

        def transform(self, X, *args, **kwargs):
            # what happens if X is dataframe with columns in different order than the fitted?
            columns = self.get_feature_names()
            input_columns = self.get_input_feature_names()
            if input_columns is not None and X is not None:
                if X.shape[1] != len(input_columns):
                    raise ValueError("shapes of X and feature_names incompatible")
                # if isinstance(X, pd.DataFrame):
                #     X = X[[input_columns]]
            Z = super().transform(X, *args, **kwargs)
            return self.cast_to_dataframe(Z)

    return NewCls


# from sklearn.preprocessing import FunctionTransformer

StandardScaler = dataframe_transformer(StandardScaler)
RobustScaler = dataframe_transformer(RobustScaler)
MinMaxScaler = dataframe_transformer(MinMaxScaler)
MaxAbsScaler = dataframe_transformer(MaxAbsScaler)
KBinsDiscretizer = dataframe_transformer(KBinsDiscretizer)
Binarizer = dataframe_transformer(Binarizer)
QuantileTransformer = dataframe_transformer(QuantileTransformer)
PolynomialFeatures = dataframe_transformer(PolynomialFeatures)
Normalizer = dataframe_transformer(Normalizer)
PowerTransformer = dataframe_transformer(PowerTransformer)

OneHotEncoder = dataframe_transformer(OneHotEncoder)
OrdinalEncoder = dataframe_transformer(OrdinalEncoder)
LabelEncoder = dataframe_transformer(LabelEncoder)
LabelBinarizer = dataframe_transformer(LabelBinarizer)

SimpleImputer = dataframe_transformer(SimpleImputer)
MissingIndicator = dataframe_transformer(MissingIndicator)

Pipeline = dataframe_transformer(Pipeline)
ColumnTransformer = dataframe_transformer(ColumnTransformer)
FeatureUnion = dataframe_transformer(FeatureUnion)


def get_feature_names(self):
    if hasattr(self, "_column_names"):
        return self._column_names
    return None


######################################################################################
######################################################################################

import unittest
import inspect


class MyTestScenario(unittest.TestCase):

    @classmethod
    def model(cls, t, X_fit, cols_fit, X_t, cols_o, *args, **kwargs):
        if isinstance(X_fit, pd.DataFrame):
            cols_fit = list(X_fit)

        schema = []
        if cols_o:
            schema = cols_o
        elif t._match_fit_io_feature_names and cols_fit:
            schema = cols_fit
            if t._match_fit_io_feature_names:
                cols_o = schema

        if X_t is not None and len(X_t):
            t = t.fit(X_fit, *args, columns=cols_o, **kwargs)
            Z = t.transform(X_t)
        else:
            Z = t.fit_transform(X_fit, *args, columns=cols_o, **kwargs)

        return t, Z, schema


    @classmethod
    def setUpClass(cls):
        ds_bunch = from_sklean_to_dataset_bunch(sklearn.datasets.load_wine(), name='wine', archive=False)
        df = fetch_dataset_bunch(ds_bunch)
        cls._df = df[['alcohol', 'magnesium', 'flavanoids', 'malic_acid', 'proline', 'ash', 'total_phenols',
                      'target']]
        dfn = TypeSelector(np.float_).fit_transform(cls._df)
        cls._dfn = dfn
        cls._dfn_permuted = dfn[['magnesium', 'alcohol', 'flavanoids', 'malic_acid', 'proline', 'ash', 'total_phenols']]
        cls._dfn_reduced = dfn[['alcohol', 'magnesium']]
        cls._dfn_expanded = df[
            ['alcohol', 'magnesium', 'flavanoids', 'malic_acid', 'proline', 'ash', 'total_phenols', 'hue']]
        cls._feature_names = list(cls._df)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.dfn = self._dfn
        self.scaler = [MyScaler(),
                       # StandardScaler(), RobustScaler(), MaxAbsScaler(), MinMaxScaler(),
                       # KBinsDiscretizer(n_bins=5, encode='ordinal'), QuantileTransformer(n_quantiles=20),
                       # PowerTransformer(),
                       # Normalizer(),
                       # PolynomialFeatures(degree=2),
                       # OrdinalEncoder(),
                       # LabelBinarizer(),
                       # LabelEncoder(),
                       # OneHotEncoder(),
                       # Binarizer(),
                       ]

    def tearDown(self):
        pass

    def show_output(self, Z, *args, **kwargs):
        return
        if isinstance(Z, np.ndarray):
            print(Z[:1, :], *args, **kwargs)
        elif isinstance(Z, pd.DataFrame):
            print(Z.head(1), *args, **kwargs)
        else:
            print(Z, *args, **kwargs)


class TestScalers(MyTestScenario):

    @unittest.expectedFailure
    def test_transform_extra_kwargs(self):
        # should not pass columns kwargs to transform()
        X, cols = self.dfn.values, list(self.dfn)
        for t in self.scaler:
            Z = t.fit(X).transform(X, columns=cols)

    def test_unfitted(self):
        for t in self.scaler:
            with self.subTest(t=t):
                t.get_feature_names()
                self.assertListEqual(t.get_feature_names(), [])
                self.assertListEqual(t.get_input_feature_names(), [])

    def test_fit_array_no_features(self):
        X = self.dfn.values
        for t in self.scaler:
            with self.subTest(t=t):
                t = t.fit(X)
                self.assertListEqual(t.get_feature_names(), [])
                self.assertListEqual(t.get_input_feature_names(), [])

            with self.subTest(t=t):
                Z = t.fit(X).transform(X)
                self.assertListEqual(t.get_feature_names(), [])
                self.assertListEqual(t.get_input_feature_names(), [])
                self.assertIsInstance(Z, np.ndarray)

            with self.subTest(t=t):
                Z = t.fit_transform(X)
                self.assertListEqual(t.get_feature_names(), [])
                self.assertListEqual(t.get_input_feature_names(), [])
                self.assertIsInstance(Z, np.ndarray)

    # @unittest.expectedFailure
    @unittest.skip("TBD")
    def test_fit_array_mismatch_input_features(self):
        pass

    def test_fit_dataframe_input_features(self):
        X, X2 = self.dfn, self._dfn_permuted
        cols, cols2 = list(X), list(X2)

        for t in self.scaler:
            with self.subTest(t=t):
                t.fit(X)
                self.assertListEqual(t.get_input_feature_names(), cols)

            with self.subTest(t=t):
                Z = t.fit(X).transform(X)
                self.assertListEqual(t.get_input_feature_names(), cols)
                # self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                Z = t.fit(X).transform(X2)
                self.assertListEqual(t.get_input_feature_names(), cols)
                # self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                Z = t.fit_transform(X)
                self.assertListEqual(t.get_input_feature_names(), cols)
                # self.assertIsInstance(Z, pd.DataFrame)


class TestScalersEqIO(MyTestScenario):
    # test scalers

    match = Bunch(fit_io_feature_names=True,
                  transform_input_feature_names=True,
                  transform_input_no_features=True)


    def setUp(self):
        super().setUp()
        self.scaler = []

        self.X, self.cols = self.dfn.values, list(self.dfn)

        # scalers with #inputs = #outputs

        self.scaler_eq_io_shapes = [StandardScaler(), RobustScaler(), MaxAbsScaler(), MinMaxScaler(),
                                    KBinsDiscretizer(n_bins=5, encode='ordinal'), QuantileTransformer(n_quantiles=20),
                                    PowerTransformer(),
                                    OrdinalEncoder(),
                                    ]
        self.scaler_neq_io_shape = []
        self.scaler.extend(self.scaler_eq_io_shapes)

        self.scaler = self.scaler_eq_io_shapes
        self.scaler = [ MyScaler(), ]
        self.match = Bunch(fit_io_feature_names=True,
                           transform_input_feature_names=True,
                           transform_input_no_features=True)

    @unittest.skipUnless(match.fit_io_feature_names, 'NA')
    def test_fit_array_with_features_1a(self):
        for t in self.scaler:
            with self.subTest(t=t):
                t = t.fit(self.X, columns=self.cols)
                self.assertListEqual(t.get_feature_names(), self.cols)
                self.assertListEqual(t.get_input_feature_names(), self.cols)

    @unittest.skipIf(match.fit_io_feature_names, 'NA')
    def test_fit_array_with_features_1b(self):
        for t in self.scaler:
            with self.subTest(t=t):
                t = t.fit(self.X, columns=self.cols)
                self.assertListEqual(t.get_input_feature_names(), self.cols)
                self.assertListEqual(t.get_feature_names(), [])


    def test_fit_array_with_features(self):
        X, cols = self.dfn.values, list(self.dfn)

        for t in self.scaler:
            with self.subTest(t=t):
                t, Z, schema = self.model(t, X, cols, [], [])
                self.assertListEqual(t.get_input_feature_names(), cols)
                self.assertListEqual(t.get_feature_names(), schema)
                if schema:
                    self.assertIsInstance(Z, pd.DataFrame)
                    self.assertListEqual(list(Z), schema)
                else:
                    self.assertIsInstance(Z, np.ndarray)

            with self.subTest(t=t):
                t, Z, schema = self.model(t, X, cols, X, [])
                self.assertListEqual(t.get_input_feature_names(), cols)
                self.assertListEqual(t.get_feature_names(), schema)
                if schema:
                    self.assertIsInstance(Z, pd.DataFrame)
                    self.assertListEqual(list(Z), schema)
                else:
                    self.assertIsInstance(Z, np.ndarray)


    @unittest.skip
    @unittest.expectedFailure
    # @unittest.skip("seems that scalers automatically trim the expanded inputs")
    def test_fit_transform_array_mismatch_features(self):
        X, cols = self._dfn_reduced.values, list(self._dfn_expanded)
        for t in self.scaler:
            t = t.fit_transform(X, columns=cols)
            self.assertListEqual(t.get_feature_names(), cols)
            self.assertIsNone(t.get_input_feature_names())


    #@unittest.skip
    def test_fit_dataframe(self):
        Xf, Xp, Xd = self.dfn, self._dfn_permuted, self._dfn_expanded
        colsf, colsp, colsd = list(Xf), list(Xp), list(Xd)

        for t in self.scaler:
            for Xt in [Xf, Xp, Xf.values]:
                for colst in [colsf, colsp, None]:
                    with self.subTest(t=t, colst=colst, Xt=Xt):

                        t, Z, schema = self.model(t, Xf, colsf, Xt, colst)

                        #t.fit(Xf, columns=colst)
                        self.assertListEqual(t.get_input_feature_names(), colsf)

                        # schema = None
                        # if t._match_fit_io_feature_names:
                        #     schema = colst if colst is not None else colsf
                        # else:
                        #     schema = colst if colst is not None else None
                        #
                        if schema is not None:
                            self.assertListEqual(t.get_feature_names(), schema)
                        else:
                            self.assertIsNone(t.get_feature_names())

                        # Z = t.transform(Xt)
                        if schema is None:
                            self.assertIsInstance(Z, np.ndarray)
                        else:
                            self.assertIsInstance(Z, pd.DataFrame)
                            self.assertListEqual(list(Z), schema)
        print('done', self.id())


    @unittest.skip
    def test_fit_dataframe_priority_of_features(self):
        # columns take priority over the dataframe in the feature names
        X, cols = self.dfn, list(self._dfn_permuted)
        for t in self.scaler_eq_io_shapes:
            t.fit(X, columns=cols)
            self.assertListEqual(t.get_feature_names(), cols)

    @unittest.skip("TBD")
    @unittest.expectedFailure
    def test_fit_dataframe_no_reordering_features(self):
        # have feature names of dataframe if fitted with it -- order of feature names is important
        X, cols = self.dfn, list(self._dfn_permuted)
        for t in self.scaler_eq_io_shapes:
            t.fit_transform(X, columns=col)
            self.assertListEqual(t.get_feature_names(), cols)


    @unittest.skip
    @unittest.expectedFailure
    def test_fit_values_with_mismatched_features(self):
        X, X2 = self.dfn.values, self._dfn_expanded
        cols, cols2 = list(X), list(X2)
        for t in self.scaler_eq_io_shapes:
            t = t.fit(X, columns=cols2)
            self.assertListEqual(t.get_feature_names(), cols2)
            self.assertListEqual(t.get_input_feature_names(), cols)

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_values_transform_mismatched_values_features(self):
        X, X2 = self.dfn.values, self._dfn_expanded
        cols, cols2 = list(X), list(X2)
        for t in self.scaler_eq_io_shapes:
            Z = t.fit(X, columns=cols).transform(X2)
            self.assertListEqual(t.get_feature_names(), cols)
            self.assertIsNone(t.get_input_feature_names())
            self.assertIsInstance(Z, pd.DataFrame)

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_transform_values_with_mismatched_features(self):
        X, X2 = self.dfn.values, self._dfn_expanded
        cols, cols2 = list(X), list(X2)
        for t in self.scaler_eq_io_shapes:
            Z = t.fit_transform(X, columns=cols2)
            self.assertListEqual(t.get_feature_names(), cols2)
            self.assertIsNone(t.get_input_feature_names())
            self.assertIsInstance(Z, pd.DataFrame)

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_dataframe_with_mismatched_features(self):
        X, X2 = self.dfn, self._dfn_expanded
        cols, cols2 = list(X), list(X2)
        for t in self.scaler_eq_io_shapes:
            t = t.fit(X, columns=cols2)
            self.assertListEqual(t.get_feature_names(), cols2)
            self.assertListEqual(t.get_input_feature_names(), cols)

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_transform_dataframe_with_mismatched_features(self):
        X, X2 = self.dfn, self._dfn_expanded
        cols, cols2 = list(X), list(X2)
        for t in self.scaler_eq_io_shapes:
            Z = t.fit_transform(X, columns=cols2)
            self.assertListEqual(t.get_feature_names(), cols2)
            self.assertListEqual(t.get_input_feature_names(), cols)
            self.assertIsInstance(Z, pd.DataFrame)
            self.assertListEqual(list(Z), cols2)

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_dataframe_transform_mismatched_dataframe(self):
        X, X2 = self.dfn, self._dfn_expanded
        cols, cols2 = list(X), list(X2)
        for t in self.scaler:
            Z = t.fit(X).transform(X2)
            # self.assertListEqual(t.get_feature_names(), cols)
            # self.assertListEqual(t.get_input_feature_names(), cols)
            self.assertIsInstance(Z, pd.DataFrame)
            # self.assertListEqual(list(Z), cols)

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_dataframe_transform_mismatched_values(self):
        X, X2 = self.dfn, self._dfn_expanded.values
        cols, cols2 = list(X), list(X2)
        for t in self.scaler_eq_io_shapes:
            Z = t.fit(X).transform(X2)
            self.assertListEqual(t.get_feature_names(), cols)
            self.assertListEqual(t.get_input_feature_names(), cols)
            self.assertIsInstance(Z, pd.DataFrame)
            self.assertListEqual(list(Z), cols)

# check the output of the transform()

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestScalers())
    suite.addTest(TestScalersEqIO())
    return suite


######################################################################################
######################################################################################

def test_pipelines(rng=0):
    # tester = unittest.TextTestRunner(verbosity=2)
    # tester.run(suite)

    unittest.main(verbosity=2)


    # fetch a dataset
    ds_bunch = from_sklean_to_dataset_bunch(sklearn.datasets.load_wine(), name='wine', archive=False)
    df = fetch_dataset_bunch(ds_bunch)
    print(df.head(5))

    fmap = get_dataframe_feature_types(df)
    print("Feature types", fmap)
    for k, v in fmap.items():
        print(f"{k}::{v}")

    df = df[['alcohol', 'magnesium', 'flavanoids', 'malic_acid', 'proline', 'ash', 'total_phenols',
             'target']]
    dfn = TypeSelector(np.float_).fit_transform(df)

    obj = MyScaler()
    print(obj)
    obj = obj.fit(dfn)
    Z = obj.transform(dfn)
    Z = obj.fit_transform(dfn)

    # dynamically add a method to an instance of a class
    t = StandardScaler()
    # t.get_feature_names = types.MethodType(get_feature_names, t)
    print("before", t.get_feature_names())
    t.fit(dfn)
    print("after", t.get_feature_names())

    q = StandardScaler().fit_transform(dfn)
    q = RobustScaler().fit_transform(dfn)
    q = PowerTransformer().fit_transform(dfn)
    q = KBinsDiscretizer(n_bins=5, encode='ordinal').fit_transform(dfn)
    q = SimpleImputer().fit_transform(dfn)
    q = MissingIndicator().fit_transform(dfn)

    enc = OneHotEncoder().fit(df)

    # ColumnTransformer fit, and fit_transform methods take a columns kwargs to create a dataframe
    colnames = [f"target_{i}" for i in np.unique(df['target'])]
    q = ColumnTransformer([('one', OneHotEncoder(), ["target"])]).fit_transform(df, columns=colnames)

    q = ColumnTransformer([('one', OrdinalEncoder(), ['target', 'magnesium'])]) \
        .fit_transform(df, columns=['target_o', 'magnesium_o'])

    enc = make_union(ColumnTransformer([('one', OneHotEncoder(), ["target"])]),
                     ColumnTransformer([('one', StandardScaler(), ['magnesium'])]))
    q = enc.fit_transform(df)

    # q = LabelEncoder().fit_transform(df['target', 'ash'])

    # define a few (composite) transformer pipelines
    txs = dict()

    # named steps
    txs['mean-robust'] = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler(quantile_range=(25.0, 75.0)))])

    # unnamed steps
    txs['median-standard'] = make_pipeline(
        make_union(SimpleImputer(strategy='median'), MissingIndicator()), StandardScaler())

    # pipelines produce dataframe if input and last stage produce a dataframe
    q = txs['mean-robust'].fit_transform(dfn)

    # print(txs['median-standard'].get_params())

    q = txs['median-standard'].fit_transform(dfn)
    q = ColumnTransformer([('am test', txs['mean-robust'], ['ash', 'magnesium'])]) \
        .fit_transform(dfn, columns=["ash_c", "magnesium_c"])

    q = DataframeCaster(columns=['ash', 'magnesium']).fit_transform(q)

    pipe1 = make_column_transformer(
        # (StandardScaler(), ['alcalinity_of_ash', 'ash']),
        # (RobustScaler(quantile_range=(25.0, 75.0)), ['total_phenols', 'flavanoids']),
        # (PowerTransformer(), ['magnesium']),
        # (txs['median-standard'], ['proanthocyanins', 'malic_acid']),
        # (KBinsDiscretizer(n_bins=5, encode='ordinal'), ['alcohol']),
        # (OrdinalEncoder(), ['magnesium']),
        (OneHotEncoder(), ['target']),
        remainder='_drop_at_depth')
    columns = []
    for t in pipe1.transformers:
        if t[2]:
            columns.extend(t[2])
            print(t[0], t[2])
    columns.extend(['target'] * 2)
    # columns = np.asarray(columns).flatten()
    pipe2 = DataframeCaster(columns=columns)
    q = pipe1.fit(df).transform(df)
    z = pipe2.fit_transform(q)
    print("here ", z[:4, :])

    classifier_pipeline = make_pipeline(
        preprocess_pipeline,
        SVC(kernel="rbf", random_state=42)
    )

    clean_df(df)


#       StandardScaler(), MaxAbsScaler() MinMaxScaler(feature_range=(0, 1))
#       RobustScaler(quantile_range=(25.0, 75.0)), KBinsDiscretizer(n_bins=5, encode='ordinal')
#       PowerTransformer() Binarizer(threshold=0.0)
#       PolynomialFeatures(degree=2)
#       Normalizer()

if __name__ == "__main__":
    rng = np.random.RandomState(2019)

    test_pipelines()
    input('press any key to exit')
