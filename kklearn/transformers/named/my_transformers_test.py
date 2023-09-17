######################################################################################
######################################################################################
import sys

import numpy as np
import pandas as pd

from sklearn.utils import Bunch

from sklearn import preprocessing, impute, pipeline, compose

from my_transformers import TransformerInterfaceMixin, get_dataset

from my_transformers import MyScaler

from my_transformers import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from my_transformers import KBinsDiscretizer, Binarizer, QuantileTransformer
from my_transformers import PolynomialFeatures, Normalizer, PowerTransformer
from my_transformers import LabelBinarizer, OneHotEncoder, LabelEncoder, OrdinalEncoder
from my_transformers import FunctionTransformer
from my_transformers import SimpleImputer, MissingIndicator

from my_transformers import ColumnTransformer
from my_transformers import Pipeline
from my_transformers import FeatureUnion

#from sklearn.compose import make_column_transformer, ColumnTransformer

# from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
# from sklearn.preprocessing import KBinsDiscretizer, Binarizer, QuantileTransformer
# from sklearn.preprocessing import PolynomialFeatures, Normalizer, PowerTransformer
# from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder, OrdinalEncoder
# from sklearn.preprocessing import FunctionTransformer

import unittest

class MyMasterTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = get_dataset()
        df.columns = list("ABCDE")
        cols = list(df)
        #cols.remove('target')
        cols.remove("E")
        dfn = df[cols]
        cols[1], cols[0] = cols[0], cols[1]

        cls._df = dfn
        cls._df_permuted = dfn[cols]
        cls._df_reduced = dfn[list(dfn)[:3]]
        cls._df_expanded = df

        cls.scaler_types = [MyScaler, StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, KBinsDiscretizer, Binarizer,
                            QuantileTransformer,
                            PolynomialFeatures,
                            Normalizer, PowerTransformer,
                            # LabelBinarizer,
                            # OneHotEncoder,
                            # LabelEncoder,
                            # OrdinalEncoder,
                            FunctionTransformer,
                            SimpleImputer, MissingIndicator,
                            ]

        cls.scaler_instance_args = {ColumnTransformer: list(('custom_col', StandardScaler(), list(cls._df)))}
        cls.scaler_instance_kwargs = {KBinsDiscretizer: dict(n_bins=5, encode='ordinal'),
                                      QuantileTransformer: dict(n_quantiles=20),
                                      PolynomialFeatures: dict(degree=1, include_bias=False),
                                      OneHotEncoder: dict(categories='auto'),
                                      OrdinalEncoder: dict(dtype=np.int_),
                                      FunctionTransformer: dict(func=lambda X: X, validate=True)
                                      }

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.df = self._df

        self.scaler = []
        for t in type(self).scaler_types:
            args = type(self).scaler_instance_args.get(t, [])
            kwargs = type(self).scaler_instance_kwargs.get(t, dict())
            obj = t(*args, **kwargs)
            self.scaler += [obj]

        t = MyScaler
        args = type(self).scaler_instance_args.get(t, [])
        kwargs = type(self).scaler_instance_kwargs.get(t, dict())
        # self.scaler = [t(*args, **kwargs), ]

        pass

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


########################################################################################################################

class TestTransformersSimple(MyMasterTest):

    def setUp(self):
        super().setUp()
        fixed = []
        for t in self.scaler:
            t._prm_fit_input_propagation = 1
            t._prm_tranform_input_validation = (2, 1)
            t._prm_transform_output_format = list((0, 4))
            t._prm_output_features_default_names_fmt = "x_{}"
            fixed.append(t)
        self.scaler = fixed
        pass

    def local_vars(self):
        X = self._df
        m = X.shape[-1]
        cols, nocols = list(X), list(range(m))
        newcols, newercols = [f"new_{x}" for x in cols], [f"newer_{x}" for x in cols]

        Xp, Xe, Xr = self._df_permuted, self._df_expanded, self._df_reduced
        colsp, colse, colsr = list(Xp), list(Xe), list(Xr)

        return X, cols, nocols, newcols, Xp, colsp, Xe, colse, Xr, colsr, newercols

    def test_not_fitted(self):
        for t in self.scaler:
            with self.subTest(t=t):
                self.assertListEqual(t.get_feature_names(0), [])
                self.assertListEqual(t.get_feature_names(), [])


    def test_fit_array_simple(self):
        X, cols, nocols = self.local_vars()[:3]

        for t in self.scaler:
            # if isinstance(t, LabelEncoder) and len(cols)>1: fails - continue
            with self.subTest(t=t):
                t = t.fit(X.values)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertListEqual(t.get_feature_names(), [])

            with self.subTest(t=t):
                Z = t.fit(X.values).transform(X.values)
                self.assertListEqual(t.get_feature_names(), nocols)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertIsInstance(Z, np.ndarray)

            with self.subTest(t=t):
                Z = t.fit_transform(X.values)
                self.assertListEqual(t.get_feature_names(), nocols)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertIsInstance(Z, np.ndarray)


    def test_fit_dataframe_simple(self):
        X, cols, nocols, newcols, Xp, colsp, Xe, colse, Xr, colsr, newercols = self.local_vars()

        for t in self.scaler:
            #if isinstance(t, LabelEncoder) and len(cols)>1 or isinstance(t, OrdinalEncoder) and t.fit(X).transform(Xp) fails
            with self.subTest(t=t):
                t.fit(X)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), [])

            with self.subTest(t=t):
                Z = t.fit(X).transform(X)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), cols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                # align transform input (permuted of fitted) to fitted input
                Z = t.fit(X).transform(Xp)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), cols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                Z = t.fit_transform(X)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), cols)
                self.assertIsInstance(Z, pd.DataFrame)


    def test_fit_dataframe_renaming_output_features(self):
        X, cols, nocols, newcols, Xp, colsp, Xe, colse, Xr, colsr, newercols = self.local_vars()

        for t in self.scaler:
            # same expected failures as the simple dataframe fit/transform tests
            with self.subTest(t=t):
                t.fit(X, columns=newcols)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), [])

            with self.subTest(t=t):
                Z = t.fit(X, columns=newcols).transform(X)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), cols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                Z = t.fit(X).transform(X, columns=newcols)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), newcols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                # transform output inherits input type - not affected by the fitted input type or its column names
                Z = t.fit(X, columns=newcols).transform(X.values)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), nocols)
                self.assertIsInstance(Z, np.ndarray)

            with self.subTest(t=t):
                # transform output becomes a DataFrame due to the columns kwarg
                Z = t.fit(X).transform(X.values, columns=newcols)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), newcols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                # latest columns kwarg take priority in transform output column names
                Z = t.fit(X, columns=newcols).transform(X.values, columns=newercols)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), newercols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                # align transform input (permuted of fitted) to fitted input
                Z = t.fit(X).transform(Xp)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), cols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                Z = t.fit_transform(X)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), cols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                Z = t.fit_transform(X, columns=newcols)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), newcols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                q, q[-1] = newcols, cols[-1]
                Z = t.fit_transform(X, columns=newcols[:-1])
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), q)
                self.assertIsInstance(Z, pd.DataFrame)

    @unittest.expectedFailure
    def test_fit_dataframe_renaming_output_features_2(self):
        # needs t._prm_transform_output_format[0] == 1 to make output dataframe
        X, cols, nocols, newcols = self.local_vars()[:4]
        for t in self.scaler:
            with self.subTest(t=t):
                Z = t.fit(X, columns=newcols).transform(X.values)
                self.assertListEqual(t.get_feature_names(0), cols)
                self.assertListEqual(t.get_feature_names(), newcols)
                self.assertIsInstance(Z, pd.DataFrame)


    def test_fit_array_renaming_output_features(self):
        X, cols, nocols, newcols, Xp, colsp, Xe, colse, Xr, colsr, newercols = self.local_vars()

        for t in self.scaler:
            # same expected failures as the dataframe fit/transform tests

            with self.subTest(t=t):
                t.fit(X.values, columns=newcols)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertListEqual(t.get_feature_names(), [])

            with self.subTest(t=t):
                Z = t.fit(X.values, columns=newcols).transform(X)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertListEqual(t.get_feature_names(), cols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                Z = t.fit(X.values).transform(X, columns=newcols)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertListEqual(t.get_feature_names(), newcols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                # transform output inherits input type - not affected by the fitted input type or its column names
                Z = t.fit(X.values, columns=newcols).transform(X)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertListEqual(t.get_feature_names(), cols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                # transform output becomes a DataFrame due to the columns kwarg
                Z = t.fit(X.values).transform(X.values, columns=newcols)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertListEqual(t.get_feature_names(), newcols)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                # latest columns kwarg take priority in transform output column names
                Z = t.fit(X.values, columns=newcols).transform(X.values, columns=newercols)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertListEqual(t.get_feature_names(), newercols)
                self.assertIsInstance(Z, pd.DataFrame)


            with self.subTest(t=t):
                # align transform input (permuted of fitted) to fitted input
                Z = t.fit(X.values).transform(Xp)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertListEqual(t.get_feature_names(), colsp)
                self.assertIsInstance(Z, pd.DataFrame)

            with self.subTest(t=t):
                Z = t.fit_transform(X.values, columns=newcols)
                self.assertListEqual(t.get_feature_names(0), nocols)
                self.assertListEqual(t.get_feature_names(), newcols)
                self.assertIsInstance(Z, pd.DataFrame)


    def test_fit_input_transform_input_mismatch(self):
        pass


    def test_column_transformer_simple(self):
        #X, cols, nocols = self.local_vars()[:3]
        X, cols, nocols, newcols, Xp, colsp, Xe, colse, Xr, colsr, newercols = self.local_vars()
        args = [('a', StandardScaler(), cols[:2]), ('b', RobustScaler(), cols[:3])]

        # compute the expected column names
        ocols, rcols = [], set(cols)
        for (name, trans, tcols) in args:
            rcols = rcols - set(tcols)
            ocols.extend([f'{name}__{f}' for f in tcols])

        ecols = ocols.copy()
        with self.subTest(ecols=ecols):
            t = ColumnTransformer(args)
            Z = t.fit_transform(X)
            self.assertListEqual(t.get_feature_names(0), cols)
            self.assertListEqual(t.get_feature_names(), ecols)
            self.assertIsInstance(Z, pd.DataFrame)

        with self.subTest(ecols=ecols):
            t = ColumnTransformer(args)
            Zp = t.fit(X).transform(Xp)
            self.assertListEqual(t.get_feature_names(0), cols)
            self.assertListEqual(t.get_feature_names(), ecols)
            self.assertIsInstance(Zp, pd.DataFrame)
            self.assertTrue(np.allclose(Zp, t.fit_transform(X)))

        ecols.extend(rcols)
        with self.subTest(ecols=ecols):
            t = ColumnTransformer(args, remainder='passthrough')
            Z = t.fit_transform(X)
            self.assertListEqual(t.get_feature_names(0), cols)
            self.assertListEqual(t.get_feature_names(), ecols)
            self.assertIsInstance(Z, pd.DataFrame)


    def test_pipeline_transformers_simple(self):
        X, cols, nocols, newcols, Xp, colsp, Xe, colse, Xr, colsr, newercols = self.local_vars()
        targs = [[('a', StandardScaler(), cols[:2]), ('b', RobustScaler(), cols[:3])]]
        tkwargs =dict(remainder='passthrough')
        pp = Pipeline(steps=[
                                ('stage0', StandardScaler()),
                                ('stage1', ColumnTransformer(*targs, **tkwargs)),
                                ('stage2', QuantileTransformer())
                                ], memory=None)

        with self.subTest(pp=pp):
            pp.fit(X)
            Z = pp.transform(X)
            ppcols = [f"{pp.steps[-1][0]}__{a}" for a in pp.steps[-1][1].get_feature_names()]
            self.assertListEqual(pp.get_feature_names(0), cols)
            self.assertListEqual(pp.get_feature_names(), ppcols)
            self.assertIsInstance(Z, pd.DataFrame)

        with self.subTest(pp=pp):
            Z = pp.fit(X).transform(Xp)
            ppcols = [f"{pp.steps[-1][0]}__{a}" for a in pp.steps[-1][1].get_feature_names()]
            self.assertListEqual(pp.get_feature_names(0), cols)
            self.assertListEqual(pp.get_feature_names(), ppcols)
            self.assertIsInstance(Z, pd.DataFrame)

        with self.subTest(pp=pp):
            Z = pp.fit_transform(X)
            ppcols = [f"{pp.steps[-1][0]}__{a}" for a in pp.steps[-1][1].get_feature_names()]
            self.assertListEqual(pp.get_feature_names(0), cols)
            self.assertListEqual(pp.get_feature_names(), ppcols)
            self.assertIsInstance(Z, pd.DataFrame)


    def test_union_transformers_simple(self):
        X, cols, nocols, newcols, Xp, colsp, Xe, colse, Xr, colsr, newercols = self.local_vars()

        X, cols, Xp = X[cols[:2]], cols[:2], Xp[cols[:2]]

        args = [[('a', StandardScaler()), ('b', RobustScaler())]]
        t = FeatureUnion(*args)

        tcols = []
        for name, _ in t.transformer_list:
            v = [f"{name}_{attr}" for attr in cols]
            tcols.extend(v)

        with self.subTest(t=t):
            t.fit(X)
            Z = t.transform(X)
            self.assertListEqual(t.get_feature_names(0), cols)
            self.assertListEqual(t.get_feature_names(), tcols)
            self.assertIsInstance(Z, pd.DataFrame)

        with self.subTest(t=t):
            t.fit(X)
            Z = t.transform(Xp)
            self.assertListEqual(t.get_feature_names(0), cols)
            self.assertListEqual(t.get_feature_names(), tcols)
            self.assertIsInstance(Z, pd.DataFrame)

        with self.subTest(t=t):
            Z = t.fit_transform(X)
            self.assertListEqual(t.get_feature_names(0), cols)
            self.assertListEqual(t.get_feature_names(), tcols)
            self.assertIsInstance(Z, pd.DataFrame)

########################################################################################################################


@unittest.skip
class TestScalersEqIO(MyMasterTest):

    def setUp(self):
        pass

    @unittest.skip
    @unittest.expectedFailure
    # @unittest.skip("seems that scalers automatically trim the expanded inputs")
    def test_fit_transform_array_mismatch_features(self):
        pass

    @unittest.skip
    def test_fit_dataframe_priority_of_features(self):
        pass

    @unittest.skip("TBD")
    @unittest.expectedFailure
    def test_fit_dataframe_no_reordering_features(self):
        pass

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_values_with_mismatched_features(self):
        pass

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_values_transform_mismatched_values_features(self):
        pass

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_dataframe_transform_mismatched_dataframe(self):
        pass

    @unittest.skip
    @unittest.expectedFailure
    def test_fit_dataframe_transform_mismatched_values(self):
        pass


######################################################################################
######################################################################################

if __name__ == "__main__":
    rng = np.random.RandomState(2019)
    sys.tracebacklimit = 4

    unittest.main(verbosity=3)

    input('press any key to exit')
