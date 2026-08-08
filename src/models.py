from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_validate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def make_ridge_pipeline(preprocessor, alpha):
    model = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )

    return TransformedTargetRegressor(
        regressor = model,
        func = np.log1p,
        inverse_func=np.expm1
    )
    