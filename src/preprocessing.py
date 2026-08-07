from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_validate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def identify_column_types(df):
    numeric_columns = (
        df.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=["object", "category", "string", "bool"]
        )
        .columns
        .tolist()
    )

    return numeric_columns, categorical_columns



def make_preprocessor(numeric_columns, categorical_columns):
    numeric_pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    categorical_pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown = "ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return preprocessor