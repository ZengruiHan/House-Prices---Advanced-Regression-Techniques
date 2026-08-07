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




def evaluate_model(model, X, y, cv):
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=-1,
    )

    train_rmse = -scores["train_score"]
    valid_rmse = -scores["test_score"]

    results = {
        "train_rmse_mean": train_rmse.mean(),
        "train_rmse_std": train_rmse.std(),
        "valid_rmse_mean": valid_rmse.mean(),
        "valid_rmse_std": valid_rmse.std(),
        "train_rmse": train_rmse,
        "valid_rmse": valid_rmse,
    }

    return results