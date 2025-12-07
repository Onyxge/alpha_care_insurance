from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb
import pandas as pd
import numpy as np


# --- MODELS ---
def train_linear(X, y):
    try:
        model = LinearRegression()
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Linear model error: {e}")
        return None


def train_logistic(X, y):
    try:
        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Logistic model error: {e}")
        return None


def train_random_forest(X, y, task='regression'):
    """
    Trains a Random Forest model.
    task: 'regression' or 'classification'
    """
    try:
        if task == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Random Forest error: {e}")
        return None


def train_xgboost(X, y, task='regression'):
    """
    Trains an XGBoost model.
    task: 'regression' or 'classification'
    """
    try:
        if task == 'regression':
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        else:
            model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

        model.fit(X, y)
        return model
    except Exception as e:
        print(f"XGBoost error: {e}")
        return None


# --- EVALUATION ---
def eval_regression(y_true, y_pred):
    """Returns a dictionary of regression metrics"""
    try:
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }
    except Exception as e:
        print(f"Regression metrics error: {e}")
        return None


def eval_classification(y_true, y_pred):
    """Returns a dictionary of classification metrics"""
    try:
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "F1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    except Exception as e:
        print(f"Classification metrics error: {e}")
        return None


def regression_metrics(y_true, y_pred):
    return eval_regression(y_true, y_pred)