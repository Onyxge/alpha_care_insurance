import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Metric & Utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------------------
# 1. FUNCTION: Preprocess Data (The bridge between Raw Data & Your Models)
# --------------------------------------------------------------
def preprocess_data(df, target_col='TotalClaims'):
    """
    Cleans data and converts categorical text columns into numbers
    so the models can process them.
    """
    print("--- Preprocessing Data ---")

    # 1. Filter: Modeling Risk (Severity) usually focuses on cases where claims exist
    data = df[df['TotalClaims'] > 0].copy()

    # 2. Select useful features (excluding IDs and dates for simplicity)
    feature_cols = [
        'VehicleType', 'Make', 'BodyType', 'Province', 'Gender',
        'CubicCapacity', 'Kilowatts', 'CustomValueEstimate', 'NumberOfDoors',
        'SumInsured', 'CalculatedPremiumPerTerm'
    ]

    # Ensure columns exist
    valid_cols = [c for c in feature_cols if c in data.columns]
    data = data[valid_cols + [target_col]].copy()

    # 3. Handle Missing Values
    # Fill numeric NaNs with mean, categorical with 'Unknown'
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna('Unknown')
        else:
            data[col] = data[col].fillna(data[col].mean())

    # 4. Encode Categorical Columns (Text -> Numbers)
    # We use LabelEncoding for tree models (RF, XGB) as it's efficient
    label_encoders = {}
    for col in valid_cols:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    X = data[valid_cols]
    y = data[target_col]

    print(f"Data ready. Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y, label_encoders


# --------------------------------------------------------------
# 2. FUNCTION: Split Data (From your file)
# --------------------------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# --------------------------------------------------------------
# 3. FUNCTION: Train All Models (From your file)
# --------------------------------------------------------------
def train_models(X_train, y_train):
    print("--- Training Models ---")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# --------------------------------------------------------------
# 4. FUNCTION: Evaluate Models
# --------------------------------------------------------------
def evaluate_models(models, X_test, y_test):
    print("--- Evaluating Models ---")
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })
        print(f"{name}: RMSE={rmse:.2f}, R2={r2:.4f}")

    return pd.DataFrame(results)


# --------------------------------------------------------------
# 5. FUNCTION: Feature Importance (SHAP Requirement)
# --------------------------------------------------------------
def plot_feature_importance(model, feature_names, model_name="Model"):
    """
    Plots the top 10 most important features for Tree-based models.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Skipping feature importance for {model_name} (not supported)")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10

    plt.figure(figsize=(10, 6))
    plt.title(f"Top 10 Feature Importances: {model_name}")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()
    print(f"Plot generated for {model_name}")