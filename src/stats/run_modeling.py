import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path so we can import internal modules
sys.path.append(os.path.abspath("src"))

# We import the modeling logic, but NOT the old data loader
from src.stats.modeling import preprocess_data, split_data, train_models, evaluate_models


def main():
    # 1. Setup Paths
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'data' / 'Processed' / 'cleaned_data.csv'
    metrics_path = project_root / 'reports' / 'model_metrics.csv'
    plot_path = project_root / 'reports' / 'figures' / 'feature_importance.png'

    # Ensure output dirs exist
    os.makedirs(metrics_path.parent, exist_ok=True)
    os.makedirs(plot_path.parent, exist_ok=True)

    print(f"Loading data from: {data_path}")
    if not data_path.exists():
        print("Error: Processed data not found. Run 'dvc repro' to generate it.")
        return

    # 2. Load Data (Use standard pandas CSV reader here)
    # Since processed data is comma-separated, we don't need the custom pipe loader
    try:
        df = pd.read_csv(data_path, low_memory=False)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    print("--- Preprocessing Data ---")
    # 3. Preprocess
    X, y, encoders = preprocess_data(df, target_col='TotalClaims')

    # 4. Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Train Models
    models = train_models(X_train, y_train)

    # 6. Evaluate
    results = evaluate_models(models, X_test, y_test)
    print("\n--- Final Results Comparison ---")
    print(results)

    # 7. Save Metrics
    print(f"Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        f.write(results.to_string())

    # 8. Feature Importance (Best Model: Random Forest)
    if "Random Forest" in models:
        rf_model = models["Random Forest"]

        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            # Get top 10 indices
            indices = importances.argsort()[::-1][:10]
            top_features = X.columns[indices]
            top_importances = importances[indices]

            plt.figure(figsize=(10, 6))
            plt.title("Random Forest Feature Importance")
            plt.bar(range(10), top_importances, align="center")
            plt.xticks(range(10), top_features, rotation=45)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved feature importance plot to {plot_path}")
        else:
            print("Random Forest model not fitted or missing feature_importances_")


if __name__ == "__main__":
    main()