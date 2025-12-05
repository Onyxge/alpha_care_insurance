import os
import sys

# Ensure the 'src' folder is visible to Python
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from src.data_loader import load_data
from src.eda_utils import generate_key_plots, generate_comprehensive_stats


def main():
    # 1. Define where the data is
    file_path = 'data/Raw/MachineLearningRating_v3.txt'

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 2. Load the data
    print("Loading data...")
    df = load_data(file_path)

    if df is None:
        print("Data loading failed.")
        return

    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Generate the plots
    # Note: These functions automatically save and show the plots.
    # We do NOT assign them to variables because they return None.
    print("Generating EDA plots...")
    generate_key_plots(df)

    print("Generating comprehensive statistics...")
    generate_comprehensive_stats(df)

    print("Analysis complete. Check the 'reports/figures' folder for your images.")


if __name__ == "__main__":
    main()