#updated to use cleaned data instead of cleaning everytime
import sys
import os
import pandas as pd  # Import pandas directly
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("src"))
# from data_loader import load_data  <-- REMOVE THIS
from src.stats.ab_testing import ABTester


def main():
    # 1. Setup Paths
    project_root = Path(__file__).resolve().parents[2]

    # Use the filename you defined in your dvc.yaml
    data_path = project_root / 'data' / 'Processed' / 'cleaned_data.csv'
    report_path = project_root / 'reports' / 'ab_test_results.csv'

    # Ensure report directory exists
    os.makedirs(report_path.parent, exist_ok=True)

    print(f"Loading data from: {data_path}")
    if not data_path.exists():
        print("Error: Processed data not found. Run 'dvc repro' to generate it.")
        return

    # 2. Load Data using STANDARD pandas read_csv (Comma separated)
    try:
        df = pd.read_csv(data_path, low_memory=False)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # 3. Initialize Tester
    print("Initializing A/B Tester...")
    tester = ABTester(df)

    # Open file to write results
    with open(report_path, "w") as f:
        f.write("=== A/B Hypothesis Testing Report ===\n\n")

        # --- HYPOTHESIS 1: PROVINCE RISK ---
        f.write("--- HYPOTHESIS 1: Province Risk ---\n")
        prov_stats = df.groupby('Province')[['TotalClaims', 'TotalPremium']].sum()
        prov_stats['LossRatio'] = prov_stats['TotalClaims'] / prov_stats['TotalPremium']
        prov_stats = prov_stats.sort_values('LossRatio', ascending=False)

        risky_prov = prov_stats.index[0]
        safe_prov = prov_stats.index[-1]

        f.write(f"Risky: {risky_prov} vs Safe: {safe_prov}\n")
        results = tester.run_hypothesis_test('Province', [risky_prov, safe_prov], 'Risk')
        for res in results:
            f.write(f"  > {res.get('Test', 'Test')}: {res['Interpretation']} (p={res.get('P-Value', 'N/A')})\n")
        f.write("\n")

        # --- HYPOTHESIS 2: ZIP CODE RISK ---
        f.write("--- HYPOTHESIS 2: Zip Code Risk ---\n")
        # Filter for volume
        zip_counts = df['PostalCode'].value_counts()
        valid_zips = zip_counts[zip_counts > 50].index

        if len(valid_zips) > 2:
            zip_stats = df[df['PostalCode'].isin(valid_zips)].groupby('PostalCode')[
                ['TotalClaims', 'TotalPremium']].sum()
            zip_stats['LossRatio'] = zip_stats['TotalClaims'] / zip_stats['TotalPremium']
            zip_stats = zip_stats.sort_values('LossRatio', ascending=False)

            risky_zip = zip_stats.index[0]
            safe_zip = zip_stats.index[-1]

            f.write(f"Risky Zip: {risky_zip} vs Safe Zip: {safe_zip}\n")
            results = tester.run_hypothesis_test('PostalCode', [risky_zip, safe_zip], 'Risk')
            for res in results:
                f.write(f"  > {res.get('Test', 'Test')}: {res['Interpretation']} (p={res.get('P-Value', 'N/A')})\n")
        else:
            f.write("Not enough volume in Zip Codes for valid testing.\n")
        f.write("\n")

        # --- HYPOTHESIS 4: GENDER RISK ---
        f.write("--- HYPOTHESIS 4: Gender Risk ---\n")
        if 'Gender' in df.columns:
            results = tester.run_hypothesis_test('Gender', ['Female', 'Male'], 'Risk')
            for res in results:
                f.write(f"  > {res.get('Test', 'Test')}: {res['Interpretation']} (p={res.get('P-Value', 'N/A')})\n")
        else:
            f.write("Gender column missing.\n")

    print(f"A/B Testing completed. Report saved to {report_path}")


if __name__ == "__main__":
    main()