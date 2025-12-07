import sys
import os

# Ensure the 'src' folder is visible to Python
# This allows us to import from data_loader and stats.ab_testing
sys.path.append(os.path.abspath("src"))

from src.data_loader import load_data
from src.stats.ab_testing import ABTester


def main():
    # 1. Load Data
    # Ensure this path matches your actual file location relative to the project root
    file_path = '../../data/Raw/MachineLearningRating_v3.txt'

    print(f"Loading data from: {file_path}")
    raw_df = load_data(file_path)

    if raw_df is None:
        print("Error: Data could not be loaded. Check the file path.")
        return

    # 2. Initialize the A/B Tester Class
    print("Initializing A/B Tester...")
    tester = ABTester(raw_df)

    # ---------------------------------------------------------
    # HYPOTHESIS 1: PROVINCE RISK (ANOVA -> A/B Test)
    # ---------------------------------------------------------
    print("\n--- HYPOTHESIS 1: Risk Differences Across Provinces ---")

    # Dynamic Step: Find the Province with Max and Min Loss Ratio
    prov_stats = raw_df.groupby('Province')[['TotalClaims', 'TotalPremium']].sum()
    prov_stats['LossRatio'] = prov_stats['TotalClaims'] / prov_stats['TotalPremium']
    prov_stats = prov_stats.sort_values('LossRatio', ascending=False)

    risky_province = prov_stats.index[0]
    safe_province = prov_stats.index[-1]

    print(f"  > Highest Risk: {risky_province} (LR: {prov_stats.iloc[0]['LossRatio']:.2f})")
    print(f"  > Lowest Risk:  {safe_province} (LR: {prov_stats.iloc[-1]['LossRatio']:.2f})")

    # Run the Test
    results_prov = tester.run_hypothesis_test(
        feature_col='Province',
        groups=[risky_province, safe_province],
        metric='Risk'
    )

    for res in results_prov:
        # FIX: Use .get('Test') OR .get('What') to handle inconsistent keys
        test_name = res.get('Test', res.get('What', 'Unknown Test'))
        print(f"  > {test_name}: {res['Interpretation']} (P-Value: {res.get('P-Value', 'N/A')})")

    # ---------------------------------------------------------
    # HYPOTHESIS 2 & 3: ZIP CODE RISK & MARGIN
    # ---------------------------------------------------------
    print("\n--- HYPOTHESIS 2 & 3: Risk & Margin Between Zip Codes ---")

    # Dynamic Step: Find Risky vs Safe Zip Codes (Filter for volume > 50 to avoid noise)
    zip_counts = raw_df['PostalCode'].value_counts()
    valid_zips = zip_counts[zip_counts > 50].index

    zip_stats = raw_df[raw_df['PostalCode'].isin(valid_zips)].groupby('PostalCode')[
        ['TotalClaims', 'TotalPremium']].sum()
    zip_stats['LossRatio'] = zip_stats['TotalClaims'] / zip_stats['TotalPremium']
    zip_stats = zip_stats.sort_values('LossRatio', ascending=False)

    risky_zip = zip_stats.index[0]
    safe_zip = zip_stats.index[-1]

    print(f"  > Risky Zip: {risky_zip} | Safe Zip: {safe_zip}")

    # Run Risk Test (Hypothesis 2)
    print("  [H2] Testing Risk Differences:")
    results_zip_risk = tester.run_hypothesis_test(
        feature_col='PostalCode',
        groups=[risky_zip, safe_zip],
        metric='Risk'
    )
    for res in results_zip_risk:
        test_name = res.get('Test', res.get('What', 'Unknown Test'))
        print(f"    > {test_name}: {res['Interpretation']}")

    # Run Margin Test (Hypothesis 3)
    print("  [H3] Testing Margin (Profit) Differences:")
    results_zip_margin = tester.run_hypothesis_test(
        feature_col='PostalCode',
        groups=[risky_zip, safe_zip],
        metric='Margin'
    )
    for res in results_zip_margin:
        test_name = res.get('Test', res.get('What', 'Unknown Test'))
        print(f"    > {test_name}: {res['Interpretation']}")

    # ---------------------------------------------------------
    # HYPOTHESIS 4: GENDER RISK
    # ---------------------------------------------------------
    print("\n--- HYPOTHESIS 4: Risk Difference Between Women and Men ---")
    results_gender = tester.run_hypothesis_test(
        feature_col='Gender',
        groups=['Female' , 'Male'],
        metric='Risk'
    )
    for res in results_gender:
        test_name = res.get('Test', res.get('What', 'Unknown Test'))
        print(f"  > {test_name}: {res['Interpretation']} (P-Value: {res.get('P-Value', 'N/A')})")


if __name__ == "__main__":
    main()