import pandas as pd
from typing import List, Dict, Any

# Assuming hypothesis.py is in the same directory (src/stats)
# and contains: t_test, chi_square, cohens_d, translate_insight
from hypothesis import t_test, chi_square, cohens_d, translate_insight


class ABTester:
    """
    Performs A/B Hypothesis Testing for AlphaCare Insurance Solutions (ACIS).
    The main goal is to test for significant differences in Claim Frequency
    and Claim Severity across categorical features (e.g., Gender, Province).
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the ABTester. Requires a DataFrame that includes the
        target risk columns.
        """
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self) -> None:
        """
        Calculates the necessary derivative metrics for risk analysis.
        1. HasClaim (for Frequency)
        2. Margin (for Profit/Loss analysis)
        """
        # Risk Metric 1: Claim Frequency (Binary)
        # Policy is "Risky" (1) if TotalClaims > 0, "Not Risky" (0) otherwise.
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)

        # Margin Metric: Total Premium - Total Claims (for Profit analysis)
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']

        # Risk Severity data: Only policies that have actually made a claim
        self.severity_df = self.df[self.df['HasClaim'] == 1].copy()

    # --------------------------------------------------------------------------
    # CORE TESTING METHODS
    # --------------------------------------------------------------------------

    def test_frequency(self, group_col: str, group_a: Any, group_b: Any) -> Dict:
        """
        Tests for a significant difference in Claim Frequency (HasClaim rate)
        between two groups using the Chi-Squared Test.
        """
        # Filter data for the two groups
        data = self.df[self.df[group_col].isin([group_a, group_b])]

        # Chi-Square Test: (Group Col) vs (HasClaim)
        try:
            stat, p_value, dof, expected = chi_square(data, group_col, 'HasClaim')
        except ValueError:
            return {"Test": "Frequency (Chi-Sq)", "Interpretation": "Not enough data/variance for test."}

        # Calculate group means (claim rates)
        rate_a = data[data[group_col] == group_a]['HasClaim'].mean()
        rate_b = data[data[group_col] == group_b]['HasClaim'].mean()

        # Interpret the result
        insight = translate_insight(
            test_name=f"Chi-Sq on Claim Frequency for {group_col}: {group_a} vs {group_b}",
            p_value=p_value,
        )
        insight['Rate A'] = f"{group_a}: {rate_a:.4f}"
        insight['Rate B'] = f"{group_b}: {rate_b:.4f}"
        insight['P-Value'] = f"{p_value:.4f}"
        return insight

    def test_severity(self, group_col: str, group_a: Any, group_b: Any) -> Dict:
        """
        Tests for a significant difference in Claim Severity (Average Claim Amount)
        between two groups using the T-Test. (Only on policies with claims).
        """
        # Filter the severity data (claims > 0) for the two groups
        data_a = self.severity_df[self.severity_df[group_col] == group_a]['TotalClaims']
        data_b = self.severity_df[self.severity_df[group_col] == group_b]['TotalClaims']

        if data_a.shape[0] < 5 or data_b.shape[0] < 5:
            return {"Test": "Severity (T-Test)", "Interpretation": "Insufficient claims data for groups."}

        # T-Test: (Group A Claims) vs (Group B Claims)
        stat, p_value = t_test(data_a, data_b)
        effect_size = cohens_d(data_a, data_b)

        # Interpret the result
        insight = translate_insight(
            test_name=f"T-Test on Claim Severity for {group_col}: {group_a} vs {group_b}",
            p_value=p_value,
            effect=f"{effect_size:.3f}"  # Cohen's D is the effect size
        )
        insight['Avg Claim A'] = f"{group_a}: R{data_a.mean():.2f}"
        insight['Avg Claim B'] = f"{group_b}: R{data_b.mean():.2f}"
        insight['P-Value'] = f"{p_value:.4f}"
        return insight

    def test_margin(self, group_col: str, group_a: Any, group_b: Any) -> Dict:
        """
        Tests for a significant difference in Margin (Profit/Loss)
        between two groups using the T-Test. (Used for Hypothesis 3).
        """
        data_a = self.df[self.df[group_col] == group_a]['Margin']
        data_b = self.df[self.df[group_col] == group_b]['Margin']

        if data_a.shape[0] < 5 or data_b.shape[0] < 5:
            return {"Test": "Margin (T-Test)", "Interpretation": "Insufficient data for groups."}

        # T-Test: (Group A Margin) vs (Group B Margin)
        stat, p_value = t_test(data_a, data_b)

        # Interpret the result
        insight = translate_insight(
            test_name=f"T-Test on Margin for {group_col}: {group_a} vs {group_b}",
            p_value=p_value,
        )
        insight['Avg Margin A'] = f"{group_a}: R{data_a.mean():.2f}"
        insight['Avg Margin B'] = f"{group_b}: R{data_b.mean():.2f}"
        insight['P-Value'] = f"{p_value:.4f}"
        return insight

    # --------------------------------------------------------------------------
    # MAIN ORCHESTRATION METHOD
    # --------------------------------------------------------------------------

    def run_hypothesis_test(self, feature_col: str, groups: List[Any], metric: str = 'Risk') -> List[Dict]:
        """
        Runs the required tests for the specified feature column and groups.
        - metric='Risk': Runs Frequency and Severity tests (Hypotheses 1, 2, 4)
        - metric='Margin': Runs only the Margin test (Hypothesis 3)
        """
        if len(groups) != 2:
            print(f"Error: run_hypothesis_test expects exactly two groups for A/B testing on {feature_col}.")
            return []

        group_a, group_b = groups
        results = []

        if metric == 'Risk':
            # 1. Test Claim Frequency (Risk Metric 1)
            results.append(self.test_frequency(feature_col, group_a, group_b))

            # 2. Test Claim Severity (Risk Metric 2)
            results.append(self.test_severity(feature_col, group_a, group_b))

        elif metric == 'Margin':
            # 3. Test Margin (Hypothesis 3)
            results.append(self.test_margin(feature_col, group_a, group_b))

        return results


# Example Usage (This would be in your main script or Jupyter notebook)
# ---------------------------------------------------------------------
"""
# 1. Load Data
from src.data_loader import load_data
df = load_data('data/Raw/MachineLearningRating_v3.txt') 

# 2. Instantiate Tester
tester = ABTester(df)

# 3. Run Hypothesis 4: Gender
print("\\n--- Running Hypothesis 4: Gender ---")
gender_results = tester.run_hypothesis_test(
    feature_col='Gender', 
    groups=['Female', 'Male'], # Ensure these match your unique values
    metric='Risk'
)
for res in gender_results:
    print(res)

# 4. Run Hypothesis 3: Margin (example for two zip codes)
print("\\n--- Running Hypothesis 3: Margin (Zip Code Example) ---")
margin_results = tester.run_hypothesis_test(
    feature_col='PostalCode',
    groups=[2000, 3000], # Replace with actual high/low zip codes from your EDA
    metric='Margin'
)
for res in margin_results:
    print(res)
"""