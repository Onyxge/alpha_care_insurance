from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, chi2_contingency
import numpy as np
import pandas as pd

def t_test(a, b):
    try:
        return ttest_ind(a.dropna(), b.dropna(), equal_var=False)
    except Exception as e:
        print(f"T-test error: {e}")
        return None


def mann_whitney(a, b):
    try:
        return mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")
    except Exception as e:
        print(f"Mann-Whitney error: {e}")
        return None


def bootstrap_ci(a, b, n=3000):
    try:
        a = a.dropna().values
        b = b.dropna().values

        diffs = []
        size = min(len(a), len(b))

        for _ in range(n):
            da = np.random.choice(a, size, replace=True)
            db = np.random.choice(b, size, replace=True)
            diffs.append(da.mean() - db.mean())

        return np.percentile(diffs, [2.5, 97.5])

    except Exception as e:
        print(f"Bootstrap CI error: {e}")
        return None


def run_anova(*groups):
    try:
        cleaned = [g.dropna() for g in groups]
        return f_oneway(*cleaned)
    except Exception as e:
        print(f"ANOVA error: {e}")
        return None


def chi_square(df, col1, col2):
    try:
        table = pd.crosstab(df[col1], df[col2])
        return chi2_contingency(table)
    except Exception as e:
        print(f"Chi-square error: {e}")
        return None


def cohens_d(a, b):
    try:
        a, b = a.dropna(), b.dropna()
        pooled = np.sqrt((a.var() + b.var()) / 2)
        return (a.mean() - b.mean()) / pooled
    except Exception as e:
        print(f"Cohen's d error: {e}")
        return None


def translate_insight(test_name, p_value, effect=None):
    try:
        significant = p_value < 0.05

        return {
            "What": f"{test_name} found p={p_value:.4f}",
            "Interpretation": "Significant difference" if significant else "No significant difference",
            "EffectSize": effect,
            "BusinessAction": (
                "Investigate driver variables and adjust pricing or risk models."
                if significant
                else "No immediate business action required."
            )
        }

    except Exception as e:
        print(f"Insight translation error: {e}")
        return None
