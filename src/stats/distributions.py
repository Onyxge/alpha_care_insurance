import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, skew, kurtosis

def plot_distribution(series, title="Distribution"):
    try:
        data = series.dropna()
        if data.empty:
            raise ValueError("Series is empty. Cannot plot distribution.")

        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True)
        plt.title(title)
        plt.tight_layout()
        return plt

    except Exception as e:
        print(f"Error plotting distribution: {e}")
        return None


def distribution_summary(series):
    try:
        data = series.dropna()
        return {
            "mean": data.mean(),
            "median": data.median(),
            "std": data.std(),
            "skewness": skew(data),
            "kurtosis": kurtosis(data)
        }
    except Exception as e:
        print(f"Error computing distribution summary: {e}")
        return None


def normality_test(series):
    try:
        data = series.dropna()

        # Shapiro limit is about 5000 rows for reliability
        sample_size = min(len(data), 5000)
        sample = data.sample(sample_size, random_state=42)

        stat, p = shapiro(sample)
        return stat, p

    except Exception as e:
        print(f"Error performing normality test: {e}")
        return None
