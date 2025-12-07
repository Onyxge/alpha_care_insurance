import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a professional style for the plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def check_columns(df, required):
    """
    Verifies that the dataframe contains the necessary columns for plotting.
    Returns True if valid, False (and prints missing) if invalid.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: The following columns are missing from the data: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    return True


def generate_key_plots(df, save_dir='reports/figures'):
    """
    Generates the 3 initial key visualizations required for the AlphaCare interim report.
    1. Geographic Risk (Loss Ratio by Province)
    2. Claim Severity Distribution (Boxen Plot)
    3. Monthly Profitability Timeline
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define required columns for these specific plots
    required_cols = ['TotalClaims', 'TotalPremium', 'Province', 'VehicleType', 'TransactionMonth']

    if not check_columns(df, required_cols):
        return

    # --- PLOT 1: Geographic Risk Analysis (Loss Ratio by Province) ---
    print("Generating Plot 1: Risk Exposure by Province...")
    plt.figure(figsize=(12, 6))

    # Calculate aggregate metrics per province
    province_risk = df.groupby('Province')[['TotalClaims', 'TotalPremium']].sum().reset_index()

    # Avoid division by zero
    province_risk['AggregateLossRatio'] = province_risk['TotalClaims'] / province_risk['TotalPremium'].replace(0, 1)
    province_risk = province_risk.sort_values('AggregateLossRatio', ascending=False)

    sns.barplot(x='AggregateLossRatio', y='Province', data=province_risk, palette='viridis', hue='Province',
                legend=False)
    plt.axvline(x=1.0, color='r', linestyle='--', label='Breakeven Point (1.0)')

    plt.title('Plot 1: Risk Exposure by Province (Aggregate Loss Ratio)', fontsize=16)
    plt.xlabel('Loss Ratio (Claims / Premium)', fontsize=12)
    plt.ylabel('Province', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/1_risk_by_province.png")
    plt.show()

    # --- PLOT 2: Outlier Detection & Risk by Vehicle Type ---
    print("Generating Plot 2: Claim Severity by Vehicle Type...")
    plt.figure(figsize=(14, 7))

    # Filter for positive claims only (severity)
    claims_only = df[df['TotalClaims'] > 0].copy()

    # Use boxenplot for better outlier visualization on large datasets
    sns.boxenplot(x='VehicleType', y='TotalClaims', data=claims_only, palette='rocket', hue='VehicleType', legend=False)

    plt.title('Plot 2: Claim Severity Distribution by Vehicle Type', fontsize=16)
    plt.xticks(rotation=45)
    plt.xlabel('Vehicle Type', fontsize=12)
    plt.ylabel('Total Claim Amount (ZAR)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/2_severity_by_vehicle.png")
    plt.show()

    # --- PLOT 3: Monthly Profitability Timeline ---
    print("Generating Plot 3: Monthly Margin...")
    plt.figure(figsize=(14, 6))

    # Resample by month
    monthly_stats = df.set_index('TransactionMonth').resample('ME')[['TotalPremium', 'TotalClaims']].sum()

    # Plotting
    sns.lineplot(data=monthly_stats[['TotalPremium', 'TotalClaims']], markers=True, dashes=False)

    # Fill areas for profit/loss visualization
    plt.fill_between(monthly_stats.index, monthly_stats['TotalClaims'], monthly_stats['TotalPremium'],
                     where=(monthly_stats['TotalPremium'] > monthly_stats['TotalClaims']),
                     interpolate=True, color='green', alpha=0.1, label='Profit')
    plt.fill_between(monthly_stats.index, monthly_stats['TotalClaims'], monthly_stats['TotalPremium'],
                     where=(monthly_stats['TotalPremium'] <= monthly_stats['TotalClaims']),
                     interpolate=True, color='red', alpha=0.1, label='Loss')

    plt.title('Plot 3: Monthly Profitability Timeline', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Amount (ZAR)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/3_monthly_profitability.png")
    plt.show()

def generate_comprehensive_stats(df, save_dir='reports/figures'):
    """
    Calculates and visualizes the specific metrics defined in the brief:
    1. Claim Frequency
    2. Claim Severity
    3. Loss Ratio
    Grouped by Gender and PostalCode.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 1. PRE-CALCULATIONS ---
    # Create a flag for "Has Claim" (1 if TotalClaims > 0, else 0)
    df['HasClaim'] = df['TotalClaims'].apply(lambda x: 1 if x > 0 else 0)

    # Filter for severity (only positive claims)
    claims_only = df[df['TotalClaims'] > 0]

    # --- 2. GENDER ANALYSIS (Risk differences) ---
    print("\n--- GENDER ANALYSIS ---")
    gender_stats = df.groupby('Gender').agg({
        'HasClaim': 'mean',  # This is Claim Frequency
        'TotalClaims': 'sum',
        'TotalPremium': 'sum'
    }).rename(columns={'HasClaim': 'ClaimFrequency'})

    # Calculate Loss Ratio and Severity manually
    gender_stats['LossRatio'] = gender_stats['TotalClaims'] / gender_stats['TotalPremium']
    gender_stats['AvgClaimSeverity'] = claims_only.groupby('Gender')['TotalClaims'].mean()

    print(gender_stats[['ClaimFrequency', 'AvgClaimSeverity', 'LossRatio']])

    # Plot Gender Risk
    plt.figure(figsize=(12, 6))

    # Subplot 1: Frequency
    plt.subplot(1, 2, 1)
    sns.barplot(x=gender_stats.index, y='ClaimFrequency', data=gender_stats, palette='pastel', hue=gender_stats.index,
                legend=False)
    plt.title('Claim Frequency by Gender')
    plt.ylabel('Frequency (Claims per Policy)')

    # Subplot 2: Severity
    plt.subplot(1, 2, 2)
    sns.barplot(x=gender_stats.index, y='AvgClaimSeverity', data=gender_stats, palette='pastel', hue=gender_stats.index,
                legend=False)
    plt.title('Avg Claim Severity by Gender')
    plt.ylabel('Severity (Avg Amount)')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/4_gender_risk.png")
    plt.show()

    # --- 3. POSTAL CODE ANALYSIS (Top 10 Riskiest Areas) ---
    print("\n--- POSTAL CODE ANALYSIS (Top 10 by Volume) ---")
    # We filter for postal codes with significant volume to avoid noise
    zip_stats = df.groupby('PostalCode').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'HasClaim': 'count'  # Total policies
    })

    # Filter for zips with at least 100 policies to be statistically valid
    zip_stats = zip_stats[zip_stats['HasClaim'] > 100].copy()

    zip_stats['LossRatio'] = zip_stats['TotalClaims'] / zip_stats['TotalPremium']

    # Sort by Loss Ratio desc
    top_risky_zips = zip_stats.sort_values('LossRatio', ascending=False).head(10)

    print("Top 10 Riskiest Postal Codes (High Loss Ratio):")
    print(top_risky_zips[['LossRatio', 'TotalPremium']])

    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_risky_zips.index, y='LossRatio', data=top_risky_zips, palette='Reds_d', hue=top_risky_zips.index,
                legend=False)
    plt.axhline(1.0, color='k', linestyle='--', label='Breakeven')
    plt.title('Top 10 Riskiest Postal Codes (Loss Ratio > 1.0)')
    plt.ylabel('Loss Ratio')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/5_zipcode_risk.png")
    plt.show()

    print(f"All stats and plots saved to {save_dir}/")