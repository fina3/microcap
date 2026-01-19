"""
Quick analysis of collected micro-cap metrics.

Provides summary statistics and identifies interesting patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def analyze_metrics(csv_path: str):
    """Analyze collected metrics and print insights."""

    print("\n" + "="*70)
    print("MICRO-CAP METRICS ANALYSIS")
    print("="*70)

    # Load data
    df = pd.read_csv(csv_path)

    # Filter out tickers with no data
    valid_df = df[df['data_completeness_score'] > 0].copy()

    print(f"\nLoaded {len(df)} tickers ({len(valid_df)} with valid data)")

    if valid_df.empty:
        print("No valid data to analyze")
        return

    # Summary statistics
    print("\n" + "-"*70)
    print("VALUATION METRICS SUMMARY")
    print("-"*70)

    metrics = ['pe_trailing', 'pe_forward', 'pb_ratio', 'price_to_sales']

    for metric in metrics:
        if metric in valid_df.columns:
            data = valid_df[metric].dropna()
            if len(data) > 0:
                print(f"\n{metric.upper().replace('_', ' ')}:")
                print(f"  Mean: {data.mean():.2f}")
                print(f"  Median: {data.median():.2f}")
                print(f"  Min: {data.min():.2f}")
                print(f"  Max: {data.max():.2f}")

    # Financial health
    print("\n" + "-"*70)
    print("FINANCIAL HEALTH")
    print("-"*70)

    debt_metrics = ['debt_to_equity', 'total_debt', 'operating_cash_flow']

    for metric in debt_metrics:
        if metric in valid_df.columns:
            data = valid_df[metric].dropna()
            if len(data) > 0:
                print(f"\n{metric.upper().replace('_', ' ')}:")
                if metric == 'debt_to_equity':
                    print(f"  Mean: {data.mean():.2f}")
                    print(f"  Median: {data.median():.2f}")
                else:
                    print(f"  Mean: ${data.mean():,.0f}")
                    print(f"  Median: ${data.median():,.0f}")

    # Ownership structure
    print("\n" + "-"*70)
    print("OWNERSHIP STRUCTURE")
    print("-"*70)

    ownership_metrics = ['insider_ownership_pct', 'institutional_ownership_pct']

    for metric in ownership_metrics:
        if metric in valid_df.columns:
            data = valid_df[metric].dropna()
            if len(data) > 0:
                print(f"\n{metric.upper().replace('_', ' ')}:")
                print(f"  Mean: {data.mean():.2f}%")
                print(f"  Median: {data.median():.2f}%")
                print(f"  Min: {data.min():.2f}%")
                print(f"  Max: {data.max():.2f}%")

    # Short interest
    print("\n" + "-"*70)
    print("SHORT INTEREST")
    print("-"*70)

    if 'short_percent_float' in valid_df.columns:
        short_data = valid_df['short_percent_float'].dropna()
        if len(short_data) > 0:
            print(f"\nSHORT % OF FLOAT:")
            print(f"  Mean: {short_data.mean():.2f}%")
            print(f"  Median: {short_data.median():.2f}%")
            print(f"  Max: {short_data.max():.2f}% ({valid_df.loc[valid_df['short_percent_float'].idxmax(), 'ticker']})")

    # Performance
    print("\n" + "-"*70)
    print("52-WEEK PERFORMANCE")
    print("-"*70)

    if '52_week_price_change_pct' in valid_df.columns:
        perf_data = valid_df['52_week_price_change_pct'].dropna()
        if len(perf_data) > 0:
            print(f"\nMean return: {perf_data.mean():+.2f}%")
            print(f"Median return: {perf_data.median():+.2f}%")

            # Best and worst performers
            best_idx = perf_data.idxmax()
            worst_idx = perf_data.idxmin()

            print(f"\nBest performer: {valid_df.loc[best_idx, 'ticker']} ({perf_data[best_idx]:+.2f}%)")
            print(f"Worst performer: {valid_df.loc[worst_idx, 'ticker']} ({perf_data[worst_idx]:+.2f}%)")

    # Market cap distribution
    print("\n" + "-"*70)
    print("MARKET CAP DISTRIBUTION")
    print("-"*70)

    if 'market_cap' in valid_df.columns:
        mc_data = valid_df['market_cap'].dropna()
        if len(mc_data) > 0:
            print(f"\nMean: ${mc_data.mean():,.0f}")
            print(f"Median: ${mc_data.median():,.0f}")
            print(f"Min: ${mc_data.min():,.0f} ({valid_df.loc[mc_data.idxmin(), 'ticker']})")
            print(f"Max: ${mc_data.max():,.0f} ({valid_df.loc[mc_data.idxmax(), 'ticker']})")

    # Interesting patterns
    print("\n" + "-"*70)
    print("INTERESTING PATTERNS")
    print("-"*70)

    # High insider ownership + strong performance
    if 'insider_ownership_pct' in valid_df.columns and '52_week_price_change_pct' in valid_df.columns:
        high_insider = valid_df[valid_df['insider_ownership_pct'] > 10]
        if len(high_insider) > 0:
            avg_return = high_insider['52_week_price_change_pct'].mean()
            print(f"\nHigh insider ownership (>10%) stocks:")
            print(f"  Count: {len(high_insider)}")
            print(f"  Avg 52-week return: {avg_return:+.2f}%")
            for _, row in high_insider.iterrows():
                print(f"    {row['ticker']}: {row['insider_ownership_pct']:.1f}% insider, {row['52_week_price_change_pct']:+.1f}% return")

    # High short interest
    if 'short_percent_float' in valid_df.columns:
        high_short = valid_df[valid_df['short_percent_float'] > 5]
        if len(high_short) > 0:
            print(f"\nHigh short interest (>5%) stocks:")
            for _, row in high_short.iterrows():
                print(f"  {row['ticker']}: {row['short_percent_float']:.2f}% short")

    # Low valuation + strong performance
    if 'pe_trailing' in valid_df.columns and '52_week_price_change_pct' in valid_df.columns:
        low_pe = valid_df[(valid_df['pe_trailing'] < 15) & (valid_df['pe_trailing'] > 0)]
        if len(low_pe) > 0:
            avg_return = low_pe['52_week_price_change_pct'].mean()
            print(f"\nLow P/E (<15) stocks:")
            print(f"  Count: {len(low_pe)}")
            print(f"  Avg 52-week return: {avg_return:+.2f}%")
            for _, row in low_pe.iterrows():
                print(f"    {row['ticker']}: P/E={row['pe_trailing']:.1f}, return={row['52_week_price_change_pct']:+.1f}%")

    print("\n" + "="*70)


if __name__ == '__main__':
    # Find most recent metrics file
    import os
    from glob import glob

    metrics_files = glob('data/raw/microcap_metrics_*.csv')
    if metrics_files:
        latest_file = max(metrics_files, key=os.path.getctime)
        print(f"Analyzing: {latest_file}")
        analyze_metrics(latest_file)
    else:
        print("No metrics files found. Run pull_microcap_metrics.py first.")
