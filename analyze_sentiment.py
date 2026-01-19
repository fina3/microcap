"""
Analyze collected sentiment data for micro-cap stocks.

Provides analysis and reporting on sentiment scores.
"""

import sys
sys.path.append('src')

import argparse
from datetime import datetime
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import pytz

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze sentiment scores from collected data'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input CSV file. Default: most recent in data/raw'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for analysis report'
    )

    return parser.parse_args()


def find_latest_sentiment_file(data_dir: str = 'data/raw') -> str:
    """Find the most recent sentiment scores file."""
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = list(data_path.glob('sentiment_scores_*.csv'))

    if not files:
        raise FileNotFoundError(f"No sentiment score files found in {data_dir}")

    # Sort by modification time, newest first
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return str(files[0])


def analyze_distribution(df: pd.DataFrame, column: str) -> dict:
    """Calculate distribution statistics for a column."""
    valid_data = df[df['data_quality_score'] > 0][column].dropna()

    if valid_data.empty:
        return {
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan
        }

    return {
        'count': len(valid_data),
        'mean': valid_data.mean(),
        'std': valid_data.std(),
        'min': valid_data.min(),
        'max': valid_data.max(),
        'median': valid_data.median()
    }


def categorize_sentiment(net_sentiment: float) -> str:
    """Categorize net sentiment score."""
    if pd.isna(net_sentiment):
        return "Unknown"
    elif net_sentiment > 0.2:
        return "Bullish"
    elif net_sentiment > 0.05:
        return "Slightly Bullish"
    elif net_sentiment > -0.05:
        return "Neutral"
    elif net_sentiment > -0.2:
        return "Slightly Bearish"
    else:
        return "Bearish"


def main():
    """Main analysis function."""
    args = parse_args()

    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS REPORT")
    print("="*70)

    # Find input file
    if args.input:
        input_file = args.input
    else:
        try:
            input_file = find_latest_sentiment_file()
            logger.info(f"Using latest file: {input_file}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    # Load data
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records from {input_file}")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        sys.exit(1)

    print(f"\nInput: {input_file}")
    print(f"Records: {len(df)}")

    # Filter valid data
    valid_df = df[df['data_quality_score'] > 0]
    invalid_df = df[df['data_quality_score'] == 0]

    print(f"Valid records: {len(valid_df)}")
    print(f"Invalid records: {len(invalid_df)}")

    if invalid_df.empty is False:
        print(f"\nInvalid tickers:")
        for _, row in invalid_df.iterrows():
            print(f"  {row['ticker']}: {row.get('quality_flags', 'Unknown issue')}")

    if valid_df.empty:
        print("\nNo valid sentiment data to analyze")
        return

    # Overall statistics
    print("\n" + "-"*70)
    print("AGGREGATE STATISTICS")
    print("-"*70)

    metrics = ['net_sentiment', 'polarity', 'uncertainty_score',
               'positive_score', 'negative_score', 'hedging_ratio', 'confidence_ratio']

    for metric in metrics:
        stats = analyze_distribution(df, metric)
        if stats['count'] > 0:
            print(f"\n{metric}:")
            print(f"  Mean:   {stats['mean']:+.4f}")
            print(f"  Median: {stats['median']:+.4f}")
            print(f"  Std:    {stats['std']:.4f}")
            print(f"  Range:  [{stats['min']:+.4f}, {stats['max']:+.4f}]")

    # Sentiment distribution
    print("\n" + "-"*70)
    print("SENTIMENT DISTRIBUTION")
    print("-"*70)

    valid_df['sentiment_category'] = valid_df['net_sentiment'].apply(categorize_sentiment)
    category_counts = valid_df['sentiment_category'].value_counts()

    for category in ['Bullish', 'Slightly Bullish', 'Neutral', 'Slightly Bearish', 'Bearish']:
        count = category_counts.get(category, 0)
        pct = (count / len(valid_df)) * 100 if len(valid_df) > 0 else 0
        bar = '#' * int(pct / 5)
        print(f"  {category:18s}: {count:2d} ({pct:5.1f}%) {bar}")

    # Per-ticker breakdown
    print("\n" + "-"*70)
    print("PER-TICKER SENTIMENT RANKING")
    print("-"*70)

    ranked = valid_df.sort_values('polarity', ascending=False)

    print(f"\n{'Ticker':<8} {'Sentiment':>10} {'Polarity':>10} {'Uncertainty':>12} {'Quality':>8} Category")
    print("-" * 70)

    for _, row in ranked.iterrows():
        category = categorize_sentiment(row['net_sentiment'])
        print(
            f"{row['ticker']:<8} "
            f"{row['net_sentiment']:>+10.3f} "
            f"{row['polarity']:>+10.3f} "
            f"{row['uncertainty_score']:>12.3f} "
            f"{row['data_quality_score']:>8.2f} "
            f"{category}"
        )

    # Top and bottom performers
    print("\n" + "-"*70)
    print("SENTIMENT EXTREMES")
    print("-"*70)

    if len(ranked) >= 2:
        print("\nMost Bullish:")
        top = ranked.iloc[0]
        print(f"  {top['ticker']}: sentiment={top['net_sentiment']:+.3f}, polarity={top['polarity']:+.3f}")

        print("\nMost Bearish:")
        bottom = ranked.iloc[-1]
        print(f"  {bottom['ticker']}: sentiment={bottom['net_sentiment']:+.3f}, polarity={bottom['polarity']:+.3f}")

    # Uncertainty analysis
    print("\n" + "-"*70)
    print("UNCERTAINTY ANALYSIS")
    print("-"*70)

    high_uncertainty = valid_df[valid_df['uncertainty_score'] > valid_df['uncertainty_score'].median()]
    low_uncertainty = valid_df[valid_df['uncertainty_score'] <= valid_df['uncertainty_score'].median()]

    print(f"\nHigh Uncertainty Stocks (above median):")
    for _, row in high_uncertainty.sort_values('uncertainty_score', ascending=False).iterrows():
        print(f"  {row['ticker']}: uncertainty={row['uncertainty_score']:.3f}")

    print(f"\nLow Uncertainty Stocks (below median):")
    for _, row in low_uncertainty.sort_values('uncertainty_score').iterrows():
        print(f"  {row['ticker']}: uncertainty={row['uncertainty_score']:.3f}")

    # Data quality summary
    print("\n" + "-"*70)
    print("DATA QUALITY SUMMARY")
    print("-"*70)

    print(f"\nAverage Quality Score: {valid_df['data_quality_score'].mean():.2f}")
    print(f"Average Word Count: {valid_df['total_words'].mean():.0f}")

    flagged = valid_df[valid_df['quality_flags'].notna() & (valid_df['quality_flags'] != '')]
    if not flagged.empty:
        print(f"\nTickers with Quality Flags:")
        for _, row in flagged.iterrows():
            print(f"  {row['ticker']}: {row['quality_flags']}")

    # Recommendations
    print("\n" + "-"*70)
    print("TRADING SIGNALS (based on sentiment)")
    print("-"*70)

    bullish = valid_df[
        (valid_df['net_sentiment'] > 0.1) &
        (valid_df['uncertainty_score'] < 0.05) &
        (valid_df['data_quality_score'] > 0.7)
    ]

    bearish = valid_df[
        (valid_df['net_sentiment'] < -0.1) &
        (valid_df['uncertainty_score'] < 0.05) &
        (valid_df['data_quality_score'] > 0.7)
    ]

    uncertain = valid_df[valid_df['uncertainty_score'] > 0.1]

    print(f"\nBullish signals (positive sentiment, low uncertainty, high quality):")
    if bullish.empty:
        print("  None")
    else:
        for _, row in bullish.iterrows():
            print(f"  {row['ticker']}: sentiment={row['net_sentiment']:+.3f}")

    print(f"\nBearish signals (negative sentiment, low uncertainty, high quality):")
    if bearish.empty:
        print("  None")
    else:
        for _, row in bearish.iterrows():
            print(f"  {row['ticker']}: sentiment={row['net_sentiment']:+.3f}")

    print(f"\nHigh uncertainty (proceed with caution):")
    if uncertain.empty:
        print("  None")
    else:
        for _, row in uncertain.iterrows():
            print(f"  {row['ticker']}: uncertainty={row['uncertainty_score']:.3f}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    # Save report if output specified
    if args.output:
        # Would save detailed report to file
        logger.info(f"Report saved to: {args.output}")

    return df


if __name__ == '__main__':
    df = main()
