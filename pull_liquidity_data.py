#!/usr/bin/env python3
"""
Pull bid-ask spread data for liquidity analysis.

Collects real-time bid/ask quotes and calculates spreads
to identify stocks that may be too illiquid to trade.
"""

import sys
sys.path.insert(0, 'src')

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

from data.liquidity_checker import (
    check_liquidity_batch,
    get_liquidity_summary,
    LIQUIDITY_GRADES
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_file(pattern: str, directory: str = 'data/raw') -> str:
    """Find the most recent file matching pattern."""
    data_path = Path(directory)
    files = list(data_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0])


def load_tickers_from_predictions() -> list:
    """Load tickers from latest predictions file."""
    try:
        predictions_file = find_latest_file('predictions_*.csv')
        df = pd.read_csv(predictions_file)
        return df['ticker'].tolist()
    except FileNotFoundError:
        logger.warning("No predictions file found")
        return []


def load_tickers_from_universe() -> list:
    """Load tickers from latest universe file."""
    try:
        universe_file = find_latest_file('universe_*.csv')
        df = pd.read_csv(universe_file)
        return df['ticker'].tolist()
    except FileNotFoundError:
        logger.warning("No universe file found")
        return []


def main():
    parser = argparse.ArgumentParser(
        description='Pull bid-ask spread data for liquidity analysis'
    )

    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        help='Specific tickers to check'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=150,
        help='Limit number of tickers (default: 150)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file (default: data/raw/liquidity_YYYYMMDD.csv)'
    )

    parser.add_argument(
        '--source',
        type=str,
        choices=['predictions', 'universe'],
        default='predictions',
        help='Source for tickers (default: predictions)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("BID-ASK SPREAD LIQUIDITY ANALYZER")
    print("=" * 70)

    # Get tickers
    if args.tickers:
        tickers = args.tickers
    elif args.source == 'predictions':
        tickers = load_tickers_from_predictions()
        if not tickers:
            tickers = load_tickers_from_universe()
    else:
        tickers = load_tickers_from_universe()

    if not tickers:
        print("ERROR: No tickers found to analyze")
        sys.exit(1)

    # Apply limit
    if args.limit and len(tickers) > args.limit:
        tickers = tickers[:args.limit]

    print(f"\nAnalyzing {len(tickers)} tickers...")
    print("-" * 70)

    # Check liquidity
    results = check_liquidity_batch(tickers)

    # Generate summary
    summary = get_liquidity_summary(results)

    # Print summary
    print("\n" + "-" * 70)
    print("LIQUIDITY SUMMARY")
    print("-" * 70)

    print(f"\nTotal stocks analyzed: {summary['total']}")
    print(f"Tradeable (grades A-D): {summary['tradeable']}")
    print(f"Untradeable (grade F): {summary['untradeable']}")

    print(f"\nAverage spread: {summary['avg_spread_pct']}%")
    print(f"Median spread: {summary['median_spread_pct']}%")

    print("\nGrade Distribution:")
    for grade, count in summary['grade_counts'].items():
        desc = LIQUIDITY_GRADES[grade]['description']
        pct = (count / summary['total']) * 100 if summary['total'] > 0 else 0
        print(f"  {grade}: {count:3} stocks ({pct:5.1f}%) - {desc}")

    # Show problematic stocks
    if summary['poor_liquidity_tickers']:
        print(f"\nPoor Liquidity (D/F grades): {len(summary['poor_liquidity_tickers'])} stocks")
        grade_d = [r for r in results if r.liquidity_grade == 'D']
        grade_f = [r for r in results if r.liquidity_grade == 'F']

        if grade_d:
            print("\n  Grade D (spread 4-7%):")
            for r in grade_d[:10]:
                spread_str = f"{r.spread_pct:.1f}%" if r.spread_pct else "N/A"
                print(f"    {r.ticker:<6} - spread: {spread_str}")
            if len(grade_d) > 10:
                print(f"    ... and {len(grade_d) - 10} more")

        if grade_f:
            print("\n  Grade F (spread >7% or no data):")
            for r in grade_f[:10]:
                spread_str = f"{r.spread_pct:.1f}%" if r.spread_pct else "ILLIQUID"
                print(f"    {r.ticker:<6} - spread: {spread_str}")
            if len(grade_f) > 10:
                print(f"    ... and {len(grade_f) - 10} more")

    # Save results
    date_str = datetime.now().strftime('%Y%m%d')
    if args.output:
        output_file = args.output
    else:
        output_file = f'data/raw/liquidity_{date_str}.csv'

    output_df = pd.DataFrame([r.to_dict() for r in results])
    output_df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70 + "\n")

    return results


if __name__ == '__main__':
    main()
