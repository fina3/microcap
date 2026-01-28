#!/usr/bin/env python3
"""
Pull options data for top-ranked micro-cap predictions.

Screens for liquid options chains on high-conviction picks.
Most micro-caps won't have options - this identifies the few that do.
"""

import sys
sys.path.insert(0, 'src')

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from screening.options_screener import (
    screen_options_batch,
    get_options_summary,
    MIN_OPEN_INTEREST,
    MAX_BID_ASK_SPREAD_PCT
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


def load_top_predictions(top_n: int = 20) -> pd.DataFrame:
    """Load top N predictions by rank."""
    try:
        predictions_file = find_latest_file('predictions_*.csv')
        df = pd.read_csv(predictions_file)

        if 'rank' in df.columns:
            df = df.sort_values('rank').head(top_n)
        else:
            df = df.sort_values('confidence', ascending=False).head(top_n)

        return df
    except FileNotFoundError:
        logger.error("No predictions file found")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description='Screen top predictions for options availability'
    )

    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of top predictions to screen (default: 20)'
    )

    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        help='Specific tickers to check (overrides --top)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/options_candidates.csv',
        help='Output CSV file (default: data/options_candidates.csv)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("OPTIONS LIQUIDITY SCREENER")
    print("=" * 70)
    print(f"\nFilters:")
    print(f"  - Open Interest >= {MIN_OPEN_INTEREST}")
    print(f"  - Bid-Ask Spread <= {MAX_BID_ASK_SPREAD_PCT}%")
    print(f"  - Expiry: 30-60 days out (or nearest available)")
    print(f"  - Strike: ATM (within 5% of current price)")

    # Get tickers to screen
    if args.tickers:
        tickers = args.tickers
        print(f"\nScreening {len(tickers)} specified tickers...")
    else:
        predictions_df = load_top_predictions(args.top)
        if predictions_df.empty:
            print("\nERROR: No predictions found to screen")
            sys.exit(1)

        tickers = predictions_df['ticker'].tolist()
        print(f"\nScreening top {len(tickers)} ranked predictions...")

    print("-" * 70)

    # Screen for options
    results = screen_options_batch(tickers)

    # Generate summary
    summary = get_options_summary(results)

    print("\n" + "-" * 70)
    print("OPTIONS SCREENING SUMMARY")
    print("-" * 70)

    print(f"\nTotal screened:     {summary['total_screened']}")
    print(f"Have options:       {summary['has_options']} ({summary['pct_with_options']:.1f}%)")
    print(f"Liquid options:     {summary['liquid_options']} ({summary['pct_liquid']:.1f}%)")

    # Show candidates with liquid options
    candidates = [r for r in results if r.options_available]
    if candidates:
        print(f"\n" + "=" * 70)
        print("OPTIONS CANDIDATES")
        print("=" * 70)
        print(f"\n{'Ticker':<8} {'Price':>8} {'Strike':>8} {'Expiry':<12} {'Premium':>8} {'OI':>8} {'IV':>8}")
        print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

        for c in candidates:
            iv_str = f"{c.implied_vol:.0f}%" if c.implied_vol else "N/A"
            premium_str = f"${c.call_premium:.2f}" if c.call_premium else "N/A"
            print(
                f"{c.ticker:<8} "
                f"${c.current_price:>6.2f} "
                f"${c.nearest_strike:>6.2f} "
                f"{c.expiry:<12} "
                f"{premium_str:>8} "
                f"{c.open_interest:>8} "
                f"{iv_str:>8}"
            )
    else:
        print("\n  No tickers with liquid options found.")
        print("  (This is expected for most micro-caps)")

    # Show tickers with options but illiquid
    has_options_illiquid = [r for r in results if r.has_options and not r.options_available]
    if has_options_illiquid:
        print(f"\n  Options exist but illiquid: {', '.join([r.ticker for r in has_options_illiquid])}")

    # Save results
    output_df = pd.DataFrame([r.to_dict() for r in results])
    output_df.to_csv(args.output, index=False)

    print("\n" + "=" * 70)
    print(f"Results saved to: {args.output}")
    print("=" * 70 + "\n")

    return results


if __name__ == '__main__':
    main()
