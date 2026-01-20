"""
Pull micro-cap stock universe from Finviz.

CLI script to build a universe of US micro-cap stocks meeting criteria:
- Market Cap: $50M - $500M
- Country: USA
- Average Volume: > 50,000
- Excludes: Financials, REITs
"""

import sys
sys.path.append('src')

import argparse
from datetime import datetime
import pytz
import logging

from data.universe_screener import UniverseScreener

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for universe screening."""
    parser = argparse.ArgumentParser(
        description='Screen for micro-cap stock universe from Finviz'
    )
    parser.add_argument(
        '--as-of-date',
        type=str,
        default=None,
        help='As-of date in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for CSV file (default: data/raw)'
    )

    args = parser.parse_args()

    # Parse as_of_date
    if args.as_of_date:
        as_of_date = datetime.strptime(args.as_of_date, '%Y-%m-%d')
        as_of_date = pytz.utc.localize(as_of_date)
    else:
        as_of_date = datetime.now(pytz.utc)

    print("\n" + "="*70)
    print("MICRO-CAP UNIVERSE SCREENER")
    print("="*70)
    print(f"Using data available as of {as_of_date.date()} to build universe")
    print("="*70 + "\n")

    # Create screener and run
    screener = UniverseScreener(as_of_date=as_of_date)
    df = screener.scrape_universe()

    if df.empty:
        print("\nNo stocks found matching criteria!")
        return 1

    # Save results
    output_path = screener.save_universe(df, args.output_dir)

    # Print summary
    print("\n" + "="*70)
    print("UNIVERSE SCREENING COMPLETE")
    print("="*70)
    print(f"Total stocks in universe: {len(df)}")
    print(f"Output saved to: {output_path}")
    print("="*70 + "\n")

    # Show first 10 tickers
    print("Sample tickers:")
    for ticker in df['ticker'].head(10).tolist():
        print(f"  {ticker}")
    if len(df) > 10:
        print(f"  ... and {len(df) - 10} more")

    return 0


if __name__ == '__main__':
    sys.exit(main())
