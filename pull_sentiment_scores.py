"""
Pull sentiment scores for micro-cap tickers from SEC 8-K filings.

Fetches 8-K filings (Item 2.02) and analyzes earnings sentiment
using Loughran-McDonald financial dictionary.
"""

import sys
sys.path.append('src')

import argparse
from datetime import datetime
import logging

import pandas as pd
import pytz

from data.sentiment_collector import SentimentCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default micro-cap tickers to analyze
DEFAULT_TICKERS = [
    "AORT", "QCRH", "NBN", "TRS", "TCBX",
    "REVG", "RDVT", "NATR", "MTRN", "AIOT"
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect sentiment scores from SEC 8-K filings'
    )

    parser.add_argument(
        '--as-of-date',
        type=str,
        default=None,
        help='Analysis date (YYYY-MM-DD). Default: today'
    )

    parser.add_argument(
        '--tickers',
        type=str,
        default=None,
        help='Comma-separated list of tickers. Default: predefined list'
    )

    parser.add_argument(
        '--lookback-days',
        type=int,
        default=365,
        help='Days to look back for filings. Default: 365'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory. Default: data/raw'
    )

    parser.add_argument(
        '--user-agent',
        type=str,
        default='MicroCapAnalysis research@example.com',
        help='User-Agent for SEC requests'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "="*70)
    print("MICRO-CAP SENTIMENT COLLECTION")
    print("="*70)

    # Parse as-of date
    if args.as_of_date:
        try:
            as_of_date = datetime.strptime(args.as_of_date, '%Y-%m-%d')
            as_of_date = pytz.utc.localize(as_of_date)
        except ValueError:
            logger.error(f"Invalid date format: {args.as_of_date}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        as_of_date = datetime.now(pytz.utc)

    # Parse tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = DEFAULT_TICKERS

    print(f"\nConfiguration:")
    print(f"  As-of date: {as_of_date.date()}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Lookback: {args.lookback_days} days")
    print(f"  Output: {args.output_dir}")
    print("="*70)

    # Create collector
    collector = SentimentCollector(
        as_of_date=as_of_date,
        user_agent=args.user_agent
    )

    # Collect sentiment for all tickers
    df = collector.collect_all_tickers(
        tickers=tickers,
        lookback_days=args.lookback_days
    )

    # Print summary
    collector.print_summary(df)

    if df.empty:
        logger.warning("No sentiment data collected")
        print("\n" + "="*70)
        print("COLLECTION COMPLETE - NO DATA")
        print("="*70)
        return None

    # Save to CSV
    date_str = as_of_date.strftime('%Y%m%d')
    output_file = f'{args.output_dir}/sentiment_scores_{date_str}.csv'

    df.to_csv(output_file, index=False)
    logger.info(f"\nSentiment scores saved to: {output_file}")

    # Save human-readable summary
    summary_file = f'{args.output_dir}/sentiment_scores_{date_str}_summary.txt'

    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MICRO-CAP SENTIMENT COLLECTION SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"\nCollection Date: {as_of_date.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        f.write(f"Tickers: {', '.join(tickers)}\n")
        f.write(f"Lookback Period: {args.lookback_days} days\n")

        valid_df = df[df['data_quality_score'] > 0]
        f.write(f"\nValid Results: {len(valid_df)}/{len(df)}\n")

        if not valid_df.empty:
            f.write(f"Average Data Quality: {valid_df['data_quality_score'].mean():.2f}\n")
            f.write(f"\nAggregate Metrics:\n")
            f.write(f"  Average Net Sentiment: {valid_df['net_sentiment'].mean():+.3f}\n")
            f.write(f"  Average Uncertainty: {valid_df['uncertainty_score'].mean():.3f}\n")
            f.write(f"  Average Polarity: {valid_df['polarity'].mean():+.3f}\n")

        f.write("\n" + "-"*70 + "\n")
        f.write("PER-TICKER RESULTS\n")
        f.write("-"*70 + "\n")

        for _, row in df.iterrows():
            ticker = row['ticker']
            quality = row.get('data_quality_score', 0)

            if quality > 0:
                f.write(f"\n{ticker}:\n")
                f.write(f"  Filing Date: {row['filing_date']}\n")
                f.write(f"  Net Sentiment: {row['net_sentiment']:+.3f}\n")
                f.write(f"  Polarity: {row['polarity']:+.3f}\n")
                f.write(f"  Uncertainty: {row['uncertainty_score']:.3f}\n")
                f.write(f"  Data Quality: {quality:.2f}\n")
                f.write(f"  Word Count: {row.get('total_words', 0)}\n")

                flags = row.get('quality_flags', '')
                if flags:
                    f.write(f"  Quality Flags: {flags}\n")
            else:
                flags = row.get('quality_flags', 'UNKNOWN')
                f.write(f"\n{ticker}: NO DATA\n")
                f.write(f"  Reason: {flags}\n")

    logger.info(f"Summary saved to: {summary_file}")

    print("\n" + "="*70)
    print("COLLECTION COMPLETE")
    print("="*70)

    return df


if __name__ == '__main__':
    df = main()
