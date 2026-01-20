#!/usr/bin/env python3
"""
Paper Trading CLI for tracking predictions without real money.

Usage:
    python paper_trade.py log          # Log today's predictions as paper trades
    python paper_trade.py update       # Update trades with current prices
    python paper_trade.py report       # Show performance summary
    python paper_trade.py log --ticker REVG --prediction 1 --confidence 0.95  # Log single trade

Commands:
    log     - Run predictions and log them as paper trades with entry prices
    update  - Check current prices and mark completed trades
    report  - Display performance summary and recent trades
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.tracking.paper_trader import PaperTrader
from src.models.predictor import MicroCapPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Clean output for CLI
)
logger = logging.getLogger(__name__)


def get_current_price(ticker: str) -> float:
    """Fetch current price for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {e}")
    return None


def log_predictions(args):
    """Log predictions as paper trades."""
    trader = PaperTrader(
        trades_file=Path(args.trades_file),
        target_days=args.target_days
    )

    if args.ticker:
        # Log a single manual trade
        if args.prediction is None or args.confidence is None:
            logger.error("--prediction and --confidence required for single ticker")
            return

        price = get_current_price(args.ticker)
        if price is None:
            logger.error(f"Could not fetch price for {args.ticker}")
            return

        trader.log_prediction(
            ticker=args.ticker,
            prediction=args.prediction,
            confidence=args.confidence,
            entry_price=price,
            notes=args.notes or ""
        )
        print(f"Logged paper trade for {args.ticker}")
        return

    # Otherwise, run predictions and log all of them
    print("\n" + "=" * 60)
    print("PAPER TRADING - LOG PREDICTIONS")
    print("=" * 60)

    # Find latest predictions file
    predictions_dir = Path("data/raw")
    prediction_files = sorted(predictions_dir.glob("predictions_*.csv"), reverse=True)

    if not prediction_files:
        logger.error("No predictions file found. Run run_prediction.py first.")
        return

    predictions_file = prediction_files[0]
    print(f"\nUsing predictions from: {predictions_file}")

    # Load predictions
    predictions_df = pd.read_csv(predictions_file)
    print(f"Found {len(predictions_df)} predictions")

    # Filter by confidence threshold
    if args.min_confidence:
        predictions_df = predictions_df[predictions_df['confidence'] >= args.min_confidence]
        print(f"After confidence filter (>={args.min_confidence}): {len(predictions_df)}")

    # Limit number of trades
    if args.limit:
        predictions_df = predictions_df.head(args.limit)
        print(f"Limited to: {len(predictions_df)} trades")

    logged = 0
    skipped = 0

    for _, row in predictions_df.iterrows():
        ticker = row['ticker']

        # Check if already logged today
        existing = [
            t for t in trader.trades
            if t.ticker == ticker and
            t.entry_date.date() == datetime.now(pytz.utc).date()
        ]
        if existing:
            print(f"  {ticker}: Already logged today, skipping")
            skipped += 1
            continue

        # Get current price
        price = get_current_price(ticker)
        if price is None:
            print(f"  {ticker}: Could not fetch price, skipping")
            skipped += 1
            continue

        trader.log_prediction(
            ticker=ticker,
            prediction=int(row['prediction']),
            predicted_direction=row['predicted_direction'],
            confidence=float(row['confidence']),
            entry_price=price,
            notes=f"Auto-logged from {predictions_file.name}"
        )

        print(f"  {ticker}: Logged {row['predicted_direction']} @ ${price:.2f} (conf: {row['confidence']:.1%})")
        logged += 1

    print(f"\n{'-' * 40}")
    print(f"Logged: {logged} trades")
    print(f"Skipped: {skipped}")
    print(f"Total trades in file: {len(trader.trades)}")
    print("=" * 60)


def update_trades(args):
    """Update trades with current prices."""
    print("\n" + "=" * 60)
    print("PAPER TRADING - UPDATE")
    print("=" * 60)

    trader = PaperTrader(trades_file=Path(args.trades_file))

    if not trader.trades:
        print("\nNo trades to update.")
        return

    print(f"\nUpdating {len(trader.trades)} trades...")

    results = trader.update_results()

    print(f"\n{'-' * 40}")
    print(f"Trades updated: {results['trades_updated']}")
    print(f"Trades completed: {results['trades_completed']}")

    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")

    print("=" * 60)


def show_report(args):
    """Show performance report."""
    trader = PaperTrader(trades_file=Path(args.trades_file))
    trader.print_report()


def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading System for tracking predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--trades-file',
        default='data/tracking/paper_trades.csv',
        help='Path to trades CSV file (default: data/tracking/paper_trades.csv)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Log command
    log_parser = subparsers.add_parser('log', help='Log predictions as paper trades')
    log_parser.add_argument(
        '--ticker',
        help='Single ticker to log (requires --prediction and --confidence)'
    )
    log_parser.add_argument(
        '--prediction',
        type=int,
        choices=[0, 1],
        help='Prediction: 1=outperform, 0=underperform'
    )
    log_parser.add_argument(
        '--confidence',
        type=float,
        help='Model confidence (0-1)'
    )
    log_parser.add_argument(
        '--notes',
        help='Optional notes for the trade'
    )
    log_parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.6,
        help='Minimum confidence to log (default: 0.6)'
    )
    log_parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Maximum number of trades to log (default: 10)'
    )
    log_parser.add_argument(
        '--target-days',
        type=int,
        default=90,
        help='Days until trade evaluation (default: 90)'
    )

    # Update command
    update_parser = subparsers.add_parser('update', help='Update trades with current prices')

    # Report command
    report_parser = subparsers.add_parser('report', help='Show performance summary')

    args = parser.parse_args()

    if args.command == 'log':
        log_predictions(args)
    elif args.command == 'update':
        update_trades(args)
    elif args.command == 'report':
        show_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
