#!/usr/bin/env python3
"""
Portfolio Builder CLI - Build a portfolio from predictions with risk management.

Usage:
    python build_portfolio.py                    # Build from latest predictions
    python build_portfolio.py --capital 50000   # Use $50,000 capital
    python build_portfolio.py --min-confidence 0.80  # Only 80%+ confidence
    python build_portfolio.py --predictions data/raw/predictions_20260119.csv

Outputs recommended portfolio with position sizes, stop losses, and take profits.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.portfolio.risk_manager import RiskManager, PositionRecommendation

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
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
        logger.warning(f"Error fetching price for {ticker}: {e}")
    return None


def get_sector(ticker: str) -> str:
    """Fetch sector for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('sector', 'Unknown')
    except Exception:
        return 'Unknown'


def load_predictions(predictions_file: Path) -> pd.DataFrame:
    """Load predictions from CSV file."""
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    df = pd.read_csv(predictions_file)
    logger.info(f"Loaded {len(df)} predictions from {predictions_file}")
    return df


def find_latest_predictions() -> Path:
    """Find the most recent predictions file."""
    predictions_dir = Path("data/raw")
    prediction_files = sorted(predictions_dir.glob("predictions_*.csv"), reverse=True)

    if not prediction_files:
        raise FileNotFoundError("No predictions files found in data/raw/")

    return prediction_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Build portfolio from predictions with risk management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Total portfolio capital (default: $10,000)'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        help='Path to predictions CSV (default: latest in data/raw/)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.50,
        help='Minimum confidence to include (default: 0.50)'
    )
    parser.add_argument(
        '--max-positions',
        type=int,
        default=10,
        help='Maximum number of positions (default: 10)'
    )
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=0.15,
        help='Stop loss percentage below entry (default: 0.15)'
    )
    parser.add_argument(
        '--take-profit',
        type=float,
        default=0.30,
        help='Take profit percentage above entry (default: 0.30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for portfolio (optional)'
    )
    parser.add_argument(
        '--fetch-prices',
        action='store_true',
        help='Fetch live prices instead of using predictions file prices'
    )
    parser.add_argument(
        '--fetch-sectors',
        action='store_true',
        help='Fetch sector data for concentration limits'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PORTFOLIO BUILDER")
    print("=" * 80)

    # Load predictions
    if args.predictions:
        predictions_file = Path(args.predictions)
    else:
        predictions_file = find_latest_predictions()

    print(f"\nUsing predictions: {predictions_file}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Min confidence: {args.min_confidence:.0%}")
    print(f"Max positions: {args.max_positions}")
    print(f"Stop loss: {args.stop_loss:.0%}")
    print(f"Take profit: {args.take_profit:.0%}")

    df = load_predictions(predictions_file)

    # Convert to list of dicts for RiskManager
    predictions = []
    for _, row in df.iterrows():
        pred = {
            'ticker': row['ticker'],
            'prediction': int(row['prediction']),
            'confidence': float(row['confidence']),
            'predicted_direction': row.get('predicted_direction', 'UNKNOWN'),
        }

        # Get price
        if args.fetch_prices:
            print(f"  Fetching price for {row['ticker']}...", end=' ')
            price = get_current_price(row['ticker'])
            if price:
                pred['current_price'] = price
                print(f"${price:.2f}")
            else:
                # Use actual_return_52w to estimate price if available
                print("FAILED")
                continue
        else:
            # Try to get price from file or fetch
            if 'current_price' in row and pd.notna(row['current_price']):
                pred['current_price'] = float(row['current_price'])
            else:
                # Fetch price for this ticker
                price = get_current_price(row['ticker'])
                if price:
                    pred['current_price'] = price
                else:
                    logger.warning(f"Could not get price for {row['ticker']}, skipping")
                    continue

        # Get sector
        if args.fetch_sectors:
            print(f"  Fetching sector for {row['ticker']}...", end=' ')
            sector = get_sector(row['ticker'])
            pred['sector'] = sector
            print(sector)

        predictions.append(pred)

    print(f"\n{len(predictions)} predictions with valid prices")

    # Initialize risk manager
    risk_manager = RiskManager(
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit
    )

    # Build portfolio
    recommendations = risk_manager.build_portfolio(
        predictions=predictions,
        portfolio_value=args.capital,
        min_confidence=args.min_confidence
    )

    # Print summary
    risk_manager.print_portfolio_summary(recommendations, args.capital)

    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_df = pd.DataFrame([r.to_dict() for r in recommendations])
        output_df.to_csv(output_path, index=False)
        print(f"\nPortfolio saved to: {output_path}")

    # Also save to default location
    today = datetime.now(pytz.utc).strftime('%Y%m%d')
    default_output = Path(f"data/raw/portfolio_{today}.csv")
    output_df = pd.DataFrame([r.to_dict() for r in recommendations])
    output_df.to_csv(default_output, index=False)
    print(f"Portfolio saved to: {default_output}")

    return recommendations


if __name__ == "__main__":
    main()
