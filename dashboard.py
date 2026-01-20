#!/usr/bin/env python3
"""
Micro-Cap Prediction System Dashboard
Displays summary of system status, predictions, and paper trading performance.
"""

import glob
import os
from datetime import datetime, timedelta
import pandas as pd
import pytz

# Optional: for live price updates
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def get_latest_file(pattern: str) -> str | None:
    """Find the most recent file matching a glob pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time (newest first) or by date in filename
    return max(files, key=os.path.getmtime)


def extract_date_from_filename(filepath: str) -> str | None:
    """Extract date string from filename like 'universe_20260120.csv'."""
    if not filepath:
        return None
    basename = os.path.basename(filepath)
    # Try to find YYYYMMDD pattern
    import re
    match = re.search(r'(\d{8})', basename)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return None


def file_age_days(filepath: str) -> int | None:
    """Return age of file in days."""
    if not filepath or not os.path.exists(filepath):
        return None
    mtime = os.path.getmtime(filepath)
    age = datetime.now() - datetime.fromtimestamp(mtime)
    return age.days


def load_csv_safe(filepath: str) -> pd.DataFrame | None:
    """Load CSV file safely, returning None if file doesn't exist."""
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        return None


def get_current_price(ticker: str) -> float | None:
    """Get current price for a ticker using yfinance."""
    if not HAS_YFINANCE:
        return None
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except Exception:
        pass
    return None


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_data_status():
    """Print data status section."""
    print_section("DATA STATUS")

    # Universe
    universe_file = get_latest_file("data/raw/universe_*.csv")
    if universe_file:
        df = load_csv_safe(universe_file)
        count = len(df) if df is not None else 0
        date = extract_date_from_filename(universe_file)
        print(f"  Universe:   {count:>4} stocks (updated {date})")
    else:
        print("  Universe:   No data found")

    # Metrics
    metrics_file = get_latest_file("data/raw/microcap_metrics_*.csv")
    if metrics_file:
        df = load_csv_safe(metrics_file)
        count = len(df) if df is not None else 0
        date = extract_date_from_filename(metrics_file)
        print(f"  Metrics:    {count:>4} stocks (updated {date})")
    else:
        print("  Metrics:    No data found")

    # Sentiment
    sentiment_file = get_latest_file("data/raw/sentiment_scores_*.csv")
    if sentiment_file:
        df = load_csv_safe(sentiment_file)
        count = len(df) if df is not None else 0
        date = extract_date_from_filename(sentiment_file)
        print(f"  Sentiment:  {count:>4} stocks (updated {date})")
    else:
        print("  Sentiment:  No data found")


def print_top_picks():
    """Print top 5 current picks section."""
    print_section("TOP 5 CURRENT PICKS")

    # Load portfolio (buy recommendations)
    portfolio_file = get_latest_file("data/raw/portfolio_*.csv")
    metrics_file = get_latest_file("data/raw/microcap_metrics_*.csv")

    if not portfolio_file:
        print("  No portfolio data found")
        return

    portfolio_df = load_csv_safe(portfolio_file)
    if portfolio_df is None or portfolio_df.empty:
        print("  No portfolio data found")
        return

    # Filter to BUY actions and sort by confidence
    buys = portfolio_df[portfolio_df['action'] == 'BUY'].copy()
    if buys.empty:
        print("  No BUY recommendations found")
        return

    buys = buys.sort_values('confidence', ascending=False).head(5)

    # Print header
    print(f"  {'Ticker':<8} {'Direction':<12} {'Conf%':>7} {'Sector':<20} {'Price':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*7} {'-'*20} {'-'*10}")

    for _, row in buys.iterrows():
        ticker = row['ticker']
        confidence = row['confidence'] * 100
        sector = row.get('sector', 'N/A')[:20]
        price = row.get('current_price', 0)

        print(f"  {ticker:<8} {'OUTPERFORM':<12} {confidence:>6.1f}% {sector:<20} ${price:>8.2f}")


def print_paper_trading_status():
    """Print paper trading status section."""
    print_section("PAPER TRADING STATUS")

    trades_file = "data/tracking/paper_trades.csv"
    if not os.path.exists(trades_file):
        print("  No paper trades file found")
        return

    df = load_csv_safe(trades_file)
    if df is None or df.empty:
        print("  No paper trades logged")
        return

    total_trades = len(df)

    # Determine completed vs active trades
    now = datetime.now(pytz.UTC)

    def is_completed(row):
        if pd.notna(row.get('is_completed')) and row['is_completed']:
            return True
        try:
            target = pd.to_datetime(row['target_date'])
            if target.tzinfo is None:
                target = target.tz_localize('UTC')
            return now >= target
        except Exception:
            return False

    df['_is_completed'] = df.apply(is_completed, axis=1)
    active_df = df[~df['_is_completed']]
    completed_df = df[df['_is_completed']]

    active_count = len(active_df)
    completed_count = len(completed_df)

    print(f"  Total trades logged:  {total_trades}")
    print(f"  Active trades:        {active_count}")
    print(f"  Completed trades:     {completed_count}")

    # Calculate current returns for active trades
    if not active_df.empty and HAS_YFINANCE:
        print()
        print("  Active Positions:")
        print(f"  {'Ticker':<8} {'Entry':>8} {'Current':>8} {'Return':>8} {'Target Date':<12}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

        for _, row in active_df.head(10).iterrows():
            ticker = row['ticker']
            entry_price = row['entry_price']

            # Get current price
            current_price = get_current_price(ticker)
            if current_price is None:
                current_price = row.get('current_price', entry_price)

            if entry_price and entry_price > 0:
                ret = ((current_price - entry_price) / entry_price) * 100
            else:
                ret = 0

            # Format target date
            try:
                target_date = pd.to_datetime(row['target_date']).strftime('%Y-%m-%d')
            except Exception:
                target_date = 'N/A'

            ret_str = f"{ret:>+7.1f}%"
            print(f"  {ticker:<8} ${entry_price:>6.2f} ${current_price:>6.2f} {ret_str} {target_date}")

    # Win rate for completed trades
    if not completed_df.empty:
        print()
        # Check if we have winner info
        if 'is_winner' in completed_df.columns:
            winners = completed_df['is_winner'].sum()
            win_rate = (winners / len(completed_df)) * 100
            print(f"  Win rate (completed): {win_rate:.1f}% ({int(winners)}/{len(completed_df)})")
        elif 'final_return_pct' in completed_df.columns:
            winners = (completed_df['final_return_pct'] > 0).sum()
            win_rate = (winners / len(completed_df)) * 100
            print(f"  Win rate (completed): {win_rate:.1f}% ({winners}/{len(completed_df)})")


def print_alerts():
    """Print alerts section."""
    print_section("ALERTS")

    alerts = []

    # Check paper trades for stop loss / take profit
    trades_file = "data/tracking/paper_trades.csv"
    if os.path.exists(trades_file):
        df = load_csv_safe(trades_file)
        if df is not None and not df.empty:
            now = datetime.now(pytz.UTC)

            for _, row in df.iterrows():
                # Check if trade is still active
                try:
                    target = pd.to_datetime(row['target_date'])
                    if target.tzinfo is None:
                        target = target.tz_localize('UTC')
                    if now >= target:
                        continue  # Skip completed trades
                except Exception:
                    pass

                ticker = row['ticker']
                entry_price = row['entry_price']

                # Get current price
                if HAS_YFINANCE:
                    current_price = get_current_price(ticker)
                else:
                    current_price = row.get('current_price', entry_price)

                if current_price is None or entry_price is None or entry_price == 0:
                    continue

                ret = ((current_price - entry_price) / entry_price) * 100

                # Stop loss threshold: -15%
                if current_price < entry_price * 0.85:
                    alerts.append(f"  ⚠ {ticker} hit stop loss ({ret:+.1f}%)")

                # Take profit threshold: +30%
                elif current_price > entry_price * 1.30:
                    alerts.append(f"  ✓ {ticker} hit take profit ({ret:+.1f}%)")

    # Check data staleness
    data_files = [
        ("Universe", "data/raw/universe_*.csv"),
        ("Metrics", "data/raw/microcap_metrics_*.csv"),
        ("Sentiment", "data/raw/sentiment_scores_*.csv"),
        ("Predictions", "data/raw/predictions_*.csv"),
    ]

    stale_files = []
    for name, pattern in data_files:
        latest = get_latest_file(pattern)
        if latest:
            age = file_age_days(latest)
            if age is not None and age > 7:
                stale_files.append(name)

    if stale_files:
        alerts.append(f"  ℹ Data is stale - run weekly analysis ({', '.join(stale_files)})")

    # Print alerts
    if alerts:
        for alert in alerts:
            print(alert)
    else:
        print("  No alerts")


def print_quick_actions():
    """Print suggested quick actions based on status."""
    print_section("QUICK ACTIONS")

    actions = []

    # Check if data is stale
    metrics_file = get_latest_file("data/raw/microcap_metrics_*.csv")
    if metrics_file:
        age = file_age_days(metrics_file)
        if age is not None and age > 7:
            actions.append("  → python run_weekly.py     # Update all data and predictions")

    # Check if paper trades exist
    trades_file = "data/tracking/paper_trades.csv"
    if os.path.exists(trades_file):
        df = load_csv_safe(trades_file)
        if df is not None and len(df) > 0:
            actions.append("  → python update_paper_trades.py  # Update current prices")

    # Always show backtest option
    actions.append("  → python run_backtest.py   # Run walk-forward validation")

    # Show portfolio generation
    portfolio_file = get_latest_file("data/raw/portfolio_*.csv")
    if not portfolio_file or file_age_days(portfolio_file) > 1:
        actions.append("  → python generate_portfolio.py  # Generate new portfolio")

    if actions:
        for action in actions:
            print(action)
    else:
        print("  System is up to date. No immediate actions needed.")


def main():
    """Main entry point."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " MICRO-CAP PREDICTION SYSTEM DASHBOARD ".center(58) + "║")
    print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M')} ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    if not HAS_YFINANCE:
        print("\n  Note: yfinance not installed - using cached prices")

    print_data_status()
    print_top_picks()
    print_paper_trading_status()
    print_alerts()
    print_quick_actions()

    print()


if __name__ == "__main__":
    main()
