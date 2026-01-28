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


def get_iwc_benchmark_data() -> dict | None:
    """
    Fetch IWC (iShares Micro-Cap ETF) data for benchmarking.

    Returns dict with:
        - current_price: Latest closing price
        - week_ago_price: Price from 7 days ago
        - weekly_change_pct: Percent change over the week
        - month_ago_price: Price from 30 days ago
        - monthly_change_pct: Percent change over the month
    """
    if not HAS_YFINANCE:
        return None

    try:
        iwc = yf.Ticker("IWC")
        # Get 35 days of history to ensure we have enough data
        hist = iwc.history(period="35d")

        if hist.empty or len(hist) < 2:
            return None

        current_price = hist['Close'].iloc[-1]

        # Find price from ~7 days ago
        week_ago_idx = max(0, len(hist) - 6)  # Approximately 5 trading days
        week_ago_price = hist['Close'].iloc[week_ago_idx]
        weekly_change_pct = ((current_price - week_ago_price) / week_ago_price) * 100

        # Find price from ~30 days ago
        month_ago_idx = 0  # Start of our 35-day window
        month_ago_price = hist['Close'].iloc[month_ago_idx]
        monthly_change_pct = ((current_price - month_ago_price) / month_ago_price) * 100

        return {
            'current_price': current_price,
            'week_ago_price': week_ago_price,
            'weekly_change_pct': weekly_change_pct,
            'month_ago_price': month_ago_price,
            'monthly_change_pct': monthly_change_pct
        }

    except Exception as e:
        print(f"  Warning: Could not fetch IWC data: {e}")
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


def load_rank_history() -> pd.DataFrame | None:
    """Load rank history for movement tracking."""
    history_file = "data/tracking/rank_history.csv"
    if not os.path.exists(history_file):
        return None
    try:
        return pd.read_csv(history_file)
    except Exception:
        return None


def get_previous_rank(ticker: str, history_df: pd.DataFrame | None) -> int | None:
    """Get previous week's rank for a ticker."""
    if history_df is None or history_df.empty:
        return None

    # Get the second most recent date's data for this ticker
    ticker_history = history_df[history_df['ticker'] == ticker].copy()
    if ticker_history.empty:
        return None

    ticker_history['date'] = pd.to_datetime(ticker_history['date'])
    ticker_history = ticker_history.sort_values('date', ascending=False)

    # Skip current date, get previous
    if len(ticker_history) >= 2:
        return int(ticker_history.iloc[1]['rank'])
    return None


def format_rank_change(current_rank: int, prev_rank: int | None) -> str:
    """Format rank change for display."""
    if prev_rank is None:
        return "NEW"
    diff = prev_rank - current_rank  # Positive = improved (moved up)
    if diff > 0:
        return f"+{diff}"
    elif diff < 0:
        return str(diff)
    else:
        return "-"


def load_liquidity_data() -> pd.DataFrame | None:
    """Load latest liquidity data."""
    liquidity_file = get_latest_file("data/raw/liquidity_*.csv")
    if not liquidity_file:
        return None
    return load_csv_safe(liquidity_file)


def print_top_picks():
    """Print top 20 ranked stocks with rank movement and liquidity."""
    print_section("TOP 20 RANKED STOCKS")

    # Load predictions (sorted by rank)
    predictions_file = get_latest_file("data/raw/predictions_*.csv")

    if not predictions_file:
        print("  No predictions data found")
        return

    predictions_df = load_csv_safe(predictions_file)
    if predictions_df is None or predictions_df.empty:
        print("  No predictions data found")
        return

    # Check if rank column exists
    if 'rank' not in predictions_df.columns:
        print("  Predictions file missing rank column - run predictions again")
        return

    # Sort by rank and take top 20
    predictions_df = predictions_df.sort_values('rank').head(20)

    # Load rank history for movement tracking
    history_df = load_rank_history()

    # Load liquidity data
    liquidity_df = load_liquidity_data()
    liquidity_map = {}
    if liquidity_df is not None:
        for _, row in liquidity_df.iterrows():
            liquidity_map[row['ticker']] = {
                'spread_pct': row.get('spread_pct'),
                'liquidity_grade': row.get('liquidity_grade', '?')
            }

    # Track poor liquidity warnings
    poor_liquidity_count = 0

    # Print header
    print(f"  {'Rank':<5} {'Ticker':<7} {'Conf%':>6} {'Move':>5} {'Spread':>7} {'Liq':>4}")
    print(f"  {'-'*5} {'-'*7} {'-'*6} {'-'*5} {'-'*7} {'-'*4}")

    for _, row in predictions_df.iterrows():
        ticker = row['ticker']
        rank = int(row['rank'])
        confidence = row['confidence'] * 100

        # Get rank movement
        prev_rank = get_previous_rank(ticker, history_df)
        rank_change = format_rank_change(rank, prev_rank)

        # Get liquidity info
        liq_info = liquidity_map.get(ticker, {})
        spread = liq_info.get('spread_pct')
        grade = liq_info.get('liquidity_grade', '?')

        if spread is not None:
            spread_str = f"{spread:.1f}%"
        else:
            spread_str = "N/A"

        # Flag poor liquidity (D or F grades)
        if grade in ('D', 'F'):
            poor_liquidity_count += 1
            grade_str = f"*{grade}*"  # Mark with asterisks
        else:
            grade_str = f" {grade} "

        print(f"  {rank:<5} {ticker:<7} {confidence:>5.1f}% {rank_change:>5} {spread_str:>7} {grade_str:>4}")

    # Show liquidity warning
    if poor_liquidity_count > 0:
        print()
        print(f"  * {poor_liquidity_count} stocks have poor liquidity (grade D/F) - trade with caution")

    # Show if no liquidity data
    if not liquidity_map:
        print()
        print("  Note: Run 'python pull_liquidity_data.py' to get spread data")


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


def print_benchmark_comparison():
    """Print portfolio performance vs IWC benchmark."""
    print_section("PERFORMANCE VS BENCHMARK (IWC)")

    if not HAS_YFINANCE:
        print("  yfinance not available - cannot fetch benchmark data")
        return

    # Get IWC benchmark data
    iwc_data = get_iwc_benchmark_data()
    if not iwc_data:
        print("  Could not fetch IWC benchmark data")
        return

    # Calculate portfolio performance from paper trades
    trades_file = "data/tracking/paper_trades.csv"
    portfolio_weekly_return = None
    portfolio_total_return = None

    if os.path.exists(trades_file):
        df = load_csv_safe(trades_file)
        if df is not None and not df.empty:
            now = datetime.now(pytz.UTC)

            # Get active trades only
            def is_active(row):
                if pd.notna(row.get('is_completed')) and row['is_completed']:
                    return False
                try:
                    target = pd.to_datetime(row['target_date'])
                    if target.tzinfo is None:
                        target = target.tz_localize('UTC')
                    return now < target
                except Exception:
                    return True

            active_df = df[df.apply(is_active, axis=1)]

            if not active_df.empty:
                # Calculate returns for each active position
                returns = []
                for _, row in active_df.iterrows():
                    entry_price = row['entry_price']
                    current_price = get_current_price(row['ticker'])
                    if current_price is None:
                        current_price = row.get('current_price', entry_price)

                    if entry_price and entry_price > 0:
                        ret = ((current_price - entry_price) / entry_price) * 100
                        returns.append(ret)

                if returns:
                    portfolio_total_return = sum(returns) / len(returns)

    # Display comparison
    print(f"\n  {'Metric':<25} {'Portfolio':>12} {'IWC':>12} {'Alpha':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    # Weekly change
    iwc_weekly = iwc_data['weekly_change_pct']
    print(f"  {'Weekly Change':<25} {'N/A':>12} {iwc_weekly:>+11.2f}% {'-':>12}")

    # Monthly change
    iwc_monthly = iwc_data['monthly_change_pct']
    print(f"  {'Monthly Change':<25} {'N/A':>12} {iwc_monthly:>+11.2f}% {'-':>12}")

    # Portfolio return (since entry)
    if portfolio_total_return is not None:
        alpha = portfolio_total_return - iwc_monthly
        alpha_str = f"{alpha:>+11.2f}%"
        port_str = f"{portfolio_total_return:>+11.2f}%"
        print(f"  {'Portfolio Return (avg)':<25} {port_str} {iwc_monthly:>+11.2f}% {alpha_str}")
    else:
        print(f"  {'Portfolio Return (avg)':<25} {'N/A':>12} {iwc_monthly:>+11.2f}% {'-':>12}")

    # IWC current price
    print(f"\n  IWC Current Price: ${iwc_data['current_price']:.2f}")

    # Performance indicator
    if portfolio_total_return is not None:
        if portfolio_total_return > iwc_monthly:
            print(f"  Status: OUTPERFORMING benchmark by {portfolio_total_return - iwc_monthly:+.2f}%")
        elif portfolio_total_return < iwc_monthly:
            print(f"  Status: UNDERPERFORMING benchmark by {iwc_monthly - portfolio_total_return:.2f}%")
        else:
            print(f"  Status: MATCHING benchmark")


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
    print_benchmark_comparison()
    print_alerts()
    print_quick_actions()

    print()


if __name__ == "__main__":
    main()
