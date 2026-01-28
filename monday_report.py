#!/usr/bin/env python3
"""
Monday Morning Report - Quick 2-minute status check.

Shows:
- Current positions and weekend price changes
- Any alerts triggered (stop loss, take profit)
- Forward test countdown
- Quick summary of system status

Usage:
    python monday_report.py

Output: Terminal display + saved to reports/monday_YYYYMMDD.txt
"""

import sys
sys.path.insert(0, 'src')

import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import pytz

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from tracking.forward_test import ForwardTestTracker, FORWARD_TEST_WEEKS


def get_latest_file(pattern: str, directory: str = "data/raw") -> Path | None:
    """Find most recent file matching pattern."""
    files = list(Path(directory).glob(pattern))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def get_price(ticker: str) -> float | None:
    """Get current price for a ticker."""
    if not HAS_YFINANCE:
        return None
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None


def get_friday_price(ticker: str) -> float | None:
    """Get Friday's closing price for weekend change calculation."""
    if not HAS_YFINANCE:
        return None
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if len(hist) >= 2:
            # Friday is typically the second-to-last trading day
            return float(hist['Close'].iloc[-2])
    except Exception:
        pass
    return None


def load_portfolio() -> pd.DataFrame | None:
    """Load current portfolio."""
    portfolio_file = get_latest_file("portfolio_*.csv")
    if portfolio_file and portfolio_file.exists():
        return pd.read_csv(portfolio_file)
    return None


def load_paper_trades() -> pd.DataFrame | None:
    """Load paper trades."""
    trades_file = Path("data/tracking/paper_trades.csv")
    if trades_file.exists():
        return pd.read_csv(trades_file)
    return None


def print_section(title: str, char: str = "-", output: list = None):
    """Print a section header."""
    line = f"\n{char * 60}\n {title}\n{char * 60}"
    print(line)
    if output is not None:
        output.append(line)


def print_line(text: str, output: list = None):
    """Print a line and optionally append to output."""
    print(text)
    if output is not None:
        output.append(text)


def main():
    now = datetime.now(pytz.utc)
    date_str = now.strftime('%Y%m%d')
    output_lines = []

    # Header
    header = f"""
{'=' * 60}
 MONDAY MORNING REPORT
 {now.strftime('%Y-%m-%d %H:%M')}
{'=' * 60}"""
    print(header)
    output_lines.append(header)

    # Forward Test Countdown
    tracker = ForwardTestTracker()
    print_section("FORWARD TEST STATUS", output=output_lines)
    countdown = tracker.get_countdown_string()
    print_line(f"  {countdown}", output_lines)

    week_num = tracker.get_current_week()
    if week_num > 0:
        stats = tracker.get_accuracy_stats()
        if stats.get('valid_for_evaluation', 0) > 0:
            acc = stats.get('overall_accuracy', 0)
            print_line(f"  Current accuracy: {acc*100:.1f}%", output_lines)
            print_line(f"  Predictions tracked: {stats['total_predictions']}", output_lines)
        else:
            print_line(f"  Predictions tracked: {stats.get('total_predictions', 0)}", output_lines)
            print_line("  Accuracy: pending price updates", output_lines)

    # Current Positions & Weekend Changes
    print_section("CURRENT POSITIONS", output=output_lines)

    portfolio_df = load_portfolio()
    paper_trades_df = load_paper_trades()

    if portfolio_df is not None and not portfolio_df.empty:
        print_line(f"  {'Ticker':<8} {'Entry':>8} {'Current':>8} {'Weekend':>8} {'Total':>8}", output_lines)
        print_line(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", output_lines)

        for _, row in portfolio_df.head(10).iterrows():
            ticker = row['ticker']
            entry_price = row.get('current_price', row.get('entry_price', 0))

            current_price = get_price(ticker)
            friday_price = get_friday_price(ticker)

            if current_price is None:
                current_price = entry_price

            # Weekend change
            if friday_price and friday_price > 0:
                weekend_change = ((current_price - friday_price) / friday_price) * 100
                weekend_str = f"{weekend_change:+.1f}%"
            else:
                weekend_str = "N/A"

            # Total change from entry
            if entry_price and entry_price > 0:
                total_change = ((current_price - entry_price) / entry_price) * 100
                total_str = f"{total_change:+.1f}%"
            else:
                total_str = "N/A"

            print_line(
                f"  {ticker:<8} ${entry_price:>6.2f} ${current_price:>6.2f} {weekend_str:>8} {total_str:>8}",
                output_lines
            )
    else:
        print_line("  No portfolio positions found", output_lines)
        print_line("  Run 'python run_weekly_analysis.py' to generate portfolio", output_lines)

    # Alerts
    print_section("ALERTS", output=output_lines)

    alerts = []

    if paper_trades_df is not None and not paper_trades_df.empty:
        for _, row in paper_trades_df.iterrows():
            # Skip completed trades
            if row.get('is_completed', False):
                continue

            ticker = row['ticker']
            entry_price = row['entry_price']

            current_price = get_price(ticker)
            if current_price is None:
                continue

            if entry_price and entry_price > 0:
                ret = ((current_price - entry_price) / entry_price) * 100

                # Stop loss: -15%
                if ret <= -15:
                    alerts.append(f"  STOP LOSS: {ticker} at {ret:+.1f}% (entry ${entry_price:.2f} -> ${current_price:.2f})")

                # Take profit: +30%
                elif ret >= 30:
                    alerts.append(f"  TAKE PROFIT: {ticker} at {ret:+.1f}% (entry ${entry_price:.2f} -> ${current_price:.2f})")

                # Warning: approaching stop (-10% to -15%)
                elif -15 < ret <= -10:
                    alerts.append(f"  WARNING: {ticker} at {ret:+.1f}% - approaching stop loss")

    if alerts:
        for alert in alerts:
            print_line(alert, output_lines)
    else:
        print_line("  No alerts - all positions within normal range", output_lines)

    # Data Freshness Check
    print_section("DATA STATUS", output=output_lines)

    data_checks = [
        ("Predictions", "predictions_*.csv"),
        ("Metrics", "microcap_metrics_*.csv"),
        ("Portfolio", "portfolio_*.csv"),
    ]

    for name, pattern in data_checks:
        latest = get_latest_file(pattern)
        if latest:
            age_days = (now - datetime.fromtimestamp(latest.stat().st_mtime, tz=pytz.utc)).days
            date_str_file = latest.stem.split('_')[-1]
            status = "OK" if age_days <= 7 else "STALE"
            print_line(f"  {name:<12}: {date_str_file} ({age_days}d old) [{status}]", output_lines)
        else:
            print_line(f"  {name:<12}: NOT FOUND", output_lines)

    # Quick Actions
    print_section("QUICK ACTIONS", output=output_lines)

    predictions_file = get_latest_file("predictions_*.csv")
    if predictions_file:
        age = (now - datetime.fromtimestamp(predictions_file.stat().st_mtime, tz=pytz.utc)).days
        if age > 7:
            print_line("  -> Run 'python run_weekly_analysis.py' (data is stale)", output_lines)
        else:
            print_line("  -> Data is fresh. Check back Wednesday.", output_lines)
    else:
        print_line("  -> Run 'python run_weekly_analysis.py' to initialize system", output_lines)

    if alerts:
        print_line("  -> Review alerts above and take action if needed", output_lines)

    # Footer
    footer = f"""
{'=' * 60}
 Time to complete: ~2 minutes
 Next check: Wednesday
{'=' * 60}
"""
    print(footer)
    output_lines.append(footer)

    # Save report
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / f"monday_{now.strftime('%Y%m%d')}.txt"
    report_file.write_text("\n".join(output_lines))
    print(f"Report saved: {report_file}")


if __name__ == "__main__":
    main()
