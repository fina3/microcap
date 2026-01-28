#!/usr/bin/env python3
"""
Wednesday Mid-Week Report.

Shows:
- Rank movements since Monday
- Any new alerts
- Liquidity changes
- Stocks that moved > 10% since Monday

Usage:
    python wednesday_report.py

Output: Terminal display + saved to reports/wednesday_YYYYMMDD.txt
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

from tracking.forward_test import ForwardTestTracker


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


def get_monday_price(ticker: str) -> float | None:
    """Get Monday's price (approximately 2 days ago)."""
    if not HAS_YFINANCE:
        return None
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if len(hist) >= 3:
            # Monday is typically 2-3 trading days back from Wednesday
            return float(hist['Close'].iloc[-3])
    except Exception:
        pass
    return None


def load_rank_history() -> pd.DataFrame | None:
    """Load rank history for movement tracking."""
    history_file = Path("data/tracking/rank_history.csv")
    if history_file.exists():
        return pd.read_csv(history_file)
    return None


def load_predictions() -> pd.DataFrame | None:
    """Load current predictions."""
    pred_file = get_latest_file("predictions_*.csv")
    if pred_file and pred_file.exists():
        return pd.read_csv(pred_file)
    return None


def load_liquidity() -> pd.DataFrame | None:
    """Load liquidity data."""
    liq_file = get_latest_file("liquidity_*.csv")
    if liq_file and liq_file.exists():
        return pd.read_csv(liq_file)
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
    output_lines = []

    # Header
    header = f"""
{'=' * 60}
 WEDNESDAY MID-WEEK REPORT
 {now.strftime('%Y-%m-%d %H:%M')}
{'=' * 60}"""
    print(header)
    output_lines.append(header)

    # Forward Test Status
    tracker = ForwardTestTracker()
    print_section("FORWARD TEST STATUS", output=output_lines)
    print_line(f"  {tracker.get_countdown_string()}", output_lines)

    # Big Movers (>10% since Monday)
    print_section("BIG MOVERS (>10% since Monday)", output=output_lines)

    predictions_df = load_predictions()
    big_movers = []

    if predictions_df is not None:
        for _, row in predictions_df.head(50).iterrows():  # Check top 50
            ticker = row['ticker']
            monday_price = get_monday_price(ticker)
            current_price = get_price(ticker)

            if monday_price and current_price and monday_price > 0:
                change_pct = ((current_price - monday_price) / monday_price) * 100

                if abs(change_pct) >= 10:
                    big_movers.append({
                        'ticker': ticker,
                        'rank': int(row['rank']),
                        'monday': monday_price,
                        'current': current_price,
                        'change': change_pct
                    })

    if big_movers:
        # Sort by absolute change
        big_movers.sort(key=lambda x: abs(x['change']), reverse=True)

        print_line(f"  {'Ticker':<8} {'Rank':>5} {'Monday':>8} {'Current':>8} {'Change':>8}", output_lines)
        print_line(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*8}", output_lines)

        for mover in big_movers[:10]:
            direction = "UP" if mover['change'] > 0 else "DOWN"
            print_line(
                f"  {mover['ticker']:<8} {mover['rank']:>5} ${mover['monday']:>6.2f} ${mover['current']:>6.2f} {mover['change']:>+7.1f}% {direction}",
                output_lines
            )
    else:
        print_line("  No stocks moved >10% since Monday", output_lines)

    # Rank Movements
    print_section("RANK MOVEMENTS (Top 20)", output=output_lines)

    rank_history_df = load_rank_history()

    if rank_history_df is not None and not rank_history_df.empty:
        rank_history_df['date'] = pd.to_datetime(rank_history_df['date'])

        # Get unique dates sorted
        dates = sorted(rank_history_df['date'].unique(), reverse=True)

        if len(dates) >= 2:
            latest_date = dates[0]
            prev_date = dates[1]

            latest_ranks = rank_history_df[rank_history_df['date'] == latest_date].set_index('ticker')['rank'].to_dict()
            prev_ranks = rank_history_df[rank_history_df['date'] == prev_date].set_index('ticker')['rank'].to_dict()

            movements = []
            for ticker, current_rank in latest_ranks.items():
                if current_rank <= 20:  # Only top 20
                    prev_rank = prev_ranks.get(ticker)
                    if prev_rank is not None:
                        change = prev_rank - current_rank  # Positive = improved
                        if change != 0:
                            movements.append({
                                'ticker': ticker,
                                'current_rank': current_rank,
                                'prev_rank': prev_rank,
                                'change': change
                            })
                    else:
                        movements.append({
                            'ticker': ticker,
                            'current_rank': current_rank,
                            'prev_rank': None,
                            'change': None
                        })

            if movements:
                # Sort by absolute change
                movements.sort(key=lambda x: abs(x['change']) if x['change'] else 0, reverse=True)

                print_line(f"  {'Ticker':<8} {'Current':>8} {'Previous':>8} {'Change':>8}", output_lines)
                print_line(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}", output_lines)

                for m in movements[:10]:
                    if m['change'] is not None:
                        change_str = f"{m['change']:+d}" if m['change'] != 0 else "="
                    else:
                        change_str = "NEW"
                    prev_str = str(m['prev_rank']) if m['prev_rank'] else "-"
                    print_line(
                        f"  {m['ticker']:<8} {m['current_rank']:>8} {prev_str:>8} {change_str:>8}",
                        output_lines
                    )
            else:
                print_line("  No rank changes in top 20", output_lines)
        else:
            print_line("  Need at least 2 weeks of data for rank comparison", output_lines)
    else:
        print_line("  No rank history available", output_lines)

    # Alerts
    print_section("ALERTS", output=output_lines)

    alerts = []
    paper_trades_df = load_paper_trades()

    if paper_trades_df is not None and not paper_trades_df.empty:
        for _, row in paper_trades_df.iterrows():
            if row.get('is_completed', False):
                continue

            ticker = row['ticker']
            entry_price = row['entry_price']
            current_price = get_price(ticker)

            if current_price is None or entry_price is None or entry_price == 0:
                continue

            ret = ((current_price - entry_price) / entry_price) * 100

            # Stop loss: -15%
            if ret <= -15:
                alerts.append(f"  STOP LOSS HIT: {ticker} at {ret:+.1f}%")

            # Take profit: +30%
            elif ret >= 30:
                alerts.append(f"  TAKE PROFIT HIT: {ticker} at {ret:+.1f}%")

            # Warning zones
            elif -15 < ret <= -10:
                alerts.append(f"  WARNING: {ticker} at {ret:+.1f}% - approaching stop")

            elif 25 <= ret < 30:
                alerts.append(f"  NEAR TARGET: {ticker} at {ret:+.1f}% - approaching take profit")

    # Check for liquidity changes
    liquidity_df = load_liquidity()
    if liquidity_df is not None and predictions_df is not None:
        top20_tickers = predictions_df.head(20)['ticker'].tolist()

        for _, row in liquidity_df.iterrows():
            if row['ticker'] in top20_tickers:
                grade = row.get('liquidity_grade', '')
                if grade in ('D', 'F'):
                    alerts.append(f"  LIQUIDITY: {row['ticker']} has poor liquidity (grade {grade})")

    if alerts:
        for alert in alerts:
            print_line(alert, output_lines)
    else:
        print_line("  No alerts", output_lines)

    # Liquidity Summary
    print_section("LIQUIDITY CHECK (Top 20)", output=output_lines)

    if liquidity_df is not None and predictions_df is not None:
        top20_tickers = predictions_df.head(20)['ticker'].tolist()
        top20_liquidity = liquidity_df[liquidity_df['ticker'].isin(top20_tickers)]

        if not top20_liquidity.empty:
            grade_counts = top20_liquidity['liquidity_grade'].value_counts().to_dict()

            print_line("  Grade distribution:", output_lines)
            for grade in ['A', 'B', 'C', 'D', 'F']:
                count = grade_counts.get(grade, 0)
                if count > 0:
                    warning = " (CAUTION)" if grade in ('D', 'F') else ""
                    print_line(f"    {grade}: {count} stocks{warning}", output_lines)

            # Average spread
            avg_spread = top20_liquidity['spread_pct'].mean()
            if pd.notna(avg_spread):
                print_line(f"\n  Average spread: {avg_spread:.2f}%", output_lines)
        else:
            print_line("  No liquidity data for top 20", output_lines)
    else:
        print_line("  No liquidity data available", output_lines)
        print_line("  Run 'python pull_liquidity_data.py' to fetch", output_lines)

    # Paper Trading Status
    print_section("PAPER TRADING STATUS", output=output_lines)

    if paper_trades_df is not None and not paper_trades_df.empty:
        active_trades = paper_trades_df[~paper_trades_df['is_completed'].fillna(False)]
        completed_trades = paper_trades_df[paper_trades_df['is_completed'].fillna(False)]

        print_line(f"  Active positions: {len(active_trades)}", output_lines)
        print_line(f"  Completed trades: {len(completed_trades)}", output_lines)

        if not active_trades.empty:
            # Calculate unrealized P&L
            returns = []
            for _, row in active_trades.iterrows():
                current = get_price(row['ticker'])
                if current and row['entry_price'] > 0:
                    ret = ((current - row['entry_price']) / row['entry_price']) * 100
                    returns.append(ret)

            if returns:
                avg_return = sum(returns) / len(returns)
                print_line(f"  Unrealized avg return: {avg_return:+.1f}%", output_lines)
    else:
        print_line("  No paper trades logged yet", output_lines)

    # Next Steps
    print_section("NEXT STEPS", output=output_lines)
    print_line("  Saturday: Run full weekly analysis", output_lines)
    print_line("    python run_weekly_analysis.py --limit 150", output_lines)
    print_line("    python forward_test_report.py", output_lines)
    print_line("    python dashboard.py", output_lines)

    # Footer
    footer = f"""
{'=' * 60}
 Time to complete: ~2-3 minutes
 Next check: Saturday (full analysis)
{'=' * 60}
"""
    print(footer)
    output_lines.append(footer)

    # Save report
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / f"wednesday_{now.strftime('%Y%m%d')}.txt"
    report_file.write_text("\n".join(output_lines))
    print(f"Report saved: {report_file}")


if __name__ == "__main__":
    main()
