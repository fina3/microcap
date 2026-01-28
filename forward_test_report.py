#!/usr/bin/env python3
"""
Forward Test Performance Report.

Generates comprehensive performance report for the 9-week forward test:
- Overall accuracy (% of predictions correct)
- Top-20 vs Bottom-20 accuracy
- Average return vs IWC benchmark
- Best and worst calls
- Accuracy by confidence bucket
- Week-over-week trend
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime
from pathlib import Path
import pandas as pd
import pytz

from tracking.forward_test import ForwardTestTracker, FORWARD_TEST_WEEKS


def print_header(title: str, char: str = "="):
    """Print a section header."""
    print()
    print(char * 70)
    print(f" {title}")
    print(char * 70)


def format_pct(value, decimals: int = 1) -> str:
    """Format percentage value."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%" if abs(value) < 1 else f"{value:.{decimals}f}%"


def format_return(value, decimals: int = 1) -> str:
    """Format return value with sign."""
    if value is None:
        return "N/A"
    return f"{value:+.{decimals}f}%"


def main():
    print()
    print("#" * 70)
    print("#" + " FORWARD TEST PERFORMANCE REPORT ".center(68) + "#")
    print("#" + f" {datetime.now().strftime('%Y-%m-%d %H:%M')} ".center(68) + "#")
    print("#" * 70)

    # Initialize tracker
    tracker = ForwardTestTracker()

    # 9-Week Countdown
    print_header("9-WEEK FORWARD TEST STATUS")
    print(f"  {tracker.get_countdown_string()}")
    print(f"  Start date: {tracker.start_date.strftime('%Y-%m-%d')}")
    print(f"  End date:   {(tracker.start_date + pd.Timedelta(weeks=FORWARD_TEST_WEEKS)).strftime('%Y-%m-%d')}")

    # Update prices first
    print()
    print("  Updating current prices...")
    update_result = tracker.update_current_prices()
    print(f"  Updated: {update_result['updated']} predictions")
    if update_result['errors']:
        print(f"  Errors: {len(update_result['errors'])}")

    # Get accuracy stats
    stats = tracker.get_accuracy_stats()

    if 'error' in stats:
        print(f"\n  {stats['error']}")
        print("\n  Run 'python run_weekly_analysis.py --forward-test' to log predictions")
        return

    print_header("PREDICTION SUMMARY")
    print(f"  Total predictions logged:    {stats['total_predictions']}")
    print(f"  Valid for evaluation:        {stats['valid_for_evaluation']}")

    if stats['valid_for_evaluation'] == 0:
        print("\n  No predictions have been evaluated yet.")
        print("  Run this report again after prices have moved.")
        return

    # Overall Accuracy
    print_header("OVERALL ACCURACY")
    print(f"  Predictions that beat IWC:   {format_pct(stats['overall_accuracy'])}")
    print()
    print(f"  Top-20 accuracy:             {format_pct(stats['top20_accuracy'])} ({stats.get('top20_count', 0)} predictions)")
    print(f"  Bottom-20 accuracy:          {format_pct(stats.get('bottom20_accuracy'))} ({stats.get('bottom20_count', 0)} predictions)")
    print()
    print("  Note: Top-20 = predicted to outperform (want high accuracy)")
    print("        Bottom-20 = predicted to underperform (want to be correct about underperformance)")

    # Returns vs Benchmark
    print_header("RETURNS VS IWC BENCHMARK")
    print(f"  Top-20 avg return:           {format_return(stats.get('top20_avg_return_pct'))}")
    print(f"  IWC benchmark return:        {format_return(stats.get('avg_benchmark_return_pct'))}")
    print(f"  Excess return (alpha):       {format_return(stats.get('excess_return_pct'))}")

    if stats.get('excess_return_pct') is not None:
        if stats['excess_return_pct'] > 0:
            print("\n  STATUS: Outperforming benchmark")
        elif stats['excess_return_pct'] < -2:
            print("\n  STATUS: Underperforming benchmark")
        else:
            print("\n  STATUS: Tracking benchmark")

    # Best and Worst Calls
    print_header("BEST & WORST CALLS")
    best, worst = tracker.get_best_worst_calls(5)

    print("  BEST CALLS:")
    if best:
        print(f"  {'Ticker':<8} {'Rank':>5} {'Return':>10} {'Days':>6} {'Week':>5}")
        print(f"  {'-'*8} {'-'*5} {'-'*10} {'-'*6} {'-'*5}")
        for call in best:
            print(f"  {call['ticker']:<8} {call['rank']:>5} {format_return(call['return_pct']):>10} {call['days_held']:>6} {call['week']:>5}")
    else:
        print("  No data yet")

    print()
    print("  WORST CALLS:")
    if worst:
        print(f"  {'Ticker':<8} {'Rank':>5} {'Return':>10} {'Days':>6} {'Week':>5}")
        print(f"  {'-'*8} {'-'*5} {'-'*10} {'-'*6} {'-'*5}")
        for call in worst:
            print(f"  {call['ticker']:<8} {call['rank']:>5} {format_return(call['return_pct']):>10} {call['days_held']:>6} {call['week']:>5}")
    else:
        print("  No data yet")

    # Accuracy by Confidence Bucket
    print_header("ACCURACY BY CONFIDENCE BUCKET")
    print(f"  {'Bucket':<20} {'Accuracy':>12} {'Count':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*8}")

    if 'high_conf_accuracy' in stats:
        print(f"  {'High (>=80%)':<20} {format_pct(stats['high_conf_accuracy']):>12} {stats.get('high_conf_count', 0):>8}")
    if 'mid_conf_accuracy' in stats:
        print(f"  {'Medium (70-80%)':<20} {format_pct(stats['mid_conf_accuracy']):>12} {stats.get('mid_conf_count', 0):>8}")
    if 'low_conf_accuracy' in stats:
        print(f"  {'Low (60-70%)':<20} {format_pct(stats['low_conf_accuracy']):>12} {stats.get('low_conf_count', 0):>8}")

    if 'high_conf_accuracy' not in stats and 'mid_conf_accuracy' not in stats:
        print("  No confidence bucket data available yet")

    # Week-over-Week Trend
    print_header("WEEK-OVER-WEEK TREND")
    weekly_trend = tracker.get_weekly_trend()

    if weekly_trend:
        print(f"  {'Week':>5} {'Predictions':>12} {'Evaluated':>10} {'Accuracy':>10} {'Avg Return':>12}")
        print(f"  {'-'*5} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")

        for week in weekly_trend:
            acc_str = format_pct(week['accuracy']) if week['accuracy'] is not None else "pending"
            ret_str = format_return(week['avg_return']) if week.get('avg_return') is not None else "pending"
            eval_count = week.get('evaluated', 0)
            print(f"  {week['week']:>5} {week['predictions']:>12} {eval_count:>10} {acc_str:>10} {ret_str:>12}")

        # Trend indicator
        accuracies = [w['accuracy'] for w in weekly_trend if w['accuracy'] is not None]
        if len(accuracies) >= 2:
            print()
            if accuracies[-1] > accuracies[0]:
                print("  TREND: Improving over time")
            elif accuracies[-1] < accuracies[0]:
                print("  TREND: Declining over time")
            else:
                print("  TREND: Stable")
    else:
        print("  No weekly data yet")

    # Cumulative Stats
    print_header("CUMULATIVE FORWARD TEST STATS")
    print(f"  Weeks completed:         {tracker.get_current_week()} of {FORWARD_TEST_WEEKS}")
    print(f"  Total predictions:       {stats['total_predictions']}")
    print(f"  Evaluated predictions:   {stats['valid_for_evaluation']}")

    if stats['overall_accuracy'] is not None:
        status = "PASSING" if stats['overall_accuracy'] >= 0.5 else "FAILING"
        print(f"  Overall accuracy:        {format_pct(stats['overall_accuracy'])} [{status}]")

    print()
    print("=" * 70)
    print(f" Report generated: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
