"""
Example backtest demonstrating lookahead bias prevention.

This script shows how to:
1. Collect financial data with temporal tracking
2. Run backtests with strict temporal controls
3. Validate for lookahead bias
"""

import sys
sys.path.append('src')

from datetime import datetime
import pytz

from backtesting import Backtester, print_backtest_summary
from utils.validation import LookaheadBiasDetector
from data import is_us_ticker


def main():
    print("\n" + "="*60)
    print("MICRO-CAP STOCK ANALYSIS - EXAMPLE BACKTEST")
    print("="*60)

    # Define test parameters
    tickers = [
        'AAPL',   # Large cap for testing (should work)
        'MSFT',   # Another large cap
        # Add micro-caps here once you identify them
    ]

    # Backtest period: Jan 2022 - Dec 2023
    start_date = datetime(2022, 1, 1, tzinfo=pytz.utc)
    end_date = datetime(2023, 12, 31, tzinfo=pytz.utc)

    # Prediction window: 90 days
    prediction_window = 90

    print(f"\nTickers: {', '.join(tickers)}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Prediction window: {prediction_window} days")
    print(f"Prediction frequency: Every 90 days")

    # Filter to US stocks only
    print("\n" + "-"*60)
    print("FILTERING NON-US STOCKS")
    print("-"*60)
    us_tickers = [t for t in tickers if is_us_ticker(t)]
    filtered = set(tickers) - set(us_tickers)

    if filtered:
        print(f"Filtered out: {', '.join(filtered)}")
    print(f"US tickers: {', '.join(us_tickers)}")

    # Create backtester
    print("\n" + "-"*60)
    print("INITIALIZING BACKTESTER")
    print("-"*60)
    backtester = Backtester(prediction_window_days=prediction_window)

    # Run backtest
    print("\n" + "-"*60)
    print("RUNNING BACKTEST")
    print("-"*60)
    print("This will:")
    print("  1. Fetch financial metrics with reporting delay")
    print("  2. Make predictions using only historical data")
    print("  3. Calculate actual forward returns")
    print("  4. Detect any temporal violations")
    print("\nStarting...")

    result = backtester.run_backtest(
        tickers=us_tickers,
        start_date=start_date,
        end_date=end_date,
        prediction_frequency_days=90,
        model_fn=None  # Uses simple baseline model
    )

    # Print results
    print_backtest_summary(result)

    # Check for lookahead bias
    print("\n" + "-"*60)
    print("CHECKING FOR LOOKAHEAD BIAS")
    print("-"*60)

    detector = LookaheadBiasDetector()

    # Convert results to DataFrame for validation
    df = result.to_dataframe()

    if not df.empty:
        # Add columns for validation
        df['data_date'] = df['as_of_date']
        df['return_start'] = df['prediction_start']

        violations = detector.check_dataframe(
            df=df,
            as_of_date_column='as_of_date',
            data_date_column='data_date',
            return_start_column='return_start'
        )

        if violations:
            print(f"\n🚨 Found {len(violations)} violations!")
            for v in violations[:5]:
                print(f"  - {v['message']}")
        else:
            print("\n✓ No lookahead bias detected in predictions!")

    # Print data quality summary
    print("\n" + "-"*60)
    print("DATA QUALITY SUMMARY")
    print("-"*60)

    quality = result.data_quality_issues
    print(f"Missing fields: {quality['missing_fields_count']}")
    print(f"Non-US tickers filtered: {quality['non_us_tickers_count']}")
    print(f"Temporal violations: {quality['temporal_violations_count']}")

    if quality['temporal_violations_count'] > 0:
        print("\nTemporal violations:")
        for v in quality['temporal_violations'][:5]:
            print(f"  - {v['ticker']}: {v['issue']}")

    # Save results
    if not df.empty:
        output_file = 'data/processed/backtest_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")

    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
