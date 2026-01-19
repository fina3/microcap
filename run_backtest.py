"""
Run walk-forward backtest on micro-cap stock predictions.

Uses collected metrics and sentiment data with temporal ordering
to validate model performance without lookahead bias.
"""

import sys
sys.path.append('src')

import argparse
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import pytz

from models.predictor import MicroCapPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_file(pattern: str, directory: str = 'data/raw') -> str:
    """Find the most recent file matching pattern."""
    data_path = Path(directory)
    files = list(data_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")

    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run walk-forward backtest on micro-cap predictions'
    )

    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help='Metrics CSV file. Default: latest in data/raw'
    )

    parser.add_argument(
        '--sentiment',
        type=str,
        default=None,
        help='Sentiment CSV file. Default: latest in data/raw'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'logistic_regression'],
        help='Model type. Default: random_forest'
    )

    parser.add_argument(
        '--benchmark-return',
        type=float,
        default=0.10,
        help='Expected benchmark annual return (decimal). Default: 0.10'
    )

    parser.add_argument(
        '--min-train-size',
        type=int,
        default=5,
        help='Minimum training samples before making predictions. Default: 5'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory. Default: data/raw'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("\n" + "="*70)
    print("MICRO-CAP WALK-FORWARD BACKTEST")
    print("="*70)

    # Find input files
    if args.metrics:
        metrics_file = args.metrics
    else:
        try:
            metrics_file = find_latest_file('microcap_metrics_*.csv')
            logger.info(f"Using latest metrics: {metrics_file}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    if args.sentiment:
        sentiment_file = args.sentiment
    else:
        try:
            sentiment_file = find_latest_file('sentiment_scores_*.csv')
            logger.info(f"Using latest sentiment: {sentiment_file}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Metrics file: {metrics_file}")
    print(f"  Sentiment file: {sentiment_file}")
    print(f"  Model: {args.model}")
    print(f"  Benchmark return: {args.benchmark_return:.1%}")
    print(f"  Min training size: {args.min_train_size}")
    print("="*70)

    # Create predictor
    predictor = MicroCapPredictor(
        model_type=args.model,
        benchmark_return=args.benchmark_return
    )

    # Load and merge data
    df = predictor.load_and_merge_data(metrics_file, sentiment_file)

    # Sort by filing date for temporal ordering
    df = df.sort_values('filing_date').reset_index(drop=True)

    print(f"\nLoaded {len(df)} stocks, ordered by filing date:")
    for i, row in df.iterrows():
        filing_date = row['filing_date'][:10] if pd.notna(row['filing_date']) else 'N/A'
        print(f"  {i+1}. {row['ticker']}: {filing_date}")

    # Run walk-forward validation
    print("\n" + "-"*70)
    print("WALK-FORWARD VALIDATION")
    print("Train on earlier filings, predict later filings (no lookahead)")
    print("-"*70)

    results = predictor.walk_forward_validate(df, min_train_size=args.min_train_size)

    print(f"\nValidation Results:")
    print(f"  Total predictions: {results['n_total']}")
    print(f"  Correct: {results['n_correct']}")
    print(f"  Accuracy: {results['accuracy']:.1%}")

    # Per-prediction breakdown
    print(f"\nPer-prediction breakdown:")
    print(f"{'Ticker':<8} {'Predicted':<14} {'Actual':<14} {'Confidence':>10} {'Result':<8}")
    print("-" * 60)

    backtest_records = []
    for i, (ticker, pred, actual, conf, split) in enumerate(zip(
        results['tickers'],
        results['predictions'],
        results['actuals'],
        results['confidences'],
        results['splits']
    )):
        pred_str = 'OUTPERFORM' if pred == 1 else 'UNDERPERFORM'
        actual_str = 'OUTPERFORM' if actual == 1 else 'UNDERPERFORM'
        result_str = 'CORRECT' if pred == actual else 'WRONG'

        print(f"{ticker:<8} {pred_str:<14} {actual_str:<14} {conf:>9.1%} {result_str:<8}")

        # Build record for CSV
        backtest_records.append({
            'ticker': ticker,
            'prediction': pred,
            'predicted_direction': pred_str,
            'actual': actual,
            'actual_direction': actual_str,
            'confidence': conf,
            'correct': pred == actual,
            'train_size': split['train_size'],
            'backtest_date': datetime.now(pytz.utc)
        })

    # Get feature importance
    print("\n" + "-"*70)
    print("FEATURE IMPORTANCE")
    print("-"*70)

    X, y, _ = predictor.prepare_features(df)
    predictor.train(X, y)
    importance = predictor.get_feature_importance()

    print(f"\nTop features driving predictions:")
    for i, (feat, imp) in enumerate(importance.items()):
        if i >= 8:
            break
        bar = '#' * int(imp * 40)
        print(f"  {feat:<25} {imp:.3f} {bar}")

    # Create results DataFrame
    backtest_df = pd.DataFrame(backtest_records)

    # Add summary row
    summary_data = {
        'ticker': 'SUMMARY',
        'prediction': None,
        'predicted_direction': f"Accuracy: {results['accuracy']:.1%}",
        'actual': None,
        'actual_direction': f"Correct: {results['n_correct']}/{results['n_total']}",
        'confidence': backtest_df['confidence'].mean(),
        'correct': None,
        'train_size': None,
        'backtest_date': datetime.now(pytz.utc)
    }

    # Save results
    date_str = datetime.now().strftime('%Y%m%d')
    output_file = f"{args.output_dir}/backtest_results_{date_str}.csv"

    backtest_df.to_csv(output_file, index=False)
    logger.info(f"Backtest results saved to: {output_file}")

    # Save detailed summary
    summary_file = f"{args.output_dir}/backtest_results_{date_str}_summary.txt"

    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MICRO-CAP WALK-FORWARD BACKTEST SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"\nBacktest Date: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Benchmark Return: {args.benchmark_return:.1%}\n")
        f.write(f"Min Training Size: {args.min_train_size}\n")
        f.write(f"\n" + "-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"\nAccuracy: {results['accuracy']:.1%} ({results['n_correct']}/{results['n_total']})\n")
        f.write(f"Average Confidence: {backtest_df['confidence'].mean():.1%}\n")
        f.write(f"\n" + "-"*70 + "\n")
        f.write("PER-TICKER RESULTS\n")
        f.write("-"*70 + "\n\n")

        for _, row in backtest_df.iterrows():
            result = "CORRECT" if row['correct'] else "WRONG"
            f.write(f"{row['ticker']}: {row['predicted_direction']} (conf: {row['confidence']:.1%}) - {result}\n")

        f.write(f"\n" + "-"*70 + "\n")
        f.write("FEATURE IMPORTANCE\n")
        f.write("-"*70 + "\n\n")

        for feat, imp in importance.items():
            f.write(f"  {feat}: {imp:.4f}\n")

    logger.info(f"Summary saved to: {summary_file}")

    # Final summary
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    print(f"\nWalk-forward accuracy: {results['accuracy']:.1%}")
    print(f"Model correctly predicted {results['n_correct']}/{results['n_total']} outcomes")
    print(f"\nResults saved to:")
    print(f"  - {output_file}")
    print(f"  - {summary_file}")
    print("="*70)

    return backtest_df


if __name__ == '__main__':
    df = main()
