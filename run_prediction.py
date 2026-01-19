"""
Run micro-cap stock predictions.

Loads metrics and sentiment data, trains model, and generates predictions
with confidence scores.
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

from models.predictor import MicroCapPredictor, PredictionResult

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
        description='Run micro-cap stock predictions'
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
        '--output',
        type=str,
        default=None,
        help='Output CSV file for predictions'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run walk-forward validation'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("\n" + "="*70)
    print("MICRO-CAP STOCK PREDICTION")
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
    print("="*70)

    # Create predictor
    predictor = MicroCapPredictor(
        model_type=args.model,
        benchmark_return=args.benchmark_return
    )

    # Load and merge data
    df = predictor.load_and_merge_data(metrics_file, sentiment_file)

    print(f"\nLoaded {len(df)} stocks")

    # Run walk-forward validation if requested
    if args.validate:
        print("\n" + "-"*70)
        print("WALK-FORWARD VALIDATION")
        print("-"*70)

        validation_results = predictor.walk_forward_validate(df)

        print(f"\nValidation Results:")
        print(f"  Accuracy: {validation_results['accuracy']:.1%}")
        print(f"  Correct: {validation_results['n_correct']}/{validation_results['n_total']}")

        print(f"\nPer-split results:")
        for i, (ticker, pred, actual, conf) in enumerate(zip(
            validation_results['tickers'],
            validation_results['predictions'],
            validation_results['actuals'],
            validation_results['confidences']
        )):
            status = "CORRECT" if pred == actual else "WRONG"
            direction = "OUTPERFORM" if pred == 1 else "UNDERPERFORM"
            print(f"  {ticker}: Predicted {direction} (conf: {conf:.2f}) - {status}")

    # Generate predictions
    print("\n" + "-"*70)
    print("GENERATING PREDICTIONS")
    print("-"*70)

    predictions = predictor.generate_predictions(df)

    # Evaluate vs actual returns
    evaluation = predictor.evaluate_vs_actual(df, predictions)

    print(f"\nModel Evaluation (vs 52-week actual returns):")
    print(f"  Accuracy: {evaluation['accuracy']:.1%} ({evaluation['correct']}/{evaluation['total']})")
    print(f"  Benchmark threshold: {evaluation['benchmark_threshold']:.1f}%")
    print(f"  Avg return (predicted OUTPERFORM): {evaluation['avg_return_predicted_outperform']:.1f}%")
    print(f"  Avg return (predicted UNDERPERFORM): {evaluation['avg_return_predicted_underperform']:.1f}%")

    # Feature importance
    feature_importance = predictor.get_feature_importance()
    print(f"\nTop Feature Importance:")
    for i, (feature, importance) in enumerate(feature_importance.items()):
        if i >= 5:
            break
        print(f"  {i+1}. {feature}: {importance:.4f}")

    # Predictions ranked by confidence
    print("\n" + "-"*70)
    print("PREDICTIONS (ranked by confidence)")
    print("-"*70)

    # Sort by confidence
    predictions_sorted = sorted(predictions, key=lambda x: x.confidence, reverse=True)

    print(f"\n{'Ticker':<8} {'Prediction':<14} {'Confidence':>10} {'Actual 52W':>12}")
    print("-" * 50)

    for pred in predictions_sorted:
        actual_str = f"{pred.actual_return_52w:+.1f}%" if not np.isnan(pred.actual_return_52w) else "N/A"
        print(
            f"{pred.ticker:<8} "
            f"{pred.predicted_direction:<14} "
            f"{pred.confidence:>10.1%} "
            f"{actual_str:>12}"
        )

    # Separate by prediction
    outperform = [p for p in predictions if p.prediction == 1]
    underperform = [p for p in predictions if p.prediction == 0]

    print(f"\nSummary:")
    print(f"  Predicted to OUTPERFORM ({len(outperform)}): {', '.join(p.ticker for p in outperform)}")
    print(f"  Predicted to UNDERPERFORM ({len(underperform)}): {', '.join(p.ticker for p in underperform)}")

    # Save to CSV if requested
    if args.output:
        output_df = pd.DataFrame([p.to_dict() for p in predictions])
        output_df.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to: {args.output}")
    else:
        # Default output
        date_str = datetime.now().strftime('%Y%m%d')
        output_file = f'data/raw/predictions_{date_str}.csv'
        output_df = pd.DataFrame([p.to_dict() for p in predictions])
        output_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to: {output_file}")

    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)

    return predictions


if __name__ == '__main__':
    predictions = main()
