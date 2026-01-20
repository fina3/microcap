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
from data.sector_analyzer import SectorAnalyzer

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

    parser.add_argument(
        '--universe',
        type=str,
        default=None,
        help='Universe CSV file with sector data. Default: latest in data/raw'
    )

    parser.add_argument(
        '--no-sector',
        action='store_true',
        help='Skip sector-relative metrics'
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

    # Add sector-relative metrics
    sector_analyzer = None
    if not args.no_sector:
        try:
            if args.universe:
                universe_file = args.universe
            else:
                universe_file = find_latest_file('universe_*.csv')

            print(f"\nAdding sector-relative metrics...")
            print(f"  Universe file: {universe_file}")

            sector_analyzer = SectorAnalyzer(universe_file=universe_file)
            df = sector_analyzer.add_sectors_to_dataframe(df)
            sector_analyzer.calculate_sector_stats(df)
            df = sector_analyzer.add_sector_relative_metrics(df)

            # Print sector summary
            sector_analyzer.print_sector_summary()

        except FileNotFoundError:
            print("  No universe file found - skipping sector analysis")
        except Exception as e:
            logger.warning(f"Error in sector analysis: {e}")

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

    # For large datasets, show top 10 and bottom 10
    show_all = len(predictions_sorted) <= 20

    # Build ticker to row mapping for sector context
    ticker_data = {row['ticker']: row for _, row in df.iterrows()}

    def format_sector_context(ticker):
        """Format sector context string for a ticker."""
        if sector_analyzer is None or ticker not in ticker_data:
            return ""

        row = ticker_data[ticker]
        sector = row.get('sector', '')
        if pd.isna(sector) or not sector:
            return ""

        pe = row.get('pe_trailing')
        sector_pe = row.get('sector_pe_median')

        if pd.notna(pe) and pd.notna(sector_pe) and sector_pe > 0:
            pct_diff = ((pe - sector_pe) / sector_pe) * 100
            if pct_diff < 0:
                return f"P/E {pe:.1f} (sector: {sector_pe:.1f}) {abs(pct_diff):.0f}% discount"
            else:
                return f"P/E {pe:.1f} (sector: {sector_pe:.1f}) {pct_diff:.0f}% premium"
        return ""

    print(f"\n{'Ticker':<8} {'Prediction':<12} {'Conf':>6} {'52W':>8}  Sector Context")
    print("-" * 80)

    if show_all:
        display_preds = predictions_sorted
    else:
        print("TOP 10 (highest confidence):")
        display_preds = predictions_sorted[:10]

    for pred in display_preds:
        actual_str = f"{pred.actual_return_52w:+.1f}%" if not np.isnan(pred.actual_return_52w) else "N/A"
        sector_ctx = format_sector_context(pred.ticker)
        print(
            f"{pred.ticker:<8} "
            f"{pred.predicted_direction:<12} "
            f"{pred.confidence:>5.0%} "
            f"{actual_str:>8}  "
            f"{sector_ctx}"
        )

    if not show_all:
        print(f"\n... {len(predictions_sorted) - 20} more predictions ...\n")
        print("BOTTOM 10 (lowest confidence):")
        print(f"{'Ticker':<8} {'Prediction':<12} {'Conf':>6} {'52W':>8}  Sector Context")
        print("-" * 80)
        for pred in predictions_sorted[-10:]:
            actual_str = f"{pred.actual_return_52w:+.1f}%" if not np.isnan(pred.actual_return_52w) else "N/A"
            sector_ctx = format_sector_context(pred.ticker)
            print(
                f"{pred.ticker:<8} "
                f"{pred.predicted_direction:<12} "
                f"{pred.confidence:>5.0%} "
                f"{actual_str:>8}  "
                f"{sector_ctx}"
            )

    # Separate by prediction
    outperform = [p for p in predictions if p.prediction == 1]
    underperform = [p for p in predictions if p.prediction == 0]

    print(f"\nSummary:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Predicted OUTPERFORM: {len(outperform)}")
    print(f"  Predicted UNDERPERFORM: {len(underperform)}")

    # Show top 5 highest confidence for each direction
    outperform_sorted = sorted(outperform, key=lambda x: x.confidence, reverse=True)
    underperform_sorted = sorted(underperform, key=lambda x: x.confidence, reverse=True)

    if outperform_sorted:
        top_out = [p.ticker for p in outperform_sorted[:5]]
        print(f"\n  Top 5 OUTPERFORM picks: {', '.join(top_out)}")

    if underperform_sorted:
        top_under = [p.ticker for p in underperform_sorted[:5]]
        print(f"  Top 5 UNDERPERFORM picks: {', '.join(top_under)}")

    # Always save full results to CSV
    date_str = datetime.now().strftime('%Y%m%d')

    if args.output:
        output_file = args.output
    else:
        output_file = f'data/raw/predictions_{date_str}.csv'

    # Create DataFrame sorted by confidence
    output_df = pd.DataFrame([p.to_dict() for p in predictions_sorted])
    output_df.to_csv(output_file, index=False)

    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"Total stocks analyzed: {len(predictions)}")
    print(f"Full results saved to: {output_file}")
    print("="*70)

    return predictions


if __name__ == '__main__':
    predictions = main()
