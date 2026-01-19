"""
Backtesting framework with strict temporal controls.

Prevents lookahead bias by enforcing that predictions use only data
available at the time of prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
import logging
import pytz

import sys
sys.path.append('..')
from utils.temporal import ensure_utc, validate_temporal_consistency
from data.collector import FinancialDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results with temporal tracking."""

    def __init__(self):
        self.predictions = []
        self.temporal_violations = []
        self.data_quality_issues = []

    def add_prediction(
        self,
        ticker: str,
        as_of_date: datetime,
        prediction_start: datetime,
        prediction_window_days: int,
        features: Dict,
        predicted_return: float,
        actual_return: Optional[float]
    ):
        """Add a prediction result."""
        self.predictions.append({
            'ticker': ticker,
            'as_of_date': as_of_date,
            'prediction_start': prediction_start,
            'prediction_window_days': prediction_window_days,
            'features': features,
            'predicted_return': predicted_return,
            'actual_return': actual_return,
            'timestamp': datetime.now(pytz.utc)
        })

    def add_violation(self, ticker: str, issue: str):
        """Log a temporal violation."""
        self.temporal_violations.append({
            'ticker': ticker,
            'issue': issue,
            'timestamp': datetime.now(pytz.utc)
        })

    def get_summary(self) -> Dict:
        """Get summary statistics of backtest."""
        df = pd.DataFrame(self.predictions)

        if df.empty:
            return {
                'total_predictions': 0,
                'temporal_violations': len(self.temporal_violations),
                'message': 'No valid predictions'
            }

        # Filter out predictions with no actual returns
        valid_df = df[df['actual_return'].notna()]

        if valid_df.empty:
            return {
                'total_predictions': len(df),
                'valid_predictions': 0,
                'temporal_violations': len(self.temporal_violations),
                'message': 'No predictions with actual returns'
            }

        # Calculate accuracy metrics
        correct_direction = (
            (valid_df['predicted_return'] > 0) ==
            (valid_df['actual_return'] > 0)
        ).sum()

        mae = np.abs(valid_df['predicted_return'] - valid_df['actual_return']).mean()
        rmse = np.sqrt(((valid_df['predicted_return'] - valid_df['actual_return']) ** 2).mean())

        return {
            'total_predictions': len(df),
            'valid_predictions': len(valid_df),
            'correct_direction': correct_direction,
            'direction_accuracy': correct_direction / len(valid_df),
            'mae': mae,
            'rmse': rmse,
            'avg_predicted_return': valid_df['predicted_return'].mean(),
            'avg_actual_return': valid_df['actual_return'].mean(),
            'temporal_violations': len(self.temporal_violations),
            'violation_list': self.temporal_violations
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        return pd.DataFrame(self.predictions)


class Backtester:
    """
    Backtesting engine with temporal safeguards.

    All predictions are validated to ensure no lookahead bias.
    """

    def __init__(self, prediction_window_days: int = 90):
        """
        Args:
            prediction_window_days: Days to predict forward (60, 90, or 180 typical)
        """
        self.collector = FinancialDataCollector()
        self.prediction_window_days = prediction_window_days
        self.result = BacktestResult()

    def run_backtest(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        prediction_frequency_days: int = 90,
        model_fn: Optional[Callable] = None
    ) -> BacktestResult:
        """
        Run backtest over a date range.

        Args:
            tickers: List of stock tickers to analyze
            start_date: Start of backtest period (timezone-aware)
            end_date: End of backtest period (timezone-aware)
            prediction_frequency_days: How often to make predictions (e.g., 90 = quarterly)
            model_fn: Function that takes features dict and returns predicted return.
                     If None, uses simple baseline model.

        Returns:
            BacktestResult with all predictions and violations
        """
        start_date = ensure_utc(start_date)
        end_date = ensure_utc(end_date)

        logger.info(
            f"Starting backtest: {len(tickers)} tickers, "
            f"{start_date.date()} to {end_date.date()}, "
            f"prediction window: {self.prediction_window_days} days"
        )

        # Generate prediction dates
        prediction_dates = []
        current_date = start_date
        while current_date <= end_date:
            prediction_dates.append(current_date)
            current_date += timedelta(days=prediction_frequency_days)

        logger.info(f"Generated {len(prediction_dates)} prediction dates")

        # Run predictions for each ticker at each date
        for ticker in tickers:
            logger.info(f"Processing {ticker}")

            for as_of_date in prediction_dates:
                self._make_prediction(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    model_fn=model_fn
                )

        # Add data quality issues from collector
        self.result.data_quality_issues = self.collector.quality_tracker.get_summary()

        logger.info("Backtest complete")
        return self.result

    def _make_prediction(
        self,
        ticker: str,
        as_of_date: datetime,
        model_fn: Optional[Callable]
    ):
        """
        Make a single prediction with temporal validation.

        CRITICAL: Documents what data was used and when prediction starts.
        """
        as_of_date = ensure_utc(as_of_date)

        # Step 1: Collect features using only data available as of as_of_date
        features = self.collector.get_fundamental_metrics(
            ticker=ticker,
            as_of_date=as_of_date
        )

        if not features:
            logger.warning(f"{ticker}: No features available as of {as_of_date.date()}")
            return

        # Step 2: Make prediction
        # Simple baseline model: predict positive return if PE < 15 and PB < 2
        if model_fn is None:
            pe = features.get('PE')
            pb = features.get('PB')

            if pe and pe.value and pb and pb.value:
                if pe.value < 15 and pb.value < 2:
                    predicted_return = 0.10  # Predict 10% gain
                else:
                    predicted_return = 0.0  # Predict flat
            else:
                predicted_return = 0.0
        else:
            # Use custom model
            feature_dict = {k: v.value for k, v in features.items()}
            predicted_return = model_fn(feature_dict)

        # Step 3: Calculate prediction window start date
        # CRITICAL: Prediction window starts AFTER as_of_date
        # Add 1 day buffer to ensure we don't use same-day data
        prediction_start = as_of_date + timedelta(days=1)

        # Validate temporal consistency
        if not validate_temporal_consistency(
            as_of_date=prediction_start,
            data_date=as_of_date,
            allow_equal=True
        ):
            self.result.add_violation(
                ticker,
                f"Prediction start {prediction_start.date()} before as_of_date {as_of_date.date()}"
            )
            return

        # Step 4: Calculate actual forward returns
        actual_return = self.collector.calculate_forward_returns(
            ticker=ticker,
            start_date=prediction_start,
            prediction_window_days=self.prediction_window_days,
            as_of_date=as_of_date
        )

        # Step 5: Log the prediction
        feature_summary = {
            k: v.value for k, v in features.items()
        }

        logger.info(
            f"{ticker}: Using data available as of {as_of_date.date()} "
            f"to predict returns over {self.prediction_window_days} days "
            f"starting {prediction_start.date()}"
        )

        self.result.add_prediction(
            ticker=ticker,
            as_of_date=as_of_date,
            prediction_start=prediction_start,
            prediction_window_days=self.prediction_window_days,
            features=feature_summary,
            predicted_return=predicted_return,
            actual_return=actual_return
        )


def print_backtest_summary(result: BacktestResult):
    """Pretty print backtest results."""
    summary = result.get_summary()

    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)

    print(f"\nTotal predictions: {summary.get('total_predictions', 0)}")
    print(f"Valid predictions: {summary.get('valid_predictions', 0)}")

    if summary.get('valid_predictions', 0) > 0:
        print(f"\nDirection accuracy: {summary.get('direction_accuracy', 0):.2%}")
        print(f"Correct direction: {summary.get('correct_direction', 0)}")
        print(f"MAE: {summary.get('mae', 0):.4f}")
        print(f"RMSE: {summary.get('rmse', 0):.4f}")
        print(f"Avg predicted return: {summary.get('avg_predicted_return', 0):.2%}")
        print(f"Avg actual return: {summary.get('avg_actual_return', 0):.2%}")

    print(f"\n🚨 Temporal violations: {summary.get('temporal_violations', 0)}")

    if summary.get('temporal_violations', 0) > 0:
        print("\nVIOLATIONS DETECTED:")
        for violation in summary.get('violation_list', []):
            print(f"  - {violation['ticker']}: {violation['issue']}")

    print("\n" + "="*60)
