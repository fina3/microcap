"""
Forward Testing System for Micro-Cap Predictions.

Tracks predictions made vs actual outcomes over the 9-week forward test period.
Maintains an append-only audit log for all predictions.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import pytz
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Forward test configuration
FORWARD_TEST_START = datetime(2026, 1, 28, tzinfo=pytz.UTC)
FORWARD_TEST_WEEKS = 9
FORWARD_TEST_END = FORWARD_TEST_START + timedelta(weeks=FORWARD_TEST_WEEKS)

# File paths
FORWARD_TEST_LOG = Path("data/tracking/forward_test_log.csv")
FORWARD_TEST_CONFIG = Path("data/tracking/forward_test_config.json")


@dataclass
class ForwardTestPrediction:
    """A single forward test prediction record."""
    prediction_date: datetime
    ticker: str
    predicted_rank: int
    confidence: float
    predicted_direction: str  # "OUTPERFORM" / "UNDERPERFORM"
    entry_price: float
    current_price: Optional[float] = None
    return_pct: Optional[float] = None
    days_held: int = 0
    benchmark_price_entry: Optional[float] = None
    benchmark_price_current: Optional[float] = None
    benchmark_return_pct: Optional[float] = None
    beat_benchmark: Optional[bool] = None
    model_version: str = "rf_v1"
    week_number: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV storage."""
        return {
            'prediction_date': self.prediction_date.isoformat() if isinstance(self.prediction_date, datetime) else self.prediction_date,
            'ticker': self.ticker,
            'predicted_rank': self.predicted_rank,
            'confidence': self.confidence,
            'predicted_direction': self.predicted_direction,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'return_pct': self.return_pct,
            'days_held': self.days_held,
            'benchmark_price_entry': self.benchmark_price_entry,
            'benchmark_price_current': self.benchmark_price_current,
            'benchmark_return_pct': self.benchmark_return_pct,
            'beat_benchmark': self.beat_benchmark,
            'model_version': self.model_version,
            'week_number': self.week_number
        }


class ForwardTestTracker:
    """
    Forward test tracking system.

    Tracks predictions over the 9-week forward test period with:
    - Append-only audit log (never overwrite)
    - Rolling accuracy calculations
    - Week-over-week trend analysis
    """

    def __init__(
        self,
        log_file: Path = FORWARD_TEST_LOG,
        benchmark_ticker: str = "IWC",
        start_date: datetime = FORWARD_TEST_START
    ):
        self.log_file = Path(log_file)
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date
        self.predictions: List[ForwardTestPrediction] = []

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing predictions
        self._load_predictions()

        logger.info(
            f"ForwardTestTracker initialized - "
            f"log_file: {self.log_file}, "
            f"benchmark: {self.benchmark_ticker}, "
            f"existing_predictions: {len(self.predictions)}"
        )

    def _load_predictions(self):
        """Load existing predictions from CSV."""
        if self.log_file.exists():
            try:
                df = pd.read_csv(self.log_file)
                if not df.empty:
                    self.predictions = []
                    for _, row in df.iterrows():
                        pred = ForwardTestPrediction(
                            prediction_date=pd.to_datetime(row['prediction_date']),
                            ticker=row['ticker'],
                            predicted_rank=int(row['predicted_rank']),
                            confidence=float(row['confidence']),
                            predicted_direction=row['predicted_direction'],
                            entry_price=float(row['entry_price']),
                            current_price=row.get('current_price'),
                            return_pct=row.get('return_pct'),
                            days_held=int(row.get('days_held', 0)),
                            benchmark_price_entry=row.get('benchmark_price_entry'),
                            benchmark_price_current=row.get('benchmark_price_current'),
                            benchmark_return_pct=row.get('benchmark_return_pct'),
                            beat_benchmark=row.get('beat_benchmark'),
                            model_version=row.get('model_version', 'rf_v1'),
                            week_number=int(row.get('week_number', 1))
                        )
                        self.predictions.append(pred)
                    logger.info(f"Loaded {len(self.predictions)} existing predictions")
            except Exception as e:
                logger.error(f"Error loading predictions: {e}")
                self.predictions = []
        else:
            self.predictions = []

    def _append_to_log(self, new_predictions: List[ForwardTestPrediction]):
        """Append new predictions to log file (never overwrite existing)."""
        if not new_predictions:
            return

        new_df = pd.DataFrame([p.to_dict() for p in new_predictions])

        if self.log_file.exists():
            # Append to existing file
            existing_df = pd.read_csv(self.log_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(self.log_file, index=False)
        else:
            # Create new file
            new_df.to_csv(self.log_file, index=False)

        logger.info(f"Appended {len(new_predictions)} predictions to {self.log_file}")

    def get_current_week(self) -> int:
        """Get current week number of forward test (1-9)."""
        now = datetime.now(pytz.utc)
        if now < self.start_date:
            return 0

        weeks_elapsed = (now - self.start_date).days // 7
        return min(weeks_elapsed + 1, FORWARD_TEST_WEEKS)

    def get_weeks_remaining(self) -> int:
        """Get weeks remaining in forward test."""
        return max(0, FORWARD_TEST_WEEKS - self.get_current_week())

    def get_countdown_string(self) -> str:
        """Get formatted countdown string."""
        week = self.get_current_week()
        remaining = self.get_weeks_remaining()

        if week == 0:
            return "Forward test not yet started"
        elif remaining == 0:
            return f"Week {week} of {FORWARD_TEST_WEEKS} | Forward test COMPLETE"
        else:
            return f"Week {week} of {FORWARD_TEST_WEEKS} | {remaining} week{'s' if remaining != 1 else ''} remaining"

    def log_weekly_predictions(
        self,
        predictions_df: pd.DataFrame,
        top_n: int = 20,
        model_version: str = "rf_v1"
    ) -> int:
        """
        Log top N predictions from weekly analysis.

        Args:
            predictions_df: DataFrame with columns: ticker, rank, confidence, predicted_direction
            top_n: Number of top predictions to log (default: 20)
            model_version: Version string for the model used

        Returns:
            Number of predictions logged
        """
        now = datetime.now(pytz.utc)
        week_num = self.get_current_week()
        today_str = now.strftime('%Y-%m-%d')

        # Check if we already logged predictions for today
        existing_today = [
            p for p in self.predictions
            if pd.to_datetime(p.prediction_date).strftime('%Y-%m-%d') == today_str
        ]

        if existing_today:
            logger.warning(f"Already logged {len(existing_today)} predictions for {today_str}")
            return 0

        # Get benchmark entry price
        benchmark_entry = self._get_price(self.benchmark_ticker)

        # Sort by rank and take top N
        df_sorted = predictions_df.sort_values('rank').head(top_n)

        new_predictions = []
        for _, row in df_sorted.iterrows():
            ticker = row['ticker']
            entry_price = self._get_price(ticker)

            if entry_price is None or entry_price <= 0:
                logger.warning(f"Could not get price for {ticker}, skipping")
                continue

            pred = ForwardTestPrediction(
                prediction_date=now,
                ticker=ticker,
                predicted_rank=int(row['rank']),
                confidence=float(row['confidence']),
                predicted_direction=row.get('predicted_direction', 'OUTPERFORM'),
                entry_price=entry_price,
                current_price=entry_price,
                return_pct=0.0,
                days_held=0,
                benchmark_price_entry=benchmark_entry,
                benchmark_price_current=benchmark_entry,
                benchmark_return_pct=0.0,
                model_version=model_version,
                week_number=week_num
            )
            new_predictions.append(pred)

        # Append to log file and memory
        self._append_to_log(new_predictions)
        self.predictions.extend(new_predictions)

        logger.info(f"Logged {len(new_predictions)} predictions for week {week_num}")
        return len(new_predictions)

    def _get_price(self, ticker: str) -> Optional[float]:
        """Fetch current price for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.debug(f"Error fetching price for {ticker}: {e}")
        return None

    def update_current_prices(self) -> Dict:
        """
        Update all predictions with current prices.

        Returns:
            Summary of updates made
        """
        now = datetime.now(pytz.utc)
        updates = {'updated': 0, 'errors': []}

        # Get current benchmark price
        benchmark_current = self._get_price(self.benchmark_ticker)

        for pred in self.predictions:
            try:
                # Get current price
                current_price = self._get_price(pred.ticker)
                if current_price is None:
                    updates['errors'].append(f"Could not fetch {pred.ticker}")
                    continue

                pred.current_price = current_price

                # Calculate return
                if pred.entry_price and pred.entry_price > 0:
                    pred.return_pct = ((current_price - pred.entry_price) / pred.entry_price) * 100

                # Calculate days held
                pred_date = pd.to_datetime(pred.prediction_date)
                if pred_date.tzinfo is None:
                    pred_date = pred_date.tz_localize('UTC')
                pred.days_held = (now - pred_date).days

                # Update benchmark
                if benchmark_current:
                    pred.benchmark_price_current = benchmark_current
                    if pred.benchmark_price_entry and pred.benchmark_price_entry > 0:
                        pred.benchmark_return_pct = (
                            (benchmark_current - pred.benchmark_price_entry) /
                            pred.benchmark_price_entry
                        ) * 100

                        # Did this stock beat the benchmark?
                        if pred.return_pct is not None:
                            pred.beat_benchmark = pred.return_pct > pred.benchmark_return_pct

                updates['updated'] += 1

            except Exception as e:
                updates['errors'].append(f"{pred.ticker}: {str(e)}")

        # Rewrite the full log with updated values
        if updates['updated'] > 0:
            df = pd.DataFrame([p.to_dict() for p in self.predictions])
            df.to_csv(self.log_file, index=False)
            logger.info(f"Updated {updates['updated']} predictions")

        return updates

    def get_accuracy_stats(self) -> Dict:
        """
        Calculate rolling accuracy statistics.

        Returns:
            Dictionary with accuracy metrics
        """
        if not self.predictions:
            return {'error': 'No predictions logged'}

        # Filter predictions with valid benchmark comparison
        valid_preds = [p for p in self.predictions if p.beat_benchmark is not None]

        if not valid_preds:
            return {
                'total_predictions': len(self.predictions),
                'valid_for_evaluation': 0,
                'note': 'Predictions need price updates to calculate accuracy'
            }

        # Overall accuracy (predictions that beat benchmark)
        beat_count = sum(1 for p in valid_preds if p.beat_benchmark)
        overall_accuracy = beat_count / len(valid_preds) if valid_preds else 0

        # Top 20 accuracy
        top20_preds = [p for p in valid_preds if p.predicted_rank <= 20]
        top20_beat = sum(1 for p in top20_preds if p.beat_benchmark)
        top20_accuracy = top20_beat / len(top20_preds) if top20_preds else 0

        # Bottom 20 accuracy (should underperform)
        max_rank = max(p.predicted_rank for p in valid_preds)
        bottom20_preds = [p for p in valid_preds if p.predicted_rank > max_rank - 20]
        bottom20_underperform = sum(1 for p in bottom20_preds if not p.beat_benchmark)
        bottom20_accuracy = bottom20_underperform / len(bottom20_preds) if bottom20_preds else 0

        # Returns
        returns = [p.return_pct for p in valid_preds if p.return_pct is not None]
        benchmark_returns = [p.benchmark_return_pct for p in valid_preds if p.benchmark_return_pct is not None]

        top20_returns = [p.return_pct for p in top20_preds if p.return_pct is not None]

        # Confidence bucket accuracy
        high_conf = [p for p in valid_preds if p.confidence >= 0.8]
        mid_conf = [p for p in valid_preds if 0.7 <= p.confidence < 0.8]
        low_conf = [p for p in valid_preds if 0.6 <= p.confidence < 0.7]

        stats = {
            'total_predictions': len(self.predictions),
            'valid_for_evaluation': len(valid_preds),
            'overall_accuracy': overall_accuracy,
            'top20_accuracy': top20_accuracy,
            'top20_count': len(top20_preds),
            'bottom20_accuracy': bottom20_accuracy,
            'bottom20_count': len(bottom20_preds),
            'avg_return_pct': np.mean(returns) if returns else None,
            'avg_benchmark_return_pct': np.mean(benchmark_returns) if benchmark_returns else None,
            'top20_avg_return_pct': np.mean(top20_returns) if top20_returns else None,
            'excess_return_pct': (np.mean(top20_returns) - np.mean(benchmark_returns)) if (top20_returns and benchmark_returns) else None,
        }

        # Confidence buckets
        if high_conf:
            stats['high_conf_accuracy'] = sum(1 for p in high_conf if p.beat_benchmark) / len(high_conf)
            stats['high_conf_count'] = len(high_conf)
        if mid_conf:
            stats['mid_conf_accuracy'] = sum(1 for p in mid_conf if p.beat_benchmark) / len(mid_conf)
            stats['mid_conf_count'] = len(mid_conf)
        if low_conf:
            stats['low_conf_accuracy'] = sum(1 for p in low_conf if p.beat_benchmark) / len(low_conf)
            stats['low_conf_count'] = len(low_conf)

        return stats

    def get_weekly_trend(self) -> List[Dict]:
        """
        Get week-over-week accuracy trend.

        Returns:
            List of dictionaries with weekly stats
        """
        weekly_stats = []

        for week in range(1, self.get_current_week() + 1):
            week_preds = [p for p in self.predictions if p.week_number == week]
            valid_preds = [p for p in week_preds if p.beat_benchmark is not None]

            if not valid_preds:
                weekly_stats.append({
                    'week': week,
                    'predictions': len(week_preds),
                    'accuracy': None,
                    'avg_return': None
                })
                continue

            accuracy = sum(1 for p in valid_preds if p.beat_benchmark) / len(valid_preds)
            returns = [p.return_pct for p in valid_preds if p.return_pct is not None]

            weekly_stats.append({
                'week': week,
                'predictions': len(week_preds),
                'evaluated': len(valid_preds),
                'accuracy': accuracy,
                'avg_return': np.mean(returns) if returns else None
            })

        return weekly_stats

    def get_best_worst_calls(self, n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Get best and worst performing predictions.

        Returns:
            Tuple of (best_calls, worst_calls) as lists of dictionaries
        """
        valid_preds = [p for p in self.predictions if p.return_pct is not None]

        if not valid_preds:
            return [], []

        sorted_preds = sorted(valid_preds, key=lambda p: p.return_pct or 0, reverse=True)

        best = []
        for p in sorted_preds[:n]:
            best.append({
                'ticker': p.ticker,
                'rank': p.predicted_rank,
                'return_pct': p.return_pct,
                'days_held': p.days_held,
                'week': p.week_number
            })

        worst = []
        for p in sorted_preds[-n:]:
            worst.append({
                'ticker': p.ticker,
                'rank': p.predicted_rank,
                'return_pct': p.return_pct,
                'days_held': p.days_held,
                'week': p.week_number
            })

        return best, worst

    def get_predictions_df(self) -> pd.DataFrame:
        """Get all predictions as a DataFrame."""
        if not self.predictions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.predictions])
