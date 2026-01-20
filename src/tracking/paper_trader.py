"""
Paper Trading System for tracking predictions without real money.

Logs predictions with entry prices and tracks performance over time.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from pathlib import Path

import pandas as pd
import numpy as np
import pytz
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.temporal import ensure_utc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default storage path
DEFAULT_TRADES_FILE = Path("data/tracking/paper_trades.csv")


@dataclass
class PaperTrade:
    """A single paper trade record."""
    ticker: str
    prediction: int  # 1 = outperform, 0 = underperform
    predicted_direction: str  # "OUTPERFORM" / "UNDERPERFORM"
    confidence: float  # 0-1 probability
    entry_price: float
    entry_date: datetime
    target_date: datetime  # When to evaluate the trade
    current_price: Optional[float] = None
    current_return_pct: Optional[float] = None
    final_price: Optional[float] = None
    final_return_pct: Optional[float] = None
    is_completed: bool = False
    is_winner: Optional[bool] = None  # True if prediction was correct
    benchmark_return_pct: Optional[float] = None  # IWC return for comparison
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV storage."""
        return {
            'ticker': self.ticker,
            'prediction': self.prediction,
            'predicted_direction': self.predicted_direction,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date,
            'target_date': self.target_date,
            'current_price': self.current_price,
            'current_return_pct': self.current_return_pct,
            'final_price': self.final_price,
            'final_return_pct': self.final_return_pct,
            'is_completed': self.is_completed,
            'is_winner': self.is_winner,
            'benchmark_return_pct': self.benchmark_return_pct,
            'notes': self.notes
        }


class PaperTrader:
    """
    Paper trading system for tracking predictions.

    Logs predictions, tracks performance, and generates reports.
    """

    def __init__(
        self,
        trades_file: Path = DEFAULT_TRADES_FILE,
        benchmark_ticker: str = "IWC",
        target_days: int = 90
    ):
        """
        Initialize paper trader.

        Args:
            trades_file: Path to CSV file for persistent storage
            benchmark_ticker: Ticker to compare against (default: IWC micro-cap ETF)
            target_days: Default days until trade evaluation
        """
        self.trades_file = Path(trades_file)
        self.benchmark_ticker = benchmark_ticker
        self.target_days = target_days
        self.trades: List[PaperTrade] = []

        # Ensure directory exists
        self.trades_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing trades
        self._load_trades()

        logger.info(
            f"PaperTrader initialized - "
            f"trades_file: {self.trades_file}, "
            f"benchmark: {self.benchmark_ticker}, "
            f"target_days: {self.target_days}, "
            f"existing_trades: {len(self.trades)}"
        )

    def _load_trades(self):
        """Load trades from CSV file."""
        if self.trades_file.exists():
            try:
                df = pd.read_csv(self.trades_file)

                # Parse dates
                date_cols = ['entry_date', 'target_date']
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])

                # Convert to PaperTrade objects
                self.trades = []
                for _, row in df.iterrows():
                    trade = PaperTrade(
                        ticker=row['ticker'],
                        prediction=int(row['prediction']),
                        predicted_direction=row['predicted_direction'],
                        confidence=float(row['confidence']),
                        entry_price=float(row['entry_price']),
                        entry_date=row['entry_date'],
                        target_date=row['target_date'],
                        current_price=row.get('current_price'),
                        current_return_pct=row.get('current_return_pct'),
                        final_price=row.get('final_price'),
                        final_return_pct=row.get('final_return_pct'),
                        is_completed=bool(row.get('is_completed', False)),
                        is_winner=row.get('is_winner'),
                        benchmark_return_pct=row.get('benchmark_return_pct'),
                        notes=row.get('notes', '')
                    )
                    self.trades.append(trade)

                logger.info(f"Loaded {len(self.trades)} existing trades")
            except Exception as e:
                logger.error(f"Error loading trades: {e}")
                self.trades = []
        else:
            logger.info("No existing trades file found, starting fresh")
            self.trades = []

    def _save_trades(self):
        """Save trades to CSV file."""
        if not self.trades:
            logger.warning("No trades to save")
            return

        df = pd.DataFrame([t.to_dict() for t in self.trades])
        df.to_csv(self.trades_file, index=False)
        logger.info(f"Saved {len(self.trades)} trades to {self.trades_file}")

    def log_prediction(
        self,
        ticker: str,
        prediction: int,
        confidence: float,
        entry_price: float,
        target_date: Optional[datetime] = None,
        predicted_direction: Optional[str] = None,
        notes: str = ""
    ) -> PaperTrade:
        """
        Log a new prediction as a paper trade.

        Args:
            ticker: Stock ticker symbol
            prediction: 1 = outperform, 0 = underperform
            confidence: Model confidence (0-1)
            entry_price: Current price at prediction time
            target_date: When to evaluate (defaults to target_days from now)
            predicted_direction: "OUTPERFORM" / "UNDERPERFORM" (auto-derived if not provided)
            notes: Optional notes about the prediction

        Returns:
            PaperTrade object
        """
        entry_date = datetime.now(pytz.utc)

        if target_date is None:
            target_date = entry_date + timedelta(days=self.target_days)

        if predicted_direction is None:
            predicted_direction = "OUTPERFORM" if prediction == 1 else "UNDERPERFORM"

        trade = PaperTrade(
            ticker=ticker,
            prediction=prediction,
            predicted_direction=predicted_direction,
            confidence=confidence,
            entry_price=entry_price,
            entry_date=entry_date,
            target_date=target_date,
            notes=notes
        )

        self.trades.append(trade)
        self._save_trades()

        logger.info(
            f"Logged paper trade: {ticker} - {predicted_direction} "
            f"(confidence: {confidence:.1%}) at ${entry_price:.2f}, "
            f"target: {target_date.date()}"
        )

        return trade

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Fetch current price for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
        return None

    def update_results(self) -> Dict:
        """
        Update all trades with current prices and mark completed ones.

        Returns:
            Summary of updates made
        """
        now = datetime.now(pytz.utc)
        updates = {
            'trades_updated': 0,
            'trades_completed': 0,
            'errors': []
        }

        # Get benchmark return
        benchmark_prices = {}
        try:
            iwc = yf.Ticker(self.benchmark_ticker)
            benchmark_hist = iwc.history(period="1y")
            if not benchmark_hist.empty:
                benchmark_prices = benchmark_hist['Close'].to_dict()
        except Exception as e:
            logger.error(f"Error fetching benchmark: {e}")

        for trade in self.trades:
            try:
                # Skip already completed trades
                if trade.is_completed:
                    continue

                # Get current price
                current_price = self.get_current_price(trade.ticker)
                if current_price is None:
                    updates['errors'].append(f"Could not fetch price for {trade.ticker}")
                    continue

                trade.current_price = current_price
                trade.current_return_pct = (
                    (current_price - trade.entry_price) / trade.entry_price
                ) * 100

                updates['trades_updated'] += 1

                # Check if trade is now complete (past target date)
                if now >= ensure_utc(trade.target_date):
                    trade.is_completed = True
                    trade.final_price = current_price
                    trade.final_return_pct = trade.current_return_pct

                    # Calculate benchmark return over same period
                    # (simplified: use most recent benchmark data)
                    if benchmark_prices:
                        try:
                            entry_benchmark = list(benchmark_prices.values())[0]
                            current_benchmark = list(benchmark_prices.values())[-1]
                            trade.benchmark_return_pct = (
                                (current_benchmark - entry_benchmark) / entry_benchmark
                            ) * 100
                        except Exception:
                            pass

                    # Determine if prediction was correct
                    # Outperform = 1 means we expected stock to beat benchmark
                    if trade.benchmark_return_pct is not None:
                        outperformed = trade.final_return_pct > trade.benchmark_return_pct
                        trade.is_winner = (
                            (trade.prediction == 1 and outperformed) or
                            (trade.prediction == 0 and not outperformed)
                        )
                    else:
                        # Without benchmark, just use 10% threshold
                        outperformed = trade.final_return_pct > 10
                        trade.is_winner = (
                            (trade.prediction == 1 and outperformed) or
                            (trade.prediction == 0 and not outperformed)
                        )

                    updates['trades_completed'] += 1
                    logger.info(
                        f"{trade.ticker}: COMPLETED - "
                        f"return: {trade.final_return_pct:+.1f}%, "
                        f"winner: {trade.is_winner}"
                    )

            except Exception as e:
                updates['errors'].append(f"{trade.ticker}: {str(e)}")
                logger.error(f"Error updating {trade.ticker}: {e}")

        self._save_trades()

        logger.info(
            f"Update complete - "
            f"updated: {updates['trades_updated']}, "
            f"completed: {updates['trades_completed']}, "
            f"errors: {len(updates['errors'])}"
        )

        return updates

    def get_performance_summary(self) -> Dict:
        """
        Calculate performance summary across all trades.

        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {'error': 'No trades logged'}

        total_trades = len(self.trades)
        completed_trades = [t for t in self.trades if t.is_completed]
        pending_trades = [t for t in self.trades if not t.is_completed]

        summary = {
            'total_trades': total_trades,
            'completed_trades': len(completed_trades),
            'pending_trades': len(pending_trades),
        }

        # Completed trade stats
        if completed_trades:
            winners = [t for t in completed_trades if t.is_winner]
            losers = [t for t in completed_trades if t.is_winner is False]

            summary['win_rate'] = len(winners) / len(completed_trades)
            summary['winners'] = len(winners)
            summary['losers'] = len(losers)

            returns = [t.final_return_pct for t in completed_trades if t.final_return_pct is not None]
            if returns:
                summary['avg_return_pct'] = np.mean(returns)
                summary['median_return_pct'] = np.median(returns)
                summary['best_return_pct'] = max(returns)
                summary['worst_return_pct'] = min(returns)
                summary['std_return_pct'] = np.std(returns)

            # Best and worst trades
            if completed_trades:
                best_trade = max(completed_trades, key=lambda t: t.final_return_pct or 0)
                worst_trade = min(completed_trades, key=lambda t: t.final_return_pct or 0)
                summary['best_trade'] = {
                    'ticker': best_trade.ticker,
                    'return_pct': best_trade.final_return_pct,
                    'prediction': best_trade.predicted_direction
                }
                summary['worst_trade'] = {
                    'ticker': worst_trade.ticker,
                    'return_pct': worst_trade.final_return_pct,
                    'prediction': worst_trade.predicted_direction
                }

            # Confidence calibration
            high_conf = [t for t in completed_trades if t.confidence >= 0.8]
            low_conf = [t for t in completed_trades if t.confidence < 0.6]

            if high_conf:
                high_conf_winners = [t for t in high_conf if t.is_winner]
                summary['high_confidence_win_rate'] = len(high_conf_winners) / len(high_conf)

            if low_conf:
                low_conf_winners = [t for t in low_conf if t.is_winner]
                summary['low_confidence_win_rate'] = len(low_conf_winners) / len(low_conf)

        # Pending trade stats
        if pending_trades:
            current_returns = [
                t.current_return_pct for t in pending_trades
                if t.current_return_pct is not None
            ]
            if current_returns:
                summary['pending_avg_return_pct'] = np.mean(current_returns)

            # Days until next completion
            now = datetime.now(pytz.utc)
            days_remaining = [
                (ensure_utc(t.target_date) - now).days
                for t in pending_trades
            ]
            summary['next_completion_days'] = min(days_remaining)

        return summary

    def get_trades_df(self) -> pd.DataFrame:
        """Get all trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def print_report(self):
        """Print a formatted performance report."""
        summary = self.get_performance_summary()

        print("\n" + "=" * 60)
        print("PAPER TRADING PERFORMANCE REPORT")
        print("=" * 60)

        if 'error' in summary:
            print(f"\n{summary['error']}")
            return

        print(f"\nTotal trades logged: {summary['total_trades']}")
        print(f"Completed trades: {summary['completed_trades']}")
        print(f"Pending trades: {summary['pending_trades']}")

        if summary['completed_trades'] > 0:
            print("\n--- COMPLETED TRADES ---")
            print(f"Win rate: {summary['win_rate']:.1%} ({summary['winners']}/{summary['completed_trades']})")

            if 'avg_return_pct' in summary:
                print(f"Average return: {summary['avg_return_pct']:+.1f}%")
                print(f"Median return: {summary['median_return_pct']:+.1f}%")
                print(f"Best return: {summary['best_return_pct']:+.1f}%")
                print(f"Worst return: {summary['worst_return_pct']:+.1f}%")
                print(f"Std deviation: {summary['std_return_pct']:.1f}%")

            if 'best_trade' in summary:
                best = summary['best_trade']
                print(f"\nBest trade: {best['ticker']} ({best['prediction']}) -> {best['return_pct']:+.1f}%")

                worst = summary['worst_trade']
                print(f"Worst trade: {worst['ticker']} ({worst['prediction']}) -> {worst['return_pct']:+.1f}%")

            if 'high_confidence_win_rate' in summary:
                print(f"\nHigh confidence (>=80%) win rate: {summary['high_confidence_win_rate']:.1%}")
            if 'low_confidence_win_rate' in summary:
                print(f"Low confidence (<60%) win rate: {summary['low_confidence_win_rate']:.1%}")

        if summary['pending_trades'] > 0:
            print("\n--- PENDING TRADES ---")
            if 'pending_avg_return_pct' in summary:
                print(f"Current avg return: {summary['pending_avg_return_pct']:+.1f}%")
            if 'next_completion_days' in summary:
                print(f"Next completion in: {summary['next_completion_days']} days")

        # List recent trades
        print("\n--- RECENT TRADES ---")
        df = self.get_trades_df()
        if not df.empty:
            # Sort by entry date descending
            df = df.sort_values('entry_date', ascending=False).head(10)
            for _, row in df.iterrows():
                status = "DONE" if row['is_completed'] else "PENDING"
                ret = row['current_return_pct'] if not row['is_completed'] else row['final_return_pct']
                ret_str = f"{ret:+.1f}%" if ret is not None else "N/A"
                winner_str = ""
                if row['is_completed'] and row['is_winner'] is not None:
                    winner_str = " [WIN]" if row['is_winner'] else " [LOSS]"
                print(
                    f"  {row['ticker']}: {row['predicted_direction']} "
                    f"(conf: {row['confidence']:.1%}) -> {ret_str} [{status}]{winner_str}"
                )

        print("\n" + "=" * 60)
