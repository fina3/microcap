"""
Paper Trading System for tracking predictions without real money.

Logs predictions with entry prices and tracks performance over time.

Exit conditions (checked in order):
1. Stop loss: -15% from entry
2. Take profit: +30% from entry
3. Max holding period: 10 trading days (~2 weeks)
4. Target date reached (legacy fallback)
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

# Trading constraints
STOP_LOSS_PCT = -15.0       # Exit if down 15%
TAKE_PROFIT_PCT = 30.0      # Exit if up 30%
MAX_HOLDING_DAYS = 10       # Max trading days to hold (10 trading days = ~2 weeks)


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
    exit_reason: Optional[str] = None  # STOP_LOSS, TAKE_PROFIT, MAX_HOLDING, TARGET_DATE
    exit_date: Optional[datetime] = None  # When the trade was closed
    trading_days_held: int = 0  # Number of trading days position was held
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
            'exit_reason': self.exit_reason,
            'exit_date': self.exit_date,
            'trading_days_held': self.trading_days_held,
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
                date_cols = ['entry_date', 'target_date', 'exit_date']
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
                        exit_reason=row.get('exit_reason'),
                        exit_date=row.get('exit_date'),
                        trading_days_held=int(row.get('trading_days_held', 0)),
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
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
        return None

    def get_price_history(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch price history for a ticker between dates.

        Returns DataFrame with Date index and Close prices.
        """
        try:
            stock = yf.Ticker(ticker)
            # Add buffer days to ensure we get enough data
            start_str = (start_date - timedelta(days=5)).strftime('%Y-%m-%d')
            end_str = (end_date + timedelta(days=5)).strftime('%Y-%m-%d')
            hist = stock.history(start=start_str, end=end_str)
            return hist
        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return pd.DataFrame()

    def count_trading_days(self, start_date: datetime, end_date: datetime, price_history: pd.DataFrame) -> int:
        """
        Count trading days between two dates using actual price history.

        Returns number of trading days (days with price data).
        """
        if price_history.empty:
            # Fallback: estimate ~5 trading days per week
            calendar_days = (end_date - start_date).days
            return int(calendar_days * 5 / 7)

        # Ensure dates are comparable
        start_dt = pd.Timestamp(start_date).tz_localize(None) if start_date.tzinfo else pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date).tz_localize(None) if end_date.tzinfo else pd.Timestamp(end_date)

        # Count trading days in the history between start and end
        hist_dates = price_history.index.tz_localize(None) if price_history.index.tzinfo else price_history.index
        trading_days = hist_dates[(hist_dates >= start_dt) & (hist_dates <= end_dt)]

        return len(trading_days)

    def get_price_on_trading_day(self, price_history: pd.DataFrame, entry_date: datetime, trading_day_num: int) -> tuple:
        """
        Get the closing price on the Nth trading day after entry.

        Returns (price, actual_date) or (None, None) if not available.
        """
        if price_history.empty:
            return None, None

        # Ensure entry_date is comparable
        entry_dt = pd.Timestamp(entry_date).tz_localize(None) if entry_date.tzinfo else pd.Timestamp(entry_date)

        # Get trading days after entry
        hist_dates = price_history.index.tz_localize(None) if price_history.index.tzinfo else price_history.index
        future_days = price_history[hist_dates >= entry_dt]

        if len(future_days) >= trading_day_num:
            target_row = future_days.iloc[trading_day_num - 1]  # 0-indexed, so day 10 is index 9
            target_date = future_days.index[trading_day_num - 1]
            return float(target_row['Close']), target_date

        return None, None

    def _check_exit_conditions(self, trade: 'PaperTrade', price_history: pd.DataFrame, benchmark_history: pd.DataFrame) -> tuple:
        """
        Check all exit conditions for a trade.

        Returns (should_exit, exit_reason, exit_price, exit_date, trading_days_held)
        """
        now = datetime.now(pytz.utc)
        entry_date = ensure_utc(trade.entry_date)

        if price_history.empty:
            return False, None, None, None, 0

        # Ensure entry_date is comparable
        entry_dt = pd.Timestamp(entry_date).tz_localize(None)

        # Get trading days after entry
        hist_index = price_history.index.tz_localize(None) if price_history.index.tzinfo else price_history.index
        future_prices = price_history[hist_index >= entry_dt]

        if future_prices.empty:
            return False, None, None, None, 0

        # Check each trading day for exit conditions
        for i, (date, row) in enumerate(future_prices.iterrows(), 1):
            close_price = float(row['Close'])
            return_pct = ((close_price - trade.entry_price) / trade.entry_price) * 100

            # Check stop loss
            if return_pct <= STOP_LOSS_PCT:
                return True, "STOP_LOSS", close_price, date, i

            # Check take profit
            if return_pct >= TAKE_PROFIT_PCT:
                return True, "TAKE_PROFIT", close_price, date, i

            # Check max holding period
            if i >= MAX_HOLDING_DAYS:
                return True, "MAX_HOLDING", close_price, date, i

        # No exit triggered yet - return current state
        current_trading_days = len(future_prices)
        current_price = float(future_prices.iloc[-1]['Close'])
        return False, None, current_price, None, current_trading_days

    def update_results(self) -> Dict:
        """
        Update all trades with current prices and check exit conditions.

        Exit conditions (checked in order each trading day):
        1. Stop loss: position down 15% from entry
        2. Take profit: position up 30% from entry
        3. Max holding: 10 trading days reached

        Returns:
            Summary of updates made
        """
        now = datetime.now(pytz.utc)
        updates = {
            'trades_updated': 0,
            'trades_completed': 0,
            'stop_loss_exits': 0,
            'take_profit_exits': 0,
            'max_holding_exits': 0,
            'errors': []
        }

        # Get benchmark history
        benchmark_history = pd.DataFrame()
        try:
            iwc = yf.Ticker(self.benchmark_ticker)
            benchmark_history = iwc.history(period="3mo")
        except Exception as e:
            logger.error(f"Error fetching benchmark: {e}")

        for trade in self.trades:
            try:
                # Skip already completed trades
                if trade.is_completed:
                    continue

                # Get price history since entry
                entry_date = ensure_utc(trade.entry_date)
                price_history = self.get_price_history(trade.ticker, entry_date, now)

                if price_history.empty:
                    updates['errors'].append(f"Could not fetch history for {trade.ticker}")
                    continue

                # Check exit conditions
                should_exit, exit_reason, exit_price, exit_date, trading_days = self._check_exit_conditions(
                    trade, price_history, benchmark_history
                )

                # Update current state
                trade.current_price = exit_price if exit_price else self.get_current_price(trade.ticker)
                if trade.current_price and trade.entry_price > 0:
                    trade.current_return_pct = (
                        (trade.current_price - trade.entry_price) / trade.entry_price
                    ) * 100
                trade.trading_days_held = trading_days

                updates['trades_updated'] += 1

                # Complete the trade if exit triggered
                if should_exit:
                    trade.is_completed = True
                    trade.exit_reason = exit_reason
                    trade.exit_date = exit_date
                    trade.final_price = exit_price
                    trade.final_return_pct = (
                        (exit_price - trade.entry_price) / trade.entry_price
                    ) * 100

                    # Calculate benchmark return over same period
                    if not benchmark_history.empty:
                        try:
                            entry_dt = pd.Timestamp(entry_date).tz_localize(None)
                            exit_dt = pd.Timestamp(exit_date).tz_localize(None) if exit_date else None

                            bench_index = benchmark_history.index.tz_localize(None) if benchmark_history.index.tzinfo else benchmark_history.index

                            entry_bench = benchmark_history[bench_index >= entry_dt]
                            if not entry_bench.empty and exit_dt:
                                exit_bench = benchmark_history[bench_index <= exit_dt]
                                if not exit_bench.empty:
                                    bench_entry_price = float(entry_bench.iloc[0]['Close'])
                                    bench_exit_price = float(exit_bench.iloc[-1]['Close'])
                                    trade.benchmark_return_pct = (
                                        (bench_exit_price - bench_entry_price) / bench_entry_price
                                    ) * 100
                        except Exception as e:
                            logger.debug(f"Could not calculate benchmark return: {e}")

                    # Determine if prediction was correct
                    if trade.benchmark_return_pct is not None:
                        outperformed = trade.final_return_pct > trade.benchmark_return_pct
                        trade.is_winner = (
                            (trade.prediction == 1 and outperformed) or
                            (trade.prediction == 0 and not outperformed)
                        )
                    else:
                        # Without benchmark, win if positive return
                        trade.is_winner = trade.final_return_pct > 0

                    updates['trades_completed'] += 1

                    # Track exit type
                    if exit_reason == "STOP_LOSS":
                        updates['stop_loss_exits'] += 1
                    elif exit_reason == "TAKE_PROFIT":
                        updates['take_profit_exits'] += 1
                    elif exit_reason == "MAX_HOLDING":
                        updates['max_holding_exits'] += 1

                    logger.info(
                        f"{trade.ticker}: {exit_reason} - "
                        f"return: {trade.final_return_pct:+.1f}%, "
                        f"days held: {trading_days}, "
                        f"winner: {trade.is_winner}"
                    )

            except Exception as e:
                updates['errors'].append(f"{trade.ticker}: {str(e)}")
                logger.error(f"Error updating {trade.ticker}: {e}")

        self._save_trades()

        logger.info(
            f"Update complete - "
            f"updated: {updates['trades_updated']}, "
            f"completed: {updates['trades_completed']} "
            f"(SL: {updates['stop_loss_exits']}, TP: {updates['take_profit_exits']}, MAX: {updates['max_holding_exits']}), "
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

            # Exit reason breakdown
            stop_loss_trades = [t for t in completed_trades if t.exit_reason == "STOP_LOSS"]
            take_profit_trades = [t for t in completed_trades if t.exit_reason == "TAKE_PROFIT"]
            max_holding_trades = [t for t in completed_trades if t.exit_reason == "MAX_HOLDING"]

            summary['exit_reasons'] = {
                'stop_loss': len(stop_loss_trades),
                'take_profit': len(take_profit_trades),
                'max_holding': len(max_holding_trades),
            }

            # Average return by exit reason
            if stop_loss_trades:
                summary['stop_loss_avg_return'] = np.mean([t.final_return_pct for t in stop_loss_trades if t.final_return_pct])
            if take_profit_trades:
                summary['take_profit_avg_return'] = np.mean([t.final_return_pct for t in take_profit_trades if t.final_return_pct])
            if max_holding_trades:
                summary['max_holding_avg_return'] = np.mean([t.final_return_pct for t in max_holding_trades if t.final_return_pct])

            # Average trading days held
            days_held = [t.trading_days_held for t in completed_trades if t.trading_days_held > 0]
            if days_held:
                summary['avg_days_held'] = np.mean(days_held)

            # Best and worst trades
            if completed_trades:
                best_trade = max(completed_trades, key=lambda t: t.final_return_pct or 0)
                worst_trade = min(completed_trades, key=lambda t: t.final_return_pct or 0)
                summary['best_trade'] = {
                    'ticker': best_trade.ticker,
                    'return_pct': best_trade.final_return_pct,
                    'prediction': best_trade.predicted_direction,
                    'exit_reason': best_trade.exit_reason,
                    'days_held': best_trade.trading_days_held
                }
                summary['worst_trade'] = {
                    'ticker': worst_trade.ticker,
                    'return_pct': worst_trade.final_return_pct,
                    'prediction': worst_trade.predicted_direction,
                    'exit_reason': worst_trade.exit_reason,
                    'days_held': worst_trade.trading_days_held
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

            # Trading days held for pending
            pending_days = [t.trading_days_held for t in pending_trades]
            if pending_days:
                summary['pending_avg_days_held'] = np.mean(pending_days)
                summary['pending_max_days_held'] = max(pending_days)

            # Days until max holding period
            days_until_forced_exit = [MAX_HOLDING_DAYS - t.trading_days_held for t in pending_trades]
            summary['next_forced_exit_days'] = min(days_until_forced_exit) if days_until_forced_exit else None

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

            if 'avg_days_held' in summary:
                print(f"Avg days held: {summary['avg_days_held']:.1f} trading days")

            # Exit reason breakdown
            if 'exit_reasons' in summary:
                print("\n--- EXIT REASONS ---")
                reasons = summary['exit_reasons']
                print(f"Stop loss (-{abs(STOP_LOSS_PCT):.0f}%):  {reasons['stop_loss']}")
                if 'stop_loss_avg_return' in summary:
                    print(f"  Avg return: {summary['stop_loss_avg_return']:+.1f}%")
                print(f"Take profit (+{TAKE_PROFIT_PCT:.0f}%): {reasons['take_profit']}")
                if 'take_profit_avg_return' in summary:
                    print(f"  Avg return: {summary['take_profit_avg_return']:+.1f}%")
                print(f"Max holding ({MAX_HOLDING_DAYS}d):    {reasons['max_holding']}")
                if 'max_holding_avg_return' in summary:
                    print(f"  Avg return: {summary['max_holding_avg_return']:+.1f}%")

            if 'best_trade' in summary:
                best = summary['best_trade']
                exit_str = f" [{best.get('exit_reason', 'N/A')}]" if best.get('exit_reason') else ""
                days_str = f" ({best.get('days_held', 0)}d)" if best.get('days_held') else ""
                print(f"\nBest trade: {best['ticker']} -> {best['return_pct']:+.1f}%{exit_str}{days_str}")

                worst = summary['worst_trade']
                exit_str = f" [{worst.get('exit_reason', 'N/A')}]" if worst.get('exit_reason') else ""
                days_str = f" ({worst.get('days_held', 0)}d)" if worst.get('days_held') else ""
                print(f"Worst trade: {worst['ticker']} -> {worst['return_pct']:+.1f}%{exit_str}{days_str}")

            if 'high_confidence_win_rate' in summary:
                print(f"\nHigh confidence (>=80%) win rate: {summary['high_confidence_win_rate']:.1%}")
            if 'low_confidence_win_rate' in summary:
                print(f"Low confidence (<60%) win rate: {summary['low_confidence_win_rate']:.1%}")

        if summary['pending_trades'] > 0:
            print("\n--- PENDING TRADES ---")
            if 'pending_avg_return_pct' in summary:
                print(f"Current avg return: {summary['pending_avg_return_pct']:+.1f}%")
            if 'pending_avg_days_held' in summary:
                print(f"Avg days held: {summary['pending_avg_days_held']:.1f} trading days")
            if 'pending_max_days_held' in summary:
                print(f"Max days held: {summary['pending_max_days_held']} trading days")
            if 'next_forced_exit_days' in summary and summary['next_forced_exit_days'] is not None:
                print(f"Next forced exit in: {summary['next_forced_exit_days']} trading days")

        # List recent trades
        print("\n--- RECENT TRADES ---")
        df = self.get_trades_df()
        if not df.empty:
            # Sort by entry date descending
            df = df.sort_values('entry_date', ascending=False).head(10)
            for _, row in df.iterrows():
                status = "DONE" if row['is_completed'] else "PENDING"
                ret = row['current_return_pct'] if not row['is_completed'] else row['final_return_pct']
                ret_str = f"{ret:+.1f}%" if ret is not None and pd.notna(ret) else "N/A"
                winner_str = ""
                exit_str = ""
                days_str = ""
                if row['is_completed']:
                    if row['is_winner'] is not None and pd.notna(row['is_winner']):
                        winner_str = " [WIN]" if row['is_winner'] else " [LOSS]"
                    if row.get('exit_reason') and pd.notna(row.get('exit_reason')):
                        exit_str = f" ({row['exit_reason']})"
                if row.get('trading_days_held', 0) > 0:
                    days_str = f" {int(row['trading_days_held'])}d"
                print(
                    f"  {row['ticker']}: {ret_str}{days_str} [{status}]{exit_str}{winner_str}"
                )

        print("\n" + "=" * 60)
