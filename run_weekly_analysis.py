#!/usr/bin/env python3
"""
Master Weekly Analysis Script for Micro-Cap Stocks.

Runs the complete analysis pipeline:
1. Update universe (if older than 7 days)
2. Pull metrics for all stocks
3. Pull sentiment for all stocks
4. Run predictions with sector analysis
5. Apply risk management / build portfolio
6. Log to paper trading tracker
7. Update past paper trades with current prices
8. Generate summary report

Usage:
    python run_weekly_analysis.py                    # Full analysis
    python run_weekly_analysis.py --limit 50        # Limit to 50 stocks
    python run_weekly_analysis.py --skip-universe   # Don't refresh universe
    python run_weekly_analysis.py --dry-run         # Show what would happen
"""

import sys
sys.path.insert(0, 'src')

import argparse
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
import numpy as np
import pytz

# Import modules that exist
from data.universe_screener import UniverseScreener
from data.sentiment_collector import SentimentCollector
from data.sector_analyzer import SectorAnalyzer
from data.liquidity_checker import check_liquidity_batch, get_liquidity_summary
from screening.options_screener import screen_options_batch, get_options_summary
from models.predictor import MicroCapPredictor
from portfolio.risk_manager import RiskManager
from tracking.paper_trader import PaperTrader
from tracking.forward_test import ForwardTestTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeeklyAnalyzer:
    """
    Master class for running weekly micro-cap analysis.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        reports_dir: str = "reports",
        capital: float = 10000,
        dry_run: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir)
        self.capital = capital
        self.dry_run = dry_run
        self.as_of_date = datetime.now(pytz.utc)
        self.date_str = self.as_of_date.strftime('%Y%m%d')

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.universe_df = None
        self.metrics_df = None
        self.sentiment_df = None
        self.predictions = None
        self.liquidity_data = {}  # ticker -> {liquidity_grade, spread_pct}
        self.liquidity_summary = None
        self.options_data = {}  # ticker -> OptionsCandidate
        self.options_candidates = []  # tickers with liquid options
        self.portfolio = None
        self.paper_trades_summary = None

        # Timing
        self.step_times: Dict[str, float] = {}

    def _log_step(self, step: int, name: str, status: str = "STARTING"):
        """Log step progress."""
        print(f"\n{'='*70}")
        print(f"STEP {step}: {name} [{status}]")
        print(f"{'='*70}")

    def _find_latest_file(self, pattern: str) -> Optional[Path]:
        """Find most recent file matching pattern."""
        files = list(self.data_dir.glob(pattern))
        if not files:
            return None
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0]

    def _file_age_days(self, filepath: Path) -> float:
        """Get age of file in days."""
        if not filepath.exists():
            return float('inf')
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=pytz.utc)
        return (self.as_of_date - mtime).total_seconds() / 86400

    # =========================================================================
    # STEP 1: Update Universe
    # =========================================================================
    def step1_update_universe(self, force: bool = False, skip: bool = False) -> bool:
        """Update stock universe if older than 7 days."""
        self._log_step(1, "UPDATE UNIVERSE")
        start_time = time.time()

        universe_file = self._find_latest_file("universe_*.csv")

        if skip:
            if universe_file:
                print(f"  Skipping universe refresh (--skip-universe)")
                self.universe_df = pd.read_csv(universe_file)
                print(f"  Using existing universe: {universe_file}")
                print(f"  Stocks in universe: {len(self.universe_df)}")
            else:
                print("  ERROR: No existing universe file found!")
                return False
        elif universe_file and not force:
            age_days = self._file_age_days(universe_file)
            if age_days < 7:
                print(f"  Universe file is {age_days:.1f} days old (< 7 days)")
                print(f"  Using existing universe: {universe_file}")
                self.universe_df = pd.read_csv(universe_file)
                print(f"  Stocks in universe: {len(self.universe_df)}")
            else:
                print(f"  Universe file is {age_days:.1f} days old (>= 7 days)")
                print(f"  Refreshing universe...")
                if not self.dry_run:
                    self._refresh_universe()
        else:
            print("  No universe file found or force refresh requested")
            if not self.dry_run:
                self._refresh_universe()

        self.step_times['universe'] = time.time() - start_time
        return True

    def _refresh_universe(self):
        """Refresh the stock universe."""
        screener = UniverseScreener(as_of_date=self.as_of_date)
        self.universe_df = screener.scrape_universe()

        output_file = self.data_dir / f"universe_{self.date_str}.csv"
        self.universe_df.to_csv(output_file, index=False)
        print(f"  Universe saved: {output_file}")
        print(f"  Total stocks: {len(self.universe_df)}")

    # =========================================================================
    # STEP 2: Pull Metrics
    # =========================================================================
    def step2_pull_metrics(self, limit: Optional[int] = None) -> bool:
        """Pull financial metrics for all stocks."""
        self._log_step(2, "PULL METRICS")
        start_time = time.time()

        if self.universe_df is None:
            print("  ERROR: No universe loaded!")
            return False

        num_stocks = limit if limit else len(self.universe_df)
        print(f"  Processing {num_stocks} tickers...")

        if self.dry_run:
            print(f"  [DRY RUN] Would collect metrics for {num_stocks} stocks")
            # Load existing metrics if available
            metrics_file = self._find_latest_file("microcap_metrics_*.csv")
            if metrics_file:
                self.metrics_df = pd.read_csv(metrics_file)
        else:
            # Call the pull_microcap_metrics.py script
            cmd = ["python", "pull_microcap_metrics.py", "--universe", "--quiet"]
            if limit:
                cmd.extend(["--limit", str(limit)])

            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  ERROR: {result.stderr}")
                return False

            # Load the generated file
            metrics_file = self._find_latest_file("microcap_metrics_*.csv")
            if metrics_file:
                self.metrics_df = pd.read_csv(metrics_file)
                print(f"  Metrics loaded: {metrics_file}")
                print(f"  Stocks with data: {len(self.metrics_df)}")
            else:
                print("  ERROR: No metrics file generated!")
                return False

        self.step_times['metrics'] = time.time() - start_time
        return True

    # =========================================================================
    # STEP 3: Pull Sentiment
    # =========================================================================
    def step3_pull_sentiment(self, limit: Optional[int] = None) -> bool:
        """Pull sentiment from 8-K and 10-Q filings."""
        self._log_step(3, "PULL SENTIMENT")
        start_time = time.time()

        if self.metrics_df is None:
            print("  ERROR: No metrics loaded!")
            return False

        num_stocks = len(self.metrics_df)
        if limit:
            num_stocks = min(limit, num_stocks)

        print(f"  Processing {num_stocks} tickers...")

        if self.dry_run:
            print(f"  [DRY RUN] Would collect sentiment for {num_stocks} stocks")
            # Load existing sentiment if available
            sentiment_file = self._find_latest_file("sentiment_scores_*.csv")
            if sentiment_file:
                self.sentiment_df = pd.read_csv(sentiment_file)
        else:
            # Call the pull_sentiment_scores.py script
            cmd = ["python", "pull_sentiment_scores.py", "--universe"]
            if limit:
                cmd.extend(["--limit", str(limit)])

            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  ERROR: {result.stderr}")
                # Don't fail completely - try to load existing file
                sentiment_file = self._find_latest_file("sentiment_scores_*.csv")
                if sentiment_file:
                    self.sentiment_df = pd.read_csv(sentiment_file)
                    print(f"  Using existing sentiment file: {sentiment_file}")
                else:
                    return False

            # Load the generated file
            sentiment_file = self._find_latest_file("sentiment_scores_*.csv")
            if sentiment_file:
                self.sentiment_df = pd.read_csv(sentiment_file)
                print(f"  Sentiment loaded: {sentiment_file}")
                print(f"  Stocks with sentiment: {len(self.sentiment_df)}")
            else:
                print("  ERROR: No sentiment file generated!")
                return False

        self.step_times['sentiment'] = time.time() - start_time
        return True

    # =========================================================================
    # STEP 4: Run Predictions
    # =========================================================================
    def step4_run_predictions(self) -> bool:
        """Run prediction model with sector analysis."""
        self._log_step(4, "RUN PREDICTIONS")
        start_time = time.time()

        if self.metrics_df is None or self.sentiment_df is None:
            print("  ERROR: Missing metrics or sentiment data!")
            return False

        # Initialize predictor
        predictor = MicroCapPredictor(model_type="random_forest")

        # Load and merge data
        metrics_file = self._find_latest_file("microcap_metrics_*.csv")
        sentiment_file = self._find_latest_file("sentiment_scores_*.csv")

        if not metrics_file or not sentiment_file:
            print("  ERROR: Cannot find metrics or sentiment files!")
            return False

        df = predictor.load_and_merge_data(str(metrics_file), str(sentiment_file))
        print(f"  Merged data: {len(df)} stocks")

        # Add sector-relative metrics
        universe_file = self._find_latest_file("universe_*.csv")
        if universe_file:
            sector_analyzer = SectorAnalyzer(universe_file=universe_file)
            df = sector_analyzer.add_sectors_to_dataframe(df)
            sector_analyzer.calculate_sector_stats(df)
            df = sector_analyzer.add_sector_relative_metrics(df)
            self.sector_stats = sector_analyzer.sector_stats
        else:
            self.sector_stats = {}

        # Generate predictions
        self.predictions = predictor.generate_predictions(df)
        self.feature_importance = predictor.get_feature_importance()
        self.merged_df = df  # Save for later use

        # Evaluate
        evaluation = predictor.evaluate_vs_actual(df, self.predictions)

        print(f"\n  Model Accuracy: {evaluation['accuracy']:.1%}")
        print(f"  Predicted OUTPERFORM: {sum(1 for p in self.predictions if p.prediction == 1)}")
        print(f"  Predicted UNDERPERFORM: {sum(1 for p in self.predictions if p.prediction == 0)}")

        # Save predictions (already sorted by rank)
        if not self.dry_run:
            predictions_sorted = sorted(self.predictions, key=lambda x: x.rank)
            output_df = pd.DataFrame([p.to_dict() for p in predictions_sorted])
            output_file = self.data_dir / f"predictions_{self.date_str}.csv"
            output_df.to_csv(output_file, index=False)
            print(f"  Predictions saved: {output_file}")

            # Save to rank history for tracking movement over time
            self._save_rank_history(predictions_sorted)

        self.step_times['predictions'] = time.time() - start_time
        return True

    def _save_rank_history(self, predictions: list) -> None:
        """
        Append current rankings to rank_history.csv for tracking movement.

        Columns: date, ticker, rank, confidence, prev_rank, rank_change
        """
        tracking_dir = Path("data/tracking")
        tracking_dir.mkdir(parents=True, exist_ok=True)
        history_file = tracking_dir / "rank_history.csv"

        # Load existing history to get previous ranks
        prev_ranks = {}
        if history_file.exists():
            try:
                existing_df = pd.read_csv(history_file)
                # Get most recent rank for each ticker
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                for ticker in existing_df['ticker'].unique():
                    ticker_df = existing_df[existing_df['ticker'] == ticker]
                    latest = ticker_df.sort_values('date', ascending=False).iloc[0]
                    prev_ranks[ticker] = int(latest['rank'])
            except Exception as e:
                logger.warning(f"Could not load rank history: {e}")

        # Build new rows
        today = self.as_of_date.strftime('%Y-%m-%d')
        new_rows = []
        for pred in predictions:
            prev_rank = prev_ranks.get(pred.ticker)
            if prev_rank is not None:
                rank_change = prev_rank - pred.rank  # Positive = improved
            else:
                rank_change = None  # NEW entry

            new_rows.append({
                'date': today,
                'ticker': pred.ticker,
                'rank': pred.rank,
                'confidence': pred.confidence,
                'prev_rank': prev_rank,
                'rank_change': rank_change
            })

        new_df = pd.DataFrame(new_rows)

        # Append to existing or create new
        if history_file.exists():
            existing_df = pd.read_csv(history_file)
            # Remove any existing entries for today (in case of re-run)
            existing_df = existing_df[existing_df['date'] != today]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_csv(history_file, index=False)
        print(f"  Rank history updated: {history_file} ({len(new_rows)} entries)")

    # =========================================================================
    # STEP 5: Check Liquidity
    # =========================================================================
    def step5_check_liquidity(self) -> bool:
        """Check bid-ask spreads to identify illiquid stocks."""
        self._log_step(5, "CHECK LIQUIDITY")
        start_time = time.time()

        if not self.predictions:
            print("  ERROR: No predictions available!")
            return False

        tickers = [p.ticker for p in self.predictions]
        print(f"  Checking liquidity for {len(tickers)} tickers...")

        if self.dry_run:
            print(f"  [DRY RUN] Would check liquidity for {len(tickers)} stocks")
            # Try to load existing liquidity data
            liquidity_file = self._find_latest_file("liquidity_*.csv")
            if liquidity_file:
                liq_df = pd.read_csv(liquidity_file)
                for _, row in liq_df.iterrows():
                    self.liquidity_data[row['ticker']] = {
                        'liquidity_grade': row.get('liquidity_grade', '?'),
                        'spread_pct': row.get('spread_pct')
                    }
        else:
            # Check liquidity for all tickers
            results = check_liquidity_batch(tickers)

            # Store results
            for r in results:
                self.liquidity_data[r.ticker] = {
                    'liquidity_grade': r.liquidity_grade,
                    'spread_pct': r.spread_pct
                }

            # Generate summary
            self.liquidity_summary = get_liquidity_summary(results)

            print(f"\n  Liquidity Summary:")
            print(f"    Tradeable (A-D): {self.liquidity_summary['tradeable']}")
            print(f"    Untradeable (F): {self.liquidity_summary['untradeable']}")
            print(f"    Average spread: {self.liquidity_summary['avg_spread_pct']}%")

            print(f"\n  Grade Distribution:")
            for grade, count in self.liquidity_summary['grade_counts'].items():
                print(f"    {grade}: {count} stocks")

            # Save liquidity data
            output_df = pd.DataFrame([r.to_dict() for r in results])
            output_file = self.data_dir / f"liquidity_{self.date_str}.csv"
            output_df.to_csv(output_file, index=False)
            print(f"\n  Liquidity data saved: {output_file}")

        # Screen top-20 for options availability
        print(f"\n  Screening top 20 for options availability...")
        top20_tickers = [p.ticker for p in sorted(self.predictions, key=lambda x: x.rank)[:20]]

        if not self.dry_run:
            options_results = screen_options_batch(top20_tickers)

            for opt in options_results:
                self.options_data[opt.ticker] = opt
                if opt.options_available:
                    self.options_candidates.append(opt.ticker)

            if self.options_candidates:
                print(f"  Options available: {', '.join(self.options_candidates)}")

                # Save options data
                options_df = pd.DataFrame([r.to_dict() for r in options_results])
                options_file = Path("data/options_candidates.csv")
                options_df.to_csv(options_file, index=False)
                print(f"  Options data saved: {options_file}")
            else:
                print("  No top-20 picks have liquid options (expected for micro-caps)")

        self.step_times['liquidity'] = time.time() - start_time
        return True

    # =========================================================================
    # STEP 6: Build Portfolio
    # =========================================================================
    def step6_build_portfolio(self) -> bool:
        """Apply risk management and build portfolio."""
        self._log_step(6, "BUILD PORTFOLIO")
        start_time = time.time()

        if not self.predictions:
            print("  ERROR: No predictions available!")
            return False

        # Get current prices for predictions
        predictions_with_prices = []
        for pred in self.predictions:
            # Get price from merged_df
            row = self.merged_df[self.merged_df['ticker'] == pred.ticker]
            if not row.empty:
                price = row.iloc[0].get('current_price')
                if pd.notna(price) and price > 0:
                    predictions_with_prices.append({
                        'ticker': pred.ticker,
                        'prediction': pred.prediction,
                        'confidence': pred.confidence,
                        'current_price': price,
                        'sector': row.iloc[0].get('sector')
                    })

        print(f"  Predictions with valid prices: {len(predictions_with_prices)}")

        # Build portfolio with liquidity filtering
        risk_manager = RiskManager(
            max_positions=10,
            stop_loss_pct=0.15,
            take_profit_pct=0.30
        )

        self.portfolio = risk_manager.build_portfolio(
            predictions=predictions_with_prices,
            portfolio_value=self.capital,
            min_confidence=0.60,
            liquidity_data=self.liquidity_data
        )

        print(f"  Portfolio positions: {len(self.portfolio)}")

        total_allocated = sum(p.dollar_amount for p in self.portfolio)
        print(f"  Total allocated: ${total_allocated:,.0f} ({total_allocated/self.capital:.0%})")

        # Save portfolio
        if not self.dry_run:
            output_df = pd.DataFrame([p.to_dict() for p in self.portfolio])
            output_file = self.data_dir / f"portfolio_{self.date_str}.csv"
            output_df.to_csv(output_file, index=False)
            print(f"  Portfolio saved: {output_file}")

        self.step_times['portfolio'] = time.time() - start_time
        return True

    # =========================================================================
    # STEP 7: Log Paper Trades
    # =========================================================================
    def step7_log_paper_trades(self) -> bool:
        """Log portfolio positions to paper trading tracker."""
        self._log_step(7, "LOG PAPER TRADES")
        start_time = time.time()

        if not self.portfolio:
            print("  ERROR: No portfolio available!")
            return False

        trader = PaperTrader()

        logged = 0
        skipped = 0

        for position in self.portfolio:
            # Check if already logged today
            existing = [
                t for t in trader.trades
                if t.ticker == position.ticker and
                t.entry_date.date() == self.as_of_date.date()
            ]

            if existing:
                print(f"  {position.ticker}: Already logged today, skipping")
                skipped += 1
                continue

            if self.dry_run:
                print(f"  [DRY RUN] Would log: {position.ticker} @ ${position.current_price:.2f}")
                logged += 1
            else:
                trader.log_prediction(
                    ticker=position.ticker,
                    prediction=1,  # All portfolio positions are BUY
                    predicted_direction="OUTPERFORM",
                    confidence=position.confidence,
                    entry_price=position.current_price,
                    notes=f"Weekly analysis {self.date_str}"
                )
                logged += 1
                print(f"  {position.ticker}: Logged @ ${position.current_price:.2f} (conf: {position.confidence:.0%})")

        print(f"\n  Logged: {logged}, Skipped: {skipped}")
        print(f"  Total trades in tracker: {len(trader.trades)}")

        self.step_times['paper_trades'] = time.time() - start_time
        return True

    # =========================================================================
    # STEP 8: Update Past Trades
    # =========================================================================
    def step8_update_past_trades(self) -> bool:
        """Update past paper trades with current prices."""
        self._log_step(8, "UPDATE PAST TRADES")
        start_time = time.time()

        trader = PaperTrader()

        if not trader.trades:
            print("  No existing trades to update")
            return True

        print(f"  Updating {len(trader.trades)} trades...")

        if self.dry_run:
            print(f"  [DRY RUN] Would update prices for {len(trader.trades)} trades")
        else:
            results = trader.update_results()
            print(f"  Trades updated: {results['trades_updated']}")
            print(f"  Trades completed: {results['trades_completed']}")
            if results['errors']:
                print(f"  Errors: {len(results['errors'])}")

        # Get performance summary
        self.paper_trades_summary = trader.get_performance_summary()

        self.step_times['update_trades'] = time.time() - start_time
        return True

    # =========================================================================
    # STEP 9: Generate Report
    # =========================================================================
    def step9_generate_report(self) -> str:
        """Generate weekly summary report."""
        self._log_step(9, "GENERATE REPORT")
        start_time = time.time()

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"MICRO-CAP WEEKLY ANALYSIS - {self.as_of_date.strftime('%Y-%m-%d')}")
        report_lines.append("=" * 70)
        report_lines.append("")

        # Universe summary
        if self.universe_df is not None:
            report_lines.append(f"Universe: {len(self.universe_df)} stocks screened")
        if self.metrics_df is not None:
            report_lines.append(f"Analyzed: {len(self.metrics_df)} stocks with valid metrics")
        report_lines.append("")

        # Top ranked stocks
        report_lines.append("-" * 70)
        report_lines.append("TOP 20 RANKED STOCKS")
        report_lines.append("-" * 70)

        if self.predictions:
            predictions_sorted = sorted(self.predictions, key=lambda x: x.rank)
            for pred in predictions_sorted[:20]:
                # Check if options available
                options_flag = ""
                if pred.ticker in self.options_candidates:
                    options_flag = " [OPTIONS]"
                report_lines.append(
                    f"Rank {pred.rank:3}: {pred.ticker:<6} ({pred.confidence:.1%} confidence){options_flag}"
                )

        report_lines.append("")

        # Options candidates
        if self.options_candidates:
            report_lines.append("-" * 70)
            report_lines.append("OPTIONS CANDIDATES")
            report_lines.append("-" * 70)
            report_lines.append(f"Top-20 picks with liquid options: {len(self.options_candidates)}")
            for ticker in self.options_candidates:
                opt = self.options_data.get(ticker)
                if opt:
                    report_lines.append(
                        f"  {ticker}: ${opt.nearest_strike} strike, "
                        f"${opt.call_premium:.2f} premium, "
                        f"OI={opt.open_interest}, "
                        f"exp {opt.expiry}"
                    )
            report_lines.append("")

        # Portfolio recommendations
        report_lines.append("-" * 70)
        report_lines.append("PORTFOLIO RECOMMENDATIONS")
        report_lines.append("-" * 70)

        if self.portfolio:
            for i, pos in enumerate(self.portfolio[:10], 1):
                sector = pos.sector or "Unknown"
                report_lines.append(
                    f"{i:2}. {pos.ticker:<6} - Buy ${pos.dollar_amount:,.0f} "
                    f"({pos.shares} shares @ ${pos.current_price:.2f}) [{sector}]"
                )
                report_lines.append(
                    f"     Stop: ${pos.stop_loss:.2f} | Target: ${pos.take_profit:.2f}"
                )
        else:
            report_lines.append("No portfolio recommendations generated")

        report_lines.append("")

        # Paper trading performance
        report_lines.append("-" * 70)
        report_lines.append("PAPER TRADING PERFORMANCE")
        report_lines.append("-" * 70)

        if self.paper_trades_summary and 'error' not in self.paper_trades_summary:
            summary = self.paper_trades_summary
            report_lines.append(f"Total positions tracked: {summary.get('total_trades', 0)}")
            report_lines.append(f"Active positions: {summary.get('pending_trades', 0)}")
            report_lines.append(f"Completed trades: {summary.get('completed_trades', 0)}")

            if summary.get('completed_trades', 0) > 0:
                report_lines.append(f"Win rate: {summary.get('win_rate', 0):.1%}")
                report_lines.append(f"Average return: {summary.get('avg_return_pct', 0):+.1f}%")

                if 'best_trade' in summary:
                    best = summary['best_trade']
                    report_lines.append(f"Best trade: {best['ticker']} ({best['return_pct']:+.1f}%)")
                if 'worst_trade' in summary:
                    worst = summary['worst_trade']
                    report_lines.append(f"Worst trade: {worst['ticker']} ({worst['return_pct']:+.1f}%)")

            if summary.get('pending_trades', 0) > 0 and 'pending_avg_return_pct' in summary:
                report_lines.append(f"Current unrealized return: {summary.get('pending_avg_return_pct', 0):+.1f}%")
        else:
            report_lines.append("No paper trading history available")

        report_lines.append("")

        # Sector breakdown
        report_lines.append("-" * 70)
        report_lines.append("SECTOR BREAKDOWN")
        report_lines.append("-" * 70)

        if self.portfolio:
            sector_counts = {}
            for pos in self.portfolio:
                sector = pos.sector or "Unknown"
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
                report_lines.append(f"  {sector}: {count} position{'s' if count > 1 else ''}")
        else:
            report_lines.append("No sector data available")

        report_lines.append("")

        # Liquidity summary
        report_lines.append("-" * 70)
        report_lines.append("LIQUIDITY SUMMARY")
        report_lines.append("-" * 70)

        if self.liquidity_summary:
            report_lines.append(f"Tradeable stocks (A-D): {self.liquidity_summary['tradeable']}")
            report_lines.append(f"Untradeable stocks (F): {self.liquidity_summary['untradeable']}")
            report_lines.append(f"Average spread: {self.liquidity_summary['avg_spread_pct']}%")
            report_lines.append(f"Median spread: {self.liquidity_summary['median_spread_pct']}%")
            report_lines.append("")
            report_lines.append("Grade distribution:")
            for grade, count in self.liquidity_summary['grade_counts'].items():
                report_lines.append(f"  {grade}: {count} stocks")
            if self.liquidity_summary['poor_liquidity_tickers']:
                report_lines.append("")
                report_lines.append(f"Poor liquidity (D/F): {len(self.liquidity_summary['poor_liquidity_tickers'])} stocks excluded/reduced")
        else:
            report_lines.append("No liquidity data available")

        report_lines.append("")

        # Feature importance
        report_lines.append("-" * 70)
        report_lines.append("TOP PREDICTIVE FEATURES")
        report_lines.append("-" * 70)

        if hasattr(self, 'feature_importance') and self.feature_importance:
            for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:10], 1):
                report_lines.append(f"  {i:2}. {feature}: {importance:.4f}")
        else:
            report_lines.append("No feature importance data available")

        report_lines.append("")

        # Timing summary
        report_lines.append("-" * 70)
        report_lines.append("EXECUTION TIMING")
        report_lines.append("-" * 70)

        total_time = sum(self.step_times.values())
        for step, duration in self.step_times.items():
            report_lines.append(f"  {step}: {duration:.1f}s")
        report_lines.append(f"  TOTAL: {total_time:.1f}s ({total_time/60:.1f} min)")

        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append(f"Report generated: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("=" * 70)

        report_text = "\n".join(report_lines)

        # Save report
        if not self.dry_run:
            report_file = self.reports_dir / f"weekly_{self.date_str}.txt"
            report_file.write_text(report_text)
            print(f"\n  Report saved: {report_file}")

        self.step_times['report'] = time.time() - start_time

        return report_text

    # =========================================================================
    # Main Run Method
    # =========================================================================
    def run(
        self,
        skip_universe: bool = False,
        limit: Optional[int] = None
    ) -> bool:
        """Run the complete weekly analysis pipeline."""

        print("\n" + "#" * 70)
        print("#" + " " * 68 + "#")
        print("#" + "  MICRO-CAP WEEKLY ANALYSIS PIPELINE".center(68) + "#")
        print("#" + f"  {self.as_of_date.strftime('%Y-%m-%d %H:%M:%S UTC')}".center(68) + "#")
        print("#" + " " * 68 + "#")
        print("#" * 70)

        if self.dry_run:
            print("\n*** DRY RUN MODE - No changes will be saved ***\n")

        # Step 1: Update Universe
        if not self.step1_update_universe(skip=skip_universe):
            return False

        # Step 2: Pull Metrics
        if not self.step2_pull_metrics(limit=limit):
            return False

        # Step 3: Pull Sentiment
        if not self.step3_pull_sentiment(limit=limit):
            return False

        # Step 4: Run Predictions
        if not self.step4_run_predictions():
            return False

        # Step 5: Check Liquidity
        if not self.step5_check_liquidity():
            return False

        # Step 6: Build Portfolio
        if not self.step6_build_portfolio():
            return False

        # Step 7: Log Paper Trades
        if not self.step7_log_paper_trades():
            return False

        # Step 8: Update Past Trades
        if not self.step8_update_past_trades():
            return False

        # Step 9: Generate Report
        report = self.step9_generate_report()

        # Print final report
        print("\n" + report)

        return True

    # =========================================================================
    # STEP 10: Log to Forward Test (Optional)
    # =========================================================================
    def step10_log_forward_test(self, top_n: int = 20, model_version: str = "rf_v1") -> bool:
        """Log top predictions to forward test tracker."""
        self._log_step(10, "LOG FORWARD TEST")
        start_time = time.time()

        if not self.predictions:
            print("  ERROR: No predictions available!")
            return False

        tracker = ForwardTestTracker()

        # Show countdown
        print(f"  {tracker.get_countdown_string()}")

        # Prepare predictions DataFrame
        predictions_sorted = sorted(self.predictions, key=lambda x: x.rank)
        pred_df = pd.DataFrame([
            {
                'ticker': p.ticker,
                'rank': p.rank,
                'confidence': p.confidence,
                'predicted_direction': p.predicted_direction
            }
            for p in predictions_sorted
        ])

        if self.dry_run:
            print(f"  [DRY RUN] Would log top {top_n} predictions to forward test")
        else:
            logged = tracker.log_weekly_predictions(
                pred_df,
                top_n=top_n,
                model_version=model_version
            )
            print(f"  Logged {logged} predictions to forward test tracker")

            if logged == 0:
                print("  Note: Predictions may already be logged for today")

        self.step_times['forward_test'] = time.time() - start_time
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run weekly micro-cap stock analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--skip-universe',
        action='store_true',
        help="Don't refresh universe (use existing)"
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=150,
        help="Limit number of stocks to process (default: 150)"
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would happen without making changes"
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help="Portfolio capital (default: $10,000)"
    )

    parser.add_argument(
        '--forward-test',
        action='store_true',
        default=True,
        help="Log predictions to forward test tracker (default: enabled)"
    )

    parser.add_argument(
        '--no-forward-test',
        action='store_true',
        help="Disable forward test logging"
    )

    parser.add_argument(
        '--model-version',
        type=str,
        default="rf_v1",
        help="Model version string for forward test tracking (default: rf_v1)"
    )

    args = parser.parse_args()

    # Handle forward test flag
    do_forward_test = args.forward_test and not args.no_forward_test

    analyzer = WeeklyAnalyzer(
        capital=args.capital,
        dry_run=args.dry_run
    )

    success = analyzer.run(
        skip_universe=args.skip_universe,
        limit=args.limit
    )

    # Log to forward test if enabled and analysis succeeded
    if success and do_forward_test:
        analyzer.step10_log_forward_test(
            top_n=20,
            model_version=args.model_version
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
