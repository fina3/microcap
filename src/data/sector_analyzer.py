"""
Sector Analyzer for comparing stocks to their sector averages.

A P/E of 15 means different things in Tech vs Utilities.
This module calculates sector-relative metrics for better comparison.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SectorStats:
    """Statistics for a sector."""
    sector: str
    count: int
    pe_median: float
    pb_median: float
    ps_median: float
    revenue_growth_median: float
    gross_margin_median: float
    operating_margin_median: float
    debt_to_equity_median: float


class SectorAnalyzer:
    """
    Analyzes stocks relative to their sector peers.

    Calculates sector medians and relative metrics:
    - pe_vs_sector = stock_pe / sector_median_pe (< 1.0 = cheaper)
    - pb_vs_sector = stock_pb / sector_median_pb
    - etc.
    """

    # Metrics to calculate sector-relative values for
    RELATIVE_METRICS = [
        'pe_trailing', 'pb_ratio', 'price_to_sales', 'debt_to_equity',
        'revenue_growth', 'gross_margin', 'operating_margin'
    ]

    def __init__(self, universe_file: Optional[Path] = None):
        """
        Initialize sector analyzer.

        Args:
            universe_file: Path to universe CSV with sector data
        """
        self.universe_file = universe_file
        self.sector_map: Dict[str, str] = {}  # ticker -> sector
        self.sector_stats: Dict[str, SectorStats] = {}  # sector -> stats

        if universe_file and Path(universe_file).exists():
            self._load_universe(universe_file)

        logger.info(f"SectorAnalyzer initialized with {len(self.sector_map)} tickers")

    def _load_universe(self, universe_file: Path):
        """Load sector mappings from universe file."""
        try:
            df = pd.read_csv(universe_file)
            if 'ticker' in df.columns and 'sector' in df.columns:
                for _, row in df.iterrows():
                    ticker = str(row['ticker']).upper()
                    sector = str(row['sector']).strip()
                    if sector and sector != 'nan':
                        self.sector_map[ticker] = sector
                logger.info(f"Loaded {len(self.sector_map)} ticker-sector mappings")
        except Exception as e:
            logger.error(f"Error loading universe file: {e}")

    def get_sector(self, ticker: str) -> Optional[str]:
        """Get sector for a ticker."""
        # Check cache first
        if ticker in self.sector_map:
            return self.sector_map[ticker]

        # Try to fetch from Yahoo Finance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector')
            if sector:
                self.sector_map[ticker] = sector
                return sector
        except Exception as e:
            logger.debug(f"Could not fetch sector for {ticker}: {e}")

        return None

    def add_sectors_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sector column to a dataframe with ticker column.

        Args:
            df: DataFrame with 'ticker' column

        Returns:
            DataFrame with 'sector' column added
        """
        if 'ticker' not in df.columns:
            logger.warning("No 'ticker' column found")
            return df

        df = df.copy()
        df['sector'] = df['ticker'].apply(self.get_sector)

        sectors_found = df['sector'].notna().sum()
        logger.info(f"Added sectors: {sectors_found}/{len(df)} tickers")

        return df

    def calculate_sector_stats(self, df: pd.DataFrame) -> Dict[str, SectorStats]:
        """
        Calculate median statistics for each sector.

        Args:
            df: DataFrame with metrics and 'sector' column

        Returns:
            Dictionary mapping sector names to SectorStats
        """
        if 'sector' not in df.columns:
            logger.warning("No 'sector' column - adding sectors first")
            df = self.add_sectors_to_dataframe(df)

        self.sector_stats = {}

        for sector, group in df.groupby('sector'):
            if pd.isna(sector) or len(group) < 2:
                continue

            stats = SectorStats(
                sector=sector,
                count=len(group),
                pe_median=self._safe_median(group, 'pe_trailing'),
                pb_median=self._safe_median(group, 'pb_ratio'),
                ps_median=self._safe_median(group, 'price_to_sales'),
                revenue_growth_median=self._safe_median(group, 'revenue_growth'),
                gross_margin_median=self._safe_median(group, 'gross_margin'),
                operating_margin_median=self._safe_median(group, 'operating_margin'),
                debt_to_equity_median=self._safe_median(group, 'debt_to_equity'),
            )
            self.sector_stats[sector] = stats

            logger.info(
                f"{sector} ({stats.count} stocks): "
                f"P/E={stats.pe_median:.1f}, P/B={stats.pb_median:.2f}, "
                f"Growth={stats.revenue_growth_median:.1%}"
            )

        return self.sector_stats

    def _safe_median(self, group: pd.DataFrame, column: str) -> float:
        """Calculate median, handling missing values."""
        if column not in group.columns:
            return np.nan
        values = group[column].dropna()
        # Filter out extreme outliers (beyond 3 std)
        if len(values) > 3:
            mean, std = values.mean(), values.std()
            if std > 0:
                values = values[(values >= mean - 3*std) & (values <= mean + 3*std)]
        return values.median() if len(values) > 0 else np.nan

    def add_sector_relative_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sector-relative metrics to a dataframe.

        For each metric, calculates: stock_value / sector_median
        Values < 1.0 indicate below-average (cheaper for valuation metrics)

        Args:
            df: DataFrame with metrics

        Returns:
            DataFrame with sector-relative columns added
        """
        if 'sector' not in df.columns:
            df = self.add_sectors_to_dataframe(df)

        if not self.sector_stats:
            self.calculate_sector_stats(df)

        df = df.copy()

        # Add sector-relative metrics
        metric_mapping = {
            'pe_trailing': 'pe_median',
            'pb_ratio': 'pb_median',
            'price_to_sales': 'ps_median',
            'revenue_growth': 'revenue_growth_median',
            'gross_margin': 'gross_margin_median',
            'operating_margin': 'operating_margin_median',
            'debt_to_equity': 'debt_to_equity_median',
        }

        for metric, stat_attr in metric_mapping.items():
            relative_col = f"{metric}_vs_sector"
            df[relative_col] = np.nan

            for idx, row in df.iterrows():
                sector = row.get('sector')
                if pd.isna(sector) or sector not in self.sector_stats:
                    continue

                stock_val = row.get(metric)
                sector_median = getattr(self.sector_stats[sector], stat_attr)

                if pd.notna(stock_val) and pd.notna(sector_median) and sector_median != 0:
                    df.at[idx, relative_col] = stock_val / sector_median

        # Add sector median columns for reference
        df['sector_pe_median'] = df['sector'].map(
            lambda s: self.sector_stats.get(s, SectorStats('', 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)).pe_median
        )
        df['sector_pb_median'] = df['sector'].map(
            lambda s: self.sector_stats.get(s, SectorStats('', 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)).pb_median
        )

        logger.info(f"Added {len(metric_mapping)} sector-relative metrics")

        return df

    def get_sector_comparison(self, ticker: str, metrics_row: pd.Series) -> Dict:
        """
        Get detailed sector comparison for a single ticker.

        Args:
            ticker: Stock ticker
            metrics_row: Series with stock metrics

        Returns:
            Dictionary with comparison details
        """
        sector = self.get_sector(ticker)
        if not sector or sector not in self.sector_stats:
            return {'sector': sector, 'comparison_available': False}

        stats = self.sector_stats[sector]

        comparison = {
            'sector': sector,
            'sector_count': stats.count,
            'comparison_available': True,
            'metrics': {}
        }

        # Compare key metrics
        comparisons = [
            ('pe_trailing', stats.pe_median, 'P/E', True),  # True = lower is better
            ('pb_ratio', stats.pb_median, 'P/B', True),
            ('price_to_sales', stats.ps_median, 'P/S', True),
            ('revenue_growth', stats.revenue_growth_median, 'Rev Growth', False),  # Higher is better
            ('gross_margin', stats.gross_margin_median, 'Gross Margin', False),
            ('operating_margin', stats.operating_margin_median, 'Op Margin', False),
        ]

        for metric, sector_median, label, lower_better in comparisons:
            stock_val = metrics_row.get(metric)
            if pd.notna(stock_val) and pd.notna(sector_median) and sector_median != 0:
                relative = stock_val / sector_median
                pct_diff = (stock_val - sector_median) / abs(sector_median) * 100

                # Determine if favorable
                if lower_better:
                    favorable = relative < 1.0
                    description = f"{abs(pct_diff):.0f}% {'discount' if favorable else 'premium'}"
                else:
                    favorable = relative > 1.0
                    description = f"{abs(pct_diff):.0f}% {'above' if favorable else 'below'} avg"

                comparison['metrics'][label] = {
                    'stock_value': stock_val,
                    'sector_median': sector_median,
                    'relative': relative,
                    'pct_diff': pct_diff,
                    'favorable': favorable,
                    'description': description
                }

        return comparison

    def print_sector_summary(self):
        """Print summary of sector statistics."""
        if not self.sector_stats:
            print("No sector statistics calculated yet.")
            return

        print("\n" + "=" * 80)
        print("SECTOR STATISTICS SUMMARY")
        print("=" * 80)
        print(f"\n{'Sector':<25} {'Count':>6} {'P/E':>8} {'P/B':>8} {'P/S':>8} {'Growth':>10}")
        print("-" * 80)

        for sector in sorted(self.sector_stats.keys()):
            stats = self.sector_stats[sector]
            growth_str = f"{stats.revenue_growth_median:.1%}" if pd.notna(stats.revenue_growth_median) else "N/A"
            print(
                f"{sector:<25} {stats.count:>6} "
                f"{stats.pe_median:>8.1f} {stats.pb_median:>8.2f} "
                f"{stats.ps_median:>8.2f} {growth_str:>10}"
            )

        print("=" * 80)
