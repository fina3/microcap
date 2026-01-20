"""
Earnings surprise fetcher for micro-cap stock analysis.

Fetches actual vs expected EPS from Yahoo Finance to calculate
earnings surprise: (actual - expected) / abs(expected)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass

import yfinance as yf
import pandas as pd
import numpy as np
import pytz

import sys
sys.path.append('..')
from utils.temporal import ensure_utc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EarningsSurprise:
    """Container for earnings surprise data."""
    ticker: str
    as_of_date: datetime
    report_date: Optional[datetime]
    actual_eps: Optional[float]
    expected_eps: Optional[float]
    surprise: Optional[float]  # (actual - expected) / abs(expected)
    surprise_pct: Optional[float]  # surprise * 100
    data_quality: float  # 0-1 quality score
    quality_flags: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'as_of_date': self.as_of_date,
            'report_date': self.report_date,
            'actual_eps': self.actual_eps,
            'expected_eps': self.expected_eps,
            'earnings_surprise': self.surprise,
            'earnings_surprise_pct': self.surprise_pct,
            'earnings_data_quality': self.data_quality,
            'earnings_quality_flags': ','.join(self.quality_flags) if self.quality_flags else ''
        }


class EarningsFetcher:
    """
    Fetches earnings surprise data from Yahoo Finance.

    Uses yfinance to get actual and expected EPS, then calculates
    the standardized earnings surprise metric.
    """

    def __init__(self):
        """Initialize fetcher."""
        logger.info("EarningsFetcher initialized")

    def fetch_earnings_surprise(
        self,
        ticker: str,
        as_of_date: datetime,
        lookback_quarters: int = 4
    ) -> EarningsSurprise:
        """
        Fetch earnings surprise for a ticker.

        Uses the most recent earnings report available as of as_of_date.
        Calculates: earnings_surprise = (actual - expected) / abs(expected)

        Args:
            ticker: Stock ticker symbol
            as_of_date: Date from which we're analyzing (timezone-aware)
            lookback_quarters: How many quarters back to search

        Returns:
            EarningsSurprise data object
        """
        as_of_date = ensure_utc(as_of_date)
        quality_flags = []

        logger.debug(f"{ticker}: Fetching earnings surprise as of {as_of_date.date()}")

        try:
            stock = yf.Ticker(ticker)

            # Get earnings history
            earnings_df = self._get_earnings_history(stock, ticker)

            if earnings_df is None or earnings_df.empty:
                logger.warning(f"{ticker}: No earnings history available")
                return self._create_empty_result(
                    ticker, as_of_date, ['NO_EARNINGS_HISTORY']
                )

            # Filter to earnings available as of as_of_date
            # Add buffer for reporting delay (earnings typically released 1-2 days after quarter end)
            earnings_df = self._filter_by_date(earnings_df, as_of_date)

            if earnings_df.empty:
                logger.warning(f"{ticker}: No earnings available as of {as_of_date.date()}")
                return self._create_empty_result(
                    ticker, as_of_date, ['NO_EARNINGS_BY_DATE']
                )

            # Get most recent earnings
            latest = earnings_df.iloc[0]

            actual_eps = self._safe_get(latest, 'epsActual')
            expected_eps = self._safe_get(latest, 'epsEstimate')
            report_date = self._get_report_date(latest)

            # Log what we found
            logger.debug(
                f"{ticker}: Found earnings - "
                f"actual={actual_eps}, expected={expected_eps}, "
                f"report_date={report_date}"
            )

            # Calculate surprise
            surprise, surprise_pct = self._calculate_surprise(
                actual_eps, expected_eps, ticker, quality_flags
            )

            # Determine data quality
            data_quality = self._calculate_quality(
                actual_eps, expected_eps, report_date, as_of_date, quality_flags
            )

            # Check if earnings are stale
            if report_date:
                days_old = (as_of_date - report_date).days
                if days_old > 120:  # More than ~1 quarter old
                    quality_flags.append('STALE_EARNINGS')
                    logger.debug(f"{ticker}: Earnings are {days_old} days old")

            logger.info(
                f"{ticker}: Earnings surprise = {surprise_pct:.1f}% "
                f"(actual={actual_eps}, expected={expected_eps}, quality={data_quality:.2f})"
                if surprise_pct is not None else
                f"{ticker}: Earnings surprise = N/A (quality={data_quality:.2f})"
            )

            return EarningsSurprise(
                ticker=ticker,
                as_of_date=as_of_date,
                report_date=report_date,
                actual_eps=actual_eps,
                expected_eps=expected_eps,
                surprise=surprise,
                surprise_pct=surprise_pct,
                data_quality=data_quality,
                quality_flags=quality_flags
            )

        except Exception as e:
            logger.error(f"{ticker}: Error fetching earnings - {e}")
            return self._create_empty_result(
                ticker, as_of_date, [f'FETCH_ERROR: {str(e)[:50]}']
            )

    def _get_earnings_history(
        self,
        stock: yf.Ticker,
        ticker: str
    ) -> Optional[pd.DataFrame]:
        """Get earnings history from Yahoo Finance."""
        try:
            # Try earnings_history first (has estimates)
            if hasattr(stock, 'earnings_history') and stock.earnings_history is not None:
                df = stock.earnings_history
                if df is not None and not df.empty:
                    logger.debug(f"{ticker}: Using earnings_history")
                    return df.sort_index(ascending=False)

            # Try earnings_dates (alternate source)
            if hasattr(stock, 'earnings_dates') and stock.earnings_dates is not None:
                df = stock.earnings_dates
                if df is not None and not df.empty:
                    logger.debug(f"{ticker}: Using earnings_dates")
                    # Rename columns if needed for consistency
                    if 'EPS Estimate' in df.columns and 'Reported EPS' in df.columns:
                        df = df.rename(columns={
                            'EPS Estimate': 'epsEstimate',
                            'Reported EPS': 'epsActual'
                        })
                    return df.sort_index(ascending=False)

            # Try quarterly_earnings as fallback
            if hasattr(stock, 'quarterly_earnings') and stock.quarterly_earnings is not None:
                df = stock.quarterly_earnings
                if df is not None and not df.empty:
                    logger.debug(f"{ticker}: Using quarterly_earnings (no estimates)")
                    # This typically only has actuals, not estimates
                    if 'Earnings' in df.columns:
                        df['epsActual'] = df['Earnings']
                    return df.sort_index(ascending=False)

            return None

        except Exception as e:
            logger.debug(f"{ticker}: Error getting earnings history - {e}")
            return None

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        as_of_date: datetime
    ) -> pd.DataFrame:
        """Filter earnings to those available as of as_of_date."""
        try:
            # Convert index to datetime if needed
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Make index timezone-aware if needed
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            # Filter to dates before as_of_date
            # Add small buffer for same-day availability
            cutoff = as_of_date + timedelta(days=1)
            filtered = df[df.index <= cutoff]

            return filtered.sort_index(ascending=False)

        except Exception as e:
            logger.debug(f"Error filtering by date: {e}")
            return df

    def _safe_get(self, row, column: str) -> Optional[float]:
        """Safely get a value from a row."""
        try:
            if column in row.index:
                val = row[column]
                if pd.notna(val):
                    return float(val)
            return None
        except Exception:
            return None

    def _get_report_date(self, row) -> Optional[datetime]:
        """Extract report date from earnings row."""
        try:
            # The index is typically the date
            if hasattr(row, 'name'):
                date = row.name
                if isinstance(date, pd.Timestamp):
                    return date.to_pydatetime().replace(tzinfo=pytz.utc)
                elif isinstance(date, datetime):
                    return ensure_utc(date)
            return None
        except Exception:
            return None

    def _calculate_surprise(
        self,
        actual: Optional[float],
        expected: Optional[float],
        ticker: str,
        quality_flags: List[str]
    ) -> tuple:
        """
        Calculate earnings surprise.

        Formula: (actual - expected) / abs(expected)

        Returns:
            Tuple of (surprise, surprise_pct) or (None, None) if cannot calculate
        """
        if actual is None:
            quality_flags.append('MISSING_ACTUAL_EPS')
            return None, None

        if expected is None:
            quality_flags.append('MISSING_EXPECTED_EPS')
            return None, None

        if expected == 0:
            quality_flags.append('ZERO_EXPECTED_EPS')
            # Can't divide by zero, but we can note direction
            if actual > 0:
                return float('inf'), float('inf')
            elif actual < 0:
                return float('-inf'), float('-inf')
            else:
                return 0.0, 0.0

        surprise = (actual - expected) / abs(expected)
        surprise_pct = surprise * 100

        # Flag extreme surprises
        if abs(surprise) > 2.0:  # More than 200% surprise
            quality_flags.append('EXTREME_SURPRISE')
            logger.debug(f"{ticker}: Extreme earnings surprise: {surprise_pct:.1f}%")

        return surprise, surprise_pct

    def _calculate_quality(
        self,
        actual: Optional[float],
        expected: Optional[float],
        report_date: Optional[datetime],
        as_of_date: datetime,
        quality_flags: List[str]
    ) -> float:
        """Calculate data quality score (0-1)."""
        score = 1.0

        if actual is None:
            score -= 0.5
        if expected is None:
            score -= 0.3
        if report_date is None:
            score -= 0.1

        # Penalize stale data
        if report_date:
            days_old = (as_of_date - report_date).days
            if days_old > 180:
                score -= 0.2
            elif days_old > 120:
                score -= 0.1

        # Penalize based on flags
        score -= len(quality_flags) * 0.05

        return max(0.0, min(1.0, score))

    def _create_empty_result(
        self,
        ticker: str,
        as_of_date: datetime,
        quality_flags: List[str]
    ) -> EarningsSurprise:
        """Create empty result with quality flags."""
        return EarningsSurprise(
            ticker=ticker,
            as_of_date=as_of_date,
            report_date=None,
            actual_eps=None,
            expected_eps=None,
            surprise=None,
            surprise_pct=None,
            data_quality=0.0,
            quality_flags=quality_flags
        )


def fetch_earnings_surprise(
    ticker: str,
    as_of_date: Optional[datetime] = None
) -> EarningsSurprise:
    """
    Convenience function to fetch earnings surprise for a ticker.

    Args:
        ticker: Stock ticker symbol
        as_of_date: Date for analysis (default: now)

    Returns:
        EarningsSurprise object
    """
    if as_of_date is None:
        as_of_date = datetime.now(pytz.utc)

    fetcher = EarningsFetcher()
    return fetcher.fetch_earnings_surprise(ticker, as_of_date)
