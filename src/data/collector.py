"""
Data collection module for micro-cap stock analysis.

Enforces temporal constraints and US-only filtering.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
import pytz

import sys
sys.path.append('..')
from utils.temporal import (
    ensure_utc,
    calculate_reporting_date,
    get_fiscal_quarter,
    validate_temporal_consistency,
    TemporalDataPoint
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityTracker:
    """Track data quality and completeness for micro-cap stocks."""

    def __init__(self):
        self.missing_fields = []
        self.non_us_tickers = []
        self.temporal_violations = []

    def log_missing(self, ticker: str, field: str, as_of_date: datetime):
        """Log a missing data field."""
        self.missing_fields.append({
            'ticker': ticker,
            'field': field,
            'as_of_date': as_of_date.date(),
            'timestamp': datetime.now(pytz.utc)
        })
        logger.warning(f"Missing data: {ticker} - {field} as of {as_of_date.date()}")

    def log_non_us_ticker(self, ticker: str):
        """Log a non-US ticker that was filtered out."""
        self.non_us_tickers.append(ticker)
        logger.warning(f"Filtered out non-US ticker: {ticker}")

    def log_temporal_violation(self, ticker: str, issue: str):
        """Log a temporal/lookahead bias violation."""
        self.temporal_violations.append({
            'ticker': ticker,
            'issue': issue,
            'timestamp': datetime.now(pytz.utc)
        })
        logger.error(f"TEMPORAL VIOLATION: {ticker} - {issue}")

    def get_summary(self) -> Dict:
        """Get summary of data quality issues."""
        return {
            'missing_fields_count': len(self.missing_fields),
            'non_us_tickers_count': len(self.non_us_tickers),
            'temporal_violations_count': len(self.temporal_violations),
            'missing_fields': self.missing_fields,
            'non_us_tickers': self.non_us_tickers,
            'temporal_violations': self.temporal_violations
        }


def is_us_ticker(ticker: str) -> bool:
    """
    Check if ticker is a US stock.

    Filters out international stocks by checking for country suffixes.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True if US stock, False otherwise
    """
    # Common non-US suffixes
    non_us_suffixes = [
        '.L',   # London
        '.TO',  # Toronto
        '.V',   # Vancouver
        '.AX',  # Australia
        '.NS',  # India NSE
        '.BO',  # India BSE
        '.HK',  # Hong Kong
        '.T',   # Tokyo
        '.PA',  # Paris
        '.DE',  # Germany
        '.SW',  # Switzerland
        '.MI',  # Milan
        '.AS',  # Amsterdam
        '.BR',  # Brussels
        '.CO',  # Copenhagen
        '.HE',  # Helsinki
        '.OL',  # Oslo
        '.ST',  # Stockholm
    ]

    ticker_upper = ticker.upper()

    for suffix in non_us_suffixes:
        if ticker_upper.endswith(suffix):
            return False

    return True


class FinancialDataCollector:
    """
    Collects financial data with strict temporal controls.

    All methods require as_of_date parameter to enforce lookahead bias prevention.
    """

    def __init__(self):
        self.quality_tracker = DataQualityTracker()

    def get_price_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        as_of_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical price data with temporal validation.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for price history (timezone-aware)
            end_date: End date for price history (timezone-aware)
            as_of_date: Date from which we're analyzing (timezone-aware)

        Returns:
            DataFrame with price data, empty if validation fails
        """
        # Validate inputs
        as_of_date = ensure_utc(as_of_date)
        start_date = ensure_utc(start_date)
        end_date = ensure_utc(end_date)

        # Check US-only filter
        if not is_us_ticker(ticker):
            self.quality_tracker.log_non_us_ticker(ticker)
            logger.error(f"Rejected non-US ticker: {ticker}")
            return pd.DataFrame()

        # Temporal validation: end_date must be <= as_of_date
        if not validate_temporal_consistency(as_of_date, end_date):
            self.quality_tracker.log_temporal_violation(
                ticker,
                f"Price end_date {end_date.date()} > as_of_date {as_of_date.date()}"
            )
            return pd.DataFrame()

        logger.info(
            f"Fetching prices for {ticker}: "
            f"{start_date.date()} to {end_date.date()} "
            f"(as of {as_of_date.date()})"
        )

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No price data returned for {ticker}")
                self.quality_tracker.log_missing(ticker, 'price_data', as_of_date)

            # Make index timezone-aware
            df.index = df.index.tz_localize('UTC')

            return df

        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            self.quality_tracker.log_missing(ticker, 'price_data', as_of_date)
            return pd.DataFrame()

    def get_fundamental_metrics(
        self,
        ticker: str,
        as_of_date: datetime
    ) -> Dict[str, TemporalDataPoint]:
        """
        Get fundamental metrics with data availability dates.

        CRITICAL: This method estimates when quarterly data became available
        based on reporting lag rules.

        Args:
            ticker: Stock ticker symbol
            as_of_date: Date from which we're analyzing (timezone-aware)

        Returns:
            Dictionary of metric name -> TemporalDataPoint
        """
        as_of_date = ensure_utc(as_of_date)

        if not is_us_ticker(ticker):
            self.quality_tracker.log_non_us_ticker(ticker)
            return {}

        logger.info(f"Fetching fundamentals for {ticker} as of {as_of_date.date()}")

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get most recent quarterly financials
            quarterly_financials = stock.quarterly_financials
            quarterly_balance = stock.quarterly_balance_sheet

            # Determine the most recent quarter end and reporting date
            if not quarterly_financials.empty:
                most_recent_quarter_end = quarterly_financials.columns[0]

                # Convert to timezone-aware datetime
                if isinstance(most_recent_quarter_end, pd.Timestamp):
                    most_recent_quarter_end = most_recent_quarter_end.to_pydatetime()
                most_recent_quarter_end = ensure_utc(most_recent_quarter_end)

                # Get fiscal quarter
                _, fiscal_quarter = get_fiscal_quarter(most_recent_quarter_end)

                # Calculate when this data became available
                reporting_date = calculate_reporting_date(
                    most_recent_quarter_end,
                    fiscal_quarter
                )

                # Only use this data if it was available by as_of_date
                if not validate_temporal_consistency(as_of_date, reporting_date):
                    logger.warning(
                        f"{ticker}: Most recent data (Q{fiscal_quarter} {most_recent_quarter_end.date()}) "
                        f"not available until {reporting_date.date()}, "
                        f"but as_of_date is {as_of_date.date()}"
                    )
                    # Use previous quarter's data instead
                    if len(quarterly_financials.columns) > 1:
                        most_recent_quarter_end = quarterly_financials.columns[1]
                        most_recent_quarter_end = ensure_utc(most_recent_quarter_end.to_pydatetime())
                        _, fiscal_quarter = get_fiscal_quarter(most_recent_quarter_end)
                        reporting_date = calculate_reporting_date(
                            most_recent_quarter_end,
                            fiscal_quarter
                        )
                    else:
                        logger.error(f"{ticker}: No valid historical data available")
                        return {}

            else:
                logger.warning(f"{ticker}: No quarterly financials available")
                self.quality_tracker.log_missing(ticker, 'quarterly_financials', as_of_date)
                # Use a conservative estimate for info fields
                reporting_date = as_of_date - timedelta(days=60)

            logger.info(
                f"{ticker}: Using data available as of {reporting_date.date()} "
                f"for analysis as of {as_of_date.date()}"
            )

            # Extract metrics with temporal tracking
            metrics = {}

            # P/E ratio
            pe_ratio = info.get('trailingPE', np.nan)
            metrics['PE'] = TemporalDataPoint(
                value=pe_ratio if not pd.isna(pe_ratio) else None,
                data_date=reporting_date,
                ticker=ticker,
                metric='PE'
            )
            if pd.isna(pe_ratio):
                self.quality_tracker.log_missing(ticker, 'PE', as_of_date)

            # P/B ratio
            pb_ratio = info.get('priceToBook', np.nan)
            metrics['PB'] = TemporalDataPoint(
                value=pb_ratio if not pd.isna(pb_ratio) else None,
                data_date=reporting_date,
                ticker=ticker,
                metric='PB'
            )
            if pd.isna(pb_ratio):
                self.quality_tracker.log_missing(ticker, 'PB', as_of_date)

            # Market cap
            market_cap = info.get('marketCap', np.nan)
            metrics['MarketCap'] = TemporalDataPoint(
                value=market_cap if not pd.isna(market_cap) else None,
                data_date=reporting_date,
                ticker=ticker,
                metric='MarketCap'
            )

            # Debt to equity
            debt_to_equity = info.get('debtToEquity', np.nan)
            metrics['DebtToEquity'] = TemporalDataPoint(
                value=debt_to_equity if not pd.isna(debt_to_equity) else None,
                data_date=reporting_date,
                ticker=ticker,
                metric='DebtToEquity'
            )
            if pd.isna(debt_to_equity):
                self.quality_tracker.log_missing(ticker, 'DebtToEquity', as_of_date)

            # Short interest
            short_ratio = info.get('shortRatio', np.nan)
            metrics['ShortRatio'] = TemporalDataPoint(
                value=short_ratio if not pd.isna(short_ratio) else None,
                data_date=reporting_date,
                ticker=ticker,
                metric='ShortRatio'
            )
            if pd.isna(short_ratio):
                self.quality_tracker.log_missing(ticker, 'ShortRatio', as_of_date)

            # Insider ownership
            insider_ownership = info.get('heldPercentInsiders', np.nan)
            metrics['InsiderOwnership'] = TemporalDataPoint(
                value=insider_ownership if not pd.isna(insider_ownership) else None,
                data_date=reporting_date,
                ticker=ticker,
                metric='InsiderOwnership'
            )
            if pd.isna(insider_ownership):
                self.quality_tracker.log_missing(ticker, 'InsiderOwnership', as_of_date)

            return metrics

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {}

    def calculate_forward_returns(
        self,
        ticker: str,
        start_date: datetime,
        prediction_window_days: int,
        as_of_date: datetime
    ) -> Optional[float]:
        """
        Calculate forward returns for backtesting.

        CRITICAL: start_date must be AFTER as_of_date to prevent lookahead bias.

        Args:
            ticker: Stock ticker symbol
            start_date: When the prediction window starts (timezone-aware)
            prediction_window_days: Length of prediction window (e.g., 60, 90, 180)
            as_of_date: Date when prediction was made (timezone-aware)

        Returns:
            Forward return as decimal (e.g., 0.15 = 15% gain), or None if invalid
        """
        as_of_date = ensure_utc(as_of_date)
        start_date = ensure_utc(start_date)

        # CRITICAL CHECK: start_date must be >= as_of_date
        if start_date < as_of_date:
            self.quality_tracker.log_temporal_violation(
                ticker,
                f"Forward return start_date {start_date.date()} < as_of_date {as_of_date.date()}"
            )
            logger.error(
                f"LOOKAHEAD BIAS DETECTED: {ticker} - "
                f"Cannot calculate returns starting {start_date.date()} "
                f"using data as of {as_of_date.date()}"
            )
            return None

        end_date = start_date + timedelta(days=prediction_window_days)

        logger.info(
            f"{ticker}: Calculating forward return from {start_date.date()} "
            f"to {end_date.date()} ({prediction_window_days} days) "
            f"using data as of {as_of_date.date()}"
        )

        try:
            # Get price data
            price_df = self.get_price_data(
                ticker=ticker,
                start_date=start_date - timedelta(days=5),  # Buffer for market closed days
                end_date=end_date + timedelta(days=5),
                as_of_date=end_date  # We can know these prices when backtesting
            )

            if price_df.empty:
                logger.warning(f"{ticker}: No price data for forward return calculation")
                return None

            # Get prices closest to start and end dates
            start_prices = price_df[price_df.index >= start_date]
            end_prices = price_df[price_df.index >= end_date]

            if start_prices.empty or end_prices.empty:
                logger.warning(f"{ticker}: Insufficient price data for return calculation")
                return None

            start_price = start_prices.iloc[0]['Close']
            end_price = end_prices.iloc[0]['Close']

            forward_return = (end_price - start_price) / start_price

            logger.info(
                f"{ticker}: Forward return = {forward_return:.2%} "
                f"(${start_price:.2f} -> ${end_price:.2f})"
            )

            return forward_return

        except Exception as e:
            logger.error(f"Error calculating forward return for {ticker}: {e}")
            return None
