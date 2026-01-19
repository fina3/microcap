"""
Temporal utilities for preventing lookahead bias in financial data analysis.

All functions enforce timezone-aware dates and track data availability dates.
"""

from datetime import datetime, timedelta
from typing import Optional
import pytz
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure a datetime object is timezone-aware in UTC.

    Args:
        dt: datetime object (naive or aware)

    Returns:
        Timezone-aware datetime in UTC

    Raises:
        ValueError: if dt is None
    """
    if dt is None:
        raise ValueError("Datetime cannot be None")

    if dt.tzinfo is None:
        logger.warning(f"Converting naive datetime {dt} to UTC")
        return pytz.utc.localize(dt)

    return dt.astimezone(pytz.utc)


def calculate_reporting_date(quarter_end: datetime, fiscal_quarter: int) -> datetime:
    """
    Calculate the earliest date when quarterly financial data becomes publicly available.

    CRITICAL: This prevents lookahead bias by enforcing realistic reporting delays.

    Rules:
    - Q1, Q2, Q3: 10-Q filed within 45 days of quarter end
    - Q4: 10-K filed within 90 days of fiscal year end (but typically ~60 days)
    - We conservatively add 45 days for all quarters, 60 days for Q4

    Args:
        quarter_end: End date of the fiscal quarter (timezone-aware)
        fiscal_quarter: Quarter number (1, 2, 3, or 4)

    Returns:
        Timezone-aware datetime when data is available
    """
    quarter_end = ensure_utc(quarter_end)

    if fiscal_quarter not in [1, 2, 3, 4]:
        raise ValueError(f"Invalid fiscal quarter: {fiscal_quarter}. Must be 1-4")

    if fiscal_quarter == 4:
        # Q4 data (10-K) typically available 60 days after year end
        delay_days = 60
        logger.info(f"Q4 data for {quarter_end.date()}: adding {delay_days} day delay")
    else:
        # Q1-Q3 data (10-Q) typically available 45 days after quarter end
        delay_days = 45
        logger.info(f"Q{fiscal_quarter} data for {quarter_end.date()}: adding {delay_days} day delay")

    reporting_date = quarter_end + timedelta(days=delay_days)
    return reporting_date


def get_fiscal_quarter(date: datetime) -> tuple[datetime, int]:
    """
    Determine fiscal quarter end date and quarter number for a given date.

    Assumes calendar year fiscal year (most common for US stocks).

    Args:
        date: Any date (timezone-aware)

    Returns:
        Tuple of (quarter_end_date, quarter_number)
    """
    date = ensure_utc(date)

    year = date.year
    month = date.month

    if month <= 3:
        quarter = 1
        quarter_end = datetime(year, 3, 31, tzinfo=pytz.utc)
    elif month <= 6:
        quarter = 2
        quarter_end = datetime(year, 6, 30, tzinfo=pytz.utc)
    elif month <= 9:
        quarter = 3
        quarter_end = datetime(year, 9, 30, tzinfo=pytz.utc)
    else:
        quarter = 4
        quarter_end = datetime(year, 12, 31, tzinfo=pytz.utc)

    return quarter_end, quarter


def validate_temporal_consistency(
    as_of_date: datetime,
    data_date: datetime,
    allow_equal: bool = True
) -> bool:
    """
    Validate that data_date comes before or equal to as_of_date.

    This prevents using future data in historical analysis.

    Args:
        as_of_date: The date from which we're analyzing (timezone-aware)
        data_date: The date when data became available (timezone-aware)
        allow_equal: Whether data_date == as_of_date is valid

    Returns:
        True if temporally consistent, False otherwise
    """
    as_of_date = ensure_utc(as_of_date)
    data_date = ensure_utc(data_date)

    if allow_equal:
        is_valid = data_date <= as_of_date
    else:
        is_valid = data_date < as_of_date

    if not is_valid:
        logger.error(
            f"TEMPORAL VIOLATION: Using data from {data_date.date()} "
            f"in analysis as of {as_of_date.date()}"
        )

    return is_valid


class TemporalDataPoint:
    """
    Represents a piece of financial data with its availability timestamp.

    Ensures all data points track when they became available to prevent lookahead bias.
    """

    def __init__(
        self,
        value: any,
        data_date: datetime,
        ticker: Optional[str] = None,
        metric: Optional[str] = None
    ):
        """
        Args:
            value: The actual data value (can be None for missing data)
            data_date: When this data became publicly available (timezone-aware)
            ticker: Stock ticker (optional)
            metric: Metric name (optional)
        """
        self.value = value
        self.data_date = ensure_utc(data_date)
        self.ticker = ticker
        self.metric = metric

    def is_available_at(self, as_of_date: datetime) -> bool:
        """Check if this data point was available at a given date."""
        return validate_temporal_consistency(
            as_of_date=as_of_date,
            data_date=self.data_date,
            allow_equal=True
        )

    def __repr__(self):
        return (
            f"TemporalDataPoint("
            f"ticker={self.ticker}, "
            f"metric={self.metric}, "
            f"value={self.value}, "
            f"available={self.data_date.date()})"
        )
