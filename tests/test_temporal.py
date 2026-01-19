"""
Unit tests for temporal utilities.

Tests the core functions that prevent lookahead bias.
"""

import unittest
from datetime import datetime
import pytz
import sys
sys.path.append('../src')

from utils.temporal import (
    ensure_utc,
    calculate_reporting_date,
    get_fiscal_quarter,
    validate_temporal_consistency,
    TemporalDataPoint
)


class TestTemporalUtils(unittest.TestCase):
    """Test temporal utility functions."""

    def test_ensure_utc_naive_datetime(self):
        """Test conversion of naive datetime to UTC."""
        naive_dt = datetime(2024, 1, 15, 12, 0, 0)
        utc_dt = ensure_utc(naive_dt)

        self.assertIsNotNone(utc_dt.tzinfo)
        self.assertEqual(utc_dt.tzinfo, pytz.utc)

    def test_ensure_utc_aware_datetime(self):
        """Test handling of already timezone-aware datetime."""
        aware_dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=pytz.utc)
        utc_dt = ensure_utc(aware_dt)

        self.assertEqual(utc_dt, aware_dt)

    def test_calculate_reporting_date_q1(self):
        """Test Q1 reporting date calculation (45 day delay)."""
        q1_end = datetime(2024, 3, 31, tzinfo=pytz.utc)
        reporting_date = calculate_reporting_date(q1_end, fiscal_quarter=1)

        expected = datetime(2024, 5, 15, tzinfo=pytz.utc)  # 45 days later
        self.assertEqual(reporting_date, expected)

    def test_calculate_reporting_date_q4(self):
        """Test Q4 reporting date calculation (60 day delay)."""
        q4_end = datetime(2023, 12, 31, tzinfo=pytz.utc)
        reporting_date = calculate_reporting_date(q4_end, fiscal_quarter=4)

        expected = datetime(2024, 2, 29, tzinfo=pytz.utc)  # 60 days later (leap year)
        self.assertEqual(reporting_date, expected)

    def test_get_fiscal_quarter(self):
        """Test fiscal quarter determination."""
        # Q1
        q1_date = datetime(2024, 2, 15, tzinfo=pytz.utc)
        q1_end, q1_num = get_fiscal_quarter(q1_date)
        self.assertEqual(q1_num, 1)
        self.assertEqual(q1_end.month, 3)

        # Q4
        q4_date = datetime(2024, 11, 15, tzinfo=pytz.utc)
        q4_end, q4_num = get_fiscal_quarter(q4_date)
        self.assertEqual(q4_num, 4)
        self.assertEqual(q4_end.month, 12)

    def test_validate_temporal_consistency_valid(self):
        """Test temporal validation with valid dates."""
        as_of_date = datetime(2024, 3, 1, tzinfo=pytz.utc)
        data_date = datetime(2024, 2, 15, tzinfo=pytz.utc)

        is_valid = validate_temporal_consistency(as_of_date, data_date)
        self.assertTrue(is_valid)

    def test_validate_temporal_consistency_invalid(self):
        """Test temporal validation with invalid dates (lookahead bias)."""
        as_of_date = datetime(2024, 2, 15, tzinfo=pytz.utc)
        data_date = datetime(2024, 3, 1, tzinfo=pytz.utc)  # Future data!

        is_valid = validate_temporal_consistency(as_of_date, data_date)
        self.assertFalse(is_valid)

    def test_temporal_datapoint_availability(self):
        """Test TemporalDataPoint availability checking."""
        data_available = datetime(2024, 2, 15, tzinfo=pytz.utc)
        dp = TemporalDataPoint(
            value=15.5,
            data_date=data_available,
            ticker='AAPL',
            metric='PE'
        )

        # Should be available after data date
        as_of_future = datetime(2024, 3, 1, tzinfo=pytz.utc)
        self.assertTrue(dp.is_available_at(as_of_future))

        # Should NOT be available before data date
        as_of_past = datetime(2024, 2, 1, tzinfo=pytz.utc)
        self.assertFalse(dp.is_available_at(as_of_past))


class TestLookaheadBiasScenarios(unittest.TestCase):
    """Test specific lookahead bias scenarios."""

    def test_q4_data_before_february(self):
        """
        CRITICAL TEST: Ensure Q4 data is not used before mid-February.

        This is the most common lookahead bias mistake.
        """
        # Q4 2023 ends Dec 31, 2023
        q4_end = datetime(2023, 12, 31, tzinfo=pytz.utc)
        reporting_date = calculate_reporting_date(q4_end, fiscal_quarter=4)

        # Reporting date should be late February 2024
        self.assertEqual(reporting_date.year, 2024)
        self.assertEqual(reporting_date.month, 2)
        self.assertGreaterEqual(reporting_date.day, 28)  # At least Feb 28

        # Using data on Jan 15, 2024 should be INVALID
        as_of_jan = datetime(2024, 1, 15, tzinfo=pytz.utc)
        is_valid = validate_temporal_consistency(as_of_jan, reporting_date)
        self.assertFalse(is_valid, "Q4 data should NOT be available in January!")

        # Using data on March 1, 2024 should be VALID
        as_of_mar = datetime(2024, 3, 1, tzinfo=pytz.utc)
        is_valid = validate_temporal_consistency(as_of_mar, reporting_date)
        self.assertTrue(is_valid, "Q4 data should be available in March")

    def test_forward_returns_after_as_of_date(self):
        """
        CRITICAL TEST: Forward returns must start AFTER as_of_date.
        """
        as_of_date = datetime(2024, 1, 15, tzinfo=pytz.utc)

        # Prediction window starting same day - should be valid (allow_equal=True)
        prediction_start_same = datetime(2024, 1, 15, tzinfo=pytz.utc)
        is_valid = validate_temporal_consistency(
            prediction_start_same,
            as_of_date,
            allow_equal=True
        )
        self.assertTrue(is_valid)

        # Prediction window starting before - should be INVALID
        prediction_start_before = datetime(2024, 1, 14, tzinfo=pytz.utc)
        is_valid = validate_temporal_consistency(
            prediction_start_before,
            as_of_date,
            allow_equal=True
        )
        self.assertFalse(is_valid, "Cannot predict returns starting before as_of_date!")


if __name__ == '__main__':
    unittest.main()
