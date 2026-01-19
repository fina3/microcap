"""
Data validation utilities for checking lookahead bias and data quality.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
import logging

from .temporal import ensure_utc, validate_temporal_consistency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LookaheadBiasDetector:
    """
    Detect potential lookahead bias in datasets.

    Checks for common violations like:
    - Financial data dated after as_of_date
    - Forward returns starting before as_of_date
    - Q4 data used before mid-February
    """

    def __init__(self):
        self.violations = []

    def check_dataframe(
        self,
        df: pd.DataFrame,
        as_of_date_column: str,
        data_date_column: str,
        return_start_column: str = None
    ) -> List[Dict]:
        """
        Check a DataFrame for temporal violations.

        Args:
            df: DataFrame to check
            as_of_date_column: Column name for analysis date
            data_date_column: Column name for when data became available
            return_start_column: Optional column for forward return start dates

        Returns:
            List of violations found
        """
        violations = []

        # Check data dates vs as_of dates
        for idx, row in df.iterrows():
            as_of = ensure_utc(pd.to_datetime(row[as_of_date_column]))
            data_date = ensure_utc(pd.to_datetime(row[data_date_column]))

            if not validate_temporal_consistency(as_of, data_date):
                violations.append({
                    'index': idx,
                    'type': 'data_date_after_as_of_date',
                    'as_of_date': as_of,
                    'data_date': data_date,
                    'message': f"Data date {data_date.date()} > as_of_date {as_of.date()}"
                })

        # Check return start dates if provided
        if return_start_column and return_start_column in df.columns:
            for idx, row in df.iterrows():
                as_of = ensure_utc(pd.to_datetime(row[as_of_date_column]))
                return_start = ensure_utc(pd.to_datetime(row[return_start_column]))

                if return_start < as_of:
                    violations.append({
                        'index': idx,
                        'type': 'return_start_before_as_of_date',
                        'as_of_date': as_of,
                        'return_start': return_start,
                        'message': f"Return start {return_start.date()} < as_of_date {as_of.date()}"
                    })

        self.violations.extend(violations)
        return violations

    def check_q4_timing(
        self,
        quarter_end: datetime,
        data_usage_date: datetime
    ) -> bool:
        """
        Check if Q4 data is being used too early.

        Q4 data (10-K) typically not available until mid-February.

        Args:
            quarter_end: Q4 quarter end date (Dec 31)
            data_usage_date: Date when data is being used

        Returns:
            True if valid timing, False if violation detected
        """
        quarter_end = ensure_utc(quarter_end)
        data_usage_date = ensure_utc(data_usage_date)

        # Check if this is Q4 (December quarter end)
        if quarter_end.month != 12:
            return True  # Not Q4, no special check needed

        # Q4 data should not be used before February 15
        earliest_q4_date = datetime(
            quarter_end.year + 1,
            2,
            15,
            tzinfo=quarter_end.tzinfo
        )

        if data_usage_date < earliest_q4_date:
            self.violations.append({
                'type': 'q4_data_used_too_early',
                'quarter_end': quarter_end,
                'usage_date': data_usage_date,
                'earliest_valid_date': earliest_q4_date,
                'message': (
                    f"Q4 {quarter_end.year} data used on {data_usage_date.date()}, "
                    f"but not available until {earliest_q4_date.date()}"
                )
            })
            return False

        return True

    def get_summary(self) -> Dict:
        """Get summary of detected violations."""
        return {
            'total_violations': len(self.violations),
            'violations': self.violations,
            'violation_types': {
                violation['type']: sum(1 for v in self.violations if v['type'] == violation['type'])
                for violation in self.violations
            }
        }

    def print_report(self):
        """Print a formatted report of violations."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("LOOKAHEAD BIAS DETECTION REPORT")
        print("="*60)

        total = summary['total_violations']

        if total == 0:
            print("\n✓ No lookahead bias detected!")
        else:
            print(f"\n🚨 FOUND {total} VIOLATIONS:")

            for v_type, count in summary['violation_types'].items():
                print(f"\n  {v_type}: {count} violations")

            print("\nDETAILS:")
            for i, violation in enumerate(self.violations[:10], 1):
                print(f"\n  {i}. {violation['message']}")

            if total > 10:
                print(f"\n  ... and {total - 10} more violations")

        print("\n" + "="*60)
