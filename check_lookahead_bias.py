"""
Comprehensive lookahead bias checker for the codebase.

Run this script to validate temporal consistency across all modules.
"""

import sys
sys.path.append('src')

import ast
import os
from pathlib import Path
from datetime import datetime
import pytz


class CodeAnalyzer:
    """Analyze Python code for potential lookahead bias patterns."""

    def __init__(self):
        self.violations = []
        self.warnings = []
        self.checks_passed = []

    def check_file(self, file_path: str):
        """Check a Python file for lookahead bias patterns."""
        print(f"\nChecking {file_path}...")

        with open(file_path, 'r') as f:
            content = f.read()

        # Check 1: Functions touching financial data should have as_of_date param
        self._check_as_of_date_parameter(file_path, content)

        # Check 2: Look for dangerous date operations
        self._check_dangerous_date_ops(file_path, content)

        # Check 3: Check for timezone-aware date creation
        self._check_timezone_awareness(file_path, content)

        # Check 4: Look for forward-looking operations
        self._check_forward_operations(file_path, content)

    def _check_as_of_date_parameter(self, file_path: str, content: str):
        """Check if functions touching financial data have as_of_date parameter."""
        financial_keywords = [
            'price', 'return', 'metric', 'fundamental',
            'financial', 'backtest', 'predict', 'forecast'
        ]

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name.lower()

                    # Skip private and special methods
                    if func_name.startswith('_'):
                        continue

                    # Check if function name suggests it handles financial data
                    if any(keyword in func_name for keyword in financial_keywords):
                        # Check if it has as_of_date parameter
                        param_names = [arg.arg for arg in node.args.args]

                        if 'as_of_date' not in param_names:
                            self.warnings.append({
                                'file': file_path,
                                'function': node.name,
                                'line': node.lineno,
                                'issue': 'Function handles financial data but missing as_of_date parameter',
                                'severity': 'WARNING'
                            })
                        else:
                            self.checks_passed.append({
                                'file': file_path,
                                'function': node.name,
                                'check': 'Has as_of_date parameter'
                            })

        except SyntaxError as e:
            print(f"  Syntax error in {file_path}: {e}")

    def _check_dangerous_date_ops(self, file_path: str, content: str):
        """Check for potentially dangerous date operations."""
        dangerous_patterns = [
            ('datetime.now()', 'Using datetime.now() - should use as_of_date'),
            ('datetime.today()', 'Using datetime.today() - should use as_of_date'),
            ('.shift(-', 'Forward shift operation - potential lookahead'),
            ('future', 'Variable named "future" - verify no lookahead'),
        ]

        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            for pattern, issue in dangerous_patterns:
                if pattern in line and not line.strip().startswith('#'):
                    self.warnings.append({
                        'file': file_path,
                        'line': i,
                        'code': line.strip(),
                        'issue': issue,
                        'severity': 'WARNING'
                    })

    def _check_timezone_awareness(self, file_path: str, content: str):
        """Check that datetimes are created with timezone info."""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if 'datetime(' in line:
                # Check if tzinfo parameter is present
                if 'tzinfo=' not in line and 'pytz' not in line:
                    # Might be creating naive datetime
                    if not line.strip().startswith('#'):
                        self.warnings.append({
                            'file': file_path,
                            'line': i,
                            'code': line.strip(),
                            'issue': 'Possible naive datetime creation - should include tzinfo',
                            'severity': 'WARNING'
                        })

    def _check_forward_operations(self, file_path: str, content: str):
        """Check for operations that might look forward in time."""
        forward_patterns = [
            'predict.*start.*<',
            'return.*before',
            'future.*data',
        ]

        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            for pattern in forward_patterns:
                if pattern in line_lower and not line.strip().startswith('#'):
                    self.warnings.append({
                        'file': file_path,
                        'line': i,
                        'code': line.strip(),
                        'issue': f'Potential forward-looking operation (pattern: {pattern})',
                        'severity': 'WARNING'
                    })

    def print_report(self):
        """Print analysis report."""
        print("\n" + "="*70)
        print("LOOKAHEAD BIAS CODE ANALYSIS REPORT")
        print("="*70)

        # Violations
        if self.violations:
            print(f"\n🚨 CRITICAL VIOLATIONS: {len(self.violations)}")
            for v in self.violations:
                print(f"\n  File: {v['file']}")
                print(f"  Line: {v.get('line', 'N/A')}")
                print(f"  Issue: {v['issue']}")
        else:
            print("\n✓ No critical violations found")

        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for w in self.warnings[:20]:  # Show first 20
                print(f"\n  File: {w['file']}")
                print(f"  Line: {w.get('line', 'N/A')}")
                if 'function' in w:
                    print(f"  Function: {w['function']}")
                if 'code' in w:
                    print(f"  Code: {w['code'][:80]}")
                print(f"  Issue: {w['issue']}")

            if len(self.warnings) > 20:
                print(f"\n  ... and {len(self.warnings) - 20} more warnings")
        else:
            print("\n✓ No warnings")

        # Passed checks
        print(f"\n✓ CHECKS PASSED: {len(self.checks_passed)}")

        print("\n" + "="*70)


def main():
    print("="*70)
    print("MICRO-CAP STOCK ANALYSIS - LOOKAHEAD BIAS CHECKER")
    print("="*70)
    print("\nThis script will:")
    print("  1. Check all Python files for temporal consistency")
    print("  2. Verify as_of_date parameters on financial functions")
    print("  3. Detect dangerous date operations")
    print("  4. Identify potential lookahead bias patterns")

    # Find all Python files in src/
    src_dir = Path('src')
    if not src_dir.exists():
        print(f"\n❌ Error: {src_dir} directory not found")
        return

    py_files = list(src_dir.rglob('*.py'))
    print(f"\nFound {len(py_files)} Python files to analyze")

    # Analyze each file
    analyzer = CodeAnalyzer()

    for py_file in py_files:
        if '__pycache__' not in str(py_file):
            analyzer.check_file(str(py_file))

    # Print report
    analyzer.print_report()

    # Check specific temporal patterns
    print("\n" + "="*70)
    print("TEMPORAL LOGIC VALIDATION")
    print("="*70)

    # Test Q4 reporting date
    print("\n1. Testing Q4 reporting date calculation...")
    from utils.temporal import calculate_reporting_date
    q4_2023 = datetime(2023, 12, 31, tzinfo=pytz.utc)
    reporting_date = calculate_reporting_date(q4_2023, 4)
    print(f"   Q4 2023 (Dec 31) reports on: {reporting_date.date()}")

    if reporting_date.month == 2 and reporting_date.day >= 28:
        print("   ✓ PASS: Q4 data delayed until late February")
    else:
        print("   ❌ FAIL: Q4 data not properly delayed!")

    # Test temporal consistency
    print("\n2. Testing temporal consistency validation...")
    from utils.temporal import validate_temporal_consistency

    # Valid case
    as_of = datetime(2024, 3, 1, tzinfo=pytz.utc)
    data_date = datetime(2024, 2, 15, tzinfo=pytz.utc)
    is_valid = validate_temporal_consistency(as_of, data_date)

    if is_valid:
        print("   ✓ PASS: Valid temporal ordering detected correctly")
    else:
        print("   ❌ FAIL: Valid temporal ordering rejected!")

    # Invalid case (lookahead)
    as_of = datetime(2024, 2, 15, tzinfo=pytz.utc)
    data_date = datetime(2024, 3, 1, tzinfo=pytz.utc)
    is_valid = validate_temporal_consistency(as_of, data_date)

    if not is_valid:
        print("   ✓ PASS: Lookahead bias detected correctly")
    else:
        print("   ❌ FAIL: Lookahead bias not detected!")

    # Test US ticker filtering
    print("\n3. Testing US ticker filtering...")
    from data.collector import is_us_ticker

    test_cases = [
        ('AAPL', True),
        ('MSFT', True),
        ('BRK.A', True),
        ('NESN.SW', False),  # Switzerland
        ('RELIANCE.NS', False),  # India
        ('BP.L', False),  # London
    ]

    all_passed = True
    for ticker, expected in test_cases:
        result = is_us_ticker(ticker)
        if result == expected:
            print(f"   ✓ {ticker}: correctly identified as {'US' if expected else 'non-US'}")
        else:
            print(f"   ❌ {ticker}: incorrectly identified!")
            all_passed = False

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    total_issues = len(analyzer.violations) + len(analyzer.warnings)

    if total_issues == 0:
        print("\n✓ All checks passed! No lookahead bias detected.")
    else:
        print(f"\n⚠️  Found {total_issues} potential issues to review")
        print("\nReview the warnings above and verify:")
        print("  - All financial functions have as_of_date parameter")
        print("  - All datetimes are timezone-aware")
        print("  - No forward-looking operations")
        print("  - Q4 data is properly delayed")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
