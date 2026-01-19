# Lookahead Bias Prevention Checklist

Use this checklist when adding features or reviewing code to ensure no lookahead bias is introduced.

## General Principles

- [ ] All datetime objects are timezone-aware (UTC)
- [ ] All functions touching financial data have `as_of_date` parameter
- [ ] Data usage is logged with timestamps
- [ ] Missing data is represented as NaN/None, never filled with zeros
- [ ] Only US stocks are included (no .NS, .L, .TO, etc. suffixes)

## Data Collection

- [ ] Financial metrics include data availability date (not just quarter end)
- [ ] Q4 data assumes 60-day reporting delay (mid-February release)
- [ ] Q1-Q3 data assumes 45-day reporting delay
- [ ] Price data end date <= as_of_date
- [ ] No future data is used (validate_temporal_consistency passes)
- [ ] Data quality issues are logged (missing fields, non-US tickers)

## Feature Engineering

- [ ] Features use only data available as of as_of_date
- [ ] No "leaking" of future information (e.g., using future price volatility)
- [ ] Target variable (forward returns) starts AFTER as_of_date
- [ ] Rolling windows look backward only, never forward
- [ ] Lag features respect data availability dates

## Model Training

- [ ] Training data split by time, not randomly
- [ ] Validation set comes after training set temporally
- [ ] No shuffling of temporal data
- [ ] Cross-validation respects time series structure
- [ ] Model sees only data available at prediction time

## Backtesting

- [ ] Prediction window starts AFTER as_of_date (add 1-day buffer)
- [ ] Document: "Using data available as of [X] to predict returns over [Y] days starting [Z]"
- [ ] Actual returns calculated from prediction start date forward
- [ ] No overlap between data used and prediction window
- [ ] Walk-forward validation (not look-ahead)

## Common Pitfalls to Avoid

### ❌ WRONG: Using quarter end date as data date
```python
# BAD: Q4 2023 ends Dec 31, but data not available until Feb 2024
as_of_date = datetime(2024, 1, 15)
q4_end = datetime(2023, 12, 31)
metrics = get_metrics(q4_end)  # Using data that doesn't exist yet!
```

### ✅ CORRECT: Using reporting date as data date
```python
# GOOD: Calculate when Q4 2023 data became available
q4_end = datetime(2023, 12, 31)
reporting_date = calculate_reporting_date(q4_end, fiscal_quarter=4)
# reporting_date = Feb 29, 2024

as_of_date = datetime(2024, 1, 15)
if reporting_date <= as_of_date:
    metrics = get_metrics(q4_end)  # Only use if available
```

### ❌ WRONG: Prediction window starting before as_of_date
```python
# BAD: Using data from Feb 1 to predict returns starting Jan 1
as_of_date = datetime(2024, 2, 1)
prediction_start = datetime(2024, 1, 1)  # In the past!
returns = calculate_returns(prediction_start, 90)
```

### ✅ CORRECT: Prediction window starting after as_of_date
```python
# GOOD: Using data from Feb 1 to predict returns starting Feb 2
as_of_date = datetime(2024, 2, 1)
prediction_start = as_of_date + timedelta(days=1)
returns = calculate_returns(prediction_start, 90)
```

### ❌ WRONG: Using all historical data without date filter
```python
# BAD: This might include future data
df = get_all_data(ticker)
df['target'] = df['price'].shift(-30)  # Future prices!
model.fit(df)
```

### ✅ CORRECT: Filtering by as_of_date
```python
# GOOD: Only use data available as of specific date
df = get_all_data(ticker)
df = df[df['data_date'] <= as_of_date]
prediction_start = as_of_date + timedelta(days=1)
df['target'] = get_forward_returns(prediction_start, 30)
model.fit(df)
```

## Code Review Questions

When reviewing code, ask:

1. **Can we prove this data was available at this time?**
2. **Are we accidentally using future information?**
3. **Is the prediction window strictly after the as_of_date?**
4. **Are dates timezone-aware?**
5. **Is Q4 data delayed until mid-February?**
6. **Are non-US stocks filtered out?**
7. **Are missing values logged and tracked?**
8. **Is temporal consistency validated?**

## Testing for Lookahead Bias

Run these checks regularly:

```python
from utils.validation import LookaheadBiasDetector

# Check a DataFrame
detector = LookaheadBiasDetector()
violations = detector.check_dataframe(
    df=your_df,
    as_of_date_column='as_of_date',
    data_date_column='data_date',
    return_start_column='prediction_start'
)

if violations:
    print(f"🚨 FOUND {len(violations)} VIOLATIONS")
    for v in violations:
        print(f"  - {v['message']}")
else:
    print("✓ No lookahead bias detected")
```

## Documentation Standards

Every prediction must be documented:

```python
logger.info(
    f"{ticker}: Using data available as of {as_of_date.date()} "
    f"to predict returns over {window_days} days "
    f"starting {prediction_start.date()}"
)
```

## Pre-Commit Checks

Before committing code:

- [ ] Run unit tests (`pytest tests/`)
- [ ] Check for temporal violations in logs
- [ ] Run LookaheadBiasDetector on sample data
- [ ] Verify Q4 timing in examples
- [ ] Confirm all dates are timezone-aware
- [ ] Review data quality summary

## References

- CLAUDE.md - Project rules and standards
- src/utils/temporal.py - Temporal utilities
- src/utils/validation.py - Lookahead bias detection
- tests/test_temporal.py - Unit tests for temporal logic
