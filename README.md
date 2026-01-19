# Micro-Cap Stock Analysis

A robust stock analysis framework for micro-cap equities with built-in lookahead bias prevention.

## Features

- **Temporal Safeguards**: All financial data is tracked with availability dates to prevent lookahead bias
- **Automatic Reporting Delay**: Q4 reports are correctly dated 60 days after year-end, Q1-Q3 reports 45 days after quarter end
- **US-Only Filtering**: Automatically filters out international stocks (.NS, .L, .TO, etc.)
- **Data Quality Tracking**: Logs missing values and incomplete data typical of micro-caps
- **Strict Backtesting**: Forward returns always start AFTER data availability date

## Project Structure

```
microcap-analysis/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── collector.py          # Data collection with temporal tracking
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── backtest.py           # Backtesting framework
│   ├── models/                   # ML models (to be added)
│   └── utils/
│       ├── __init__.py
│       ├── temporal.py           # Temporal utilities
│       └── validation.py         # Lookahead bias detection
├── tests/                        # Unit tests
├── notebooks/                    # Analysis notebooks
├── data/
│   ├── raw/                      # Raw data
│   └── processed/                # Processed data
├── logs/                         # Log files
├── example_backtest.py           # Example usage
├── requirements.txt
├── CLAUDE.md                     # Project guidelines
└── README.md
```

## Installation

```bash
cd microcap-analysis
pip install -r requirements.txt
```

## Quick Start

Run the example backtest:

```bash
python example_backtest.py
```

This will:
1. Fetch financial metrics with proper reporting delays
2. Make predictions using only historical data
3. Calculate actual forward returns
4. Check for temporal violations

## Critical Rules (From CLAUDE.md)

1. **NO LOOKAHEAD BIAS**: Never use data before its public release date
2. **US STOCKS ONLY**: Filter out any non-US tickers
3. **HANDLE SPARSE DATA**: Micro-caps have incomplete data - use NaN, not zeros
4. **TRACK PREDICTION WINDOWS**: Always document data usage dates

## Code Standards

All functions that touch financial data must:
- Accept an `as_of_date` parameter (timezone-aware UTC)
- Validate temporal consistency before using data
- Log data quality issues
- Document prediction windows clearly

## Example Usage

```python
from datetime import datetime
import pytz
from backtesting import Backtester
from utils.validation import LookaheadBiasDetector

# Create backtester
backtester = Backtester(prediction_window_days=90)

# Run backtest
result = backtester.run_backtest(
    tickers=['AAPL', 'MSFT'],
    start_date=datetime(2022, 1, 1, tzinfo=pytz.utc),
    end_date=datetime(2023, 12, 31, tzinfo=pytz.utc),
    prediction_frequency_days=90
)

# Check for lookahead bias
detector = LookaheadBiasDetector()
df = result.to_dataframe()
violations = detector.check_dataframe(
    df=df,
    as_of_date_column='as_of_date',
    data_date_column='data_date'
)

if violations:
    print(f"🚨 Found {len(violations)} violations!")
else:
    print("✓ No lookahead bias detected!")
```

## Data Sources

- **Yahoo Finance** (yfinance): Financial metrics and price data
- **SEC EDGAR**: 13F filings (to be implemented)
- **Seeking Alpha / FMP**: Earnings transcripts (to be implemented)
- **Whale Wisdom**: Fund holdings (to be implemented)

## Testing

```bash
# Run unit tests (to be added)
pytest tests/

# Check specific file for lookahead bias
python -c "from src.utils.validation import LookaheadBiasDetector; ..."
```

## Logging

All temporal operations and data quality issues are logged:
- Missing data points
- Non-US tickers filtered
- Temporal violations
- Data availability dates

Check `logs/` directory for detailed logs.

## Next Steps

1. Add more micro-cap tickers to analyze
2. Implement earnings call sentiment analysis
3. Add ML models for predictions
4. Integrate SEC EDGAR for 13F filings
5. Add unit tests for all modules

## License

MIT
