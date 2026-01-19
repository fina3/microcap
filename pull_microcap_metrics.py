"""
Pull comprehensive metrics for micro-cap tickers.

Fetches all required metrics with temporal tracking and data quality monitoring.
"""

import sys
sys.path.append('src')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import logging
from typing import Dict, List, Optional

from utils.temporal import ensure_utc, calculate_reporting_date, get_fiscal_quarter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Micro-cap tickers to analyze
TICKERS = ["AORT", "QCRH", "NBN", "TRS", "TCBX", "REVG", "RDVT", "NATR", "MTRN", "AIOT"]


class MicroCapMetricsCollector:
    """
    Collects comprehensive metrics for micro-cap stocks with data quality tracking.
    """

    def __init__(self, as_of_date: datetime):
        """
        Args:
            as_of_date: Date from which we're collecting data (timezone-aware)
        """
        self.as_of_date = ensure_utc(as_of_date)
        self.data_retrieved_date = datetime.now(pytz.utc)
        self.results = []
        self.missing_metrics = {}

    def collect_all_metrics(self, ticker: str) -> Dict:
        """
        Collect all required metrics for a single ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with all metrics and metadata
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting metrics for {ticker}")
        logger.info(f"{'='*60}")

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Initialize result dictionary
            result = {
                'ticker': ticker,
                'data_retrieved_date': self.data_retrieved_date,
                'as_of_date': self.as_of_date,
            }

            # Track missing metrics for this ticker
            missing = []

            # Get quarterly financials for reporting date calculation
            quarterly_financials = stock.quarterly_financials
            quarterly_balance = stock.quarterly_balance_sheet

            # Determine data availability date
            if not quarterly_financials.empty:
                most_recent_quarter_end = quarterly_financials.columns[0]
                if isinstance(most_recent_quarter_end, pd.Timestamp):
                    most_recent_quarter_end = most_recent_quarter_end.to_pydatetime()
                most_recent_quarter_end = ensure_utc(most_recent_quarter_end)

                _, fiscal_quarter = get_fiscal_quarter(most_recent_quarter_end)
                reporting_date = calculate_reporting_date(most_recent_quarter_end, fiscal_quarter)

                result['most_recent_quarter_end'] = most_recent_quarter_end
                result['fiscal_quarter'] = fiscal_quarter
                result['estimated_reporting_date'] = reporting_date

                logger.info(f"Most recent quarter: Q{fiscal_quarter} {most_recent_quarter_end.date()}")
                logger.info(f"Estimated reporting date: {reporting_date.date()}")
            else:
                logger.warning(f"No quarterly financials available for {ticker}")
                result['most_recent_quarter_end'] = None
                result['fiscal_quarter'] = None
                result['estimated_reporting_date'] = None
                missing.append('quarterly_financials')

            # 1. P/E Ratio (Trailing)
            pe_trailing = info.get('trailingPE', np.nan)
            if pd.notna(pe_trailing):
                result['pe_trailing'] = pe_trailing
                logger.info(f"P/E (Trailing): {pe_trailing:.2f}")
            else:
                result['pe_trailing'] = None
                missing.append('pe_trailing')
                logger.warning(f"Missing: P/E (Trailing)")

            # 2. P/E Ratio (Forward)
            pe_forward = info.get('forwardPE', np.nan)
            if pd.notna(pe_forward):
                result['pe_forward'] = pe_forward
                logger.info(f"P/E (Forward): {pe_forward:.2f}")
            else:
                result['pe_forward'] = None
                missing.append('pe_forward')
                logger.warning(f"Missing: P/E (Forward)")

            # 3. P/B Ratio
            pb_ratio = info.get('priceToBook', np.nan)
            if pd.notna(pb_ratio):
                result['pb_ratio'] = pb_ratio
                logger.info(f"P/B Ratio: {pb_ratio:.2f}")
            else:
                result['pb_ratio'] = None
                missing.append('pb_ratio')
                logger.warning(f"Missing: P/B Ratio")

            # 4. Price to Sales
            price_to_sales = info.get('priceToSalesTrailing12Months', np.nan)
            if pd.notna(price_to_sales):
                result['price_to_sales'] = price_to_sales
                logger.info(f"Price to Sales: {price_to_sales:.2f}")
            else:
                result['price_to_sales'] = None
                missing.append('price_to_sales')
                logger.warning(f"Missing: Price to Sales")

            # 5. Operating Cash Flow
            operating_cash_flow = info.get('operatingCashflow', np.nan)
            if pd.notna(operating_cash_flow):
                result['operating_cash_flow'] = operating_cash_flow
                logger.info(f"Operating Cash Flow: ${operating_cash_flow:,.0f}")
            else:
                result['operating_cash_flow'] = None
                missing.append('operating_cash_flow')
                logger.warning(f"Missing: Operating Cash Flow")

            # 6. Total Debt
            total_debt = info.get('totalDebt', np.nan)
            if pd.notna(total_debt):
                result['total_debt'] = total_debt
                logger.info(f"Total Debt: ${total_debt:,.0f}")
            else:
                result['total_debt'] = None
                missing.append('total_debt')
                logger.warning(f"Missing: Total Debt")

            # 7. Debt to Equity
            debt_to_equity = info.get('debtToEquity', np.nan)
            if pd.notna(debt_to_equity):
                result['debt_to_equity'] = debt_to_equity
                logger.info(f"Debt to Equity: {debt_to_equity:.2f}")
            else:
                result['debt_to_equity'] = None
                missing.append('debt_to_equity')
                logger.warning(f"Missing: Debt to Equity")

            # 8. Short Interest / Short % of Float
            short_percent_float = info.get('shortPercentOfFloat', np.nan)
            if pd.notna(short_percent_float):
                result['short_percent_float'] = short_percent_float * 100  # Convert to percentage
                logger.info(f"Short % of Float: {short_percent_float * 100:.2f}%")
            else:
                result['short_percent_float'] = None
                missing.append('short_percent_float')
                logger.warning(f"Missing: Short % of Float")

            # Alternative: shortRatio
            short_ratio = info.get('shortRatio', np.nan)
            if pd.notna(short_ratio):
                result['short_ratio'] = short_ratio
                logger.info(f"Short Ratio (days to cover): {short_ratio:.2f}")
            else:
                result['short_ratio'] = None
                if 'short_percent_float' in missing:  # Only log if both are missing
                    missing.append('short_ratio')

            # 9. Insider Ownership %
            insider_ownership = info.get('heldPercentInsiders', np.nan)
            if pd.notna(insider_ownership):
                result['insider_ownership_pct'] = insider_ownership * 100  # Convert to percentage
                logger.info(f"Insider Ownership: {insider_ownership * 100:.2f}%")
            else:
                result['insider_ownership_pct'] = None
                missing.append('insider_ownership_pct')
                logger.warning(f"Missing: Insider Ownership %")

            # 10. Institutional Ownership %
            institutional_ownership = info.get('heldPercentInstitutions', np.nan)
            if pd.notna(institutional_ownership):
                result['institutional_ownership_pct'] = institutional_ownership * 100  # Convert to percentage
                logger.info(f"Institutional Ownership: {institutional_ownership * 100:.2f}%")
            else:
                result['institutional_ownership_pct'] = None
                missing.append('institutional_ownership_pct')
                logger.warning(f"Missing: Institutional Ownership %")

            # 11. Market Cap
            market_cap = info.get('marketCap', np.nan)
            if pd.notna(market_cap):
                result['market_cap'] = market_cap
                logger.info(f"Market Cap: ${market_cap:,.0f}")
            else:
                result['market_cap'] = None
                missing.append('market_cap')
                logger.warning(f"Missing: Market Cap")

            # 12. 52-Week Price Change %
            # Get current price
            current_price = info.get('currentPrice', np.nan)

            # Get 52-week high/low for reference
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', np.nan)
            fifty_two_week_low = info.get('fiftyTwoWeekLow', np.nan)

            if pd.notna(current_price):
                result['current_price'] = current_price
                logger.info(f"Current Price: ${current_price:.2f}")
            else:
                result['current_price'] = None
                missing.append('current_price')
                logger.warning(f"Missing: Current Price")

            if pd.notna(fifty_two_week_high) and pd.notna(fifty_two_week_low):
                result['52_week_high'] = fifty_two_week_high
                result['52_week_low'] = fifty_two_week_low
                logger.info(f"52-Week Range: ${fifty_two_week_low:.2f} - ${fifty_two_week_high:.2f}")
            else:
                result['52_week_high'] = None
                result['52_week_low'] = None

            # Calculate 52-week price change using historical data
            try:
                hist = stock.history(period='1y')
                if not hist.empty and len(hist) > 0:
                    price_52w_ago = hist.iloc[0]['Close']
                    if pd.notna(current_price) and pd.notna(price_52w_ago):
                        price_change_52w_pct = ((current_price - price_52w_ago) / price_52w_ago) * 100
                        result['52_week_price_change_pct'] = price_change_52w_pct
                        logger.info(f"52-Week Price Change: {price_change_52w_pct:+.2f}%")
                    else:
                        result['52_week_price_change_pct'] = None
                        missing.append('52_week_price_change_pct')
                else:
                    result['52_week_price_change_pct'] = None
                    missing.append('52_week_price_change_pct')
                    logger.warning(f"Missing: 52-Week Price Change %")
            except Exception as e:
                result['52_week_price_change_pct'] = None
                missing.append('52_week_price_change_pct')
                logger.warning(f"Error calculating 52-week price change: {e}")

            # Calculate data completeness score
            total_metrics = 11  # Total number of key metrics we're tracking
            metrics_retrieved = total_metrics - len(set(missing))  # Use set to avoid double-counting
            completeness_score = (metrics_retrieved / total_metrics) * 100

            result['metrics_retrieved'] = metrics_retrieved
            result['total_metrics'] = total_metrics
            result['data_completeness_score'] = completeness_score
            result['missing_metrics'] = ', '.join(missing) if missing else 'None'

            logger.info(f"\nData Completeness: {metrics_retrieved}/{total_metrics} ({completeness_score:.1f}%)")
            if missing:
                logger.warning(f"Missing metrics: {', '.join(missing)}")

            # Store missing metrics for summary
            self.missing_metrics[ticker] = missing

            return result

        except Exception as e:
            logger.error(f"Error collecting metrics for {ticker}: {e}")
            return {
                'ticker': ticker,
                'data_retrieved_date': self.data_retrieved_date,
                'as_of_date': self.as_of_date,
                'error': str(e),
                'data_completeness_score': 0.0
            }

    def collect_all_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Collect metrics for all tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with all collected metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"MICRO-CAP METRICS COLLECTION")
        logger.info(f"{'='*60}")
        logger.info(f"Tickers: {', '.join(tickers)}")
        logger.info(f"As of date: {self.as_of_date.date()}")
        logger.info(f"Data retrieved: {self.data_retrieved_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"{'='*60}")

        for ticker in tickers:
            result = self.collect_all_metrics(ticker)
            self.results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        return df

    def print_summary(self, df: pd.DataFrame):
        """Print collection summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTION SUMMARY")
        logger.info(f"{'='*60}")

        total_tickers = len(df)
        avg_completeness = df['data_completeness_score'].mean()

        logger.info(f"\nTotal tickers processed: {total_tickers}")
        logger.info(f"Average data completeness: {avg_completeness:.1f}%")

        # Tickers by completeness
        logger.info(f"\nData completeness by ticker:")
        for _, row in df.iterrows():
            ticker = row['ticker']
            score = row.get('data_completeness_score', 0)
            logger.info(f"  {ticker}: {score:.1f}%")

        # Most common missing metrics
        all_missing = []
        for missing_list in self.missing_metrics.values():
            all_missing.extend(missing_list)

        if all_missing:
            from collections import Counter
            missing_counts = Counter(all_missing)
            logger.info(f"\nMost commonly missing metrics:")
            for metric, count in missing_counts.most_common(10):
                pct = (count / total_tickers) * 100
                logger.info(f"  {metric}: {count}/{total_tickers} tickers ({pct:.0f}%)")
        else:
            logger.info(f"\n✓ No missing metrics - 100% data completeness!")

        logger.info(f"\n{'='*60}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("MICRO-CAP METRICS COLLECTION")
    print("="*70)

    # Use current date as as_of_date
    as_of_date = datetime.now(pytz.utc)

    # Create collector
    collector = MicroCapMetricsCollector(as_of_date=as_of_date)

    # Collect metrics for all tickers
    df = collector.collect_all_tickers(TICKERS)

    # Print summary
    collector.print_summary(df)

    # Save to CSV
    date_str = as_of_date.strftime('%Y%m%d')
    output_file = f'data/raw/microcap_metrics_{date_str}.csv'

    df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Metrics saved to: {output_file}")

    # Also save a human-readable summary
    summary_file = f'data/raw/microcap_metrics_{date_str}_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MICRO-CAP METRICS COLLECTION SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"\nCollection Date: {as_of_date.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        f.write(f"Tickers: {', '.join(TICKERS)}\n")
        f.write(f"\nAverage Data Completeness: {df['data_completeness_score'].mean():.1f}%\n")
        f.write("\n" + "-"*70 + "\n")
        f.write("DATA COMPLETENESS BY TICKER\n")
        f.write("-"*70 + "\n")

        for _, row in df.iterrows():
            ticker = row['ticker']
            score = row.get('data_completeness_score', 0)
            missing = row.get('missing_metrics', 'None')
            f.write(f"\n{ticker}: {score:.1f}%\n")
            if missing != 'None':
                f.write(f"  Missing: {missing}\n")

    logger.info(f"✓ Summary saved to: {summary_file}")

    print("\n" + "="*70)
    print("COLLECTION COMPLETE")
    print("="*70)

    return df


if __name__ == '__main__':
    df = main()
