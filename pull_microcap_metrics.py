"""
Pull comprehensive metrics for micro-cap tickers.

Fetches all required metrics with temporal tracking and data quality monitoring.
"""

import sys
sys.path.append('src')

import argparse
import glob as globlib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pytz
import logging
from typing import Dict, List, Optional

from utils.temporal import ensure_utc, calculate_reporting_date, get_fiscal_quarter
from data.earnings_fetcher import EarningsFetcher
from data.insider_fetcher import InsiderFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default micro-cap tickers to analyze
DEFAULT_TICKERS = ["AORT", "QCRH", "NBN", "TRS", "TCBX", "REVG", "RDVT", "NATR", "MTRN", "AIOT"]


def find_latest_universe_file(directory: str = 'data/raw') -> Optional[str]:
    """Find the most recent universe CSV file."""
    pattern = f'{directory}/universe_*.csv'
    files = globlib.glob(pattern)

    if not files:
        return None

    # Sort by filename (date is in filename YYYYMMDD)
    files.sort(reverse=True)
    return files[0]


def load_universe_tickers(filepath: str) -> List[str]:
    """Load tickers from universe CSV file."""
    df = pd.read_csv(filepath)
    return df['ticker'].tolist()


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
        self.earnings_fetcher = EarningsFetcher()
        self.insider_fetcher = InsiderFetcher()

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

            # 13. Earnings Surprise
            try:
                earnings_data = self.earnings_fetcher.fetch_earnings_surprise(
                    ticker=ticker,
                    as_of_date=self.as_of_date
                )
                result['earnings_surprise'] = earnings_data.surprise
                result['earnings_surprise_pct'] = earnings_data.surprise_pct
                result['actual_eps'] = earnings_data.actual_eps
                result['expected_eps'] = earnings_data.expected_eps
                result['earnings_report_date'] = earnings_data.report_date
                result['earnings_data_quality'] = earnings_data.data_quality

                if earnings_data.surprise is not None:
                    logger.info(f"Earnings Surprise: {earnings_data.surprise_pct:+.1f}%")
                else:
                    missing.append('earnings_surprise')
                    logger.warning(f"Missing: Earnings Surprise")
            except Exception as e:
                result['earnings_surprise'] = None
                result['earnings_surprise_pct'] = None
                result['actual_eps'] = None
                result['expected_eps'] = None
                result['earnings_report_date'] = None
                result['earnings_data_quality'] = 0.0
                missing.append('earnings_surprise')
                logger.warning(f"Error fetching earnings surprise: {e}")

            # 14. Accruals Ratio (earnings quality metric)
            # accruals_ratio = (net_income - operating_cash_flow) / total_assets
            # Lower is better (cash earnings > paper earnings)
            try:
                # Get net income from info or financials
                net_income = info.get('netIncomeToCommon', np.nan)
                if pd.isna(net_income):
                    # Try from quarterly financials
                    if not quarterly_financials.empty and 'Net Income' in quarterly_financials.index:
                        net_income = quarterly_financials.loc['Net Income'].iloc[0]

                # Get total assets from balance sheet (not in info)
                total_assets = np.nan
                if not quarterly_balance.empty:
                    # Try different possible row names
                    for asset_name in ['Total Assets', 'TotalAssets']:
                        if asset_name in quarterly_balance.index:
                            total_assets = quarterly_balance.loc[asset_name].iloc[0]
                            break

                # operating_cash_flow already fetched above from info

                result['net_income'] = net_income if pd.notna(net_income) else None
                result['total_assets'] = total_assets if pd.notna(total_assets) else None

                if pd.notna(net_income):
                    logger.info(f"Net Income: ${net_income:,.0f}")
                else:
                    logger.warning(f"Missing: Net Income")

                if pd.notna(total_assets):
                    logger.info(f"Total Assets: ${total_assets:,.0f}")
                else:
                    logger.warning(f"Missing: Total Assets")

                # Calculate accruals ratio
                if (pd.notna(net_income) and pd.notna(operating_cash_flow) and
                    pd.notna(total_assets) and total_assets != 0):
                    accruals_ratio = (net_income - operating_cash_flow) / total_assets
                    result['accruals_ratio'] = accruals_ratio
                    logger.info(f"Accruals Ratio: {accruals_ratio:.4f}")
                else:
                    result['accruals_ratio'] = None
                    missing.append('accruals_ratio')
                    logger.warning(f"Missing: Accruals Ratio (insufficient data)")

            except Exception as e:
                result['net_income'] = None
                result['total_assets'] = None
                result['accruals_ratio'] = None
                missing.append('accruals_ratio')
                logger.warning(f"Error calculating accruals ratio: {e}")

            # 15. Revenue Growth, Gross Margin, Operating Margin
            # revenue_growth = (revenue_ttm - revenue_prior_year) / revenue_prior_year
            # gross_margin = gross_profit / revenue
            # operating_margin = operating_income / revenue
            try:
                # Get annual financials for YoY comparison
                annual_financials = stock.financials

                # Initialize values
                revenue_ttm = np.nan
                revenue_prior = np.nan
                gross_profit = np.nan
                operating_income = np.nan

                # Get revenue (TTM from info, or from financials)
                revenue_ttm = info.get('totalRevenue', np.nan)
                if pd.isna(revenue_ttm) and not annual_financials.empty:
                    for rev_name in ['Total Revenue', 'TotalRevenue', 'Revenue']:
                        if rev_name in annual_financials.index:
                            revenue_ttm = annual_financials.loc[rev_name].iloc[0]
                            break

                # Get prior year revenue for growth calculation
                if not annual_financials.empty and len(annual_financials.columns) >= 2:
                    for rev_name in ['Total Revenue', 'TotalRevenue', 'Revenue']:
                        if rev_name in annual_financials.index:
                            revenue_prior = annual_financials.loc[rev_name].iloc[1]
                            break

                # Get gross profit
                gross_profit = info.get('grossProfit', np.nan)
                if pd.isna(gross_profit) and not annual_financials.empty:
                    for gp_name in ['Gross Profit', 'GrossProfit']:
                        if gp_name in annual_financials.index:
                            gross_profit = annual_financials.loc[gp_name].iloc[0]
                            break

                # Get operating income
                operating_income = info.get('operatingIncome', np.nan)
                if pd.isna(operating_income) and not annual_financials.empty:
                    for oi_name in ['Operating Income', 'OperatingIncome', 'EBIT']:
                        if oi_name in annual_financials.index:
                            operating_income = annual_financials.loc[oi_name].iloc[0]
                            break

                # Store raw values
                result['revenue_ttm'] = revenue_ttm if pd.notna(revenue_ttm) else None
                result['revenue_prior_year'] = revenue_prior if pd.notna(revenue_prior) else None
                result['gross_profit'] = gross_profit if pd.notna(gross_profit) else None
                result['operating_income'] = operating_income if pd.notna(operating_income) else None

                # Calculate revenue growth
                if pd.notna(revenue_ttm) and pd.notna(revenue_prior) and revenue_prior != 0:
                    revenue_growth = (revenue_ttm - revenue_prior) / abs(revenue_prior)
                    result['revenue_growth'] = revenue_growth
                    logger.info(f"Revenue Growth: {revenue_growth*100:+.1f}%")
                else:
                    result['revenue_growth'] = None
                    missing.append('revenue_growth')
                    if pd.isna(revenue_prior):
                        logger.warning(f"Missing: Revenue Growth (no prior year data)")
                    else:
                        logger.warning(f"Missing: Revenue Growth")

                # Calculate gross margin
                if pd.notna(gross_profit) and pd.notna(revenue_ttm) and revenue_ttm != 0:
                    gross_margin = gross_profit / revenue_ttm
                    result['gross_margin'] = gross_margin
                    logger.info(f"Gross Margin: {gross_margin*100:.1f}%")
                else:
                    result['gross_margin'] = None
                    missing.append('gross_margin')
                    logger.warning(f"Missing: Gross Margin")

                # Calculate operating margin
                if pd.notna(operating_income) and pd.notna(revenue_ttm) and revenue_ttm != 0:
                    operating_margin = operating_income / revenue_ttm
                    result['operating_margin'] = operating_margin
                    logger.info(f"Operating Margin: {operating_margin*100:.1f}%")
                else:
                    result['operating_margin'] = None
                    missing.append('operating_margin')
                    logger.warning(f"Missing: Operating Margin")

            except Exception as e:
                result['revenue_ttm'] = None
                result['revenue_prior_year'] = None
                result['revenue_growth'] = None
                result['gross_profit'] = None
                result['gross_margin'] = None
                result['operating_income'] = None
                result['operating_margin'] = None
                missing.extend(['revenue_growth', 'gross_margin', 'operating_margin'])
                logger.warning(f"Error calculating revenue metrics: {e}")

            # 16. Insider Activity (from SEC Form 4 filings)
            # Tracks insider buying/selling over last 90 days
            try:
                insider_activity = self.insider_fetcher.get_insider_activity(
                    ticker=ticker,
                    as_of_date=self.as_of_date,
                    lookback_days=90
                )

                result['insider_net_purchases_90d'] = insider_activity.net_purchases
                result['insider_buy_ratio'] = insider_activity.buy_ratio
                result['insider_shares_bought'] = insider_activity.total_bought
                result['insider_shares_sold'] = insider_activity.total_sold
                result['insider_num_transactions'] = insider_activity.num_transactions
                result['insider_data_quality'] = insider_activity.data_quality

                if insider_activity.buy_ratio is not None:
                    logger.info(
                        f"Insider Activity: net={insider_activity.net_purchases:+,.0f} shares, "
                        f"buy_ratio={insider_activity.buy_ratio:.2f}"
                    )
                else:
                    if insider_activity.num_transactions == 0:
                        # No transactions is valid data, not missing
                        logger.info(f"Insider Activity: No transactions in 90 days")
                    else:
                        missing.append('insider_buy_ratio')
                        logger.warning(f"Missing: Insider Buy Ratio")

            except Exception as e:
                result['insider_net_purchases_90d'] = None
                result['insider_buy_ratio'] = None
                result['insider_shares_bought'] = None
                result['insider_shares_sold'] = None
                result['insider_num_transactions'] = None
                result['insider_data_quality'] = 0.0
                missing.append('insider_activity')
                logger.warning(f"Error fetching insider activity: {e}")

            # Calculate data completeness score
            total_metrics = 17  # Total number of key metrics we're tracking
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

    def collect_all_tickers(self, tickers: List[str], show_progress: bool = True) -> pd.DataFrame:
        """
        Collect metrics for all tickers.

        Args:
            tickers: List of ticker symbols
            show_progress: Whether to show progress logging

        Returns:
            DataFrame with all collected metrics
        """
        total = len(tickers)

        logger.info(f"\n{'='*60}")
        logger.info(f"MICRO-CAP METRICS COLLECTION")
        logger.info(f"{'='*60}")
        logger.info(f"Total tickers to process: {total}")
        logger.info(f"As of date: {self.as_of_date.date()}")
        logger.info(f"Data retrieved: {self.data_retrieved_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"{'='*60}")

        for i, ticker in enumerate(tickers, 1):
            if show_progress:
                pct = (i / total) * 100
                print(f"\rProgress: {i}/{total} ({pct:.1f}%) - Processing {ticker}...", end='', flush=True)

            result = self.collect_all_metrics(ticker)
            self.results.append(result)

        if show_progress:
            print()  # New line after progress

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pull comprehensive metrics for micro-cap tickers'
    )

    parser.add_argument(
        '--universe',
        action='store_true',
        help='Use tickers from latest universe file (data/raw/universe_*.csv)'
    )

    parser.add_argument(
        '--universe-file',
        type=str,
        default=None,
        help='Specific universe CSV file to use'
    )

    parser.add_argument(
        '--tickers',
        type=str,
        default=None,
        help='Comma-separated list of tickers (overrides --universe)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit to first N tickers (for testing)'
    )

    parser.add_argument(
        '--as-of-date',
        type=str,
        default=None,
        help='As-of date in YYYY-MM-DD format (default: today)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory (default: data/raw)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce logging verbosity'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("\n" + "="*70)
    print("MICRO-CAP METRICS COLLECTION")
    print("="*70)

    # Determine tickers to process
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
        source = "command line"
    elif args.universe or args.universe_file:
        if args.universe_file:
            universe_file = args.universe_file
        else:
            universe_file = find_latest_universe_file()

        if universe_file is None:
            logger.error("No universe file found. Run pull_universe.py first.")
            sys.exit(1)

        tickers = load_universe_tickers(universe_file)
        source = universe_file
    else:
        tickers = DEFAULT_TICKERS
        source = "default list"

    # Apply limit if specified
    if args.limit and args.limit > 0:
        tickers = tickers[:args.limit]
        print(f"Limited to first {args.limit} tickers")

    # Parse as-of date
    if args.as_of_date:
        try:
            as_of_date = datetime.strptime(args.as_of_date, '%Y-%m-%d')
            as_of_date = pytz.utc.localize(as_of_date)
        except ValueError:
            logger.error(f"Invalid date format: {args.as_of_date}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        as_of_date = datetime.now(pytz.utc)

    print(f"\nConfiguration:")
    print(f"  Source: {source}")
    print(f"  Tickers to process: {len(tickers)}")
    print(f"  As-of date: {as_of_date.date()}")
    print(f"  Output directory: {args.output_dir}")
    print("="*70)

    # Reduce logging if quiet mode
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Create collector
    collector = MicroCapMetricsCollector(as_of_date=as_of_date)

    # Collect metrics for all tickers
    df = collector.collect_all_tickers(tickers, show_progress=True)

    # Print summary
    collector.print_summary(df)

    # Save to CSV
    date_str = as_of_date.strftime('%Y%m%d')
    output_file = f'{args.output_dir}/microcap_metrics_{date_str}.csv'

    df.to_csv(output_file, index=False)
    logger.info(f"\nMetrics saved to: {output_file}")

    # Also save a human-readable summary
    summary_file = f'{args.output_dir}/microcap_metrics_{date_str}_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MICRO-CAP METRICS COLLECTION SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"\nCollection Date: {as_of_date.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        f.write(f"Source: {source}\n")
        f.write(f"Tickers processed: {len(tickers)}\n")
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

    logger.info(f"Summary saved to: {summary_file}")

    print("\n" + "="*70)
    print("COLLECTION COMPLETE")
    print(f"Processed {len(df)} tickers")
    print(f"Output: {output_file}")
    print("="*70)

    return df


if __name__ == '__main__':
    df = main()
