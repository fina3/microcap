"""
Universe screener for micro-cap stocks using Finviz.

Scrapes finviz.com screener to build a universe of US micro-cap stocks
meeting specified criteria.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging
import pytz
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sectors to exclude (different accounting standards)
EXCLUDED_SECTORS = ['Financial', 'Real Estate']

# Finviz screener base URL
FINVIZ_BASE_URL = 'https://finviz.com/screener.ashx'

# Headers to mimic browser request
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}


class UniverseScreener:
    """
    Screens for micro-cap stocks from Finviz with specified criteria.

    Criteria:
    - Market Cap: $50M - $500M
    - Country: USA only
    - Average Volume: > 50,000
    - Excludes: Financials, REITs
    """

    def __init__(self, as_of_date: datetime):
        """
        Args:
            as_of_date: Date for which we're building the universe (timezone-aware)
        """
        if as_of_date.tzinfo is None:
            as_of_date = pytz.utc.localize(as_of_date)
        self.as_of_date = as_of_date
        self.data_retrieved_date = datetime.now(pytz.utc)
        self.stocks: List[Dict] = []
        self.filtered_count = {
            'sector_excluded': 0,
            'market_cap_too_high': 0,
            'market_cap_too_low': 0,
            'non_us': 0,
        }

    def _build_screener_url(self, offset: int = 0) -> str:
        """
        Build Finviz screener URL with filters.

        Filters:
        - cap_microover: Market Cap > $50M
        - cap_smallunder: Market Cap < $2B (will filter to $500M in code)
        - geo_usa: USA only
        - sh_avgvol_o50: Average Volume > 50K
        """
        params = {
            'v': '111',  # Overview view
            'f': 'cap_microover,cap_smallunder,geo_usa,sh_avgvol_o50',
            'ft': '4',   # Show all matches
            'r': str(offset + 1),  # Starting row (1-indexed)
        }

        param_str = '&'.join(f'{k}={v}' for k, v in params.items())
        return f'{FINVIZ_BASE_URL}?{param_str}'

    def _parse_market_cap(self, market_cap_str: str) -> Optional[float]:
        """
        Parse market cap string to float in millions.

        Args:
            market_cap_str: String like '125.5M' or '1.2B'

        Returns:
            Market cap in millions, or None if parsing fails
        """
        if not market_cap_str or market_cap_str == '-':
            return None

        market_cap_str = market_cap_str.strip().upper()

        try:
            if market_cap_str.endswith('B'):
                return float(market_cap_str[:-1]) * 1000
            elif market_cap_str.endswith('M'):
                return float(market_cap_str[:-1])
            elif market_cap_str.endswith('K'):
                return float(market_cap_str[:-1]) / 1000
            else:
                return float(market_cap_str) / 1_000_000
        except (ValueError, TypeError):
            logger.warning(f"Could not parse market cap: {market_cap_str}")
            return None

    def _parse_volume(self, volume_str: str) -> Optional[int]:
        """
        Parse volume string to integer.

        Args:
            volume_str: String like '1.5M' or '500K'

        Returns:
            Volume as integer, or None if parsing fails
        """
        if not volume_str or volume_str == '-':
            return None

        volume_str = volume_str.strip().upper().replace(',', '')

        try:
            if volume_str.endswith('M'):
                return int(float(volume_str[:-1]) * 1_000_000)
            elif volume_str.endswith('K'):
                return int(float(volume_str[:-1]) * 1_000)
            else:
                return int(float(volume_str))
        except (ValueError, TypeError):
            logger.warning(f"Could not parse volume: {volume_str}")
            return None

    def _scrape_page(self, url: str) -> List[Dict]:
        """
        Scrape a single page of Finviz screener results.

        Args:
            url: Finviz screener URL

        Returns:
            List of stock dictionaries from this page
        """
        stocks = []

        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the screener results table
            table = soup.find('table', class_='screener_table')
            if not table:
                # Try alternative class name
                table = soup.find('table', {'class': re.compile(r'styled-table-new')})

            if not table:
                logger.warning("Could not find screener table on page")
                return stocks

            rows = table.find_all('tr')[1:]  # Skip header row

            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 11:
                    continue

                try:
                    # Finviz overview columns (v=111):
                    # 0: No, 1: Ticker, 2: Company, 3: Sector, 4: Industry,
                    # 5: Country, 6: Market Cap, 7: P/E, 8: Price, 9: Change, 10: Volume

                    ticker = cells[1].get_text(strip=True)
                    company_name = cells[2].get_text(strip=True)
                    sector = cells[3].get_text(strip=True)
                    industry = cells[4].get_text(strip=True)
                    country = cells[5].get_text(strip=True)
                    market_cap_str = cells[6].get_text(strip=True)
                    avg_volume_str = cells[10].get_text(strip=True)

                    market_cap_millions = self._parse_market_cap(market_cap_str)
                    avg_volume = self._parse_volume(avg_volume_str)

                    stocks.append({
                        'ticker': ticker,
                        'company_name': company_name,
                        'sector': sector,
                        'industry': industry,
                        'country': country,
                        'market_cap_millions': market_cap_millions,
                        'market_cap_str': market_cap_str,
                        'avg_volume': avg_volume,
                    })

                except (IndexError, AttributeError) as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue

            return stocks

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return stocks

    def _get_total_count(self, url: str) -> int:
        """Get total number of results from first page."""
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for total count in the page
            total_div = soup.find('td', {'id': 'screener_total'})
            if total_div:
                match = re.search(r'Total:\s*(\d+)', total_div.get_text())
                if match:
                    return int(match.group(1))

            # Alternative: count from page info
            page_info = soup.find('td', class_='count-text')
            if page_info:
                match = re.search(r'/\s*(\d+)', page_info.get_text())
                if match:
                    return int(match.group(1))

            # Fallback: estimate from first page
            return 1000

        except Exception as e:
            logger.warning(f"Could not get total count: {e}")
            return 1000

    def scrape_universe(self) -> pd.DataFrame:
        """
        Scrape full universe of stocks meeting criteria.

        Returns:
            DataFrame with filtered universe
        """
        logger.info(f"{'='*60}")
        logger.info("MICRO-CAP UNIVERSE SCREENER")
        logger.info(f"{'='*60}")
        logger.info(f"As of date: {self.as_of_date.date()}")
        logger.info(f"Criteria: Market Cap $50M-$500M, USA, Avg Vol > 50K")
        logger.info(f"Excluding sectors: {', '.join(EXCLUDED_SECTORS)}")
        logger.info(f"{'='*60}")

        all_stocks = []
        offset = 0
        page_size = 20  # Finviz shows 20 results per page

        # Get first page to estimate total
        first_url = self._build_screener_url(0)
        total_estimate = self._get_total_count(first_url)
        logger.info(f"Estimated total stocks to scan: {total_estimate}")

        while True:
            url = self._build_screener_url(offset)
            logger.info(f"Scraping page at offset {offset}...")

            page_stocks = self._scrape_page(url)

            if not page_stocks:
                logger.info("No more results found")
                break

            all_stocks.extend(page_stocks)
            logger.info(f"  Found {len(page_stocks)} stocks (total: {len(all_stocks)})")

            if len(page_stocks) < page_size:
                break

            offset += page_size

            # Rate limiting
            time.sleep(1)

            # Safety limit
            if offset > 2000:
                logger.warning("Reached safety limit of 2000 stocks")
                break

        logger.info(f"\nTotal stocks scraped: {len(all_stocks)}")

        # Apply filters
        filtered_stocks = self._apply_filters(all_stocks)

        # Create DataFrame
        df = pd.DataFrame(filtered_stocks)

        if not df.empty:
            # Add metadata
            df['as_of_date'] = self.as_of_date
            df['data_retrieved_date'] = self.data_retrieved_date

            # Select and order columns
            df = df[[
                'ticker',
                'company_name',
                'sector',
                'market_cap_millions',
                'avg_volume',
                'industry',
                'as_of_date',
                'data_retrieved_date'
            ]]

            # Rename for output
            df = df.rename(columns={'market_cap_millions': 'market_cap'})

        self._log_summary(df)

        return df

    def _apply_filters(self, stocks: List[Dict]) -> List[Dict]:
        """
        Apply additional filters to scraped stocks.

        Filters:
        - Market cap <= $500M
        - Exclude Financial and Real Estate sectors
        """
        filtered = []

        for stock in stocks:
            # Skip if missing critical data
            if stock['market_cap_millions'] is None:
                continue

            # Filter by market cap (already > $50M from Finviz, need <= $500M)
            if stock['market_cap_millions'] > 500:
                self.filtered_count['market_cap_too_high'] += 1
                continue

            if stock['market_cap_millions'] < 50:
                self.filtered_count['market_cap_too_low'] += 1
                continue

            # Exclude sectors
            if stock['sector'] in EXCLUDED_SECTORS:
                self.filtered_count['sector_excluded'] += 1
                continue

            # Verify US only
            if stock['country'] != 'USA':
                self.filtered_count['non_us'] += 1
                continue

            filtered.append(stock)

        return filtered

    def _log_summary(self, df: pd.DataFrame):
        """Log screening summary."""
        logger.info(f"\n{'='*60}")
        logger.info("SCREENING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Final universe size: {len(df)} stocks")
        logger.info(f"\nFiltered out:")
        logger.info(f"  - Market cap > $500M: {self.filtered_count['market_cap_too_high']}")
        logger.info(f"  - Market cap < $50M: {self.filtered_count['market_cap_too_low']}")
        logger.info(f"  - Excluded sectors: {self.filtered_count['sector_excluded']}")
        logger.info(f"  - Non-US: {self.filtered_count['non_us']}")

        if not df.empty:
            logger.info(f"\nSector breakdown:")
            sector_counts = df['sector'].value_counts()
            for sector, count in sector_counts.items():
                logger.info(f"  {sector}: {count}")

            logger.info(f"\nMarket cap range:")
            logger.info(f"  Min: ${df['market_cap'].min():.1f}M")
            logger.info(f"  Max: ${df['market_cap'].max():.1f}M")
            logger.info(f"  Median: ${df['market_cap'].median():.1f}M")

        logger.info(f"{'='*60}")

    def save_universe(self, df: pd.DataFrame, output_dir: str = 'data/raw') -> str:
        """
        Save universe to CSV file.

        Args:
            df: Universe DataFrame
            output_dir: Output directory path

        Returns:
            Path to saved file
        """
        date_str = self.as_of_date.strftime('%Y%m%d')
        output_path = f'{output_dir}/universe_{date_str}.csv'

        # Save with specific columns for the output
        output_df = df[['ticker', 'company_name', 'sector', 'market_cap', 'avg_volume']].copy()
        output_df.to_csv(output_path, index=False)

        logger.info(f"\nUniverse saved to: {output_path}")

        return output_path


def screen_universe(as_of_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Convenience function to screen for micro-cap universe.

    Args:
        as_of_date: Date for screening (defaults to now)

    Returns:
        DataFrame with universe stocks
    """
    if as_of_date is None:
        as_of_date = datetime.now(pytz.utc)

    screener = UniverseScreener(as_of_date=as_of_date)
    df = screener.scrape_universe()

    return df
