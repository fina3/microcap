"""
Insider Transaction Fetcher for SEC EDGAR Form 4 filings.

Fetches Form 4 filings to analyze insider buying/selling patterns.
Implements rate limiting and temporal validation.
"""

import re
import time
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET

import requests
import pytz

import sys
sys.path.append('..')
from utils.temporal import ensure_utc, validate_temporal_consistency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# SEC EDGAR API endpoints
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"


@dataclass
class InsiderTransaction:
    """A single insider transaction from Form 4."""
    ticker: str
    filing_date: datetime
    transaction_date: Optional[datetime]
    insider_name: str
    insider_title: str
    transaction_type: str  # 'P' = Purchase, 'S' = Sale, 'A' = Award, etc.
    shares: float
    price_per_share: Optional[float]
    shares_owned_after: Optional[float]
    is_direct: bool  # Direct vs indirect ownership

    @property
    def is_purchase(self) -> bool:
        """Check if this is a purchase transaction."""
        return self.transaction_type in ('P', 'A', 'M', 'G', 'I', 'J')

    @property
    def is_sale(self) -> bool:
        """Check if this is a sale transaction."""
        return self.transaction_type in ('S', 'D', 'F')


@dataclass
class InsiderActivity:
    """Aggregated insider activity for a ticker."""
    ticker: str
    as_of_date: datetime
    lookback_days: int
    total_bought: float  # Total shares purchased
    total_sold: float  # Total shares sold
    net_purchases: float  # bought - sold
    buy_ratio: Optional[float]  # bought / (bought + sold)
    num_buyers: int
    num_sellers: int
    num_transactions: int
    transactions: List[InsiderTransaction] = field(default_factory=list)
    data_quality: float = 1.0
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'as_of_date': self.as_of_date,
            'lookback_days': self.lookback_days,
            'insider_shares_bought': self.total_bought,
            'insider_shares_sold': self.total_sold,
            'insider_net_purchases': self.net_purchases,
            'insider_buy_ratio': self.buy_ratio,
            'insider_num_buyers': self.num_buyers,
            'insider_num_sellers': self.num_sellers,
            'insider_num_transactions': self.num_transactions,
            'insider_data_quality': self.data_quality,
            'insider_quality_flags': ','.join(self.quality_flags) if self.quality_flags else ''
        }


class RateLimiter:
    """Rate limiter for SEC EDGAR API compliance."""

    def __init__(self, requests_per_second: float = 8.0):
        self.min_interval = 1.0 / min(requests_per_second, 10.0)
        self.last_request_time = 0.0

    def wait_if_needed(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class InsiderFetcher:
    """
    Fetches Form 4 filings from SEC EDGAR to analyze insider transactions.

    Form 4 must be filed within 2 business days of an insider transaction.
    Tracks purchases (P), sales (S), and other transaction types.
    """

    def __init__(
        self,
        user_agent: str = "MicroCapAnalysis research@example.com",
        requests_per_second: float = 8.0,
        cache_dir: str = "data/raw/form4_cache"
    ):
        """
        Initialize fetcher with SEC-compliant settings.

        Args:
            user_agent: Required User-Agent header
            requests_per_second: Rate limit (max 10, default 8)
            cache_dir: Directory to cache fetched filings
        """
        self.user_agent = user_agent
        self.rate_limiter = RateLimiter(requests_per_second)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate'
        })

        self._ticker_cik_map: Optional[Dict[str, str]] = None

        logger.info(f"InsiderFetcher initialized")

    def _make_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Make rate-limited request with retry logic."""
        for attempt in range(max_retries):
            self.rate_limiter.wait_if_needed()

            try:
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    logger.debug(f"Not found: {url}")
                    return None
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def get_ticker_cik_map(self) -> Dict[str, str]:
        """Get or refresh ticker to CIK mapping."""
        if self._ticker_cik_map is not None:
            return self._ticker_cik_map

        cache_file = self.cache_dir / "ticker_cik_map.json"
        cache_age_days = 7

        if cache_file.exists():
            cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=pytz.utc)
            if (datetime.now(pytz.utc) - cache_mtime).days < cache_age_days:
                try:
                    with open(cache_file, 'r') as f:
                        self._ticker_cik_map = json.load(f)
                        return self._ticker_cik_map
                except (json.JSONDecodeError, IOError):
                    pass

        logger.info("Fetching ticker-CIK mapping from SEC...")
        response = self._make_request(SEC_COMPANY_TICKERS_URL)

        if response is None:
            self._ticker_cik_map = {}
            return self._ticker_cik_map

        try:
            data = response.json()
            self._ticker_cik_map = {}

            for entry in data.values():
                ticker = entry.get('ticker', '').upper()
                cik = str(entry.get('cik_str', ''))
                if ticker and cik:
                    self._ticker_cik_map[ticker] = cik.zfill(10)

            with open(cache_file, 'w') as f:
                json.dump(self._ticker_cik_map, f)

        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing ticker data: {e}")
            self._ticker_cik_map = {}

        return self._ticker_cik_map

    def get_cik_for_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK for a ticker symbol."""
        cik_map = self.get_ticker_cik_map()
        return cik_map.get(ticker.upper())

    def fetch_form4_filings(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        as_of_date: datetime
    ) -> List[Dict]:
        """
        Fetch Form 4 filing metadata for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start of date range
            end_date: End of date range
            as_of_date: Analysis date for temporal validation

        Returns:
            List of filing metadata dictionaries
        """
        start_date = ensure_utc(start_date)
        end_date = ensure_utc(end_date)
        as_of_date = ensure_utc(as_of_date)

        cik = self.get_cik_for_ticker(ticker)
        if not cik:
            logger.warning(f"{ticker}: No CIK found")
            return []

        submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)
        response = self._make_request(submissions_url)

        if response is None:
            logger.error(f"{ticker}: Failed to fetch submissions")
            return []

        try:
            data = response.json()
        except ValueError:
            logger.error(f"{ticker}: Invalid JSON response")
            return []

        filings = []
        recent = data.get('filings', {}).get('recent', {})

        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])

        for i in range(len(forms)):
            if forms[i] != '4':
                continue

            try:
                filing_date = datetime.strptime(dates[i], '%Y-%m-%d')
                filing_date = ensure_utc(filing_date)
            except (ValueError, IndexError):
                continue

            if filing_date < start_date or filing_date > end_date:
                continue

            # Temporal validation
            if not validate_temporal_consistency(as_of_date, filing_date):
                continue

            accession = accessions[i]
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""

            # Form 4 XML is usually the primary document or named with .xml
            xml_filename = primary_doc
            if not xml_filename.endswith('.xml'):
                # Try to find the XML file
                xml_filename = primary_doc.replace('.htm', '.xml').replace('.html', '.xml')

            filings.append({
                'ticker': ticker,
                'cik': cik,
                'accession': accession,
                'filing_date': filing_date,
                'primary_doc': primary_doc,
                'xml_filename': xml_filename
            })

        logger.debug(f"{ticker}: Found {len(filings)} Form 4 filings")
        return filings

    def parse_form4_xml(
        self,
        ticker: str,
        cik: str,
        accession: str,
        filing_date: datetime
    ) -> List[InsiderTransaction]:
        """
        Parse Form 4 XML to extract transactions.

        Args:
            ticker: Stock ticker
            cik: Company CIK
            accession: Filing accession number
            filing_date: Filing date

        Returns:
            List of InsiderTransaction objects
        """
        transactions = []

        # Build URL for the XML file
        accession_formatted = accession.replace('-', '')
        cik_stripped = cik.lstrip('0')

        # First, fetch the index page to find the XML file
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{accession_formatted}/"
        response = self._make_request(index_url)

        xml_content = None
        if response:
            # Look for Form 4 XML file (typically named with form4 or wk-form4)
            xml_matches = re.findall(r'href="([^"]+\.xml)"', response.text, re.IGNORECASE)
            for xml_href in xml_matches:
                # Skip index files
                if 'index' in xml_href.lower():
                    continue

                # Handle both absolute and relative paths
                if xml_href.startswith('/'):
                    xml_url = f"https://www.sec.gov{xml_href}"
                elif xml_href.startswith('http'):
                    xml_url = xml_href
                else:
                    xml_url = f"{index_url}{xml_href}"

                xml_response = self._make_request(xml_url)
                if xml_response and xml_response.status_code == 200:
                    # Check if it looks like a Form 4
                    if '<ownershipDocument' in xml_response.text or '<nonDerivativeTransaction' in xml_response.text:
                        xml_content = xml_response.text
                        break

        if not xml_content:
            logger.debug(f"{ticker}: Could not fetch Form 4 XML for {accession}")
            return transactions

        try:
            root = ET.fromstring(xml_content)

            # Extract reporting owner info
            owner_name = ""
            owner_title = ""

            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

                if tag == 'rptOwnerName':
                    owner_name = elem.text or ""
                elif tag == 'officerTitle':
                    owner_title = elem.text or ""

            # Extract non-derivative transactions
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

                if tag in ('nonDerivativeTransaction', 'derivativeTransaction'):
                    txn = self._parse_transaction_element(
                        elem, ticker, filing_date, owner_name, owner_title
                    )
                    if txn:
                        transactions.append(txn)

            logger.debug(f"{ticker}: Parsed {len(transactions)} transactions from {accession}")

        except ET.ParseError as e:
            logger.warning(f"{ticker}: XML parse error for {accession}: {e}")
        except Exception as e:
            logger.warning(f"{ticker}: Error parsing Form 4: {e}")

        return transactions

    def _parse_transaction_element(
        self,
        element: ET.Element,
        ticker: str,
        filing_date: datetime,
        owner_name: str,
        owner_title: str
    ) -> Optional[InsiderTransaction]:
        """Parse a transaction element from Form 4 XML."""
        try:
            trans_date = None
            trans_code = None
            acquired_disposed = None
            shares = 0.0
            price = None
            shares_after = None
            is_direct = True

            # Iterate through all descendants
            for child in element.iter():
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

                # Transaction date - get from value subelement
                if tag == 'transactionDate':
                    for val in child.iter():
                        val_tag = val.tag.split('}')[-1] if '}' in val.tag else val.tag
                        if val_tag == 'value' and val.text:
                            try:
                                trans_date = datetime.strptime(val.text, '%Y-%m-%d')
                                trans_date = ensure_utc(trans_date)
                            except (ValueError, TypeError):
                                pass

                # Transaction code (P=Purchase, S=Sale, etc.)
                elif tag == 'transactionCoding':
                    for val in child.iter():
                        val_tag = val.tag.split('}')[-1] if '}' in val.tag else val.tag
                        if val_tag == 'transactionCode' and val.text:
                            trans_code = val.text.strip()

                # Acquired or Disposed
                elif tag == 'transactionAcquiredDisposedCode':
                    for val in child.iter():
                        val_tag = val.tag.split('}')[-1] if '}' in val.tag else val.tag
                        if val_tag == 'value' and val.text:
                            acquired_disposed = val.text.strip()

                # Shares
                elif tag == 'transactionShares':
                    for val in child.iter():
                        val_tag = val.tag.split('}')[-1] if '}' in val.tag else val.tag
                        if val_tag == 'value' and val.text:
                            try:
                                shares = float(val.text)
                            except (ValueError, TypeError):
                                pass

                # Price per share
                elif tag == 'transactionPricePerShare':
                    for val in child.iter():
                        val_tag = val.tag.split('}')[-1] if '}' in val.tag else val.tag
                        if val_tag == 'value' and val.text:
                            try:
                                price = float(val.text)
                            except (ValueError, TypeError):
                                pass

                # Shares owned after
                elif tag == 'sharesOwnedFollowingTransaction':
                    for val in child.iter():
                        val_tag = val.tag.split('}')[-1] if '}' in val.tag else val.tag
                        if val_tag == 'value' and val.text:
                            try:
                                shares_after = float(val.text)
                            except (ValueError, TypeError):
                                pass

                # Direct or indirect
                elif tag == 'directOrIndirectOwnership':
                    for val in child.iter():
                        val_tag = val.tag.split('}')[-1] if '}' in val.tag else val.tag
                        if val_tag == 'value' and val.text:
                            is_direct = val.text.strip() == 'D'

            # Determine effective transaction type
            # If we have acquired_disposed code, use it to clarify
            if trans_code and shares > 0:
                # S with D (disposed) = sale, P with A (acquired) = purchase
                # A code (award) is typically an acquisition
                effective_type = trans_code

                return InsiderTransaction(
                    ticker=ticker,
                    filing_date=filing_date,
                    transaction_date=trans_date,
                    insider_name=owner_name,
                    insider_title=owner_title,
                    transaction_type=effective_type,
                    shares=shares,
                    price_per_share=price,
                    shares_owned_after=shares_after,
                    is_direct=is_direct
                )

        except Exception as e:
            logger.debug(f"Error parsing transaction: {e}")

        return None

    def get_insider_activity(
        self,
        ticker: str,
        as_of_date: datetime,
        lookback_days: int = 90
    ) -> InsiderActivity:
        """
        Get aggregated insider activity for a ticker.

        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis date (timezone-aware)
            lookback_days: Days to look back for transactions

        Returns:
            InsiderActivity with aggregated metrics
        """
        as_of_date = ensure_utc(as_of_date)
        start_date = as_of_date - timedelta(days=lookback_days)
        end_date = as_of_date

        logger.info(
            f"{ticker}: Fetching insider activity from {start_date.date()} "
            f"to {end_date.date()} (as of {as_of_date.date()})"
        )

        quality_flags = []

        # Fetch Form 4 filings
        filings = self.fetch_form4_filings(ticker, start_date, end_date, as_of_date)

        if not filings:
            logger.info(f"{ticker}: No Form 4 filings found in period")
            return InsiderActivity(
                ticker=ticker,
                as_of_date=as_of_date,
                lookback_days=lookback_days,
                total_bought=0,
                total_sold=0,
                net_purchases=0,
                buy_ratio=None,
                num_buyers=0,
                num_sellers=0,
                num_transactions=0,
                transactions=[],
                data_quality=0.5,
                quality_flags=['NO_FORM4_FILINGS']
            )

        # Parse all transactions
        all_transactions = []
        for filing in filings:
            transactions = self.parse_form4_xml(
                ticker=filing['ticker'],
                cik=filing['cik'],
                accession=filing['accession'],
                filing_date=filing['filing_date']
            )
            all_transactions.extend(transactions)

        if not all_transactions:
            logger.info(f"{ticker}: No transactions parsed from {len(filings)} filings")
            quality_flags.append('NO_TRANSACTIONS_PARSED')
            return InsiderActivity(
                ticker=ticker,
                as_of_date=as_of_date,
                lookback_days=lookback_days,
                total_bought=0,
                total_sold=0,
                net_purchases=0,
                buy_ratio=None,
                num_buyers=0,
                num_sellers=0,
                num_transactions=len(filings),
                transactions=[],
                data_quality=0.5,
                quality_flags=quality_flags
            )

        # Aggregate transactions
        total_bought = sum(t.shares for t in all_transactions if t.is_purchase)
        total_sold = sum(t.shares for t in all_transactions if t.is_sale)
        net_purchases = total_bought - total_sold

        # Calculate buy ratio
        total_volume = total_bought + total_sold
        if total_volume > 0:
            buy_ratio = total_bought / total_volume
        else:
            buy_ratio = None

        # Count unique buyers/sellers
        buyers = set(t.insider_name for t in all_transactions if t.is_purchase)
        sellers = set(t.insider_name for t in all_transactions if t.is_sale)

        # Assess data quality
        data_quality = 1.0
        if len(all_transactions) < 3:
            data_quality -= 0.2
            quality_flags.append('FEW_TRANSACTIONS')

        buy_ratio_str = f"{buy_ratio:.2f}" if buy_ratio is not None else "N/A"
        logger.info(
            f"{ticker}: Insider activity - "
            f"bought={total_bought:,.0f}, sold={total_sold:,.0f}, "
            f"net={net_purchases:+,.0f}, buy_ratio={buy_ratio_str}"
        )

        return InsiderActivity(
            ticker=ticker,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            total_bought=total_bought,
            total_sold=total_sold,
            net_purchases=net_purchases,
            buy_ratio=buy_ratio,
            num_buyers=len(buyers),
            num_sellers=len(sellers),
            num_transactions=len(all_transactions),
            transactions=all_transactions,
            data_quality=data_quality,
            quality_flags=quality_flags
        )


def fetch_insider_activity(
    ticker: str,
    as_of_date: Optional[datetime] = None,
    lookback_days: int = 90
) -> InsiderActivity:
    """
    Convenience function to fetch insider activity.

    Args:
        ticker: Stock ticker symbol
        as_of_date: Analysis date (default: now)
        lookback_days: Days to look back

    Returns:
        InsiderActivity object
    """
    if as_of_date is None:
        as_of_date = datetime.now(pytz.utc)

    fetcher = InsiderFetcher()
    return fetcher.get_insider_activity(ticker, as_of_date, lookback_days)
