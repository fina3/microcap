"""
Transcript Fetcher for SEC EDGAR 8-K filings.

Fetches earnings-related 8-K filings (Item 2.02) from SEC EDGAR API.
Implements rate limiting and temporal validation.
"""

import os
import re
import time
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

import requests
from bs4 import BeautifulSoup
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
class FilingMetadata:
    """Metadata for an SEC filing."""
    ticker: str
    cik: str
    accession_number: str
    filing_date: datetime  # PUBLIC AVAILABILITY DATE
    form_type: str
    items: List[str]
    primary_document: str
    file_url: str

    def __post_init__(self):
        self.filing_date = ensure_utc(self.filing_date)


@dataclass
class FilingContent:
    """Content extracted from an SEC filing."""
    metadata: FilingMetadata
    raw_html: str = ""
    extracted_text: str = ""
    item_202_text: str = ""  # Item 2.02 specific content
    text_length: int = 0
    extraction_quality: float = 1.0
    quality_notes: List[str] = field(default_factory=list)


class RateLimiter:
    """
    Rate limiter for SEC EDGAR API compliance.

    SEC allows max 10 requests/second. Default is 8 for safety margin.
    """

    def __init__(self, requests_per_second: float = 8.0):
        """
        Args:
            requests_per_second: Maximum requests per second (max 10)
        """
        self.min_interval = 1.0 / min(requests_per_second, 10.0)
        self.last_request_time = 0.0

    def wait_if_needed(self):
        """Block until it's safe to make another request."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class TranscriptFetcher:
    """
    Fetches 8-K filings from SEC EDGAR.

    Focuses on Item 2.02 "Results of Operations and Financial Condition"
    which contains earnings announcement information.
    """

    def __init__(
        self,
        user_agent: str = "MicroCapAnalysis research@example.com",
        requests_per_second: float = 8.0,
        cache_dir: str = "data/raw/filings_cache"
    ):
        """
        Initialize fetcher with SEC-compliant settings.

        Args:
            user_agent: Required User-Agent header (format: "AppName contact@email.com")
            requests_per_second: Rate limit (max 10, default 8 for safety)
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

        logger.info(f"TranscriptFetcher initialized with User-Agent: {user_agent}")

    def _make_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """
        Make rate-limited request with retry logic.

        Args:
            url: URL to fetch
            max_retries: Maximum retry attempts

        Returns:
            Response object or None if failed
        """
        for attempt in range(max_retries):
            self.rate_limiter.wait_if_needed()

            try:
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    logger.warning(f"Not found: {url}")
                    return None
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def get_ticker_cik_map(self) -> Dict[str, str]:
        """
        Get or refresh ticker to CIK mapping.

        Returns:
            Dictionary mapping ticker symbols to CIK numbers (padded to 10 digits)
        """
        if self._ticker_cik_map is not None:
            return self._ticker_cik_map

        cache_file = self.cache_dir / "ticker_cik_map.json"
        cache_age_days = 7

        # Check cache
        if cache_file.exists():
            cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=pytz.utc)
            if (datetime.now(pytz.utc) - cache_mtime).days < cache_age_days:
                try:
                    with open(cache_file, 'r') as f:
                        self._ticker_cik_map = json.load(f)
                        logger.info(f"Loaded {len(self._ticker_cik_map)} tickers from cache")
                        return self._ticker_cik_map
                except (json.JSONDecodeError, IOError):
                    pass

        # Fetch from SEC
        logger.info("Fetching ticker-CIK mapping from SEC...")
        response = self._make_request(SEC_COMPANY_TICKERS_URL)

        if response is None:
            logger.error("Failed to fetch ticker-CIK mapping")
            self._ticker_cik_map = {}
            return self._ticker_cik_map

        try:
            data = response.json()
            self._ticker_cik_map = {}

            for entry in data.values():
                ticker = entry.get('ticker', '').upper()
                cik = str(entry.get('cik_str', ''))

                if ticker and cik:
                    # Pad CIK to 10 digits
                    self._ticker_cik_map[ticker] = cik.zfill(10)

            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(self._ticker_cik_map, f)

            logger.info(f"Cached {len(self._ticker_cik_map)} ticker mappings")

        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing ticker data: {e}")
            self._ticker_cik_map = {}

        return self._ticker_cik_map

    def get_cik_for_ticker(self, ticker: str) -> Optional[str]:
        """
        Get CIK for a ticker symbol.

        Args:
            ticker: Stock ticker symbol

        Returns:
            10-digit CIK or None if not found
        """
        cik_map = self.get_ticker_cik_map()
        return cik_map.get(ticker.upper())

    def find_8k_filings(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        as_of_date: datetime,
        item_filter: Optional[List[str]] = None
    ) -> List[FilingMetadata]:
        """
        Find 8-K filings for a ticker within date range.

        CRITICAL: Only returns filings with filing_date <= as_of_date
        to prevent lookahead bias.

        Args:
            ticker: Stock ticker symbol
            start_date: Start of date range (timezone-aware)
            end_date: End of date range (timezone-aware)
            as_of_date: Analysis date for temporal validation (timezone-aware)
            item_filter: Optional list of item numbers to filter (e.g., ["2.02"])

        Returns:
            List of FilingMetadata for matching 8-K filings
        """
        start_date = ensure_utc(start_date)
        end_date = ensure_utc(end_date)
        as_of_date = ensure_utc(as_of_date)

        if item_filter is None:
            item_filter = ["2.02"]

        cik = self.get_cik_for_ticker(ticker)
        if not cik:
            logger.warning(f"{ticker}: No CIK found")
            return []

        # Fetch company submissions
        submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)
        response = self._make_request(submissions_url)

        if response is None:
            logger.error(f"{ticker}: Failed to fetch submissions")
            return []

        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"{ticker}: Invalid JSON response: {e}")
            return []

        filings = []
        recent = data.get('filings', {}).get('recent', {})

        # Get arrays from recent filings
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])
        items_list = recent.get('items', [])

        for i in range(len(forms)):
            if forms[i] != '8-K':
                continue

            try:
                filing_date = datetime.strptime(dates[i], '%Y-%m-%d')
                filing_date = ensure_utc(filing_date)
            except (ValueError, IndexError):
                continue

            # Check date range
            if filing_date < start_date or filing_date > end_date:
                continue

            # TEMPORAL VALIDATION: Only include if available as of as_of_date
            if not validate_temporal_consistency(as_of_date, filing_date):
                logger.debug(
                    f"{ticker}: Skipping 8-K from {filing_date.date()} - "
                    f"not available as of {as_of_date.date()}"
                )
                continue

            # Parse items
            items_str = items_list[i] if i < len(items_list) else ""
            items = [item.strip() for item in items_str.split(',') if item.strip()]

            # Check item filter
            if item_filter:
                has_target_item = any(
                    any(target in item for target in item_filter)
                    for item in items
                )
                if not has_target_item:
                    continue

            # Build file URL
            accession = accessions[i]
            accession_formatted = accession.replace('-', '')
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""
            file_url = SEC_ARCHIVES_URL.format(
                cik=cik.lstrip('0'),
                accession=accession_formatted,
                filename=primary_doc
            )

            filing = FilingMetadata(
                ticker=ticker,
                cik=cik,
                accession_number=accession,
                filing_date=filing_date,
                form_type='8-K',
                items=items,
                primary_document=primary_doc,
                file_url=file_url
            )
            filings.append(filing)

        logger.info(
            f"{ticker}: Found {len(filings)} 8-K filings "
            f"from {start_date.date()} to {end_date.date()} "
            f"available as of {as_of_date.date()}"
        )

        # Sort by date descending (most recent first)
        filings.sort(key=lambda x: x.filing_date, reverse=True)

        return filings

    def fetch_filing_content(
        self,
        filing: FilingMetadata,
        as_of_date: datetime
    ) -> Optional[FilingContent]:
        """
        Fetch and parse 8-K filing content.

        Args:
            filing: FilingMetadata for the filing to fetch
            as_of_date: Analysis date for temporal validation

        Returns:
            FilingContent or None if not available/fetchable
        """
        as_of_date = ensure_utc(as_of_date)

        # Validate temporal consistency
        if not validate_temporal_consistency(as_of_date, filing.filing_date):
            logger.warning(
                f"{filing.ticker}: Filing from {filing.filing_date.date()} "
                f"not available as of {as_of_date.date()}"
            )
            return None

        # Check cache
        cache_key = f"{filing.ticker}_{filing.accession_number.replace('-', '')}.html"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            logger.debug(f"{filing.ticker}: Loading cached filing")
            try:
                raw_html = cache_path.read_text(encoding='utf-8')
            except IOError:
                raw_html = None
        else:
            # Fetch from SEC
            logger.info(f"{filing.ticker}: Fetching filing from {filing.file_url}")
            response = self._make_request(filing.file_url)

            if response is None:
                logger.error(f"{filing.ticker}: Failed to fetch filing content")
                return None

            raw_html = response.text

            # Cache the result
            try:
                cache_path.write_text(raw_html, encoding='utf-8')
            except IOError as e:
                logger.warning(f"Failed to cache filing: {e}")

        # Extract content
        quality_notes = []

        extracted_text = self._extract_text_from_html(raw_html)
        item_202_text = self.extract_item_202_text(raw_html)

        if not item_202_text:
            quality_notes.append("ITEM_202_NOT_FOUND")
            # Fall back to full extracted text
            item_202_text = extracted_text

        if len(item_202_text) < 100:
            quality_notes.append("SHORT_TEXT")

        extraction_quality = 1.0
        if "ITEM_202_NOT_FOUND" in quality_notes:
            extraction_quality -= 0.3
        if "SHORT_TEXT" in quality_notes:
            extraction_quality -= 0.2

        return FilingContent(
            metadata=filing,
            raw_html=raw_html,
            extracted_text=extracted_text,
            item_202_text=item_202_text,
            text_length=len(item_202_text),
            extraction_quality=max(0.0, extraction_quality),
            quality_notes=quality_notes
        )

    def _extract_text_from_html(self, html: str) -> str:
        """
        Extract plain text from HTML content.

        Args:
            html: Raw HTML content

        Returns:
            Cleaned plain text
        """
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Remove script and style elements
            for element in soup(['script', 'style', 'meta', 'link']):
                element.decompose()

            text = soup.get_text(separator=' ')

            # Clean whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            return text

        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""

    def extract_item_202_text(self, html: str) -> str:
        """
        Extract Item 2.02 section from 8-K HTML.

        Item 2.02 contains "Results of Operations and Financial Condition".

        Args:
            html: Raw HTML content of 8-K filing

        Returns:
            Extracted Item 2.02 text or empty string if not found
        """
        try:
            soup = BeautifulSoup(html, 'lxml')
            text = soup.get_text(separator='\n')

            # Patterns to find Item 2.02 section
            item_patterns = [
                r'(?i)item\s*2\.02[.\s]*results\s+of\s+operations',
                r'(?i)item\s*2\.02\b',
                r'(?i)ITEM\s*2\.02',
            ]

            # Patterns for end of section
            end_patterns = [
                r'(?i)item\s*[3-9]\.',
                r'(?i)ITEM\s*[3-9]\.',
                r'(?i)signature[s]?\s*$',
                r'(?i)pursuant\s+to\s+the\s+requirements',
            ]

            start_pos = None
            for pattern in item_patterns:
                match = re.search(pattern, text)
                if match:
                    start_pos = match.start()
                    break

            if start_pos is None:
                return ""

            # Find end of section
            remaining_text = text[start_pos:]
            end_pos = len(remaining_text)

            for pattern in end_patterns:
                match = re.search(pattern, remaining_text[100:])  # Skip past Item 2.02 header
                if match:
                    end_pos = min(end_pos, match.start() + 100)

            section_text = remaining_text[:end_pos]

            # Clean the text
            section_text = re.sub(r'\s+', ' ', section_text)
            section_text = section_text.strip()

            # Remove common boilerplate
            boilerplate_patterns = [
                r'forward-looking statements?.*?(?:securities act|safe harbor)',
                r'this (current )?report (on form 8-k )?is being filed',
            ]

            for pattern in boilerplate_patterns:
                section_text = re.sub(pattern, '', section_text, flags=re.IGNORECASE | re.DOTALL)

            return section_text.strip()

        except Exception as e:
            logger.error(f"Error extracting Item 2.02: {e}")
            return ""
