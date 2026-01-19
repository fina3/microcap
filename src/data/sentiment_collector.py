"""
Sentiment Collector for micro-cap stock analysis.

Orchestrates the collection of sentiment data from SEC 8-K filings
with temporal tracking and data quality monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import pandas as pd
import pytz

from .transcript_fetcher import TranscriptFetcher, FilingMetadata, FilingContent
from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from .collector import is_us_ticker, DataQualityTracker

import sys
sys.path.append('..')
from utils.temporal import ensure_utc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentCollector:
    """
    Orchestrates sentiment data collection with temporal tracking and quality monitoring.

    Pattern matches existing MicroCapMetricsCollector for consistency.
    """

    def __init__(
        self,
        as_of_date: datetime,
        fetcher: Optional[TranscriptFetcher] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
        user_agent: str = "MicroCapAnalysis research@example.com"
    ):
        """
        Initialize collector.

        Args:
            as_of_date: Date for temporal validation (timezone-aware UTC)
            fetcher: Optional TranscriptFetcher instance
            analyzer: Optional SentimentAnalyzer instance
            user_agent: User-Agent for SEC requests
        """
        self.as_of_date = ensure_utc(as_of_date)
        self.data_retrieved_date = datetime.now(pytz.utc)

        self.fetcher = fetcher or TranscriptFetcher(user_agent=user_agent)
        self.analyzer = analyzer or SentimentAnalyzer()

        self.quality_tracker = DataQualityTracker()
        self.results: List[SentimentResult] = []
        self.filing_metadata: Dict[str, FilingMetadata] = {}

        logger.info(
            f"SentimentCollector initialized - "
            f"as_of_date: {self.as_of_date.date()}, "
            f"data_retrieved: {self.data_retrieved_date.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

    def collect_sentiment(
        self,
        ticker: str,
        lookback_days: int = 365
    ) -> Optional[SentimentResult]:
        """
        Collect sentiment for most recent available 8-K for a ticker.

        Args:
            ticker: Stock ticker symbol
            lookback_days: How far back to search for filings

        Returns:
            SentimentResult or None if no valid filings found
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting sentiment for {ticker}")
        logger.info(f"{'='*60}")

        # Validate US-only
        if not is_us_ticker(ticker):
            self.quality_tracker.log_non_us_ticker(ticker)
            logger.warning(f"{ticker}: Filtered out - non-US ticker")
            return None

        # Calculate date range
        start_date = self.as_of_date - timedelta(days=lookback_days)
        end_date = self.as_of_date

        logger.info(
            f"{ticker}: Searching for 8-K filings "
            f"from {start_date.date()} to {end_date.date()} "
            f"available as of {self.as_of_date.date()}"
        )

        # Find 8-K filings
        filings = self.fetcher.find_8k_filings(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of_date=self.as_of_date,
            item_filter=["2.02"]
        )

        if not filings:
            self.quality_tracker.log_missing(ticker, '8k_filing', self.as_of_date)
            logger.warning(f"{ticker}: No 8-K filings with Item 2.02 found")

            # Return result with quality flags
            return SentimentResult(
                ticker=ticker,
                as_of_date=self.as_of_date,
                filing_date=self.as_of_date,  # Placeholder
                positive_score=0.0,
                negative_score=0.0,
                net_sentiment=0.0,
                uncertainty_score=0.0,
                hedging_ratio=0.0,
                confidence_ratio=0.0,
                polarity=0.0,
                data_quality_score=0.0,
                quality_flags=['NO_8K_FOUND']
            )

        # Use most recent filing
        filing = filings[0]
        self.filing_metadata[ticker] = filing

        logger.info(
            f"{ticker}: Using 8-K filed {filing.filing_date.date()} "
            f"available as of {self.as_of_date.date()}"
        )

        # Check if filing is old
        days_old = (self.as_of_date - filing.filing_date).days
        quality_flags = []
        if days_old > 180:
            quality_flags.append('OLD_FILING')
            logger.warning(f"{ticker}: Most recent 8-K is {days_old} days old")

        # Fetch content
        content = self.fetcher.fetch_filing_content(filing, self.as_of_date)

        if content is None:
            self.quality_tracker.log_missing(ticker, '8k_content', self.as_of_date)
            logger.error(f"{ticker}: Failed to fetch filing content")

            return SentimentResult(
                ticker=ticker,
                as_of_date=self.as_of_date,
                filing_date=filing.filing_date,
                positive_score=0.0,
                negative_score=0.0,
                net_sentiment=0.0,
                uncertainty_score=0.0,
                hedging_ratio=0.0,
                confidence_ratio=0.0,
                polarity=0.0,
                data_quality_score=0.0,
                quality_flags=['FETCH_FAILED']
            )

        # Add quality notes from content extraction
        quality_flags.extend(content.quality_notes)

        # Analyze sentiment
        text_to_analyze = content.item_202_text or content.extracted_text

        if not text_to_analyze:
            logger.warning(f"{ticker}: No text content to analyze")
            return SentimentResult(
                ticker=ticker,
                as_of_date=self.as_of_date,
                filing_date=filing.filing_date,
                positive_score=0.0,
                negative_score=0.0,
                net_sentiment=0.0,
                uncertainty_score=0.0,
                hedging_ratio=0.0,
                confidence_ratio=0.0,
                polarity=0.0,
                data_quality_score=0.0,
                quality_flags=['NO_TEXT_CONTENT']
            )

        result = self.analyzer.analyze_text(
            text=text_to_analyze,
            as_of_date=self.as_of_date,
            filing_date=filing.filing_date,
            ticker=ticker
        )

        # Merge quality flags
        existing_flags = result.quality_flags or []
        result.quality_flags = list(set(existing_flags + quality_flags))

        # Recalculate quality score with merged flags
        if result.quality_flags:
            penalty = len(result.quality_flags) * 0.1
            result.data_quality_score = max(0.0, result.data_quality_score - penalty)

        logger.info(
            f"{ticker}: Sentiment analysis complete - "
            f"net_sentiment={result.net_sentiment:.3f}, "
            f"polarity={result.polarity:.3f}, "
            f"quality={result.data_quality_score:.2f}"
        )

        return result

    def collect_all_tickers(
        self,
        tickers: List[str],
        lookback_days: int = 365
    ) -> pd.DataFrame:
        """
        Collect sentiment for all tickers.

        Args:
            tickers: List of ticker symbols
            lookback_days: How far back to search for filings

        Returns:
            DataFrame with sentiment scores and metadata
        """
        logger.info(f"\n{'='*60}")
        logger.info("SENTIMENT DATA COLLECTION")
        logger.info(f"{'='*60}")
        logger.info(f"Tickers: {', '.join(tickers)}")
        logger.info(f"As of date: {self.as_of_date.date()}")
        logger.info(f"Lookback: {lookback_days} days")
        logger.info(f"Data retrieved: {self.data_retrieved_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"{'='*60}")

        self.results = []

        for ticker in tickers:
            result = self.collect_sentiment(ticker, lookback_days)
            if result:
                self.results.append(result)

        # Convert to DataFrame
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame([r.to_dict() for r in self.results])

        return df

    def print_summary(self, df: pd.DataFrame):
        """Print collection summary."""
        logger.info(f"\n{'='*60}")
        logger.info("COLLECTION SUMMARY")
        logger.info(f"{'='*60}")

        if df.empty:
            logger.warning("No data collected")
            return

        total_tickers = len(df)

        # Filter valid results
        valid_df = df[df['data_quality_score'] > 0]
        valid_count = len(valid_df)

        logger.info(f"\nTotal tickers processed: {total_tickers}")
        logger.info(f"Valid results: {valid_count}/{total_tickers}")

        if not valid_df.empty:
            avg_quality = valid_df['data_quality_score'].mean()
            logger.info(f"Average data quality: {avg_quality:.2f}")

            avg_sentiment = valid_df['net_sentiment'].mean()
            avg_uncertainty = valid_df['uncertainty_score'].mean()
            avg_polarity = valid_df['polarity'].mean()

            logger.info(f"\nAggregate sentiment metrics:")
            logger.info(f"  Average net sentiment: {avg_sentiment:+.3f}")
            logger.info(f"  Average uncertainty: {avg_uncertainty:.3f}")
            logger.info(f"  Average polarity: {avg_polarity:+.3f}")

        # Per-ticker summary
        logger.info(f"\nPer-ticker sentiment:")
        for _, row in df.iterrows():
            ticker = row['ticker']
            quality = row.get('data_quality_score', 0)
            flags = row.get('quality_flags', '')

            if quality > 0:
                sentiment = row['net_sentiment']
                polarity = row['polarity']
                logger.info(
                    f"  {ticker}: sentiment={sentiment:+.3f}, "
                    f"polarity={polarity:+.3f}, quality={quality:.2f}"
                )
            else:
                logger.info(f"  {ticker}: NO DATA - {flags}")

        # Quality tracker summary
        quality_summary = self.quality_tracker.get_summary()
        if quality_summary['missing_fields_count'] > 0:
            logger.info(f"\nMissing data fields: {quality_summary['missing_fields_count']}")
        if quality_summary['non_us_tickers_count'] > 0:
            logger.info(f"Non-US tickers filtered: {quality_summary['non_us_tickers_count']}")

        logger.info(f"\n{'='*60}")
