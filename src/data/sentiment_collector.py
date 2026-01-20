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
        Collect sentiment from both 8-K and 10-Q filings for a ticker.

        Analyzes:
        - 8-K Item 2.02 (earnings announcements)
        - 10-Q MD&A section (more comprehensive quarterly analysis)

        Combined sentiment = 0.4 * 8K + 0.6 * 10Q (10-Q weighted higher)

        Args:
            ticker: Stock ticker symbol
            lookback_days: How far back to search for filings

        Returns:
            SentimentResult with combined sentiment, or None if no valid filings
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

        quality_flags = []

        # ===== ANALYZE 8-K FILINGS =====
        logger.info(f"{ticker}: Searching for 8-K filings...")

        filings_8k = self.fetcher.find_8k_filings(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of_date=self.as_of_date,
            item_filter=["2.02"]
        )

        sentiment_8k = None
        filing_date_8k = None
        words_8k = 0
        result_8k = None

        if filings_8k:
            current_8k = filings_8k[0]
            prior_8k = filings_8k[1] if len(filings_8k) > 1 else None
            self.filing_metadata[ticker] = current_8k

            logger.info(f"{ticker}: Found 8-K filed {current_8k.filing_date.date()}")

            content_8k = self.fetcher.fetch_filing_content(current_8k, self.as_of_date)
            if content_8k:
                text_8k = content_8k.item_202_text or content_8k.extracted_text
                if text_8k:
                    result_8k = self.analyzer.analyze_text(
                        text=text_8k,
                        as_of_date=self.as_of_date,
                        filing_date=current_8k.filing_date,
                        ticker=ticker
                    )
                    sentiment_8k = result_8k.net_sentiment
                    filing_date_8k = current_8k.filing_date
                    words_8k = result_8k.total_words
                    quality_flags.extend(content_8k.quality_notes)
                    logger.info(f"{ticker}: 8-K sentiment = {sentiment_8k:.3f} ({words_8k} words)")
        else:
            quality_flags.append('NO_8K_FOUND')
            logger.warning(f"{ticker}: No 8-K filings with Item 2.02 found")

        # ===== ANALYZE 10-Q FILINGS =====
        logger.info(f"{ticker}: Searching for 10-Q filings...")

        filings_10q = self.fetcher.find_10q_filings(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            as_of_date=self.as_of_date
        )

        sentiment_10q = None
        filing_date_10q = None
        words_10q = 0
        result_10q = None

        if filings_10q:
            current_10q = filings_10q[0]
            logger.info(f"{ticker}: Found 10-Q filed {current_10q.filing_date.date()}")

            content_10q = self.fetcher.fetch_10q_content(current_10q, self.as_of_date)
            if content_10q:
                text_10q = content_10q.mda_text or content_10q.extracted_text
                if text_10q:
                    result_10q = self.analyzer.analyze_text(
                        text=text_10q,
                        as_of_date=self.as_of_date,
                        filing_date=current_10q.filing_date,
                        ticker=ticker
                    )
                    sentiment_10q = result_10q.net_sentiment
                    filing_date_10q = current_10q.filing_date
                    words_10q = result_10q.total_words
                    quality_flags.extend(content_10q.quality_notes)
                    logger.info(f"{ticker}: 10-Q sentiment = {sentiment_10q:.3f} ({words_10q} words)")
        else:
            quality_flags.append('NO_10Q_FOUND')
            logger.warning(f"{ticker}: No 10-Q filings found")

        # ===== CALCULATE COMBINED SENTIMENT =====
        # Weights: 10-Q gets more weight (0.6) because MD&A is more comprehensive
        WEIGHT_8K = 0.4
        WEIGHT_10Q = 0.6

        if sentiment_8k is not None and sentiment_10q is not None:
            # Both available - weighted average
            sentiment_combined = WEIGHT_8K * sentiment_8k + WEIGHT_10Q * sentiment_10q
            logger.info(
                f"{ticker}: Combined sentiment = {sentiment_combined:.3f} "
                f"(8K: {sentiment_8k:.3f} * {WEIGHT_8K} + 10Q: {sentiment_10q:.3f} * {WEIGHT_10Q})"
            )
        elif sentiment_10q is not None:
            # Only 10-Q available
            sentiment_combined = sentiment_10q
            quality_flags.append('10Q_ONLY')
            logger.info(f"{ticker}: Using 10-Q sentiment only: {sentiment_combined:.3f}")
        elif sentiment_8k is not None:
            # Only 8-K available
            sentiment_combined = sentiment_8k
            quality_flags.append('8K_ONLY')
            logger.info(f"{ticker}: Using 8-K sentiment only: {sentiment_combined:.3f}")
        else:
            # No filings found
            logger.warning(f"{ticker}: No valid filings found")
            return SentimentResult(
                ticker=ticker,
                as_of_date=self.as_of_date,
                filing_date=self.as_of_date,
                positive_score=0.0,
                negative_score=0.0,
                net_sentiment=0.0,
                uncertainty_score=0.0,
                hedging_ratio=0.0,
                confidence_ratio=0.0,
                polarity=0.0,
                data_quality_score=0.0,
                quality_flags=['NO_FILINGS_FOUND']
            )

        # ===== BUILD RESULT =====
        # Use the more comprehensive source (10-Q if available) as the base result
        base_result = result_10q if result_10q else result_8k
        primary_filing_date = filing_date_10q or filing_date_8k

        # Calculate sentiment change from prior 8-K (if available)
        prior_sentiment = None
        prior_filing_date = None
        sentiment_change = None
        sentiment_momentum = 0

        if filings_8k and len(filings_8k) > 1:
            prior_8k = filings_8k[1]
            prior_content = self.fetcher.fetch_filing_content(prior_8k, self.as_of_date)
            if prior_content:
                prior_text = prior_content.item_202_text or prior_content.extracted_text
                if prior_text:
                    prior_result = self.analyzer.analyze_text(
                        text=prior_text,
                        as_of_date=self.as_of_date,
                        filing_date=prior_8k.filing_date,
                        ticker=ticker
                    )
                    prior_sentiment = prior_result.net_sentiment
                    prior_filing_date = prior_8k.filing_date
                    sentiment_change = sentiment_combined - prior_sentiment

                    if sentiment_change > 0.05:
                        sentiment_momentum = 1
                    elif sentiment_change < -0.05:
                        sentiment_momentum = -1
                    else:
                        sentiment_momentum = 0

        # Create final result
        result = SentimentResult(
            ticker=ticker,
            as_of_date=self.as_of_date,
            filing_date=primary_filing_date,
            positive_score=base_result.positive_score,
            negative_score=base_result.negative_score,
            net_sentiment=sentiment_combined,  # Use combined as primary
            uncertainty_score=base_result.uncertainty_score,
            hedging_ratio=base_result.hedging_ratio,
            confidence_ratio=base_result.confidence_ratio,
            polarity=base_result.polarity,
            positive_count=base_result.positive_count,
            negative_count=base_result.negative_count,
            uncertainty_count=base_result.uncertainty_count,
            hedging_count=base_result.hedging_count,
            confidence_count=base_result.confidence_count,
            total_words=words_8k + words_10q,
            prior_sentiment=prior_sentiment,
            prior_filing_date=prior_filing_date,
            sentiment_change=sentiment_change,
            sentiment_momentum=sentiment_momentum,
            sentiment_8k=sentiment_8k,
            sentiment_10q=sentiment_10q,
            sentiment_combined=sentiment_combined,
            filing_date_8k=filing_date_8k,
            filing_date_10q=filing_date_10q,
            words_8k=words_8k,
            words_10q=words_10q,
            text_length=base_result.text_length,
            data_quality_score=1.0,
            quality_flags=list(set(quality_flags))
        )

        # Adjust quality score based on flags
        if result.quality_flags:
            penalty = len(result.quality_flags) * 0.1
            result.data_quality_score = max(0.0, 1.0 - penalty)

        logger.info(
            f"{ticker}: Analysis complete - "
            f"combined={sentiment_combined:.3f}, "
            f"8K={sentiment_8k if sentiment_8k else 'N/A'}, "
            f"10Q={sentiment_10q if sentiment_10q else 'N/A'}, "
            f"words={words_8k + words_10q}, "
            f"quality={result.data_quality_score:.2f}"
        )

        return result

    def collect_all_tickers(
        self,
        tickers: List[str],
        lookback_days: int = 365,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Collect sentiment for all tickers.

        Args:
            tickers: List of ticker symbols
            lookback_days: How far back to search for filings
            show_progress: Whether to show progress logging

        Returns:
            DataFrame with sentiment scores and metadata
        """
        total = len(tickers)

        logger.info(f"\n{'='*60}")
        logger.info("SENTIMENT DATA COLLECTION")
        logger.info(f"{'='*60}")
        logger.info(f"Total tickers to process: {total}")
        logger.info(f"As of date: {self.as_of_date.date()}")
        logger.info(f"Lookback: {lookback_days} days")
        logger.info(f"Data retrieved: {self.data_retrieved_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"{'='*60}")

        self.results = []

        for i, ticker in enumerate(tickers, 1):
            if show_progress:
                pct = (i / total) * 100
                print(f"\rProgress: {i}/{total} ({pct:.1f}%) - Processing {ticker}...", end='', flush=True)

            result = self.collect_sentiment(ticker, lookback_days)
            if result:
                self.results.append(result)

        if show_progress:
            print()  # New line after progress

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
