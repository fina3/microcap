"""
Sentiment Analyzer for financial transcripts and filings.

Uses Loughran-McDonald dictionary for financial sentiment analysis.
All operations require as_of_date for temporal tracking.
"""

import re
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import pytz

import sys
sys.path.append('..')
from utils.sentiment_dictionary import SentimentDictionary
from utils.temporal import ensure_utc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """
    Result of sentiment analysis on a text.

    All scores are normalized (0-1 or -1 to 1 for net_sentiment).
    """
    ticker: str
    as_of_date: datetime
    filing_date: datetime

    # Core normalized scores
    positive_score: float  # 0-1
    negative_score: float  # 0-1
    net_sentiment: float   # -1 to 1
    uncertainty_score: float  # 0-1
    hedging_ratio: float   # 0-1
    confidence_ratio: float  # 0-1
    polarity: float  # Adjusted sentiment

    # Raw counts for transparency
    positive_count: int = 0
    negative_count: int = 0
    uncertainty_count: int = 0
    hedging_count: int = 0
    confidence_count: int = 0
    total_words: int = 0

    # Data quality
    text_length: int = 0
    data_quality_score: float = 1.0
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'ticker': self.ticker,
            'as_of_date': self.as_of_date,
            'filing_date': self.filing_date,
            'positive_score': self.positive_score,
            'negative_score': self.negative_score,
            'net_sentiment': self.net_sentiment,
            'uncertainty_score': self.uncertainty_score,
            'hedging_ratio': self.hedging_ratio,
            'confidence_ratio': self.confidence_ratio,
            'polarity': self.polarity,
            'positive_count': self.positive_count,
            'negative_count': self.negative_count,
            'uncertainty_count': self.uncertainty_count,
            'hedging_count': self.hedging_count,
            'confidence_count': self.confidence_count,
            'total_words': self.total_words,
            'text_length': self.text_length,
            'data_quality_score': self.data_quality_score,
            'quality_flags': ','.join(self.quality_flags) if self.quality_flags else ''
        }


class SentimentAnalyzer:
    """
    Analyzes text sentiment using Loughran-McDonald financial dictionary.

    Designed for earnings call transcripts and SEC filings.
    """

    def __init__(self, sentiment_dict: Optional[SentimentDictionary] = None):
        """
        Initialize analyzer.

        Args:
            sentiment_dict: Optional pre-initialized SentimentDictionary.
                           Creates new one if not provided.
        """
        self.dictionary = sentiment_dict or SentimentDictionary()
        logger.info("SentimentAnalyzer initialized")

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for sentiment analysis.

        Args:
            text: Raw text input

        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation but keep word boundaries
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize on whitespace
        tokens = text.split()

        # Filter: keep only alphabetic tokens of length >= 2
        tokens = [t for t in tokens if t.isalpha() and len(t) >= 2]

        return tokens

    def _calculate_quality_score(
        self,
        word_count: int,
        quality_flags: List[str]
    ) -> float:
        """
        Calculate data quality score (0-1).

        Args:
            word_count: Total words analyzed
            quality_flags: List of quality issues

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Penalize short text
        if word_count < 100:
            score -= 0.3
        elif word_count < 500:
            score -= 0.1

        # Penalize flags
        flag_penalties = {
            'PARSING_ERRORS': 0.2,
            'SHORT_TEXT': 0.15,
            'OLD_FILING': 0.1,
            'ITEM_202_NOT_FOUND': 0.25,
        }

        for flag in quality_flags:
            score -= flag_penalties.get(flag, 0.05)

        return max(0.0, score)

    def analyze_text(
        self,
        text: str,
        as_of_date: datetime,
        filing_date: datetime,
        ticker: str = "UNKNOWN"
    ) -> SentimentResult:
        """
        Analyze text and return sentiment scores.

        Args:
            text: Raw text to analyze
            as_of_date: Date when this data becomes available (timezone-aware UTC)
            filing_date: Date of the filing/transcript
            ticker: Stock ticker for logging

        Returns:
            SentimentResult with all metrics and metadata
        """
        as_of_date = ensure_utc(as_of_date)
        filing_date = ensure_utc(filing_date)

        quality_flags = []
        text_length = len(text)

        # Check for empty or very short text
        if not text or text_length < 50:
            logger.warning(f"{ticker}: Empty or very short text ({text_length} chars)")
            quality_flags.append('SHORT_TEXT')
            return SentimentResult(
                ticker=ticker,
                as_of_date=as_of_date,
                filing_date=filing_date,
                positive_score=0.0,
                negative_score=0.0,
                net_sentiment=0.0,
                uncertainty_score=0.0,
                hedging_ratio=0.0,
                confidence_ratio=0.0,
                polarity=0.0,
                text_length=text_length,
                data_quality_score=0.0,
                quality_flags=quality_flags
            )

        # Preprocess
        tokens = self._preprocess_text(text)
        total_words = len(tokens)

        if total_words < 50:
            quality_flags.append('SHORT_TEXT')

        # Count words by category
        positive_count = 0
        negative_count = 0
        uncertainty_count = 0
        hedging_count = 0
        confidence_count = 0

        for token in tokens:
            if self.dictionary.is_positive(token):
                positive_count += 1
            if self.dictionary.is_negative(token):
                negative_count += 1
            if self.dictionary.is_uncertainty(token):
                uncertainty_count += 1
            if self.dictionary.is_hedging(token):
                hedging_count += 1
            if self.dictionary.is_confidence(token):
                confidence_count += 1

        # Calculate normalized scores
        if total_words > 0:
            positive_score = positive_count / total_words
            negative_score = negative_count / total_words
            uncertainty_score = uncertainty_count / total_words
            hedging_ratio = hedging_count / total_words
            confidence_ratio = confidence_count / total_words
        else:
            positive_score = 0.0
            negative_score = 0.0
            uncertainty_score = 0.0
            hedging_ratio = 0.0
            confidence_ratio = 0.0

        # Calculate net sentiment (-1 to 1)
        sentiment_total = positive_count + negative_count
        if sentiment_total > 0:
            net_sentiment = (positive_count - negative_count) / sentiment_total
        else:
            net_sentiment = 0.0

        # Calculate polarity (adjusted for uncertainty)
        polarity = net_sentiment * (1 - uncertainty_score)

        # Calculate quality score
        data_quality_score = self._calculate_quality_score(total_words, quality_flags)

        logger.info(
            f"{ticker}: Analyzed {total_words} words - "
            f"net_sentiment={net_sentiment:.3f}, polarity={polarity:.3f}, "
            f"uncertainty={uncertainty_score:.3f}"
        )

        return SentimentResult(
            ticker=ticker,
            as_of_date=as_of_date,
            filing_date=filing_date,
            positive_score=positive_score,
            negative_score=negative_score,
            net_sentiment=net_sentiment,
            uncertainty_score=uncertainty_score,
            hedging_ratio=hedging_ratio,
            confidence_ratio=confidence_ratio,
            polarity=polarity,
            positive_count=positive_count,
            negative_count=negative_count,
            uncertainty_count=uncertainty_count,
            hedging_count=hedging_count,
            confidence_count=confidence_count,
            total_words=total_words,
            text_length=text_length,
            data_quality_score=data_quality_score,
            quality_flags=quality_flags
        )

    def get_word_breakdown(self, text: str) -> Dict[str, List[str]]:
        """
        Get categorized word lists for debugging/verification.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with word lists by category
        """
        tokens = self._preprocess_text(text)

        breakdown = {
            'positive': [],
            'negative': [],
            'uncertainty': [],
            'hedging': [],
            'confidence': [],
            'neutral': []
        }

        for token in tokens:
            category = self.dictionary.get_word_category(token)
            breakdown[category].append(token)

        return breakdown
