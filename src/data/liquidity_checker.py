"""
Bid-Ask Spread Analyzer for Micro-Cap Stocks.

Checks liquidity by analyzing bid-ask spreads to avoid
stocks that are too expensive to trade.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime

import yfinance as yf
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LiquidityResult:
    """Result of liquidity check for a single ticker."""
    ticker: str
    bid: Optional[float]
    ask: Optional[float]
    spread_pct: Optional[float]
    avg_volume: Optional[int]
    liquidity_grade: str
    is_tradeable: bool
    as_of_date: datetime

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'bid': self.bid,
            'ask': self.ask,
            'spread_pct': self.spread_pct,
            'avg_volume': self.avg_volume,
            'liquidity_grade': self.liquidity_grade,
            'is_tradeable': self.is_tradeable,
            'as_of_date': self.as_of_date.strftime('%Y-%m-%d %H:%M:%S')
        }


# Liquidity grade thresholds
LIQUIDITY_GRADES = {
    'A': {'max_spread': 1.0, 'description': 'Excellent - trade freely'},
    'B': {'max_spread': 2.0, 'description': 'Good - acceptable'},
    'C': {'max_spread': 4.0, 'description': 'Caution - factor into sizing'},
    'D': {'max_spread': 7.0, 'description': 'Poor - small positions only'},
    'F': {'max_spread': float('inf'), 'description': 'DO NOT TRADE - too expensive'},
}


def get_liquidity_grade(spread_pct: Optional[float]) -> str:
    """
    Assign liquidity grade based on bid-ask spread.

    Args:
        spread_pct: Spread as percentage of midpoint

    Returns:
        Grade letter (A, B, C, D, or F)
    """
    if spread_pct is None:
        return 'F'  # No data = illiquid

    if spread_pct < 1.0:
        return 'A'
    elif spread_pct < 2.0:
        return 'B'
    elif spread_pct < 4.0:
        return 'C'
    elif spread_pct < 7.0:
        return 'D'
    else:
        return 'F'


def get_bid_ask_spread(ticker: str) -> LiquidityResult:
    """
    Get bid-ask spread for a ticker.

    Formula: spread_pct = (ask - bid) / midpoint * 100

    Args:
        ticker: Stock ticker symbol

    Returns:
        LiquidityResult with spread data and grade
    """
    now = datetime.now(pytz.utc)

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        bid = info.get('bid')
        ask = info.get('ask')
        avg_volume = info.get('averageVolume') or info.get('averageDailyVolume10Day')

        # Handle missing or zero bid/ask
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            logger.warning(f"{ticker}: Missing bid/ask data - marking as ILLIQUID")
            return LiquidityResult(
                ticker=ticker,
                bid=bid,
                ask=ask,
                spread_pct=None,
                avg_volume=avg_volume,
                liquidity_grade='F',
                is_tradeable=False,
                as_of_date=now
            )

        # Calculate spread
        midpoint = (bid + ask) / 2
        spread_pct = ((ask - bid) / midpoint) * 100

        # Sanity check - spread shouldn't be negative
        if spread_pct < 0:
            logger.warning(f"{ticker}: Negative spread ({spread_pct:.2f}%) - data error")
            spread_pct = abs(spread_pct)

        # Get grade
        grade = get_liquidity_grade(spread_pct)
        is_tradeable = grade != 'F'

        return LiquidityResult(
            ticker=ticker,
            bid=bid,
            ask=ask,
            spread_pct=round(spread_pct, 2),
            avg_volume=avg_volume,
            liquidity_grade=grade,
            is_tradeable=is_tradeable,
            as_of_date=now
        )

    except Exception as e:
        logger.error(f"{ticker}: Error fetching liquidity data - {e}")
        return LiquidityResult(
            ticker=ticker,
            bid=None,
            ask=None,
            spread_pct=None,
            avg_volume=None,
            liquidity_grade='F',
            is_tradeable=False,
            as_of_date=now
        )


def check_liquidity_batch(
    tickers: List[str],
    progress_callback=None
) -> List[LiquidityResult]:
    """
    Check liquidity for multiple tickers.

    Args:
        tickers: List of ticker symbols
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of LiquidityResult objects
    """
    results = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        result = get_bid_ask_spread(ticker)
        results.append(result)

        if progress_callback:
            progress_callback(i, total)

        # Log progress every 25 tickers
        if i % 25 == 0:
            logger.info(f"Liquidity check progress: {i}/{total}")

    return results


def get_liquidity_summary(results: List[LiquidityResult]) -> Dict:
    """
    Generate summary statistics for liquidity results.

    Args:
        results: List of LiquidityResult objects

    Returns:
        Summary dictionary with grade counts and stats
    """
    grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    spreads = []
    tradeable_count = 0

    for r in results:
        grade_counts[r.liquidity_grade] += 1
        if r.spread_pct is not None:
            spreads.append(r.spread_pct)
        if r.is_tradeable:
            tradeable_count += 1

    avg_spread = sum(spreads) / len(spreads) if spreads else None
    median_spread = sorted(spreads)[len(spreads) // 2] if spreads else None

    return {
        'total': len(results),
        'tradeable': tradeable_count,
        'untradeable': len(results) - tradeable_count,
        'grade_counts': grade_counts,
        'avg_spread_pct': round(avg_spread, 2) if avg_spread else None,
        'median_spread_pct': round(median_spread, 2) if median_spread else None,
        'poor_liquidity_tickers': [r.ticker for r in results if r.liquidity_grade in ('D', 'F')]
    }
