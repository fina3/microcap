"""
Options Liquidity Screener for Micro-Cap Stocks.

Checks if high-conviction picks have tradeable options chains.
Most micro-caps won't have options - this flags the ones that do.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptionsCandidate:
    """Result of options screening for a single ticker."""
    ticker: str
    has_options: bool
    options_available: bool  # Passes liquidity filters
    current_price: Optional[float]
    nearest_strike: Optional[float]
    expiry: Optional[str]
    call_bid: Optional[float]
    call_ask: Optional[float]
    call_premium: Optional[float]  # Mid price
    open_interest: Optional[int]
    volume: Optional[int]
    implied_vol: Optional[float]
    bid_ask_spread_pct: Optional[float]
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'has_options': self.has_options,
            'options_available': self.options_available,
            'current_price': self.current_price,
            'nearest_strike': self.nearest_strike,
            'expiry': self.expiry,
            'call_bid': self.call_bid,
            'call_ask': self.call_ask,
            'call_premium': self.call_premium,
            'open_interest': self.open_interest,
            'volume': self.volume,
            'implied_vol': self.implied_vol,
            'bid_ask_spread_pct': self.bid_ask_spread_pct,
            'notes': self.notes
        }


# Liquidity thresholds
MIN_OPEN_INTEREST = 100
MAX_BID_ASK_SPREAD_PCT = 20.0  # 20% max spread
MIN_DAYS_TO_EXPIRY = 30
MAX_DAYS_TO_EXPIRY = 60
ATM_STRIKE_TOLERANCE = 0.05  # Within 5% of current price


def check_options_liquidity(ticker: str) -> OptionsCandidate:
    """
    Check if a ticker has liquid options available.

    Filters:
    - Open interest > 100
    - Bid-ask spread < 20%
    - Expiry 30-60 days out
    - Strike within 5% of current price (ATM)

    Args:
        ticker: Stock ticker symbol

    Returns:
        OptionsCandidate with availability and details
    """
    if not HAS_YFINANCE:
        return OptionsCandidate(
            ticker=ticker,
            has_options=False,
            options_available=False,
            current_price=None,
            nearest_strike=None,
            expiry=None,
            call_bid=None,
            call_ask=None,
            call_premium=None,
            open_interest=None,
            volume=None,
            implied_vol=None,
            bid_ask_spread_pct=None,
            notes="yfinance not available"
        )

    try:
        stock = yf.Ticker(ticker)

        # Get current price
        hist = stock.history(period="1d")
        if hist.empty:
            return _no_options_result(ticker, "Could not fetch current price")

        current_price = hist['Close'].iloc[-1]

        # Get available expiration dates
        try:
            expiry_dates = stock.options
        except Exception:
            return _no_options_result(ticker, "No options chain available", current_price)

        if not expiry_dates or len(expiry_dates) == 0:
            return _no_options_result(ticker, "No options expiries", current_price)

        # Find expiry in 30-60 day window
        today = datetime.now().date()
        target_expiry = None

        for exp_str in expiry_dates:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                days_to_exp = (exp_date - today).days

                if MIN_DAYS_TO_EXPIRY <= days_to_exp <= MAX_DAYS_TO_EXPIRY:
                    target_expiry = exp_str
                    break
            except ValueError:
                continue

        if not target_expiry:
            # Try to find closest expiry if none in window
            for exp_str in expiry_dates:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    days_to_exp = (exp_date - today).days
                    if days_to_exp > 14:  # At least 2 weeks out
                        target_expiry = exp_str
                        break
                except ValueError:
                    continue

        if not target_expiry:
            return _no_options_result(ticker, "No suitable expiry dates", current_price)

        # Get option chain for target expiry
        try:
            chain = stock.option_chain(target_expiry)
            calls = chain.calls
        except Exception as e:
            return _no_options_result(ticker, f"Could not fetch chain: {e}", current_price)

        if calls.empty:
            return _no_options_result(ticker, "No call options", current_price)

        # Find ATM strike (within 5% of current price)
        atm_min = current_price * (1 - ATM_STRIKE_TOLERANCE)
        atm_max = current_price * (1 + ATM_STRIKE_TOLERANCE)

        atm_calls = calls[(calls['strike'] >= atm_min) & (calls['strike'] <= atm_max)]

        if atm_calls.empty:
            # Find nearest strike
            calls['strike_diff'] = abs(calls['strike'] - current_price)
            nearest_idx = calls['strike_diff'].idxmin()
            atm_calls = calls.loc[[nearest_idx]]

        # Check liquidity filters
        best_option = None
        for _, opt in atm_calls.iterrows():
            oi = opt.get('openInterest', 0)
            oi = int(oi) if pd.notna(oi) else 0
            bid = opt.get('bid', 0)
            bid = float(bid) if pd.notna(bid) else 0
            ask = opt.get('ask', 0)
            ask = float(ask) if pd.notna(ask) else 0

            # Calculate bid-ask spread
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread_pct = ((ask - bid) / mid) * 100
            else:
                spread_pct = 100  # No valid quotes

            # Check filters
            if oi >= MIN_OPEN_INTEREST and spread_pct <= MAX_BID_ASK_SPREAD_PCT:
                best_option = opt
                break
            elif best_option is None:
                best_option = opt  # Keep first as fallback

        if best_option is None:
            return _no_options_result(ticker, "No options pass liquidity filters", current_price)

        # Extract details
        strike = best_option['strike']
        bid = best_option.get('bid', 0)
        bid = float(bid) if pd.notna(bid) else 0
        ask = best_option.get('ask', 0)
        ask = float(ask) if pd.notna(ask) else 0
        oi = best_option.get('openInterest', 0)
        oi = int(oi) if pd.notna(oi) else 0
        vol = best_option.get('volume', 0)
        vol = int(vol) if pd.notna(vol) else 0
        iv = best_option.get('impliedVolatility', None)
        iv = float(iv) if pd.notna(iv) else None

        if bid > 0 and ask > 0:
            premium = (bid + ask) / 2
            spread_pct = ((ask - bid) / premium) * 100
        else:
            premium = ask if ask > 0 else None
            spread_pct = None

        # Determine if it passes all filters
        passes_filters = (
            oi >= MIN_OPEN_INTEREST and
            spread_pct is not None and
            spread_pct <= MAX_BID_ASK_SPREAD_PCT
        )

        notes = "LIQUID" if passes_filters else f"OI={oi}, Spread={spread_pct:.1f}%" if spread_pct else "Low liquidity"

        return OptionsCandidate(
            ticker=ticker,
            has_options=True,
            options_available=passes_filters,
            current_price=round(current_price, 2),
            nearest_strike=strike,
            expiry=target_expiry,
            call_bid=round(bid, 2) if bid else None,
            call_ask=round(ask, 2) if ask else None,
            call_premium=round(premium, 2) if premium else None,
            open_interest=oi,
            volume=vol,
            implied_vol=round(iv * 100, 1) if iv else None,
            bid_ask_spread_pct=round(spread_pct, 1) if spread_pct else None,
            notes=notes
        )

    except Exception as e:
        logger.error(f"{ticker}: Error checking options - {e}")
        return _no_options_result(ticker, f"Error: {e}")


def _no_options_result(ticker: str, notes: str, current_price: float = None) -> OptionsCandidate:
    """Create a result for tickers without options."""
    return OptionsCandidate(
        ticker=ticker,
        has_options=False,
        options_available=False,
        current_price=current_price,
        nearest_strike=None,
        expiry=None,
        call_bid=None,
        call_ask=None,
        call_premium=None,
        open_interest=None,
        volume=None,
        implied_vol=None,
        bid_ask_spread_pct=None,
        notes=notes
    )


def screen_options_batch(tickers: List[str]) -> List[OptionsCandidate]:
    """
    Screen multiple tickers for options availability.

    Args:
        tickers: List of ticker symbols

    Returns:
        List of OptionsCandidate objects
    """
    results = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        result = check_options_liquidity(ticker)
        results.append(result)

        status = "OPTIONS" if result.options_available else "no options"
        logger.info(f"[{i}/{total}] {ticker}: {status}")

    return results


def get_options_summary(results: List[OptionsCandidate]) -> Dict:
    """Generate summary of options screening results."""
    total = len(results)
    has_options = sum(1 for r in results if r.has_options)
    liquid_options = sum(1 for r in results if r.options_available)

    candidates = [r for r in results if r.options_available]

    return {
        'total_screened': total,
        'has_options': has_options,
        'liquid_options': liquid_options,
        'candidates': [r.ticker for r in candidates],
        'pct_with_options': (has_options / total * 100) if total > 0 else 0,
        'pct_liquid': (liquid_options / total * 100) if total > 0 else 0
    }
