"""
Risk Management Module for position sizing and portfolio construction.

Provides position sizing based on confidence, portfolio limits enforcement,
and stop-loss/take-profit calculations.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Action(Enum):
    """Trading action."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class PositionRecommendation:
    """A recommended portfolio position."""
    ticker: str
    action: Action
    confidence: float
    current_price: float
    dollar_amount: float
    shares: int
    position_pct: float  # Percentage of portfolio
    stop_loss: float
    take_profit: float
    sector: Optional[str] = None
    liquidity_grade: str = "?"
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'action': self.action.value,
            'confidence': self.confidence,
            'current_price': self.current_price,
            'dollar_amount': self.dollar_amount,
            'shares': self.shares,
            'position_pct': self.position_pct,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'sector': self.sector,
            'liquidity_grade': self.liquidity_grade,
            'notes': self.notes
        }


@dataclass
class PortfolioLimitCheck:
    """Result of portfolio limit check."""
    allowed: bool
    reason: str
    warnings: List[str]


class RiskManager:
    """
    Risk management for micro-cap portfolio construction.

    Handles position sizing, portfolio limits, and risk calculations.
    """

    # Position sizing tiers based on confidence
    CONFIDENCE_TIERS = {
        'high': {'min': 0.90, 'position_pct': 0.20},    # 90%+ -> 20% position
        'medium': {'min': 0.70, 'position_pct': 0.10},  # 70-90% -> 10% position
        'low': {'min': 0.0, 'position_pct': 0.05},      # <70% -> 5% position
    }

    # Portfolio limits
    MAX_POSITIONS = 10
    MAX_SINGLE_POSITION_PCT = 0.20  # 20% max in any single stock
    MAX_SECTOR_PCT = 0.30           # 30% max in any sector

    # Risk parameters
    DEFAULT_STOP_LOSS_PCT = 0.15    # 15% stop loss
    DEFAULT_TAKE_PROFIT_PCT = 0.30  # 30% take profit

    def __init__(
        self,
        max_positions: int = MAX_POSITIONS,
        max_single_position_pct: float = MAX_SINGLE_POSITION_PCT,
        max_sector_pct: float = MAX_SECTOR_PCT,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT
    ):
        """
        Initialize risk manager.

        Args:
            max_positions: Maximum number of positions in portfolio
            max_single_position_pct: Maximum percentage in any single stock
            max_sector_pct: Maximum percentage in any sector
            stop_loss_pct: Default stop loss percentage (below entry)
            take_profit_pct: Default take profit percentage (above entry)
        """
        self.max_positions = max_positions
        self.max_single_position_pct = max_single_position_pct
        self.max_sector_pct = max_sector_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        logger.info(
            f"RiskManager initialized - "
            f"max_positions: {max_positions}, "
            f"max_single: {max_single_position_pct:.0%}, "
            f"max_sector: {max_sector_pct:.0%}, "
            f"stop_loss: {stop_loss_pct:.0%}, "
            f"take_profit: {take_profit_pct:.0%}"
        )

    def calculate_position_size(
        self,
        confidence: float,
        portfolio_value: float,
        max_position_pct: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate position size based on confidence level.

        Higher confidence = larger position:
        - 90%+ confidence -> 20% of portfolio
        - 70-90% confidence -> 10% of portfolio
        - <70% confidence -> 5% of portfolio

        Args:
            confidence: Model confidence (0-1)
            portfolio_value: Total portfolio value in dollars
            max_position_pct: Override maximum position percentage

        Returns:
            Tuple of (dollar_amount, position_percentage)
        """
        max_pct = max_position_pct or self.max_single_position_pct

        # Determine position size based on confidence tier
        if confidence >= self.CONFIDENCE_TIERS['high']['min']:
            position_pct = self.CONFIDENCE_TIERS['high']['position_pct']
        elif confidence >= self.CONFIDENCE_TIERS['medium']['min']:
            position_pct = self.CONFIDENCE_TIERS['medium']['position_pct']
        else:
            position_pct = self.CONFIDENCE_TIERS['low']['position_pct']

        # Cap at maximum allowed position
        position_pct = min(position_pct, max_pct)

        dollar_amount = portfolio_value * position_pct

        logger.debug(
            f"Position size: confidence={confidence:.1%} -> "
            f"{position_pct:.0%} = ${dollar_amount:,.2f}"
        )

        return dollar_amount, position_pct

    def calculate_shares(
        self,
        dollar_amount: float,
        price: float,
        round_down: bool = True
    ) -> int:
        """
        Calculate number of shares for a given dollar amount.

        Args:
            dollar_amount: Dollar amount to invest
            price: Current share price
            round_down: If True, round down to whole shares

        Returns:
            Number of shares (whole number)
        """
        if price <= 0:
            return 0

        shares = dollar_amount / price

        if round_down:
            return int(shares)
        else:
            return round(shares)

    def calculate_stop_loss(
        self,
        entry_price: float,
        risk_tolerance: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price per share
            risk_tolerance: Override stop loss percentage (default: 15%)

        Returns:
            Stop loss price
        """
        pct = risk_tolerance or self.stop_loss_pct
        stop_loss = entry_price * (1 - pct)
        return round(stop_loss, 2)

    def calculate_take_profit(
        self,
        entry_price: float,
        target_gain: Optional[float] = None
    ) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price per share
            target_gain: Override take profit percentage (default: 30%)

        Returns:
            Take profit price
        """
        pct = target_gain or self.take_profit_pct
        take_profit = entry_price * (1 + pct)
        return round(take_profit, 2)

    def check_portfolio_limits(
        self,
        current_positions: List[Dict],
        new_position: Dict,
        portfolio_value: float
    ) -> PortfolioLimitCheck:
        """
        Check if a new position would violate portfolio limits.

        Limits:
        - Max 10 positions
        - Max 30% in any sector
        - Max 20% in any single stock

        Args:
            current_positions: List of current positions with keys:
                - ticker, dollar_amount, sector (optional)
            new_position: New position to check with keys:
                - ticker, dollar_amount, sector (optional)
            portfolio_value: Total portfolio value

        Returns:
            PortfolioLimitCheck with allowed flag and reason
        """
        warnings = []

        # Check max positions
        current_count = len(current_positions)
        if current_count >= self.max_positions:
            return PortfolioLimitCheck(
                allowed=False,
                reason=f"Maximum positions reached ({self.max_positions})",
                warnings=warnings
            )

        # Check if ticker already in portfolio
        current_tickers = {p.get('ticker') for p in current_positions}
        if new_position.get('ticker') in current_tickers:
            warnings.append(f"Already holding {new_position.get('ticker')}")

        # Check single position limit
        new_amount = new_position.get('dollar_amount', 0)
        new_pct = new_amount / portfolio_value if portfolio_value > 0 else 0

        if new_pct > self.max_single_position_pct:
            return PortfolioLimitCheck(
                allowed=False,
                reason=f"Position too large ({new_pct:.0%} > {self.max_single_position_pct:.0%} max)",
                warnings=warnings
            )

        # Check sector concentration
        new_sector = new_position.get('sector')
        if new_sector:
            sector_total = new_amount
            for pos in current_positions:
                if pos.get('sector') == new_sector:
                    sector_total += pos.get('dollar_amount', 0)

            sector_pct = sector_total / portfolio_value if portfolio_value > 0 else 0

            if sector_pct > self.max_sector_pct:
                return PortfolioLimitCheck(
                    allowed=False,
                    reason=f"Sector concentration too high ({new_sector}: {sector_pct:.0%} > {self.max_sector_pct:.0%} max)",
                    warnings=warnings
                )

            if sector_pct > self.max_sector_pct * 0.8:  # Warning at 80% of limit
                warnings.append(f"Sector {new_sector} approaching limit ({sector_pct:.0%})")

        # Check remaining allocation
        current_allocated = sum(p.get('dollar_amount', 0) for p in current_positions)
        total_allocated = current_allocated + new_amount

        if total_allocated > portfolio_value:
            return PortfolioLimitCheck(
                allowed=False,
                reason=f"Insufficient capital (need ${new_amount:,.2f}, have ${portfolio_value - current_allocated:,.2f} remaining)",
                warnings=warnings
            )

        return PortfolioLimitCheck(
            allowed=True,
            reason="Position allowed",
            warnings=warnings
        )

    def build_portfolio(
        self,
        predictions: List[Dict],
        portfolio_value: float,
        min_confidence: float = 0.50,
        liquidity_data: Optional[Dict[str, Dict]] = None
    ) -> List[PositionRecommendation]:
        """
        Build a portfolio from predictions.

        Args:
            predictions: List of predictions with keys:
                - ticker, prediction, confidence, current_price
                - sector (optional)
            portfolio_value: Total capital available
            min_confidence: Minimum confidence to include
            liquidity_data: Optional dict mapping ticker -> {'liquidity_grade': 'A', 'spread_pct': 1.5}

        Returns:
            List of PositionRecommendation objects
        """
        logger.info(f"Building portfolio from {len(predictions)} predictions")
        logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
        logger.info(f"Min confidence: {min_confidence:.0%}")

        liquidity_data = liquidity_data or {}

        # Filter by confidence and sort by confidence (highest first)
        filtered = [
            p for p in predictions
            if p.get('confidence', 0) >= min_confidence
            and p.get('prediction') == 1  # Only BUY recommendations
        ]
        sorted_predictions = sorted(
            filtered,
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )

        logger.info(f"Filtered to {len(sorted_predictions)} qualifying predictions")

        recommendations = []
        current_positions = []
        excluded_illiquid = 0
        reduced_positions = 0

        for pred in sorted_predictions:
            ticker = pred.get('ticker')
            confidence = pred.get('confidence', 0)
            price = pred.get('current_price', 0)
            sector = pred.get('sector')

            if price <= 0:
                logger.warning(f"{ticker}: Invalid price ({price}), skipping")
                continue

            # Check liquidity grade
            liq_info = liquidity_data.get(ticker, {})
            liquidity_grade = liq_info.get('liquidity_grade', '?')

            # Exclude F-grade stocks (too illiquid to trade)
            if liquidity_grade == 'F':
                logger.info(f"{ticker}: Skipped - liquidity grade F (DO NOT TRADE)")
                excluded_illiquid += 1
                continue

            # Calculate position size
            dollar_amount, position_pct = self.calculate_position_size(
                confidence=confidence,
                portfolio_value=portfolio_value
            )

            # Reduce position size by 50% for D-grade stocks
            if liquidity_grade == 'D':
                dollar_amount *= 0.5
                position_pct *= 0.5
                reduced_positions += 1
                logger.info(f"{ticker}: Position reduced 50% due to poor liquidity (grade D)")

            # Check portfolio limits
            new_position = {
                'ticker': ticker,
                'dollar_amount': dollar_amount,
                'sector': sector
            }

            limit_check = self.check_portfolio_limits(
                current_positions=current_positions,
                new_position=new_position,
                portfolio_value=portfolio_value
            )

            if not limit_check.allowed:
                logger.info(f"{ticker}: Skipped - {limit_check.reason}")
                continue

            # Calculate shares and actual amount
            shares = self.calculate_shares(dollar_amount, price)
            actual_amount = shares * price

            if shares == 0:
                logger.info(f"{ticker}: Skipped - price too high for position size")
                continue

            # Calculate stop loss and take profit
            stop_loss = self.calculate_stop_loss(price)
            take_profit = self.calculate_take_profit(price)

            # Build notes
            notes_list = limit_check.warnings if limit_check.warnings else []
            if liquidity_grade == 'D':
                notes_list.append("Position reduced 50% (poor liquidity)")

            # Create recommendation
            rec = PositionRecommendation(
                ticker=ticker,
                action=Action.BUY,
                confidence=confidence,
                current_price=price,
                dollar_amount=actual_amount,
                shares=shares,
                position_pct=actual_amount / portfolio_value,
                stop_loss=stop_loss,
                take_profit=take_profit,
                sector=sector,
                liquidity_grade=liquidity_grade,
                notes='; '.join(notes_list) if notes_list else ''
            )

            recommendations.append(rec)
            current_positions.append(new_position)

            logger.info(
                f"{ticker}: {shares} shares @ ${price:.2f} = ${actual_amount:,.2f} "
                f"({rec.position_pct:.0%} of portfolio) [Liq: {liquidity_grade}]"
            )

        # Summary
        total_allocated = sum(r.dollar_amount for r in recommendations)
        cash_remaining = portfolio_value - total_allocated

        logger.info(f"\nPortfolio built: {len(recommendations)} positions")
        logger.info(f"Total allocated: ${total_allocated:,.2f} ({total_allocated/portfolio_value:.0%})")
        logger.info(f"Cash remaining: ${cash_remaining:,.2f} ({cash_remaining/portfolio_value:.0%})")
        if excluded_illiquid > 0:
            logger.info(f"Excluded for illiquidity (grade F): {excluded_illiquid}")
        if reduced_positions > 0:
            logger.info(f"Positions reduced 50% (grade D): {reduced_positions}")

        return recommendations

    def print_portfolio_summary(
        self,
        recommendations: List[PositionRecommendation],
        portfolio_value: float
    ):
        """Print formatted portfolio summary."""
        print("\n" + "=" * 80)
        print(f"PORTFOLIO RECOMMENDATIONS (${portfolio_value:,.0f} capital)")
        print("=" * 80)

        if not recommendations:
            print("\nNo recommendations generated.")
            return

        # Header
        print(f"\n{'Ticker':<8} {'Action':<6} {'Conf':>7} {'Price':>9} {'Shares':>7} "
              f"{'Amount':>10} {'Stop':>9} {'Target':>9} {'%Port':>6}")
        print("-" * 80)

        # Positions
        for rec in recommendations:
            print(
                f"{rec.ticker:<8} {rec.action.value:<6} {rec.confidence:>6.1%} "
                f"${rec.current_price:>7.2f} {rec.shares:>7} "
                f"${rec.dollar_amount:>8,.0f} ${rec.stop_loss:>7.2f} "
                f"${rec.take_profit:>7.2f} {rec.position_pct:>5.0%}"
            )

        # Summary
        total_allocated = sum(r.dollar_amount for r in recommendations)
        cash_remaining = portfolio_value - total_allocated
        avg_confidence = sum(r.confidence for r in recommendations) / len(recommendations)

        print("-" * 80)
        print(f"{'TOTAL':<8} {'':<6} {avg_confidence:>6.1%} {'':<9} {'':<7} "
              f"${total_allocated:>8,.0f} {'':<9} {'':<9} {total_allocated/portfolio_value:>5.0%}")
        print(f"{'CASH':<8} {'':<6} {'':<7} {'':<9} {'':<7} "
              f"${cash_remaining:>8,.0f} {'':<9} {'':<9} {cash_remaining/portfolio_value:>5.0%}")

        print("\n" + "=" * 80)

        # Risk summary
        print("\nRISK PARAMETERS:")
        print(f"  Stop Loss: {self.stop_loss_pct:.0%} below entry")
        print(f"  Take Profit: {self.take_profit_pct:.0%} above entry")
        print(f"  Max Position: {self.max_single_position_pct:.0%}")
        print(f"  Max Sector: {self.max_sector_pct:.0%}")
        print(f"  Max Positions: {self.max_positions}")

        # Max loss calculation
        max_loss = sum(
            (rec.current_price - rec.stop_loss) * rec.shares
            for rec in recommendations
        )
        print(f"\nMax Portfolio Loss (if all hit stop): ${max_loss:,.2f} ({max_loss/portfolio_value:.1%})")

        print("=" * 80)
