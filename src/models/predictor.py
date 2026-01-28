"""
Micro-cap stock prediction model with walk-forward validation.

Predicts whether stocks will outperform the micro-cap index (IWC)
over a specified prediction window using financial metrics and sentiment data.
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('..')
from utils.temporal import ensure_utc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a stock prediction."""
    ticker: str
    as_of_date: datetime
    prediction: int  # 1 = outperform, 0 = underperform
    confidence: float  # 0-1 probability
    predicted_direction: str  # DEPRECATED - use rank instead
    actual_return_52w: float  # Historical return for comparison
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    rank: int = 0  # Strict ordering 1 to N (1 = best)
    momentum_score: float = 0.0  # For tiebreaking
    sentiment_score: float = 0.0  # For tiebreaking
    market_cap: float = 0.0  # For tiebreaking (smaller wins)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'rank': self.rank,
            'as_of_date': self.as_of_date,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'predicted_direction': self.predicted_direction,
            'actual_return_52w': self.actual_return_52w,
            'momentum_score': self.momentum_score,
            'sentiment_score': self.sentiment_score,
            'market_cap': self.market_cap,
            'feature_contributions': str(self.feature_contributions)
        }


class MicroCapPredictor:
    """
    Prediction model for micro-cap stock outperformance.

    Uses walk-forward validation to prevent lookahead bias.
    """

    # Feature columns from metrics
    METRIC_FEATURES = [
        'pe_ratio', 'pb_ratio', 'price_to_sales', 'debt_to_equity',
        'short_interest', 'insider_ownership', 'institutional_ownership',
        'momentum', 'earnings_surprise', 'accruals_ratio',
        'revenue_growth', 'gross_margin', 'operating_margin',
        'insider_buy_ratio'
    ]

    # Feature columns from sentiment
    SENTIMENT_FEATURES = [
        'sentiment', 'polarity', 'uncertainty', 'sentiment_change', 'sentiment_momentum'
    ]

    # Sector-relative features (stock value / sector median)
    # Values < 1.0 = cheaper than sector average (for valuation metrics)
    SECTOR_RELATIVE_FEATURES = [
        'pe_vs_sector', 'pb_vs_sector', 'ps_vs_sector',
        'growth_vs_sector', 'margin_vs_sector'
    ]

    # Derived features
    DERIVED_FEATURES = ['value_score', 'quality_score', 'sector_value_score']

    def __init__(
        self,
        model_type: str = "random_forest",
        prediction_window: int = 90,
        benchmark_return: float = 0.10  # Default IWC annual return assumption
    ):
        """
        Initialize predictor.

        Args:
            model_type: "random_forest" or "logistic_regression"
            prediction_window: Days to predict forward
            benchmark_return: Expected benchmark return for comparison
        """
        self.model_type = model_type
        self.prediction_window = prediction_window
        self.benchmark_return = benchmark_return

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False

        logger.info(
            f"MicroCapPredictor initialized - model: {model_type}, "
            f"prediction_window: {prediction_window} days"
        )

    def load_and_merge_data(
        self,
        metrics_file: str,
        sentiment_file: str
    ) -> pd.DataFrame:
        """
        Load and merge metrics and sentiment data.

        Args:
            metrics_file: Path to metrics CSV
            sentiment_file: Path to sentiment CSV

        Returns:
            Merged DataFrame
        """
        logger.info(f"Loading metrics from {metrics_file}")
        metrics_df = pd.read_csv(metrics_file)

        logger.info(f"Loading sentiment from {sentiment_file}")
        sentiment_df = pd.read_csv(sentiment_file)

        logger.info(f"Metrics: {len(metrics_df)} rows, Sentiment: {len(sentiment_df)} rows")

        # Select relevant columns from sentiment
        sentiment_cols = [
            'ticker', 'as_of_date', 'filing_date', 'net_sentiment',
            'polarity', 'uncertainty_score', 'data_quality_score', 'quality_flags'
        ]
        sentiment_subset = sentiment_df[[c for c in sentiment_cols if c in sentiment_df.columns]]

        # Merge on ticker
        merged = metrics_df.merge(
            sentiment_subset,
            on='ticker',
            how='left',
            suffixes=('_metrics', '_sentiment')
        )

        logger.info(f"Merged data: {len(merged)} rows")

        # Use metrics as_of_date as primary
        if 'as_of_date_metrics' in merged.columns:
            merged['as_of_date'] = merged['as_of_date_metrics']

        return merged

    def prepare_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare feature matrix and target variable.

        Args:
            df: Merged DataFrame

        Returns:
            Tuple of (X features, y target, feature_names)
        """
        features = pd.DataFrame(index=df.index)

        # Extract metrics features
        # PE ratio: use trailing, fallback to forward
        features['pe_ratio'] = df['pe_trailing'].fillna(df['pe_forward'])

        # Direct mappings
        features['pb_ratio'] = df['pb_ratio']
        features['price_to_sales'] = df['price_to_sales']
        features['debt_to_equity'] = df['debt_to_equity']
        features['short_interest'] = df['short_percent_float']
        features['insider_ownership'] = df['insider_ownership_pct']
        features['institutional_ownership'] = df['institutional_ownership_pct']
        features['momentum'] = df['52_week_price_change_pct']

        # Earnings surprise (handle NaN - will be imputed below)
        if 'earnings_surprise' in df.columns:
            features['earnings_surprise'] = df['earnings_surprise']
        else:
            features['earnings_surprise'] = np.nan
            logger.warning("earnings_surprise column not found, using NaN")

        # Accruals ratio (lower is better - cash earnings > paper earnings)
        if 'accruals_ratio' in df.columns:
            features['accruals_ratio'] = df['accruals_ratio']
        else:
            features['accruals_ratio'] = np.nan
            logger.warning("accruals_ratio column not found, using NaN")

        # Revenue growth (YoY)
        if 'revenue_growth' in df.columns:
            features['revenue_growth'] = df['revenue_growth']
        else:
            features['revenue_growth'] = np.nan
            logger.warning("revenue_growth column not found, using NaN")

        # Gross margin (gross_profit / revenue)
        if 'gross_margin' in df.columns:
            features['gross_margin'] = df['gross_margin']
        else:
            features['gross_margin'] = np.nan
            logger.warning("gross_margin column not found, using NaN")

        # Operating margin (operating_income / revenue)
        if 'operating_margin' in df.columns:
            features['operating_margin'] = df['operating_margin']
        else:
            features['operating_margin'] = np.nan
            logger.warning("operating_margin column not found, using NaN")

        # Insider buy ratio (from Form 4 filings)
        if 'insider_buy_ratio' in df.columns:
            features['insider_buy_ratio'] = df['insider_buy_ratio']
        else:
            features['insider_buy_ratio'] = np.nan
            logger.warning("insider_buy_ratio column not found, using NaN")

        # Sentiment features
        features['sentiment'] = df['net_sentiment'].fillna(0)
        features['polarity'] = df['polarity'].fillna(0)
        features['uncertainty'] = df['uncertainty_score'].fillna(0)

        # Sentiment change features (more predictive than absolute levels)
        if 'sentiment_change' in df.columns:
            features['sentiment_change'] = df['sentiment_change'].fillna(0)
        else:
            features['sentiment_change'] = 0
            logger.warning("sentiment_change column not found, using 0")

        if 'sentiment_momentum' in df.columns:
            features['sentiment_momentum'] = df['sentiment_momentum'].fillna(0)
        else:
            features['sentiment_momentum'] = 0
            logger.warning("sentiment_momentum column not found, using 0")

        # Sector-relative features (stock value / sector median)
        # Values < 1.0 = cheaper than sector average
        if 'pe_trailing_vs_sector' in df.columns:
            features['pe_vs_sector'] = df['pe_trailing_vs_sector'].fillna(1.0)
        else:
            features['pe_vs_sector'] = 1.0
            logger.warning("pe_trailing_vs_sector not found, using 1.0")

        if 'pb_ratio_vs_sector' in df.columns:
            features['pb_vs_sector'] = df['pb_ratio_vs_sector'].fillna(1.0)
        else:
            features['pb_vs_sector'] = 1.0
            logger.warning("pb_ratio_vs_sector not found, using 1.0")

        if 'price_to_sales_vs_sector' in df.columns:
            features['ps_vs_sector'] = df['price_to_sales_vs_sector'].fillna(1.0)
        else:
            features['ps_vs_sector'] = 1.0
            logger.warning("price_to_sales_vs_sector not found, using 1.0")

        if 'revenue_growth_vs_sector' in df.columns:
            features['growth_vs_sector'] = df['revenue_growth_vs_sector'].fillna(1.0)
        else:
            features['growth_vs_sector'] = 1.0
            logger.warning("revenue_growth_vs_sector not found, using 1.0")

        if 'operating_margin_vs_sector' in df.columns:
            features['margin_vs_sector'] = df['operating_margin_vs_sector'].fillna(1.0)
        else:
            features['margin_vs_sector'] = 1.0
            logger.warning("operating_margin_vs_sector not found, using 1.0")

        # Derived features
        # Value score: higher = more undervalued (absolute)
        pe_safe = features['pe_ratio'].replace(0, np.nan)
        pb_safe = features['pb_ratio'].replace(0, np.nan)
        features['value_score'] = (1 / pe_safe).fillna(0) + (1 / pb_safe).fillna(0)

        # Sector-relative value score: lower pe_vs_sector + lower pb_vs_sector = better value
        # Invert so higher = more undervalued relative to sector
        features['sector_value_score'] = (
            (2.0 - features['pe_vs_sector'].clip(0, 2)) +
            (2.0 - features['pb_vs_sector'].clip(0, 2))
        ) / 2.0

        # Quality score from sentiment data quality
        features['quality_score'] = df['data_quality_score'].fillna(0.5)

        # Handle missing values - impute with median
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                if pd.isna(median_val):
                    median_val = 0
                features[col] = features[col].fillna(median_val)
                logger.info(f"Imputed {col} missing values with median: {median_val:.4f}")

        # Create target variable
        # Outperform = 1 if 52-week return > benchmark, else 0
        target = (df['52_week_price_change_pct'] > (self.benchmark_return * 100)).astype(int)

        feature_names = list(features.columns)
        self.feature_names = feature_names

        logger.info(f"Prepared {len(feature_names)} features: {feature_names}")
        logger.info(f"Target distribution: {target.value_counts().to_dict()}")

        return features, target, feature_names

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        min_train_size: int = 5
    ) -> Dict:
        """
        Perform walk-forward validation.

        CRITICAL: Always train on earlier data, predict later data.

        Args:
            df: Full dataset
            min_train_size: Minimum samples for training

        Returns:
            Validation results
        """
        logger.info("Starting walk-forward validation")

        # Sort by filing date (temporal ordering)
        if 'filing_date' in df.columns:
            df = df.sort_values('filing_date').reset_index(drop=True)
        else:
            logger.warning("No filing_date column, using row order")

        X, y, feature_names = self.prepare_features(df)

        results = {
            'predictions': [],
            'actuals': [],
            'tickers': [],
            'confidences': [],
            'splits': []
        }

        n_samples = len(df)

        # Expanding window validation
        for i in range(min_train_size, n_samples):
            train_idx = list(range(i))
            test_idx = [i]

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Log temporal info
            if 'filing_date' in df.columns:
                train_dates = df.iloc[train_idx]['filing_date']
                test_date = df.iloc[test_idx]['filing_date'].values[0]
                logger.info(
                    f"Split {i-min_train_size+1}: Train on {len(train_idx)} samples "
                    f"(through {train_dates.max()}), "
                    f"predict {df.iloc[test_idx]['ticker'].values[0]} "
                    f"(filing: {test_date})"
                )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            model = self._create_model()
            model.fit(X_train_scaled, y_train)

            # Predict
            pred = model.predict(X_test_scaled)[0]
            prob = model.predict_proba(X_test_scaled)[0]
            confidence = max(prob)

            results['predictions'].append(pred)
            results['actuals'].append(y_test.values[0])
            results['tickers'].append(df.iloc[test_idx]['ticker'].values[0])
            results['confidences'].append(confidence)
            results['splits'].append({
                'train_size': len(train_idx),
                'test_ticker': df.iloc[test_idx]['ticker'].values[0]
            })

        # Calculate metrics
        predictions = np.array(results['predictions'])
        actuals = np.array(results['actuals'])

        accuracy = (predictions == actuals).mean()
        n_correct = (predictions == actuals).sum()

        results['accuracy'] = accuracy
        results['n_correct'] = n_correct
        results['n_total'] = len(predictions)

        logger.info(
            f"Walk-forward validation complete: "
            f"Accuracy = {accuracy:.2%} ({n_correct}/{len(predictions)})"
        )

        return results

    def _create_model(self):
        """Create a new model instance."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model on full dataset.

        Args:
            X: Feature matrix
            y: Target variable
        """
        logger.info(f"Training {self.model_type} on {len(X)} samples")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)

        self.is_trained = True
        logger.info("Training complete")

    def predict(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Confidence is the probability of the predicted class
        confidences = np.max(probabilities, axis=1)

        return predictions, confidences

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            return {}

        if self.model_type == "random_forest":
            importances = self.model.feature_importances_
        elif self.model_type == "logistic_regression":
            importances = np.abs(self.model.coef_[0])
        else:
            return {}

        importance_dict = dict(zip(self.feature_names, importances))
        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict

    def generate_predictions(
        self,
        df: pd.DataFrame
    ) -> List[PredictionResult]:
        """
        Generate predictions for all tickers in dataset.

        Args:
            df: Merged dataset

        Returns:
            List of PredictionResult objects with strict rank ordering
        """
        X, y, _ = self.prepare_features(df)

        # Train on full data
        self.train(X, y)

        # Make predictions
        predictions, confidences = self.predict(X)

        # Get feature importance
        feature_importance = self.get_feature_importance()

        # Build results
        results = []
        for i, row in df.iterrows():
            idx = df.index.get_loc(i)

            pred = predictions[idx]
            conf = confidences[idx]

            # Extract tiebreaker values
            momentum = row.get('52_week_price_change_pct', 0.0)
            if pd.isna(momentum):
                momentum = 0.0
            sentiment = row.get('net_sentiment', 0.0)
            if pd.isna(sentiment):
                sentiment = 0.0
            market_cap = row.get('market_cap', float('inf'))
            if pd.isna(market_cap):
                market_cap = float('inf')

            result = PredictionResult(
                ticker=row['ticker'],
                as_of_date=datetime.now(pytz.utc),
                prediction=int(pred),
                confidence=float(conf),
                predicted_direction="OUTPERFORM" if pred == 1 else "UNDERPERFORM",
                actual_return_52w=row.get('52_week_price_change_pct', np.nan),
                feature_contributions=feature_importance,
                momentum_score=float(momentum),
                sentiment_score=float(sentiment),
                market_cap=float(market_cap)
            )
            results.append(result)

        # Apply strict ranking with tiebreakers
        results = self._apply_strict_ranking(results)

        return results

    def _apply_strict_ranking(
        self,
        results: List[PredictionResult]
    ) -> List[PredictionResult]:
        """
        Apply strict 1-to-N ranking with tiebreakers.

        Tiebreaker order (when confidence is equal):
        1. momentum_score (higher is better)
        2. sentiment_score (higher is better)
        3. market_cap (smaller wins - favors smaller micro-caps)

        Args:
            results: List of PredictionResult with confidence scores

        Returns:
            Same list with rank field populated (1 = best)
        """
        # Sort by: confidence DESC, momentum DESC, sentiment DESC, market_cap ASC
        sorted_results = sorted(
            results,
            key=lambda x: (
                -x.confidence,
                -x.momentum_score,
                -x.sentiment_score,
                x.market_cap  # Smaller wins (ASC)
            )
        )

        # Assign strict ranks 1 to N
        for rank, result in enumerate(sorted_results, start=1):
            result.rank = rank

        logger.info(
            f"Applied strict ranking to {len(results)} predictions. "
            f"Top 3: {[r.ticker for r in sorted_results[:3]]}"
        )

        return sorted_results

    def evaluate_vs_actual(
        self,
        df: pd.DataFrame,
        predictions: List[PredictionResult]
    ) -> Dict:
        """
        Evaluate predictions against actual 52-week returns.

        Args:
            df: Original dataset
            predictions: List of predictions

        Returns:
            Evaluation metrics
        """
        pred_df = pd.DataFrame([p.to_dict() for p in predictions])
        pred_df['actual_outperform'] = (
            df['52_week_price_change_pct'] > (self.benchmark_return * 100)
        ).astype(int).values

        correct = (pred_df['prediction'] == pred_df['actual_outperform']).sum()
        total = len(pred_df)
        accuracy = correct / total if total > 0 else 0

        # Separate by prediction type
        outperform_preds = pred_df[pred_df['prediction'] == 1]
        underperform_preds = pred_df[pred_df['prediction'] == 0]

        avg_return_predicted_outperform = (
            outperform_preds['actual_return_52w'].mean()
            if len(outperform_preds) > 0 else np.nan
        )
        avg_return_predicted_underperform = (
            underperform_preds['actual_return_52w'].mean()
            if len(underperform_preds) > 0 else np.nan
        )

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_return_predicted_outperform': avg_return_predicted_outperform,
            'avg_return_predicted_underperform': avg_return_predicted_underperform,
            'benchmark_threshold': self.benchmark_return * 100
        }
