import os
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
import traceback  # Add import for traceback module
from scipy.optimize import minimize
from typing import Tuple, Dict, List, Any
import logging
import yfinance as yf

class PortfolioOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ef = None
        self.weights = None
        self.performance = None
        self.selected_stocks = []
        # Get current risk-free rate from 13-week Treasury yield
        self.risk_free_rate = self._get_current_risk_free_rate()
        self.regime_predictions = []
        self.model_confidence = []
        self.feature_importance = {}
        self.current_regime = None
        
    def _get_current_risk_free_rate(self) -> float:
        """Get current risk-free rate from 13-week Treasury yield"""
        try:
            # Use ^IRX (13-week Treasury Bill) as risk-free rate
            irx = yf.Ticker("^IRX")
            current_rate = irx.info.get('regularMarketPrice', 2.0) / 100  # Convert from percentage
            return current_rate if current_rate > 0 else 0.02  # Default to 2% if unable to get rate
        except:
            return 0.02  # Default to 2% if there's an error
            
    def detect_market_regime(self, returns: pd.DataFrame) -> str:
        """
        Detect the current market regime using multiple indicators
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            str: Market regime ('bull', 'bear', or 'neutral')
        """
        try:
            # Calculate recent (60-day) and long-term (252-day) metrics
            recent_window = 60
            longterm_window = 252
            
            # Get market index (S&P 500) data
            spy = yf.Ticker("^GSPC")
            market_data = spy.history(period="2y")
            market_returns = market_data['Close'].pct_change()
            
            # Calculate metrics
            recent_return = market_returns[-recent_window:].mean() * 252  # Annualized
            longterm_return = market_returns[-longterm_window:].mean() * 252  # Annualized
            recent_vol = market_returns[-recent_window:].std() * np.sqrt(252)  # Annualized
            longterm_vol = market_returns[-longterm_window:].std() * np.sqrt(252)  # Annualized
            
            # Calculate technical indicators
            sma_50 = market_data['Close'].rolling(window=50).mean()
            sma_200 = market_data['Close'].rolling(window=200).mean()
            current_price = market_data['Close'].iloc[-1]
            
            # Score different factors (-1 for bearish, 0 for neutral, 1 for bullish)
            scores = []
            
            # 1. Recent vs Long-term Returns
            scores.append(1 if recent_return > longterm_return else -1 if recent_return < 0 else 0)
            
            # 2. Volatility Regime
            scores.append(-1 if recent_vol > longterm_vol * 1.2 else 1 if recent_vol < longterm_vol * 0.8 else 0)
            
            # 3. Moving Average Trends
            scores.append(1 if current_price > sma_50.iloc[-1] > sma_200.iloc[-1] 
                        else -1 if current_price < sma_50.iloc[-1] < sma_200.iloc[-1] 
                        else 0)
            
            # 4. Recent Momentum
            recent_momentum = market_returns[-20:].sum()  # 20-day momentum
            scores.append(1 if recent_momentum > 0.02 else -1 if recent_momentum < -0.02 else 0)
            
            # Calculate average score
            avg_score = sum(scores) / len(scores)
            
            # Determine regime
            if avg_score > 0.3:
                regime = 'bull'
            elif avg_score < -0.3:
                regime = 'bear'
            else:
                regime = 'neutral'
                
            self.logger.info(f"Detected market regime: {regime} (score: {avg_score:.2f})")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return 'neutral'  # Default to neutral regime on error
            
    def _calculate_expected_returns(self, returns: pd.DataFrame, regime: str) -> pd.Series:
        """
        Calculate expected returns based on market regime and multiple factors
        
        Args:
            returns: DataFrame of asset returns
            regime: Current market regime ('bull', 'bear', or 'neutral')
            
        Returns:
            pd.Series: Expected returns for each asset
        """
        try:
            # 1. Calculate historical mean returns (base estimate)
            hist_returns = returns.mean() * 252  # Annualize
            
            # 2. Calculate recent momentum (last 60 days)
            recent_returns = returns.iloc[-60:].mean() * 252
            
            # 3. Get market data for beta calculation
            spy = yf.Ticker("^GSPC")
            market_data = spy.history(period="1y")
            market_returns = market_data['Close'].pct_change()
            
            # 4. Calculate betas for each stock
            betas = {}
            for col in returns.columns:
                if len(returns[col]) > 0 and len(market_returns) > 0:
                    # Align dates
                    aligned_data = pd.concat([returns[col], market_returns], axis=1).dropna()
                    if len(aligned_data) > 0:
                        beta = np.cov(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])[0,1] / np.var(aligned_data.iloc[:, 1])
                        betas[col] = beta
                    else:
                        betas[col] = 1.0
                else:
                    betas[col] = 1.0
                    
            # 5. Get current market expected return
            current_market_return = market_returns.mean() * 252
            risk_premium = current_market_return - self.risk_free_rate
            
            # 6. Calculate CAPM expected returns
            capm_returns = pd.Series({ticker: self.risk_free_rate + beta * risk_premium 
                                    for ticker, beta in betas.items()})
            
            # 7. Combine different return estimates with regime-based weights
            if regime == 'bull':
                # In bull markets, give more weight to momentum and CAPM
                weights = {'hist': 0.2, 'recent': 0.5, 'capm': 0.3}
                adj_factor = 1.2  # Increase expected returns
            elif regime == 'bear':
                # In bear markets, give more weight to historical returns and less to momentum
                weights = {'hist': 0.5, 'recent': 0.2, 'capm': 0.3}
                adj_factor = 0.8  # Decrease expected returns
            else:
                # In neutral markets, balance between all factors
                weights = {'hist': 0.4, 'recent': 0.3, 'capm': 0.3}
                adj_factor = 1.0  # No adjustment
                
            # Combine return estimates
            exp_returns = (hist_returns * weights['hist'] + 
                         recent_returns * weights['recent'] + 
                         capm_returns * weights['capm']) * adj_factor
            
            # Apply sanity checks
            exp_returns = exp_returns.clip(-0.5, 0.5)  # Cap at ±50% annual return
            
            return exp_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating expected returns: {str(e)}")
            return pd.Series(0.0, index=returns.columns)
            
    def optimize_portfolio(self, market_data: pd.DataFrame, selected_stocks: List[str]) -> Dict:
        """
        Optimize portfolio weights using mean-variance optimization
        
        Args:
            market_data: DataFrame with market data
            selected_stocks: List of stock tickers to include in portfolio
            
        Returns:
            Dict containing portfolio metrics and weights
        """
        self.logger.info("\nOptimizing portfolio...")
        result_dict = {}
        
        try:
            # Create price DataFrame
            prices_df = self._create_price_df(market_data, selected_stocks)
            if prices_df.empty:
                raise ValueError("Could not create price DataFrame")
                
            # Calculate returns
            returns = prices_df.pct_change().dropna()
            
            # Detect market regime
            regime = self.detect_market_regime(returns)
            self.logger.info(f"Detected market regime: {regime}")
            
            # Calculate expected returns (adjusted for regime)
            exp_returns = self._calculate_expected_returns(returns, regime)
            
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252  # Annualized
            
            # Define dynamic constraints based on regime and portfolio size
            num_stocks = len(selected_stocks)
            equal_weight = 1.0 / num_stocks
            
            # Calculate dynamic weight bounds based on regime
            if regime == 'bull':
                min_weight = max(0.01, equal_weight * 0.5)  # Min 1% or half equal weight
                max_weight = min(0.40, equal_weight * 2.0)  # Max 40% or double equal weight
            elif regime == 'bear':
                min_weight = max(0.02, equal_weight * 0.8)  # Min 2% or 80% equal weight
                max_weight = min(0.25, equal_weight * 1.5)  # Max 25% or 1.5x equal weight
            else:
                min_weight = max(0.015, equal_weight * 0.7)  # Min 1.5% or 70% equal weight
                max_weight = min(0.35, equal_weight * 1.8)   # Max 35% or 1.8x equal weight
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'ineq', 'fun': lambda x: x - min_weight},  # Minimum weight
                {'type': 'ineq', 'fun': lambda x: max_weight - x}  # Maximum weight
            ]
            
            bounds = tuple((min_weight, max_weight) for _ in range(num_stocks))
            
            # Initial guess (equal weights with small random perturbation)
            initial_weights = np.array([equal_weight] * num_stocks)
            initial_weights += np.random.normal(0, 0.01, num_stocks)
            initial_weights = np.clip(initial_weights, min_weight, max_weight)
            initial_weights = initial_weights / initial_weights.sum()  # Renormalize
            
            # Optimize portfolio
            result = minimize(
                lambda w: -self._calculate_sharpe_ratio(w, exp_returns, cov_matrix),
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not result.success:
                self.logger.warning(f"Portfolio optimization failed: {result.message}")
                weights = initial_weights
            else:
                weights = result.x
                
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * exp_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            # Create dictionary of weights
            weights_dict = {stock: weight for stock, weight in zip(selected_stocks, weights)}
            
            # Store results
            self.weights = weights_dict
            self.expected_return = portfolio_return
            self.volatility = portfolio_std
            self.sharpe_ratio = sharpe_ratio
            
            # Log results
            self.logger.info("\nPortfolio Optimization Results:")
            self.logger.info(f"Expected Annual Return: {portfolio_return*100:.2f}%")
            self.logger.info(f"Annual Volatility: {portfolio_std*100:.2f}%")
            self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            self.logger.info(f"Market Regime: {regime}")
            self.logger.info(f"Risk-free Rate: {self.risk_free_rate*100:.2f}%")
            self.logger.info(f"Min Weight: {min_weight*100:.2f}%")
            self.logger.info(f"Max Weight: {max_weight*100:.2f}%")
            
            result_dict = {
                'weights': weights_dict,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio,
                'regime': regime,
                'constraints': {
                    'min_weight': min_weight,
                    'max_weight': max_weight,
                    'equal_weight': equal_weight
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            self.logger.error(traceback.format_exc())
            
        return result_dict
            
    def _create_price_df(self, market_data: pd.DataFrame, selected_stocks: List[str]) -> pd.DataFrame:
        """
        Create a price DataFrame from market data for selected stocks.
        
        Args:
            market_data: DataFrame with MultiIndex (Date, Ticker) and columns including 'Close'
            selected_stocks: List of stock tickers to include
            
        Returns:
            DataFrame with dates as index and selected stocks as columns, containing closing prices
        """
        try:
            # Verify market data is not empty
            if market_data.empty:
                self.logger.error("Market data is empty")
                return pd.DataFrame()
                
            # Get unique dates from the index
            dates = market_data.index.get_level_values('Date').unique()
            
            # Create an empty DataFrame with dates as index
            prices_df = pd.DataFrame(index=dates)
            
            # For each stock, get its closing prices and add to the DataFrame
            for ticker in selected_stocks:
                try:
                    # Get closing prices for this ticker
                    stock_data = market_data.xs(ticker, level='Ticker')['Close']
                    prices_df[ticker] = stock_data
                except KeyError:
                    self.logger.warning(f"No data found for {ticker}")
                    continue
                    
            # Drop any dates with missing data
            prices_df = prices_df.dropna()
            
            if prices_df.empty:
                self.logger.error("No valid price data found for selected stocks")
                return pd.DataFrame()
                
            return prices_df
            
        except Exception as e:
            self.logger.error(f"Error creating price DataFrame: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    def _calculate_sharpe_ratio(self, weights: np.ndarray, exp_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Calculate the Sharpe ratio for a given portfolio allocation.
        
        Args:
            weights: Array of portfolio weights
            exp_returns: Series of expected returns for each asset
            cov_matrix: Covariance matrix of asset returns
            
        Returns:
            float: Portfolio Sharpe ratio
        """
        try:
            # Calculate portfolio return
            portfolio_return = np.sum(weights * exp_returns)
            
            # Calculate portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Calculate Sharpe ratio
            if portfolio_vol > 0:
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            else:
                sharpe = 0.0
                
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0  # Return zero Sharpe ratio on error
            
    def _calculate_regime_metrics(self, returns: pd.DataFrame, regime: str) -> Dict:
        """Calculate detailed regime analysis metrics"""
        try:
            # Get market data
            spy = yf.Ticker("^GSPC")
            market_data = spy.history(period="2y")
            market_returns = market_data['Close'].pct_change()
            
            # Calculate regime indicators
            recent_return = market_returns[-60:].mean() * 252
            longterm_return = market_returns[-252:].mean() * 252
            recent_vol = market_returns[-60:].std() * np.sqrt(252)
            longterm_vol = market_returns[-252:].std() * np.sqrt(252)
            
            # Technical indicators
            sma_50 = market_data['Close'].rolling(window=50).mean()
            sma_200 = market_data['Close'].rolling(window=200).mean()
            current_price = market_data['Close'].iloc[-1]
            
            # Calculate regime score components
            return_score = 1 if recent_return > longterm_return else -1 if recent_return < 0 else 0
            vol_score = -1 if recent_vol > longterm_vol * 1.2 else 1 if recent_vol < longterm_vol * 0.8 else 0
            trend_score = 1 if current_price > sma_50.iloc[-1] > sma_200.iloc[-1] else -1 if current_price < sma_50.iloc[-1] < sma_200.iloc[-1] else 0
            momentum_score = 1 if market_returns[-20:].sum() > 0.02 else -1 if market_returns[-20:].sum() < -0.02 else 0
            
            # Combine scores
            regime_score = (return_score + vol_score + trend_score + momentum_score) / 4
            
            return {
                'regime': regime,
                'regime_score': regime_score,
                'metrics': {
                    'recent_return': recent_return,
                    'longterm_return': longterm_return,
                    'recent_volatility': recent_vol,
                    'longterm_volatility': longterm_vol,
                    'price_vs_sma50': current_price / sma_50.iloc[-1] - 1,
                    'price_vs_sma200': current_price / sma_200.iloc[-1] - 1
                },
                'scores': {
                    'return_score': return_score,
                    'volatility_score': vol_score,
                    'trend_score': trend_score,
                    'momentum_score': momentum_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating regime metrics: {str(e)}")
            return {
                'regime': regime,
                'regime_score': 0,
                'metrics': {},
                'scores': {}
            }

    def generate_portfolio_report(self, optimization_result, data, factor_scores, management_scores):
        """Generate a comprehensive portfolio report with allocations, metrics and visualizations"""
        try:
            print("Generating portfolio report...")
            
            # Make sure the reports directory exists
            os.makedirs('reports', exist_ok=True)
            
            # Check if optimization_result is valid
            if optimization_result is None:
                print("Warning: No optimization result provided")
                return None
                
            # Extract portfolio data
            if isinstance(optimization_result, dict):
                weights = optimization_result.get('weights', {})
                expected_return = optimization_result.get('expected_return', 0.0)
                volatility = optimization_result.get('volatility', 0.0)
                sharpe_ratio = optimization_result.get('sharpe_ratio', 0.0)
                regime = optimization_result.get('regime', 'unknown')
                regime_metrics = optimization_result.get('regime_metrics', {})
                constraints = optimization_result.get('constraints', {})
            else:
                # Handle if optimization_result is an object
                weights = getattr(optimization_result, 'weights', {})
                expected_return = getattr(optimization_result, 'expected_return', 0.0)
                volatility = getattr(optimization_result, 'volatility', 0.0)
                sharpe_ratio = getattr(optimization_result, 'sharpe_ratio', 0.0)
                regime = getattr(optimization_result, 'regime', 'unknown')
                regime_metrics = getattr(optimization_result, 'regime_metrics', {})
                constraints = getattr(optimization_result, 'constraints', {})
            
            # Write regime analysis report
            with open('reports/regime_analysis.txt', 'w') as f:
                f.write("Market Regime Analysis:\n")
                f.write("=====================\n\n")
                f.write(f"Current Regime: {regime}\n")
                f.write(f"Regime Score: {regime_metrics.get('regime_score', 0):.2f}\n\n")
                
                # Write detailed metrics
                metrics = regime_metrics.get('metrics', {})
                f.write("Key Metrics:\n")
                for name, value in metrics.items():
                    f.write(f"{name}: {value:.2%}\n")
                
                f.write("\nComponent Scores:\n")
                scores = regime_metrics.get('scores', {})
                for name, value in scores.items():
                    f.write(f"{name}: {value:.2f}\n")
                
                f.write("\nOptimization Constraints:\n")
                f.write(f"Minimum Weight: {constraints.get('min_weight', 0)*100:.2f}%\n")
                f.write(f"Maximum Weight: {constraints.get('max_weight', 0)*100:.2f}%\n")
                f.write(f"Equal Weight: {constraints.get('equal_weight', 0)*100:.2f}%\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio report: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def get_regime_predictions(self, market_data):
        """Get market regime predictions for visualization"""
        if not isinstance(self.regime_predictions, np.ndarray) or len(self.regime_predictions) == 0:
            # Calculate regime predictions if not already done
            self.regime_predictions = self._calculate_regime_predictions(market_data)
        return self.regime_predictions
    
    def get_model_confidence(self):
        """Get model confidence scores for visualization"""
        if isinstance(self.model_confidence, pd.Series):
            # For pandas Series
            if self.model_confidence.empty:
                self._calculate_model_confidence()
            return self.model_confidence.values
        elif isinstance(self.model_confidence, np.ndarray):
            # For numpy arrays
            if len(self.model_confidence) == 0:
                self._calculate_model_confidence()
            return self.model_confidence
        elif not self.model_confidence:  # For empty lists
            # Calculate confidence scores if not already done
            self._calculate_model_confidence()
            return self.model_confidence
        else:
            # For any other case, return what we have
            return self.model_confidence
    
    def get_feature_importance(self):
        """Get feature importance scores for visualization"""
        if not self.feature_importance:
            # Calculate feature importance if not already done
            self._calculate_feature_importance()
        return self.feature_importance
    
    def _calculate_regime_predictions(self, market_data):
        """Calculate market regime predictions using ML"""
        try:
            # Extract features for regime detection
            features = pd.DataFrame()
            
            # 1. Price momentum features
            returns = market_data.groupby('Date')['Close'].mean().pct_change()
            features['momentum_1m'] = returns.rolling(21).mean()
            features['momentum_3m'] = returns.rolling(63).mean()
            features['momentum_6m'] = returns.rolling(126).mean()
            
            # 2. Volatility features
            features['volatility'] = returns.rolling(21).std()
            features['volatility_change'] = features['volatility'].pct_change()
            
            # 3. Market breadth features
            # Using try/except to handle division by zero
            try:
                up_down_ratio = market_data.groupby('Date')['Close'].apply(
                    lambda x: len(x[x > 0]) / max(1, len(x[x < 0])))  # Add max() to avoid division by zero
                features['market_breadth'] = up_down_ratio.rolling(21).mean()
            except Exception as e:
                print(f"Warning: Error calculating market breadth: {str(e)}")
                features['market_breadth'] = 0.0  # Use a default value
            
            # Fill NaN values for calculations
            features = features.fillna(0)
            
            # Simple regime detection based on features
            regime_score = (
                0.4 * features['momentum_1m'] +
                0.3 * features['momentum_3m'] +
                0.2 * features['momentum_6m'] +
                0.1 * features['market_breadth'] -
                0.2 * features['volatility_change']
            )
            
            # Handle missing values
            regime_score = regime_score.fillna(0)
            
            # Store feature importance
            self.feature_importance = {
                'Short-term Momentum': 0.4,
                'Medium-term Momentum': 0.3,
                'Long-term Momentum': 0.2,
                'Market Breadth': 0.1,
                'Volatility Change': 0.2
            }
            
            # Calculate model confidence safely
            try:
                # Generate synthetic confidence values based on volatility
                volatility_series = features['volatility'].fillna(0)
                confidence_series = 1 - (volatility_series / volatility_series.max()).fillna(0)
                
                # Ensure confidence is between 0.3 and 1.0
                confidence_series = 0.3 + (0.7 * confidence_series)
                self.model_confidence = confidence_series.values
            except Exception as e:
                print(f"Warning: Error calculating model confidence: {str(e)}")
                # Use default confidence values if calculation fails
                self.model_confidence = np.ones(len(regime_score)) * 0.7
            
            # Normalize scores to [-1, 1] range, safely handling division by zero
            max_abs_value = regime_score.abs().max()
            if max_abs_value > 0:  # Ensure we don't divide by zero
                regime_score = regime_score / max_abs_value
            else:
                # If all values are 0, keep them as 0
                regime_score = regime_score * 0
            
            # Ensure we return numpy arrays of same length
            regime_predictions = regime_score.values
            if len(self.model_confidence) != len(regime_predictions):
                print(f"Warning: Length mismatch between regime predictions ({len(regime_predictions)}) and model confidence ({len(self.model_confidence)})")
                # Adjust model confidence length to match regime predictions
                self.model_confidence = np.ones(len(regime_predictions)) * 0.7
            
            return regime_predictions
            
        except Exception as e:
            print(f"Warning: Error in regime prediction: {str(e)}")
            # Return neutral regime if calculation fails
            return np.zeros(len(market_data.index.get_level_values('Date').unique()))
    
    def _calculate_model_confidence(self):
        """Calculate model confidence if not already done"""
        if not self.model_confidence:
            # Default confidence if not calculated during regime prediction
            self.model_confidence = np.ones(len(self.regime_predictions)) * 0.8
    
    def _calculate_feature_importance(self):
        """Calculate feature importance if not already done"""
        if not self.feature_importance:
            # Default feature importance if not calculated during regime prediction
            self.feature_importance = {
                'Short-term Momentum': 0.4,
                'Medium-term Momentum': 0.3,
                'Long-term Momentum': 0.2,
                'Market Breadth': 0.1,
                'Volatility Change': 0.2
            }

if __name__ == "__main__":
    # Example usage
    from data_collector import MarketDataCollector
    from factor_analysis import FactorAnalyzer
    from management_analyzer import ManagementAnalyzer
    
    # Collect data
    collector = MarketDataCollector()
    data = collector.get_market_data()
    metrics = collector.calculate_metrics(data)
    
    # Analyze factors
    factor_analyzer = FactorAnalyzer()
    factor_scores = factor_analyzer.calculate_factor_scores(metrics, metrics.index[-1].date())
    
    # Get management scores
    mgmt_analyzer = ManagementAnalyzer()
    management_scores = {ticker: mgmt_analyzer.get_management_score(ticker) 
                        for ticker in factor_scores.index[:20]}
    
    # Optimize portfolio
    optimizer = PortfolioOptimizer()
    result = optimizer.optimize_portfolio(metrics, 10)
    
    # Generate report
    report = optimizer.generate_portfolio_report(result, metrics, factor_scores, 
                                              management_scores) 