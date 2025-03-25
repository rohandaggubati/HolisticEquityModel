import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FactorAnalyzer:
    """Analyze factor performance and backtest investment strategies"""
    
    def __init__(self):
        """Initialize the factor analyzer"""
        self.factors = ['momentum', 'volatility', 'value', 'quality', 'growth']
        self.factor_weights = {factor: 1.0/len(self.factors) for factor in self.factors}
        self.optimal_weights = None
        self.backtest_performance = None
        self.backtest_periods = None
        self.factor_returns = pd.DataFrame()
        
    def calculate_factor_scores(self, data):
        """
        Calculate factor scores for each stock
        
        Args:
            data (pd.DataFrame): Market data with MultiIndex (Date, Ticker)
            
        Returns:
            pd.DataFrame: Factor scores for each stock
        """
        try:
            if data is None or data.empty:
                logger.error("No data provided for factor analysis")
                return None
            
            # Get the latest date for each ticker
            latest_date = data.index.get_level_values('Date').max()
            
            # Get unique tickers
            tickers = data.index.get_level_values('Ticker').unique()
            
            # Create a DataFrame to store factor scores
            factor_scores = pd.DataFrame(index=tickers, columns=self.factors)
            
            # Calculate factor scores for each ticker
            for ticker in tickers:
                try:
                    # Get the ticker's data
                    ticker_data = data.xs(ticker, level='Ticker')
                    
                    # Calculate individual factor scores
                    momentum_score = self._calculate_momentum(ticker_data)
                    volatility_score = self._calculate_volatility(ticker_data)
                    value_score = self._calculate_value(ticker_data)
                    quality_score = self._calculate_quality(ticker_data)
                    growth_score = self._calculate_growth(ticker_data)
                    
                    # Store the scores
                    factor_scores.loc[ticker, 'momentum'] = momentum_score
                    factor_scores.loc[ticker, 'volatility'] = volatility_score
                    factor_scores.loc[ticker, 'value'] = value_score
                    factor_scores.loc[ticker, 'quality'] = quality_score
                    factor_scores.loc[ticker, 'growth'] = growth_score
                    
                except Exception as e:
                    logger.warning(f"Error calculating factor scores for {ticker}: {str(e)}")
                    # Set all factor scores to NaN for this ticker
                    factor_scores.loc[ticker, :] = np.nan
            
            # Fill any NaN values with the median score for each factor
            for factor in self.factors:
                # Get the median score for this factor
                median_score = factor_scores[factor].median()
                if pd.isna(median_score):
                    median_score = 0  # Default to 0 if all values are NaN
                
                # Handle missing values manually without ffill/bfill
                # Create a new Series to avoid in-place modification warnings
                values = factor_scores[factor].copy()
                
                # First, identify NaN values
                nan_mask = values.isna()
                
                # For each NaN value, try to fill with values from neighboring stocks
                if nan_mask.any():
                    values.loc[nan_mask] = median_score
                
                # Assign back to factor_scores
                factor_scores[factor] = values
            
            # Apply z-score normalization to each factor
            for factor in self.factors:
                mean = factor_scores[factor].mean()
                std = factor_scores[factor].std()
                if std != 0 and not pd.isna(std):  # Avoid division by zero and NaN
                    factor_scores[factor] = (factor_scores[factor] - mean) / std
                else:
                    factor_scores[factor] = 0  # Default to 0 if std is 0 or NaN
            
            return factor_scores
            
        except Exception as e:
            logger.error(f"Error in calculate_factor_scores: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_momentum(self, data, momentum_windows=[21, 63, 126, 252]):
        """
        Calculate momentum factor score
        
        Args:
            data (pd.DataFrame): Market data for a single ticker
            momentum_windows (list): List of lookback periods for momentum calculation
            
        Returns:
            float: Momentum factor score
        """
        try:
            if data.empty or len(data) < min(momentum_windows):
                return np.nan
                
            # Ensure data is sorted by date
            data = data.sort_index()
            
            # Get the most recent price data
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                return np.nan
                
            # Calculate momentum for different windows
            momentum_scores = []
            
            for window in momentum_windows:
                if len(prices) > window:
                    # Calculate return over the period
                    returns = prices.pct_change(periods=window).dropna()
                    if len(returns) > 0:
                        momentum = returns.iloc[-1]
                        momentum_scores.append(momentum)
                    
            # If we have no valid momentum calculations, return NaN
            if not momentum_scores:
                return np.nan
                
            # Combine momentum scores with more weight to shorter timeframes
            weights = np.array([4, 3, 2, 1])[:len(momentum_scores)]
            weights = weights / weights.sum()  # Normalize weights
            
            return np.average(momentum_scores, weights=weights)
            
        except Exception as e:
            logger.warning(f"Error calculating momentum factor: {str(e)}")
            return np.nan
            
    def _calculate_volatility(self, data, volatility_windows=[20, 60, 120]):
        """
        Calculate volatility factor score (lower is better)
        
        Args:
            data (pd.DataFrame): Market data for a single ticker
            volatility_windows (list): List of lookback periods for volatility calculation
            
        Returns:
            float: Volatility factor score (negative because lower volatility is better)
        """
        try:
            if data.empty or len(data) < min(volatility_windows):
                return np.nan
                
            # Ensure data is sorted by date
            data = data.sort_index()
            
            # Get the most recent price data
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                return np.nan
                
            # Calculate daily returns
            returns = prices.pct_change().dropna()
            
            # Calculate volatility for different windows (annualized)
            volatility_scores = []
            
            for window in volatility_windows:
                if len(returns) > window:
                    window_returns = returns.tail(window)
                    if not window_returns.empty and len(window_returns) > 1:
                        # Calculate annualized standard deviation
                        vol = window_returns.std() * np.sqrt(252)
                        if not pd.isna(vol):
                            volatility_scores.append(vol)
                    
            # If we have no valid volatility calculations, return NaN
            if not volatility_scores:
                return np.nan
                
            # Take the average volatility across different windows
            avg_volatility = np.mean(volatility_scores)
            
            # Return negative volatility (lower is better)
            return -avg_volatility
            
        except Exception as e:
            logger.warning(f"Error calculating volatility factor: {str(e)}")
            return np.nan
            
    def _calculate_value(self, data):
        """
        Calculate value factor score
        
        Args:
            data (pd.DataFrame): Market data for a single ticker
            
        Returns:
            float: Value factor score
        """
        try:
            if data.empty:
                return np.nan
                
            # Get the most recent data point
            latest_data = data.iloc[-1]
            
            # Calculate value score based on P/E, P/B, and other value metrics
            value_metrics = []
            
            # P/E ratio (lower is better for value)
            if 'P/E' in latest_data and not np.isnan(latest_data['P/E']) and latest_data['P/E'] > 0:
                pe_score = -np.log(latest_data['P/E'])  # Negative log to handle large values
                value_metrics.append(pe_score)
                
            # Forward P/E (if available)
            if 'ForwardP/E' in latest_data and not np.isnan(latest_data['ForwardP/E']) and latest_data['ForwardP/E'] > 0:
                forward_pe_score = -np.log(latest_data['ForwardP/E'])
                value_metrics.append(forward_pe_score)
                
            # P/B ratio (lower is better for value)
            if 'P/B' in latest_data and not np.isnan(latest_data['P/B']) and latest_data['P/B'] > 0:
                pb_score = -np.log(latest_data['P/B'])
                value_metrics.append(pb_score)
                
            # If we have no valid value metrics, return NaN
            if not value_metrics:
                return np.nan
                
            # Return the average of all value metrics
            return np.mean(value_metrics)
            
        except Exception as e:
            logger.warning(f"Error calculating value factor: {str(e)}")
            return np.nan
            
    def _calculate_quality(self, data):
        """
        Calculate quality factor score
        
        Args:
            data (pd.DataFrame): Market data for a single ticker
            
        Returns:
            float: Quality factor score
        """
        try:
            if data.empty:
                return np.nan
                
            # Get the most recent data point
            latest_data = data.iloc[-1]
            
            # Calculate quality score based on ROE, ROA, profit margin, debt-to-equity
            quality_metrics = []
            
            # Return on Equity (higher is better)
            if 'ROE' in latest_data and not np.isnan(latest_data['ROE']):
                roe_score = latest_data['ROE']
                quality_metrics.append(roe_score)
                
            # Return on Assets (higher is better)
            if 'ROA' in latest_data and not np.isnan(latest_data['ROA']):
                roa_score = latest_data['ROA']
                quality_metrics.append(roa_score)
                
            # Profit Margin (higher is better)
            if 'ProfitMargin' in latest_data and not np.isnan(latest_data['ProfitMargin']):
                margin_score = latest_data['ProfitMargin']
                quality_metrics.append(margin_score)
                
            # Debt-to-Equity (lower is better)
            if 'Debt/Equity' in latest_data and not np.isnan(latest_data['Debt/Equity']) and latest_data['Debt/Equity'] > 0:
                debt_score = -np.log(latest_data['Debt/Equity'])
                quality_metrics.append(debt_score)
                
            # If we have no valid quality metrics, return NaN
            if not quality_metrics:
                return np.nan
                
            # Return the average of all quality metrics
            return np.mean(quality_metrics)
            
        except Exception as e:
            logger.warning(f"Error calculating quality factor: {str(e)}")
            return np.nan
            
    def _calculate_growth(self, data):
        """
        Calculate growth factor score
        
        Args:
            data (pd.DataFrame): Market data for a single ticker
            
        Returns:
            float: Growth factor score
        """
        try:
            if data.empty:
                return np.nan
                
            # Get the most recent data point
            latest_data = data.iloc[-1]
            
            # Calculate growth score based on EPS growth, revenue growth
            growth_metrics = []
            
            # EPS Growth (if available)
            if 'EPSGrowth' in latest_data and not np.isnan(latest_data['EPSGrowth']):
                eps_growth_score = latest_data['EPSGrowth']
                growth_metrics.append(eps_growth_score)
                
            # Revenue Growth (if available)
            if 'RevenueGrowth' in latest_data and not np.isnan(latest_data['RevenueGrowth']):
                revenue_growth_score = latest_data['RevenueGrowth']
                growth_metrics.append(revenue_growth_score)
                
            # Earnings Growth (if available)
            if 'EarningsGrowth' in latest_data and not np.isnan(latest_data['EarningsGrowth']):
                earnings_growth_score = latest_data['EarningsGrowth']
                growth_metrics.append(earnings_growth_score)
                
            # If we have no valid growth metrics, use momentum as a proxy
            if not growth_metrics and 'Momentum6M' in latest_data and not np.isnan(latest_data['Momentum6M']):
                return latest_data['Momentum6M']
                
            # If we still have no valid growth metrics, return NaN
            if not growth_metrics:
                return np.nan
                
            # Return the average of all growth metrics
            return np.mean(growth_metrics)
            
        except Exception as e:
            logger.warning(f"Error calculating growth factor: {str(e)}")
            return np.nan
    
    def optimize_factor_weights(self, data, lookback_periods=3, test_periods=1, forward_period_months=1):
        """
        Optimize factor weights based on historical performance
        
        Args:
            data (pd.DataFrame): Market data
            lookback_periods (int): Number of periods to look back
            test_periods (int): Number of periods to test
            forward_period_months (int): Number of months to look forward
            
        Returns:
            dict: Optimized factor weights
        """
        try:
            if data is None or data.empty:
                logger.error("No data provided for factor optimization")
                return {factor: 1.0 / len(self.factors) for factor in self.factors}
                
            # Make sure the index is sorted for time-based slicing
            if isinstance(data.index, pd.MultiIndex):
                data = data.sort_index()
                
            # Convert forward period to days
            forward_days = forward_period_months * 30  # Approximate
            
            # Get all dates in the data
            dates = sorted(data.index.get_level_values('Date').unique())
            
            if len(dates) < 2:
                logger.error("Not enough historical data for optimization")
                return self.factor_weights
                
            # Determine the period start dates (going backward from the most recent date)
            latest_date = dates[-1]
            period_dates = []
            
            for i in range(lookback_periods + test_periods):
                if i == 0:
                    period_dates.append(latest_date)
                else:
                    # Calculate the start of the period
                    period_start = latest_date - timedelta(days=forward_days * i)
                    # Find the closest actual date in the data
                    closest_date = min(dates, key=lambda d: abs((d - period_start).total_seconds()))
                    period_dates.append(closest_date)
            
            # Reverse so we have oldest to newest
            period_dates.reverse()
            
            # Initialize storage for factor returns
            all_factor_returns = []
            
            # For each period, calculate factor returns
            for i in range(len(period_dates) - 1):
                period_start = period_dates[i]
                period_end = period_dates[i + 1]
                
                # Get data for this period
                period_data = data.loc[period_start:period_end]
                
                # Calculate factor scores at the beginning of the period
                factor_scores = self.calculate_factor_scores(period_data)
                
                if factor_scores is None or factor_scores.empty:
                    continue
                
                # Calculate forward returns for each stock
                forward_returns = self._calculate_forward_returns(period_data, period_end, forward_days // 2)
                
                if forward_returns.empty:
                    continue
                
                # Merge factor scores with forward returns
                factor_data = pd.merge(factor_scores, forward_returns, left_index=True, right_index=True)
                
                # Calculate factor returns (correlation with forward returns)
                factor_returns = {}
                for factor in self.factors:
                    corr = factor_data[[factor, 'forward_return']].corr().iloc[0, 1]
                    if not np.isnan(corr):
                        factor_returns[factor] = corr
                
                all_factor_returns.append(factor_returns)
                
            # If we don't have enough data, use equal weights
            if not all_factor_returns:
                logger.warning("Not enough data for factor optimization, using equal weights")
                return {factor: 1.0 / len(self.factors) for factor in self.factors}
                
            # Convert to DataFrame
            factor_returns_df = pd.DataFrame(all_factor_returns)
            
            # Calculate mean factor returns for training periods
            train_returns = factor_returns_df.iloc[:lookback_periods].mean()
            
            # Use the training factor returns to determine optimal weights
            # Optimize using the average returns (simple approach)
            weights = train_returns.to_dict()
            
            # Ensure all weights are positive
            for factor in weights:
                weights[factor] = max(0.0, weights[factor])
                
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {factor: weight / total_weight for factor, weight in weights.items()}
            else:
                # Default to equal weights if all weights are negative or zero
                weights = {factor: 1.0 / len(self.factors) for factor in self.factors}
                
            # Store factor returns for analysis
            self.factor_returns = factor_returns_df.set_index(period_dates[:-1])
            
            return weights
            
        except Exception as e:
            logger.error(f"Error optimizing factor weights: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fall back to equal weights
            return {factor: 1.0 / len(self.factors) for factor in self.factors}
            
    def _calculate_forward_returns(self, data, period_end, forward_days):
        """
        Calculate forward returns for each stock from period_end date
        
        Args:
            data (pd.DataFrame): Market data
            period_end (datetime): End date of the period
            forward_days (int): Number of days to look forward
            
        Returns:
            pd.Series: Forward returns for each stock
        """
        try:
            # Get all dates after period_end
            future_dates = [d for d in data.index.get_level_values('Date').unique() if d > period_end]
            
            if not future_dates:
                return pd.Series()
                
            # Find the future date that's approximately forward_days ahead
            target_date = period_end + timedelta(days=forward_days)
            future_date = min(future_dates, key=lambda d: abs((d - target_date).total_seconds()))
            
            # Get all tickers in the data
            tickers = data.index.get_level_values('Ticker').unique()
            
            # Calculate forward returns for each ticker
            forward_returns = {}
            
            for ticker in tickers:
                try:
                    # Get the ticker's data
                    ticker_data = data.xs(ticker, level='Ticker')
                    
                    # Get prices at period_end and future_date
                    price_col = 'Adj Close' if 'Adj Close' in ticker_data.columns else 'Close'
                    
                    # Find the closest dates to period_end and future_date
                    period_end_date = min(ticker_data.index, key=lambda d: abs((d - period_end).total_seconds()))
                    future_date_idx = [i for i, d in enumerate(ticker_data.index) if d >= future_date]
                    
                    if not future_date_idx:
                        continue
                        
                    future_date_idx = future_date_idx[0]
                    if future_date_idx >= len(ticker_data):
                        continue
                        
                    future_date_actual = ticker_data.index[future_date_idx]
                    
                    # Calculate return
                    start_price = ticker_data.loc[period_end_date, price_col]
                    end_price = ticker_data.loc[future_date_actual, price_col]
                    
                    forward_return = (end_price / start_price) - 1
                    forward_returns[ticker] = forward_return
                    
                except Exception as e:
                    logger.debug(f"Error calculating forward return for {ticker}: {str(e)}")
                    continue
                    
            return pd.Series(forward_returns, name='forward_return')
            
        except Exception as e:
            logger.error(f"Error calculating forward returns: {str(e)}")
            return pd.Series()
            
    def calculate_weighted_scores(self, factor_scores, weights=None):
        """
        Calculate weighted factor scores for each stock
        
        Args:
            factor_scores (pd.DataFrame): Factor scores for each stock
            weights (dict): Factor weights to use
            
        Returns:
            pd.DataFrame: Weighted factor scores for each stock
        """
        try:
            if factor_scores is None or factor_scores.empty:
                logger.error("No factor scores provided")
                return None
                
            if weights is None:
                weights = self.factor_weights
                
            # Create a copy of the factor scores
            weighted_scores = factor_scores.copy()
            
            # Ensure all factor values are numeric
            for factor in self.factors:
                if factor in weighted_scores.columns:
                    # Convert to float to ensure numeric type
                    weighted_scores[factor] = pd.to_numeric(weighted_scores[factor], errors='coerce')
            
            # Add a column for the total score (initialized to 0.0)
            weighted_scores['total_score'] = 0.0
            
            # Calculate the weighted score
            for factor in self.factors:
                if factor in weights and factor in weighted_scores.columns:
                    # Ensure the contribution is numeric
                    weighted_contribution = weights[factor] * weighted_scores[factor]
                    weighted_scores['total_score'] += weighted_contribution
            
            # Ensure total_score is float
            weighted_scores['total_score'] = pd.to_numeric(weighted_scores['total_score'], errors='coerce')
            
            # Sort by total score (descending)
            weighted_scores = weighted_scores.sort_values('total_score', ascending=False)
            
            return weighted_scores
            
        except Exception as e:
            logger.error(f"Error calculating weighted scores: {str(e)}")
            return None
    
    def backtest_strategy(self, data, periods=5, months_per_period=6, top_pct=0.1, weights=None):
        """
        Backtest the investment strategy over multiple periods
        
        Args:
            data (pd.DataFrame): Market data
            periods (int): Number of backtest periods
            months_per_period (int): Length of each period in months
            top_pct (float): Percentage of top stocks to include in the portfolio
            weights (dict): Factor weights to use
            
        Returns:
            pd.DataFrame: Backtest results for each period
        """
        try:
            if data is None or data.empty:
                logger.error("No data provided for backtesting")
                return None
                
            if weights is None:
                weights = self.factor_weights
                
            # Make sure the index is sorted for time-based slicing
            if isinstance(data.index, pd.MultiIndex):
                data = data.sort_index()
                
            # Convert months_per_period to days
            period_days = months_per_period * 30  # Approximate
            
            # Get all dates in the data
            dates = sorted(data.index.get_level_values('Date').unique())
            
            if len(dates) < 2:
                logger.error("Not enough historical data for backtesting")
                return None
                
            # Determine the period start dates (going backward from the most recent date)
            latest_date = dates[-1]
            period_dates = []
            
            for i in range(periods + 1):  # We need periods + 1 dates to get periods backtest windows
                if i == 0:
                    period_dates.append(latest_date)
                else:
                    # Calculate the start of the period
                    period_start = latest_date - timedelta(days=period_days * i)
                    # Find the closest actual date in the data
                    closest_date = min(dates, key=lambda d: abs((d - period_start).total_seconds()))
                    period_dates.append(closest_date)
            
            # Reverse so we have oldest to newest
            period_dates.reverse()
            
            # Initialize results
            results = []
            
            print(f"\nBacktesting strategy over {periods} periods...")
            
            # For each period, run the strategy
            for i in range(len(period_dates) - 1):
                period_start = period_dates[i]
                period_end = period_dates[i + 1]
                
                print(f"\nPeriod {i+1}: {period_start.date()} to {period_end.date()}")
                
                # Get data for this period
                period_data = data.loc[period_start:period_end]
                
                # Calculate factor scores at the beginning of the period
                start_data = data.loc[:period_start]
                factor_scores = self.calculate_factor_scores(start_data)
                
                if factor_scores is None or factor_scores.empty:
                    print(f"  No factor scores available for period {i+1}, skipping")
                    continue
                
                # Calculate weighted scores
                weighted_scores = self.calculate_weighted_scores(factor_scores, weights)
                
                if weighted_scores is None or weighted_scores.empty:
                    print(f"  No weighted scores available for period {i+1}, skipping")
                    continue
                
                # Select top stocks based on weighted scores
                top_count = max(1, int(len(weighted_scores) * top_pct))
                top_stocks = weighted_scores.nlargest(top_count, 'total_score').index.tolist()
                
                print(f"  Selected {len(top_stocks)} stocks for portfolio")
                
                # Calculate portfolio return
                portfolio_return = self._calculate_period_return(period_data, top_stocks)
                
                # Calculate benchmark return (all stocks equally weighted)
                all_stocks = period_data.index.get_level_values('Ticker').unique()
                benchmark_return = self._calculate_period_return(period_data, all_stocks)
                
                # Store results
                results.append({
                    'period': i + 1,
                    'start_date': period_start,
                    'end_date': period_end,
                    'portfolio_return': portfolio_return,
                    'benchmark_return': benchmark_return,
                    'excess_return': portfolio_return - benchmark_return
                })
                
                print(f"  Portfolio Return: {portfolio_return*100:.2f}%")
                print(f"  Benchmark Return: {benchmark_return*100:.2f}%")
                print(f"  Excess Return: {(portfolio_return-benchmark_return)*100:.2f}%")
                
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Calculate average returns
            if not results_df.empty:
                avg_portfolio_return = results_df['portfolio_return'].mean()
                avg_benchmark_return = results_df['benchmark_return'].mean()
                avg_excess_return = results_df['excess_return'].mean()
                
                print("\nBacktest Summary:")
                print(f"Average Portfolio Return: {avg_portfolio_return*100:.2f}%")
                print(f"Average Benchmark Return: {avg_benchmark_return*100:.2f}%")
                print(f"Average Excess Return: {avg_excess_return*100:.2f}%")
                
                # Annualize returns assuming months_per_period
                annualized_portfolio = ((1 + avg_portfolio_return) ** (12 / months_per_period)) - 1
                annualized_benchmark = ((1 + avg_benchmark_return) ** (12 / months_per_period)) - 1
                annualized_excess = ((1 + avg_excess_return) ** (12 / months_per_period)) - 1
                
                print(f"Annualized Portfolio Return: {annualized_portfolio*100:.2f}%")
                print(f"Annualized Benchmark Return: {annualized_benchmark*100:.2f}%")
                print(f"Annualized Excess Return: {annualized_excess*100:.2f}%")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error in backtest_strategy: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _calculate_period_return(self, period_data, tickers):
        """
        Calculate the return for a set of tickers over a period
        
        Args:
            period_data (pd.DataFrame): Market data for the period
            tickers (list): List of tickers to include in the calculation
            
        Returns:
            float: Portfolio return
        """
        try:
            if not tickers:
                return 0.0
                
            # Filter data for the selected tickers
            ticker_mask = period_data.index.get_level_values('Ticker').isin(tickers)
            tickers_data = period_data[ticker_mask]
            
            # Get the first and last date in the period
            dates = sorted(tickers_data.index.get_level_values('Date').unique())
            if len(dates) < 2:
                return 0.0
                
            start_date = dates[0]
            end_date = dates[-1]
            
            # Calculate returns for each ticker
            returns = []
            
            for ticker in tickers:
                try:
                    # Get the ticker's data
                    try:
                        ticker_data = tickers_data.xs(ticker, level='Ticker')
                    except KeyError:
                        # Ticker not found in the data
                        continue
                    
                    if ticker_data.empty:
                        continue
                        
                    # Get prices at the start and end of the period
                    price_col = 'Adj Close' if 'Adj Close' in ticker_data.columns else 'Close'
                    if price_col not in ticker_data.columns:
                        continue
                    
                    # Get ticker dates
                    ticker_dates = ticker_data.index.tolist()
                    
                    # Find the closest dates to start_date and end_date
                    start_idx = None
                    end_idx = None
                    
                    # Find the first date >= start_date
                    for i, date in enumerate(ticker_dates):
                        if date >= start_date:
                            start_idx = i
                            break
                            
                    # Find the last date <= end_date
                    for i in range(len(ticker_dates)-1, -1, -1):
                        if ticker_dates[i] <= end_date:
                            end_idx = i
                            break
                    
                    # Skip if we couldn't find valid indices
                    if start_idx is None or end_idx is None or start_idx > end_idx:
                        continue
                    
                    # Get start and end prices
                    start_price = ticker_data.iloc[start_idx][price_col]
                    end_price = ticker_data.iloc[end_idx][price_col]
                    
                    # Calculate return
                    if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                        ticker_return = (end_price / start_price) - 1
                        returns.append(ticker_return)
                    
                except Exception as e:
                    logger.debug(f"Error calculating return for {ticker}: {str(e)}")
                    continue
                    
            # Equal-weighted portfolio return
            if returns:
                return np.mean(returns)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating period return: {str(e)}")
            return 0.0

if __name__ == "__main__":
    # Example usage
    from data_collector import MarketDataCollector
    
    collector = MarketDataCollector()
    data = collector.get_market_data()
    metrics = collector.calculate_metrics(data)
    
    analyzer = FactorAnalyzer()
    results = analyzer.backtest_strategy(metrics)
    performance = analyzer.calculate_performance_metrics(results['portfolio_returns']) 