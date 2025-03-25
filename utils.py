"""
Utility functions for the Druck Equity Model
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logger
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the project"""
    os.makedirs('reports', exist_ok=True)
    os.makedirs('data_cache', exist_ok=True)
    return True

def format_percentage(value, decimal_places=2):
    """Format a decimal as a percentage string"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value * 100:.{decimal_places}f}%"

def format_currency(value, currency="$", decimal_places=2):
    """Format a value as a currency string"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{currency}{value:,.{decimal_places}f}"

def format_decimal(value, decimal_places=2):
    """Format a decimal value with specified decimal places"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:.{decimal_places}f}"

def format_large_number(value):
    """Format large numbers with K, M, B, T suffixes"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    abs_value = abs(value)
    sign = -1 if value < 0 else 1
    
    if abs_value < 1000:
        return f"{value:.2f}"
    elif abs_value < 1000000:
        return f"{sign * abs_value / 1000:.2f}K"
    elif abs_value < 1000000000:
        return f"{sign * abs_value / 1000000:.2f}M"
    elif abs_value < 1000000000000:
        return f"{sign * abs_value / 1000000000:.2f}B"
    else:
        return f"{sign * abs_value / 1000000000000:.2f}T"

def calculate_returns(prices, period=1):
    """Calculate returns over specified period"""
    return prices.pct_change(period).dropna()

def calculate_rolling_volatility(returns, window=20):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def calculate_drawdowns(prices):
    """Calculate drawdowns from price series"""
    # Calculate the cumulative maximum
    rolling_max = prices.cummax()
    # Calculate drawdowns
    drawdowns = prices / rolling_max - 1.0
    return drawdowns

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    """Calculate Sortino ratio"""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    downside_returns = returns[returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    if downside_deviation == 0:
        return np.nan
    
    return np.sqrt(252) * excess_returns.mean() / downside_deviation

def calculate_maximum_drawdown(prices):
    """Calculate maximum drawdown"""
    drawdowns = calculate_drawdowns(prices)
    return drawdowns.min()

def set_plotting_style():
    """Set consistent plotting style for visualizations"""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    
def save_dataframe_to_csv(df, filename, directory="reports"):
    """Save DataFrame to CSV file"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath)
    logger.info(f"Saved DataFrame to {filepath}")
    return filepath

def read_dataframe_from_csv(filename, directory="reports"):
    """Read DataFrame from CSV file"""
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded DataFrame from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error reading {filepath}: {str(e)}")
        return None

def get_timestamp_string():
    """Get a timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory 