import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import time
import logging
import pandas_datareader.data as web

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataCollector:
    """Class to collect market data and calculate financial metrics"""
    
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        self.sp500_tickers = []
        self.data_cache = {}  # Cache for storing retrieved data
    
    def get_market_data(self, lookback_years=5, max_stocks=100):
        """
        Collect market data for stocks in the S&P 500
        
        Args:
            lookback_years (int): Number of years to look back for data
            max_stocks (int): Maximum number of stocks to analyze
            
        Returns:
            pandas.DataFrame: Market data for S&P 500 stocks
        """
        # Get S&P 500 tickers
        print("Fetching S&P 500 tickers from Wikipedia...")
        self.sp500_tickers = self._get_sp500_tickers()
        
        if self.test_mode:
            # Use a smaller subset in test mode
            self.sp500_tickers = self.sp500_tickers[:20]
            print(f"Test mode: Using {len(self.sp500_tickers)} stocks")
        elif max_stocks and max_stocks < len(self.sp500_tickers):
            # Limit to top market cap stocks if requested
            top_tickers = self.get_top_market_cap_stocks(max_stocks)
            self.sp500_tickers = top_tickers
            print(f"Using top {len(self.sp500_tickers)} stocks by market cap")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * lookback_years)
        
        # Create a cache directory if it doesn't exist
        os.makedirs('data_cache', exist_ok=True)
        cache_file = f'data_cache/market_data_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.pkl'
        
        # Check if data is already cached
        if os.path.exists(cache_file):
            print(f"Loading market data from cache: {cache_file}")
            return pd.read_pickle(cache_file)
        
        # Get stock price data
        all_data = []
        
        for i, ticker in enumerate(self.sp500_tickers):
            try:
                print(f"Getting data for {ticker} ({i+1}/{len(self.sp500_tickers)})...")
                
                # Get stock data directly from yfinance
                ticker_obj = yf.Ticker(ticker)
                stock_data = ticker_obj.history(period=f"{lookback_years}y")
                
                if stock_data.empty:
                    print(f"No data found for {ticker}, skipping...")
                    continue
                
                # Add a 'Ticker' column before resetting index to preserve it
                stock_data = stock_data.assign(Ticker=ticker)
                
                # Reset index to have Date as a column
                stock_data = stock_data.reset_index()
                
                # Get company info from Yahoo Finance
                try:
                    info = ticker_obj.info
                    
                    # Add market cap and sector
                    if 'marketCap' in info:
                        stock_data['MarketCap'] = info.get('marketCap', np.nan)
                    else:
                        stock_data['MarketCap'] = np.nan
                        
                    if 'sector' in info:
                        stock_data['Sector'] = info.get('sector', 'Unknown')
                    else:
                        stock_data['Sector'] = 'Unknown'
                        
                    # Add fundamental data if available
                    for key in ['trailingPE', 'forwardPE', 'priceToBook', 'profitMargins', 
                                'returnOnAssets', 'returnOnEquity', 'debtToEquity',
                                'trailingEps', 'forwardEps', 'bookValue']:
                        if key in info:
                            stock_data[key] = info.get(key, np.nan)
                    
                    # Quarterly financial data
                    try:
                        financials = ticker_obj.quarterly_financials
                        if not financials.empty:
                            latest_quarter = financials.columns[0]
                            stock_data['Revenues'] = financials.loc['Total Revenue', latest_quarter] if 'Total Revenue' in financials.index else np.nan
                            stock_data['NetIncome'] = financials.loc['Net Income', latest_quarter] if 'Net Income' in financials.index else np.nan
                            stock_data['OperatingIncome'] = financials.loc['Operating Income', latest_quarter] if 'Operating Income' in financials.index else np.nan
                    except Exception as e:
                        logger.warning(f"Error getting financials for {ticker}: {str(e)}")
                    
                    # Balance sheet data
                    try:
                        balance_sheet = ticker_obj.quarterly_balance_sheet
                        if not balance_sheet.empty:
                            latest_quarter = balance_sheet.columns[0]
                            stock_data['TotalAssets'] = balance_sheet.loc['Total Assets', latest_quarter] if 'Total Assets' in balance_sheet.index else np.nan
                            stock_data['TotalLiabilities'] = balance_sheet.loc['Total Liabilities Net Minority Interest', latest_quarter] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else np.nan
                            stock_data['TotalEquity'] = balance_sheet.loc['Total Equity Gross Minority Interest', latest_quarter] if 'Total Equity Gross Minority Interest' in balance_sheet.index else np.nan
                    except Exception as e:
                        logger.warning(f"Error getting balance sheet for {ticker}: {str(e)}")
                    
                except Exception as e:
                    logger.warning(f"Error getting additional info for {ticker}: {str(e)}")
                
                all_data.append(stock_data)
                
                # Don't overload the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("Failed to retrieve any valid market data.")
            return pd.DataFrame()
        
        # Combine all data
        market_data = pd.concat(all_data, ignore_index=True)
        
        # Debug output
        print(f"Market data columns: {market_data.columns.tolist()}")
        print(f"First few rows of market data:\n{market_data.head()}")
        
        # Set index to MultiIndex (Date, Ticker) for easier filtering
        if 'Date' in market_data.columns and 'Ticker' in market_data.columns:
            market_data = market_data.set_index(['Date', 'Ticker'])
        else:
            logger.error("Missing required columns 'Date' or 'Ticker' in market data")
            if 'Date' not in market_data.columns:
                logger.error("'Date' column is missing in market_data")
            if 'Ticker' not in market_data.columns:
                logger.error("'Ticker' column is missing in market_data")
            # Print column names for debugging
            logger.error(f"Available columns: {market_data.columns.tolist()}")
            return pd.DataFrame()
        
        # Cache the data for future use
        market_data.to_pickle(cache_file)
        
        return market_data

    def get_specified_stocks(self, tickers, lookback_years=5):
        """
        Collect market data for specific stocks
        
        Args:
            tickers (list): List of stock tickers to collect data for
            lookback_years (int): Number of years to look back for data
            
        Returns:
            pandas.DataFrame: Market data for specified stocks
        """
        print(f"Getting data for {len(tickers)} specified stocks...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * lookback_years)
        
        # Create a cache key
        tickers_str = '-'.join(sorted(tickers))
        cache_key = f"{hash(tickers_str)}_{lookback_years}y"
        
        # Create a cache directory if it doesn't exist
        os.makedirs('data_cache', exist_ok=True)
        cache_file = f'data_cache/specified_data_{cache_key}.pkl'
        
        # Check if data is already cached and less than 7 days old
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            if (time.time() - cache_time) < 7 * 86400:  # 7 days in seconds
                print(f"Loading specified stocks data from cache (created {(time.time() - cache_time) / 86400:.1f} days ago)")
                return pd.read_pickle(cache_file)
        
        # Get stock price data
        all_data = []
        
        for i, ticker in enumerate(tickers):
            try:
                print(f"Getting data for {ticker} ({i+1}/{len(tickers)})...")
                
                # Get stock data directly from yfinance
                ticker_obj = yf.Ticker(ticker)
                stock_data = ticker_obj.history(period=f"{lookback_years}y")
                
                if stock_data.empty:
                    print(f"No data found for {ticker}, skipping...")
                    continue
                
                # Add a 'Ticker' column before resetting index to preserve it
                stock_data = stock_data.assign(Ticker=ticker)
                
                # Reset index to have Date as a column
                stock_data = stock_data.reset_index()
                
                # Get company info from Yahoo Finance
                try:
                    info = ticker_obj.info
                    
                    # Add market cap and sector
                    if 'marketCap' in info:
                        stock_data['MarketCap'] = info.get('marketCap', np.nan)
                    else:
                        stock_data['MarketCap'] = np.nan
                        
                    if 'sector' in info:
                        stock_data['Sector'] = info.get('sector', 'Unknown')
                    else:
                        stock_data['Sector'] = 'Unknown'
                        
                    # Add fundamental data if available
                    for key in ['trailingPE', 'forwardPE', 'priceToBook', 'profitMargins', 
                                'returnOnAssets', 'returnOnEquity', 'debtToEquity',
                                'trailingEps', 'forwardEps', 'bookValue']:
                        if key in info:
                            stock_data[key] = info.get(key, np.nan)
                    
                    # Quarterly financial data
                    try:
                        financials = ticker_obj.quarterly_financials
                        if not financials.empty:
                            latest_quarter = financials.columns[0]
                            stock_data['Revenues'] = financials.loc['Total Revenue', latest_quarter] if 'Total Revenue' in financials.index else np.nan
                            stock_data['NetIncome'] = financials.loc['Net Income', latest_quarter] if 'Net Income' in financials.index else np.nan
                            stock_data['OperatingIncome'] = financials.loc['Operating Income', latest_quarter] if 'Operating Income' in financials.index else np.nan
                    except Exception as e:
                        logger.warning(f"Error getting financials for {ticker}: {str(e)}")
                    
                    # Balance sheet data
                    try:
                        balance_sheet = ticker_obj.quarterly_balance_sheet
                        if not balance_sheet.empty:
                            latest_quarter = balance_sheet.columns[0]
                            stock_data['TotalAssets'] = balance_sheet.loc['Total Assets', latest_quarter] if 'Total Assets' in balance_sheet.index else np.nan
                            stock_data['TotalLiabilities'] = balance_sheet.loc['Total Liabilities Net Minority Interest', latest_quarter] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else np.nan
                            stock_data['TotalEquity'] = balance_sheet.loc['Total Equity Gross Minority Interest', latest_quarter] if 'Total Equity Gross Minority Interest' in balance_sheet.index else np.nan
                    except Exception as e:
                        logger.warning(f"Error getting balance sheet for {ticker}: {str(e)}")
                    
                except Exception as e:
                    logger.warning(f"Error getting additional info for {ticker}: {str(e)}")
                
                all_data.append(stock_data)
                
                # Don't overload the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("Failed to retrieve any valid market data.")
            return pd.DataFrame()
        
        # Combine all data
        market_data = pd.concat(all_data, ignore_index=True)
        
        # Debug output
        print(f"Market data columns: {market_data.columns.tolist()}")
        print(f"First few rows of market data:\n{market_data.head()}")
        
        # Set index to MultiIndex (Date, Ticker) for easier filtering
        if 'Date' in market_data.columns and 'Ticker' in market_data.columns:
            market_data = market_data.set_index(['Date', 'Ticker'])
        else:
            logger.error("Missing required columns 'Date' or 'Ticker' in market data")
            if 'Date' not in market_data.columns:
                logger.error("'Date' column is missing in market_data")
            if 'Ticker' not in market_data.columns:
                logger.error("'Ticker' column is missing in market_data")
            # Print column names for debugging
            logger.error(f"Available columns: {market_data.columns.tolist()}")
            return pd.DataFrame()
        
        # Cache the data for future use
        try:
            market_data.to_pickle(cache_file)
            print(f"Saved specified stocks data to cache: {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save market data to cache: {str(e)}")
        
        return market_data
    
    def get_top_market_cap_stocks(self, n=100):
        """
        Get the top n stocks by market cap
        
        Args:
            n (int): Number of top stocks to return
            
        Returns:
            list: List of top n stock tickers by market cap
        """
        try:
            print(f"Getting top {n} stocks by market cap...")
            
            # If we already have S&P 500 tickers, get them, otherwise fetch from Wikipedia
            if not self.sp500_tickers:
                self.sp500_tickers = self._get_sp500_tickers()
            
            # Create a cache directory if it doesn't exist
            os.makedirs('data_cache', exist_ok=True)
            cache_file = f'data_cache/top_market_cap_{n}.pkl'
            
            # Check if data is already cached and less than 1 day old
            if os.path.exists(cache_file):
                cache_time = os.path.getmtime(cache_file)
                if (time.time() - cache_time) < 86400:  # 1 day in seconds
                    print(f"Loading top market cap stocks from cache: {cache_file}")
                    return pd.read_pickle(cache_file)
            
            # Get market cap data for all tickers
            market_caps = {}
            for i, ticker in enumerate(self.sp500_tickers):
                try:
                    print(f"Getting market cap for {ticker} ({i+1}/{len(self.sp500_tickers)})...")
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    if 'marketCap' in info and info['marketCap'] is not None:
                        market_caps[ticker] = info['marketCap']
                    time.sleep(0.5)  # Don't overload the API
                except Exception as e:
                    logger.warning(f"Error getting market cap for {ticker}: {str(e)}")
                    continue
            
            # Sort by market cap and get top n
            sorted_tickers = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
            top_tickers = [t[0] for t in sorted_tickers[:n]]
            
            # Cache the result
            pd.to_pickle(top_tickers, cache_file)
            
            return top_tickers
            
        except Exception as e:
            logger.error(f"Error getting top market cap stocks: {str(e)}")
            # Return a small subset of S&P 500 if we fail
            if self.sp500_tickers:
                return self.sp500_tickers[:n]
            return []
    
    def calculate_metrics(self, market_data):
        """
        Calculate financial metrics from market data
        
        Args:
            market_data (pandas.DataFrame): Market data for stocks
            
        Returns:
            pandas.DataFrame: Financial metrics for stocks
        """
        print("Calculating financial metrics...")
        
        # Create a copy to avoid modifying the original
        metrics = market_data.copy()
        
        # Convert price data columns to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in metrics.columns:
                metrics[col] = pd.to_numeric(metrics[col], errors='coerce')
        
        # Reset index for calculation
        metrics = metrics.reset_index()
        
        # Group by ticker and date
        grouped = metrics.groupby('Ticker')
        
        all_metrics = []
        
        for ticker, group in grouped:
            # Sort by date
            group = group.sort_values('Date')
            
            # Calculate basic metrics
            try:
                # Daily returns
                group['DailyReturn'] = group['Adj Close'].pct_change()
                
                # 20-day moving average (roughly 1 month of trading days)
                group['MA20'] = group['Adj Close'].rolling(window=20).mean()
                
                # 50-day moving average
                group['MA50'] = group['Adj Close'].rolling(window=50).mean()
                
                # 200-day moving average
                group['MA200'] = group['Adj Close'].rolling(window=200).mean()
                
                # Relative Strength Index (RSI) - 14-day
                delta = group['Adj Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                group['RSI'] = 100 - (100 / (1 + rs))
                
                # Volatility (20-day standard deviation of returns)
                group['Volatility'] = group['DailyReturn'].rolling(window=20).std() * np.sqrt(252)  # Annualized
                
                # Price/Earnings Ratio
                if 'trailingEps' in group.columns:
                    group['P/E'] = group['Adj Close'] / group['trailingEps']
                
                # Forward P/E
                if 'forwardEps' in group.columns:
                    group['ForwardP/E'] = group['Adj Close'] / group['forwardEps']
                
                # Price to Book Ratio
                if 'bookValue' in group.columns:
                    group['P/B'] = group['Adj Close'] / group['bookValue']
                
                # Debt to Equity Ratio
                if 'debtToEquity' in group.columns:
                    group['Debt/Equity'] = group['debtToEquity']
                
                # Return on Equity
                if 'returnOnEquity' in group.columns:
                    group['ROE'] = group['returnOnEquity']
                
                # Return on Assets
                if 'returnOnAssets' in group.columns:
                    group['ROA'] = group['returnOnAssets']
                
                # Profit Margin
                if 'profitMargins' in group.columns:
                    group['ProfitMargin'] = group['profitMargins']
                
                # 1-month momentum (21 trading days)
                group['Momentum1M'] = group['Adj Close'].pct_change(periods=21)
                
                # 3-month momentum (63 trading days)
                group['Momentum3M'] = group['Adj Close'].pct_change(periods=63)
                
                # 6-month momentum (126 trading days)
                group['Momentum6M'] = group['Adj Close'].pct_change(periods=126)
                
                # 12-month momentum (252 trading days)
                group['Momentum12M'] = group['Adj Close'].pct_change(periods=252)
                
                # Trend indicator (MA50 > MA200)
                group['TrendIndicator'] = (group['MA50'] > group['MA200']).astype(int)
                
                # Add fundamental metrics if available
                # Calculate EPS growth if possible
                if 'trailingEps' in group.columns and 'forwardEps' in group.columns:
                    mask = (group['trailingEps'] > 0) & (group['forwardEps'] > 0)
                    group.loc[mask, 'EPSGrowth'] = (group.loc[mask, 'forwardEps'] / group.loc[mask, 'trailingEps']) - 1
                
                # Add to results
                all_metrics.append(group)
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {ticker}: {str(e)}")
                continue
        
        if not all_metrics:
            logger.error("Failed to calculate metrics for any stocks.")
            return pd.DataFrame()
        
        # Combine all metrics
        metrics_df = pd.concat(all_metrics)
        
        # Set index back to (Date, Ticker)
        metrics_df = metrics_df.set_index(['Date', 'Ticker'])
        
        return metrics_df
    
    def _get_sp500_tickers(self):
        """
        Get the list of S&P 500 tickers from Wikipedia
        
        Returns:
            list: List of S&P 500 stock tickers
        """
        try:
            # Create a cache directory if it doesn't exist
            os.makedirs('data_cache', exist_ok=True)
            cache_file = 'data_cache/sp500_tickers.pkl'
            
            # Check if data is already cached and less than 1 day old
            if os.path.exists(cache_file):
                cache_time = os.path.getmtime(cache_file)
                if (time.time() - cache_time) < 86400:  # 1 day in seconds
                    return pd.read_pickle(cache_file)
            
            # Get S&P 500 table from Wikipedia
            resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            soup = BeautifulSoup(resp.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            
            tickers = []
            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text.strip()
                tickers.append(ticker)
            
            # Clean tickers (remove .XX suffixes)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            # Cache the result
            pd.to_pickle(tickers, cache_file)
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting S&P 500 tickers: {str(e)}")
            # Return a small subset of common stocks if we fail
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG']

    def get_sp500_stocks(self, lookback_years=5):
        """
        Get historical data for S&P 500 stocks
        
        Args:
            lookback_years: Number of years of historical data to collect
            
        Returns:
            DataFrame with historical market data for S&P 500 stocks
        """
        # Create a cache directory if it doesn't exist
        os.makedirs('data_cache', exist_ok=True)
        cache_file = f'data_cache/sp500_data_{lookback_years}y.pkl'
        
        # Check if data is already cached and less than 7 days old
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            if (time.time() - cache_time) < 7 * 86400:  # 7 days in seconds
                print(f"Loading SP500 market data from cache (created {(time.time() - cache_time) / 86400:.1f} days ago)")
                return pd.read_pickle(cache_file)
        
        # Try to get S&P 500 constituent list
        try:
            print("Fetching S&P 500 constituents...")
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            sp500_table = pd.read_html(sp500_url)
            sp500_tickers = sp500_table[0]['Symbol'].tolist()
            
            # Clean up tickers
            sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_tickers]
            print(f"Found {len(sp500_tickers)} S&P 500 stocks")
            
        except Exception as e:
            print(f"Error fetching S&P 500 constituents: {str(e)}")
            print("Using fallback list of major stocks...")
            
            # Fallback to a subset of large cap stocks if Wikipedia fetch fails
            from main import DOW_JONES_TICKERS
            sp500_tickers = DOW_JONES_TICKERS + [
                "AMZN", "GOOGL", "META", "NFLX", "TSLA", "NVDA", "AMD", "PYPL",
                "ADBE", "COST", "CMCSA", "PEP", "AVGO", "TXN", "QCOM", "GILD",
                "MDLZ", "CHTR", "SBUX", "BKNG", "INTU", "ISRG"
            ]
        
        # Limit tickers for test mode
        if self.test_mode:
            sp500_tickers = sp500_tickers[:50]  # Use only 50 stocks for test mode
            print(f"Test mode: Limited to {len(sp500_tickers)} stocks")
            
        # Set the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(365.25 * lookback_years))
        
        # Initialize empty DataFrame for market data
        market_data = pd.DataFrame()
        
        # Use batch processing to speed up data collection
        batch_size = 20
        num_batches = (len(sp500_tickers) + batch_size - 1) // batch_size
        
        print(f"Collecting data for {len(sp500_tickers)} stocks in {num_batches} batches...")
        
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(sp500_tickers))
            batch_tickers = sp500_tickers[batch_start:batch_end]
            
            print(f"Processing batch {i+1}/{num_batches}: {batch_start+1}-{batch_end} of {len(sp500_tickers)}")
            
            try:
                # Fetch batch data
                batch_data = yf.download(
                    " ".join(batch_tickers),
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False
                )
                
                # Process each ticker in the batch
                for ticker in batch_tickers:
                    try:
                        if isinstance(batch_data, pd.DataFrame):
                            if len(batch_tickers) == 1:
                                # If only one ticker, yfinance returns a different format
                                ticker_data = batch_data.copy()
                                # Add ticker column
                                ticker_data['Ticker'] = ticker
                            else:
                                # Extract data for this ticker
                                ticker_data = batch_data[ticker].copy()
                                ticker_data['Ticker'] = ticker
                        else:
                            # Skip if no data
                            continue
                        
                        # Reset index to make Date a column
                        ticker_data = ticker_data.reset_index()
                        
                        # Set MultiIndex with (Date, Ticker)
                        ticker_data = ticker_data.set_index(['Date', 'Ticker'])
                        
                        # Concatenate to main DataFrame
                        if market_data.empty:
                            market_data = ticker_data
                        else:
                            market_data = pd.concat([market_data, ticker_data])
                    except Exception as e:
                        print(f"Error processing {ticker}: {str(e)}")
                
                # Add a small delay to avoid hitting API limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching batch {i+1}: {str(e)}")
                time.sleep(2)  # Longer delay after an error
        
        print(f"Collected data for {len(market_data.index.get_level_values('Ticker').unique())} stocks")
            
        # Get market cap for each stock
        self._add_market_cap_info(market_data)
        
        # Cache the data for future use
        try:
            market_data.to_pickle(cache_file)
            print(f"Saved S&P 500 market data to cache: {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save market data to cache: {str(e)}")
        
        return market_data

    def _add_market_cap_info(self, market_data):
        """
        Add market cap information to the market data DataFrame
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            DataFrame with market cap info added
        """
        # Check if MarketCap column already exists and has data
        if 'MarketCap' in market_data.columns and market_data['MarketCap'].notna().any():
            print("Market cap data already exists in DataFrame, skipping fetch")
            return market_data
        
        import yfinance as yf
        
        try:
            # Get unique tickers from the market data
            tickers = market_data.index.get_level_values('Ticker').unique()
            
            print(f"Fetching market cap data for {len(tickers)} stocks...")
            
            # Create a market cap cache file
            os.makedirs('data_cache', exist_ok=True)
            mcap_cache_file = f'data_cache/market_cap_data_{datetime.now().strftime("%Y%m%d")}.pkl'
            market_caps = {}
            
            # Check if we have recent market cap data cached
            if os.path.exists(mcap_cache_file):
                try:
                    market_caps = pd.read_pickle(mcap_cache_file)
                    print(f"Loaded market cap data from cache for {len(market_caps)} stocks")
                except Exception as e:
                    print(f"Could not load market cap cache: {e}")
                    market_caps = {}
            
            # Process in batches to avoid API limits
            batch_size = 20
            num_batches = (len(tickers) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(tickers))
                batch_tickers = tickers[batch_start:batch_end]
                
                if self.test_mode and i > 2:  # Limit API calls in test mode
                    print(f"Test mode: Skipping remaining market cap data after {batch_end} stocks")
                    break
                
                for ticker in batch_tickers:
                    # Skip if we already have market cap for this ticker
                    if ticker in market_caps:
                        continue
                    
                    try:
                        # Get ticker info from yfinance
                        ticker_obj = yf.Ticker(ticker)
                        ticker_info = ticker_obj.info
                        
                        # Extract market cap
                        if 'marketCap' in ticker_info and ticker_info['marketCap'] is not None:
                            market_caps[ticker] = ticker_info['marketCap']
                    except Exception as e:
                        print(f"Warning: Could not get market cap for {ticker}: {str(e)}")
                
                # Small delay to avoid hitting API limits
                time.sleep(0.5)
            
            # Save the market cap data for future use
            try:
                pd.to_pickle(market_caps, mcap_cache_file)
                print(f"Saved market cap data cache with {len(market_caps)} entries")
            except Exception as e:
                print(f"Warning: Could not save market cap cache: {str(e)}")
            
            # Add market cap data to the DataFrame
            if 'MarketCap' not in market_data.columns:
                market_data['MarketCap'] = None
            
            for ticker, mcap in market_caps.items():
                ticker_indices = market_data.index.get_level_values('Ticker') == ticker
                market_data.loc[ticker_indices, 'MarketCap'] = mcap
            
            print("Market cap data added to DataFrame")
            
        except Exception as e:
            print(f"Error fetching market cap data: {str(e)}")
            print("Continuing without market cap information")
        
        return market_data

if __name__ == "__main__":
    # Test the collector with a small subset of stocks
    collector = MarketDataCollector(test_mode=True)
    market_data = collector.get_market_data(lookback_years=2)  # Reduced lookback period for testing
    metrics = collector.calculate_metrics(market_data)
    print(f"Collected data for {len(metrics['Ticker'].unique())} stocks") 