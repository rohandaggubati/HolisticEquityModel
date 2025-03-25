"""
Constants used throughout the equity model
"""

# Ticker lists for different universes
DOW_JONES_TICKERS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", 
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", 
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]

SOFTWARE_TICKERS = [
    "ADBE", "ADSK", "ANSS", "AZPN", "CDNS", "CRM", "CTXS", "INTU", 
    "MSFT", "NOW", "ORCL", "PANW", "PAYC", "SSNC", "TEAM", "WDAY"
]

TMT_TICKERS = [
    "AAPL", "ADBE", "AMD", "AVGO", "CRM", "CSCO", "DIS", "FB", "GOOG", 
    "GOOGL", "IBM", "INTC", "META", "MSFT", "NFLX", "NVDA", "ORCL", "QCOM", 
    "T", "TMUS", "TSLA", "VZ"
]

AI_TICKERS = [
    "AAPL", "AMD", "AMZN", "GOOG", "GOOGL", "IBM", "META", "MSFT", 
    "NVDA", "ORCL", "CRM", "PLTR", "TSLA", "U", "AI", "SNPS"
]

SEMICONDUCTOR_TICKERS = [
    "AMD", "AVGO", "INTC", "MCHP", "MU", "NVDA", "QCOM", "TXN", "TSM", "AMAT",
    "ASML", "LRCX", "ADI", "SWKS", "NXPI", "ON"
]

BIOTECH_TICKERS = [
    "AMGN", "BIIB", "GILD", "ILMN", "REGN", "VRTX", "MRNA", "INCY", 
    "BMRN", "ALNY", "SGEN", "TECH", "IONS", "JAZZ", "NBIX", "RARE"
]

# Universe mapping for easy lookup
UNIVERSE_MAPPING = {
    "DOW": {"name": "Dow Jones Industrial Average", "tickers": DOW_JONES_TICKERS},
    "SP500": {"name": "S&P 500", "tickers": None},  # Will be fetched dynamically
    "SOFTWARE": {"name": "Software Companies", "tickers": SOFTWARE_TICKERS},
    "TMT": {"name": "Technology, Media & Telecom", "tickers": TMT_TICKERS},
    "AI": {"name": "Artificial Intelligence", "tickers": AI_TICKERS},
    "SEMI": {"name": "Semiconductor", "tickers": SEMICONDUCTOR_TICKERS},
    "BIOTECH": {"name": "Biotechnology", "tickers": BIOTECH_TICKERS}
} 