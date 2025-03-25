# Druck Equity Model

A quantitative equity portfolio optimization model inspired by Stanley Druckenmiller's investment philosophy, combining factor analysis, fundamental research, and technical indicators to discover alpha-generating opportunities.

## Overview

The Druck Model implements a sophisticated multi-factor approach to equity portfolio construction:

- **Factor Analysis**: Evaluates momentum, value, quality, growth, and volatility factors
- **Technical Analysis**: Incorporates market regime detection and relative strength
- **Fundamental Research**: Analyzes management quality, financial statements, and sector dynamics
- **Portfolio Optimization**: Constructs efficient portfolios optimized for Sharpe ratio and risk-adjusted returns

## Features

- Dynamic stock universe selection (Dow Jones, S&P 500, sector-specific portfolios)
- Interactive dashboard for portfolio visualization and analysis
- Multi-period backtesting across various market conditions
- Data caching system for efficient model execution
- Customizable factor weights and optimization parameters

## Installation

### Prerequisites

- Python 3.9+
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/HolisticEquityModel.git
cd HolisticEquityModel
```

2. Create and activate a virtual environment (recommended):

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Execution

Run the model with default settings (Dow Jones universe):

**macOS/Linux:**
```bash
python main.py
```

**Windows:**
```powershell
python main.py
```

### Command Line Options

The model supports various command-line arguments for customization:

```bash
python main.py --universe sp500 --stocks 100 --years 5 --test
```

Available options:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--universe` | `-u` | Stock universe (dow, sp500, software, tmt, ai, semi, biotech, all) | dow |
| `--stocks` | `-s` | Number of stocks to analyze | 30 |
| `--years` | `-y` | Lookback period in years | 5 |
| `--test` | `-t` | Run in test mode with smaller dataset | False |
| `--optimize-periods` | `-op` | Number of optimization periods | 3 |
| `--backtest-periods` | `-bp` | Number of backtest periods | 5 |
| `--reuse-data` | `-r` | Reuse the last batch of collected market data | False |

### Example Use Cases

#### Quick Test Run

Test the model quickly with a small dataset:

```bash
python main.py --test --universe dow
```

#### Full S&P 500 Analysis

Run a comprehensive analysis on the S&P 500:

```bash
python main.py --universe sp500 --stocks 100 --years 3
```

#### Technology Sector Focus

Analyze technology, media, and telecom stocks:

```bash
python main.py --universe tmt --stocks 20
```

#### Performance Optimization

After initial data collection, reuse data to test different parameters:

```bash
python main.py --universe sp500 --reuse-data --optimize-periods 5
```

### Dashboard Generation

Generate the dashboard without running the full model:

```bash
python generate_dashboard.py
```

## Output

The model generates various outputs in the `reports` directory:

- Interactive dashboard (HTML) with portfolio metrics and visualizations
- Portfolio allocation and weights
- Factor analysis and exposure
- Optimization results and efficient frontier
- Backtest performance statistics

## Model Structure

- `main.py`: Entry point and primary logic coordination
- `data_collector.py`: Market data acquisition and processing
- `factor_analysis.py`: Factor scoring and analysis
- `portfolio_optimizer.py`: Portfolio optimization and construction
- `management_analyzer.py`: Management quality evaluation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Stanley Druckenmiller's investment philosophy and approach
- Implements academic research on factor investing and quantitative portfolio management 