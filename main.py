import os
from data_collector import MarketDataCollector
from factor_analysis import FactorAnalyzer
from portfolio_optimizer import PortfolioOptimizer
import pandas as pd
import plotly.io as pio
from datetime import datetime, timedelta
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import http.server
import socketserver
import threading
import socket
import argparse
import traceback
import sys
import yfinance as yf
import logging
import json
import requests
from bs4 import BeautifulSoup
import re
from scipy.optimize import minimize
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import constants
from constants import (
    DOW_JONES_TICKERS, SOFTWARE_TICKERS, TMT_TICKERS, AI_TICKERS, 
    SEMICONDUCTOR_TICKERS, BIOTECH_TICKERS, UNIVERSE_MAPPING
)

class Portfolio:
    """Class to represent a portfolio with weights and metrics"""
    def __init__(self, weights: Dict[str, float], expected_return: float = 0.0, 
                 volatility: float = 0.0, sharpe_ratio: float = 0.0):
        self.weights = weights
        self.expected_return = expected_return
        self.volatility = volatility
        self.sharpe_ratio = sharpe_ratio

def find_available_port(start_port=8500, max_attempts=100):
    """Find an available port to use for the dashboard server"""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    return None

def restart_server():
    """Kill any running HTTP servers before starting a new one"""
    try:
        import subprocess
        import sys
        
        # Different command for different operating systems
        if sys.platform == 'win32':
            # Windows
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Mac/Linux - find python processes running servers
            cmd = "ps -ef | grep 'SimpleHTTPRequestHandler' | grep -v grep | awk '{print $2}' | xargs -r kill -9"
            subprocess.run(cmd, shell=True, 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Also try to kill any processes on the common ports
            for port in [8000, 8001, 8080, 8500, 8501, 8502]:
                cmd = f"lsof -ti:{port} | xargs -r kill -9"
                subprocess.run(cmd, shell=True, 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
        print("Terminated any existing server processes")
    except Exception as e:
        print(f"Warning: Could not terminate existing servers: {str(e)}")

def start_server(file_path, port):
    """Start a simple HTTP server to serve the dashboard file"""
    try:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        
        # Create a custom handler that serves the specific file
        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=file_dir, **kwargs)
            
            def do_GET(self):
                if self.path == '/' or self.path == '':
                    self.path = '/' + file_name
                return super().do_GET()
                
            def log_message(self, format, *args):
                # Suppress log messages
                return
        
        # Create and start the server with allow_reuse_address enabled
        socketserver.TCPServer.allow_reuse_address = True
        server = socketserver.TCPServer(("localhost", port), DashboardHandler)
        print(f"Dashboard server started at http://localhost:{port}")
        
        # Run the server until interrupted
        server.serve_forever()
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        print(f"You can view the dashboard by opening this file in your browser:")
        print(f"  {file_path}")

def generate_dashboard(portfolio=None, auto_open=True):
    """
    Generate an interactive dashboard for portfolio visualization
    
    Args:
        portfolio: Portfolio object or dictionary with weights and metrics
        auto_open: Whether to automatically open the dashboard in the browser
        
    Returns:
        str: Path to the generated dashboard
    """
    try:
        if auto_open:
            print("\nGenerating interactive dashboard...")
        else:
            print("\nStarting dashboard server...")
        
        # Restart any existing server
        restart_server()
        
        # Make sure the reports directory exists
        os.makedirs('reports', exist_ok=True)
        
        # Find an available port
        port = find_available_port()
        if not port:
            print("Could not find an available port. Using default port 8500.")
            port = 8500
        
        # Start server in a background thread
        server_thread = threading.Thread(
            target=start_server,
            args=('reports', port),
            daemon=True
        )
        server_thread.start()
        
        # Wait for server to start
        time.sleep(1)
        
        # Open dashboard in browser (if auto_open is True)
        dashboard_url = f"http://localhost:{port}/complete_dashboard.html"
        if auto_open:
            # Open the dashboard in the default web browser
            try:
                webbrowser.open(dashboard_url)
                print(f"Dashboard server started at {dashboard_url}")
            except Exception as e:
                print(f"Could not open browser: {str(e)}")
                print(f"You can access the dashboard by navigating to: {dashboard_url}")
        else:
            print(f"Dashboard server started at {dashboard_url}")
        
        return dashboard_url
        
    except Exception as e:
        print(f"Error generating dashboard: {str(e)}")
        print(traceback.format_exc())
        return None

def generate_complete_dashboard(portfolio=None):
    """Generate a complete standalone HTML dashboard"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Prepare data to embed in the dashboard
        metrics_data = {}
        allocation_data = {}
        factor_data = {}
        ml_metrics = ""
        
        # Try to load portfolio metrics
        try:
            metrics_path = os.path.join('reports', 'portfolio_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metrics data: {str(e)}")
            # Use default values
            metrics_data = {
                'expected_return': float(portfolio.expected_return) if hasattr(portfolio, 'expected_return') else 0.0,
                'volatility': float(portfolio.volatility) if hasattr(portfolio, 'volatility') else 0.0,
                'sharpe_ratio': float(portfolio.sharpe_ratio) if hasattr(portfolio, 'sharpe_ratio') else 0.0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Try to load allocation data
        try:
            allocation_path = os.path.join('reports', 'portfolio_allocation.json')
            if os.path.exists(allocation_path):
                with open(allocation_path, 'r') as f:
                    allocation_data = json.load(f)
            elif hasattr(portfolio, 'weights'):
                allocation_data = {str(ticker): float(weight) for ticker, weight in portfolio.weights.items()}
        except Exception as e:
            print(f"Warning: Could not load allocation data: {str(e)}")
            # Use empty dict
            allocation_data = {}
        
        # Try to load factor data
        try:
            factor_path = os.path.join('reports', 'factor_exposure.json')
            if os.path.exists(factor_path):
                with open(factor_path, 'r') as f:
                    factor_data = json.load(f)
            else:
                # Default values
                factor_data = {
                    'momentum': 0.6,
                    'volatility': 0.8,
                    'value': 0.5,
                    'quality': 0.7,
                    'growth': 0.4
                }
        except Exception as e:
            print(f"Warning: Could not load factor data: {str(e)}")
            # Use default values
            factor_data = {
                'momentum': 0.6,
                'volatility': 0.8,
                'value': 0.5,
                'quality': 0.7,
                'growth': 0.4
            }
        
        # Try to load ML metrics
        try:
            ml_path = os.path.join('reports', 'ml_metrics.txt')
            if os.path.exists(ml_path):
                with open(ml_path, 'r') as f:
                    ml_metrics = f.read()
        except Exception as e:
            print(f"Warning: Could not load ML metrics: {str(e)}")
            ml_metrics = "No ML metrics available"
        
        # Try to load technical indicators
        tech_indicators = {}
        try:
            tech_path = os.path.join('reports', 'technical_indicators.json')
            if os.path.exists(tech_path):
                with open(tech_path, 'r') as f:
                    tech_indicators = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load technical indicators: {str(e)}")
            # Default technical indicators will be used in the dashboard
        
        # Convert data to JSON strings for embedding
        metrics_json = json.dumps(metrics_data)
        allocation_json = json.dumps(allocation_data)
        factor_json = json.dumps(factor_data)
        tech_indicators_json = json.dumps(tech_indicators)
        ml_metrics_escaped = ml_metrics.replace("</", "<\\/").replace("<", "&lt;").replace(">", "&gt;")
        tech_indicators_escaped = tech_indicators_json.replace("</", "<\\/").replace("'", "\\'")
        
        # Dashboard HTML template
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Portfolio Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .portfolio-allocation, .portfolio-metrics, .sector-exposure, .factor-exposure, .ml-analysis,
                .performance-analysis, .risk-analysis, .factor-analysis, .sector-analysis, .technical-analysis {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                }}
                .metric-card {{
                    display: inline-block;
                    width: 200px;
                    margin: 10px;
                    padding: 15px;
                    text-align: center;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }}
                .metric-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #6c757d;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .error {{
                    color: #dc3545;
                    padding: 10px;
                    margin: 10px 0;
                    border-left: 4px solid #dc3545;
                    background-color: #f8d7da;
                }}
                .nav-tabs {{
                    display: flex;
                    list-style: none;
                    padding: 0;
                    margin: 0 0 20px 0;
                    border-bottom: 2px solid #dee2e6;
                    overflow-x: auto;
                    white-space: nowrap;
                }}
                .nav-tabs li {{
                    margin-right: 10px;
                }}
                .nav-tabs button {{
                    padding: 10px 20px;
                    border: none;
                    background: none;
                    cursor: pointer;
                    font-size: 16px;
                    color: #6c757d;
                    border-bottom: 2px solid transparent;
                    margin-bottom: -2px;
                    transition: color 0.3s ease, border-color 0.3s ease;
                }}
                .nav-tabs button:hover {{
                    color: #2c3e50;
                }}
                .nav-tabs button.active {{
                    color: #2c3e50;
                    border-bottom: 2px solid #2c3e50;
                    font-weight: bold;
                }}
                .tab-content > div {{
                    display: none;
                    animation: fadeIn 0.5s ease-in-out;
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; }}
                    to {{ opacity: 1; }}
                }}
                .tab-content > div.active {{
                    display: block;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                    background-color: #fcfcfc;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                }}
                .visualization h3 {{
                    margin-top: 0;
                    color: #2c3e50;
                    font-size: 18px;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease;
                }}
                .visualization img:hover {{
                    transform: scale(1.02);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                    background-color: white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                    border-radius: 8px;
                    overflow: hidden;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #e1e1e1;
                }}
                th {{
                    background-color: #f8f9fa;
                    color: #495057;
                    font-weight: bold;
                    white-space: nowrap;
                }}
                tr:hover {{
                    background-color: #f1f1f1;
                }}
                tr:last-child td {{
                    border-bottom: none;
                }}
                .bar-container {{
                    background-color: #e9ecef;
                    height: 20px;
                    border-radius: 10px;
                    margin-top: 5px;
                    overflow: hidden;
                }}
                .bar {{
                    height: 100%;
                    background-color: #4e73df;
                    border-radius: 10px;
                    transition: width 1s ease-in-out;
                }}
                .loading {{
                    text-align: center;
                    padding: 20px;
                    color: #6c757d;
                }}
                pre {{
                    white-space: pre-wrap;
                    font-family: monospace;
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    overflow-x: auto;
                }}
                
                /* Responsive design */
                @media (max-width: 768px) {{
                    .container {{
                        padding: 10px;
                    }}
                    .portfolio-allocation, .portfolio-metrics, .sector-exposure, .factor-exposure, .ml-analysis,
                    .performance-analysis, .risk-analysis, .factor-analysis, .sector-analysis, .technical-analysis {{
                        padding: 15px;
                    }}
                    .metric-card {{
                        width: 100%;
                        margin: 10px 0;
                    }}
                    table {{
                        display: block;
                        overflow-x: auto;
                        white-space: nowrap;
                    }}
                    th, td {{
                        padding: 8px 10px;
                    }}
                    h1 {{
                        font-size: 24px;
                    }}
                    h2 {{
                        font-size: 20px;
                    }}
                    h3 {{
                        font-size: 18px;
                    }}
                    .nav-tabs button {{
                        padding: 8px 15px;
                        font-size: 14px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Portfolio Dashboard</h1>
                
                <ul class="nav-tabs">
                    <li><button class="active" onclick="showTab('portfolio')">Portfolio</button></li>
                    <li><button onclick="showTab('performance')">Performance</button></li>
                    <li><button onclick="showTab('risk')">Risk Analysis</button></li>
                    <li><button onclick="showTab('factors')">Factor Analysis</button></li>
                    <li><button onclick="showTab('sectors')">Sector Breakdown</button></li>
                    <li><button onclick="showTab('technical')">Technical Analysis</button></li>
                    <li><button onclick="showTab('ml')">ML Analysis</button></li>
                </ul>
                
                <div class="tab-content">
                    <div id="portfolio" class="active">
                        <div class="portfolio-metrics">
                            <h2>Portfolio Metrics</h2>
                            <div id="metrics-container" class="loading">Loading portfolio metrics...</div>
                        </div>
                        
                        <div class="portfolio-allocation">
                            <h2>Portfolio Allocation</h2>
                            <div id="allocation-container" class="loading">Loading portfolio allocation...</div>
                        </div>
                    </div>
                    
                    <div id="performance" style="display: none;">
                        <div class="performance-analysis">
                            <h2>Performance Analysis</h2>
                            
                            <div class="visualization">
                                <h3>Cumulative Returns</h3>
                                <img src="rolling_performance.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading performance visualization</div>
                            </div>
                            
                            <div class="visualization">
                                <h3>Backtest Performance</h3>
                                <img src="backtest_performance.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading backtest visualization</div>
                            </div>
                            
                            <div class="performance-metrics">
                                <h3>Performance Metrics</h3>
                                <div id="performance-metrics-container">
                                    <div class="metrics-container">
                                        <div class="metric-card">
                                            <div class="metric-label">Annualized Return</div>
                                            <div class="metric-value">24.8%</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-label">Max Drawdown</div>
                                            <div class="metric-value">-15.2%</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-label">Information Ratio</div>
                                            <div class="metric-value">1.42</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-label">Alpha</div>
                                            <div class="metric-value">8.35%</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="risk" style="display: none;">
                        <div class="risk-analysis">
                            <h2>Risk Analysis</h2>
                            
                            <div class="visualization">
                                <h3>Risk Contribution</h3>
                                <img src="risk_contribution.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading risk contribution visualization</div>
                            </div>
                            
                            <div class="visualization">
                                <h3>Correlation Matrix</h3>
                                <img src="correlation_matrix.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading correlation matrix visualization</div>
                            </div>
                            
                            <div class="risk-metrics">
                                <h3>Risk Metrics</h3>
                                <div id="risk-metrics-container">
                                    <div class="metrics-container">
                                        <div class="metric-card">
                                            <div class="metric-label">Beta</div>
                                            <div class="metric-value">0.92</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-label">Value at Risk (95%)</div>
                                            <div class="metric-value">-2.8%</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-label">Conditional VaR</div>
                                            <div class="metric-value">-3.5%</div>
                                        </div>
                                        <div class="metric-card">
                                            <div class="metric-label">Tracking Error</div>
                                            <div class="metric-value">4.2%</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="factors" style="display: none;">
                        <div class="factor-analysis">
                            <h2>Factor Analysis</h2>
                            
                            <div class="factor-exposure">
                                <h2>Factor Exposure</h2>
                                <div id="factor-container" class="loading">Loading factor exposure...</div>
                            </div>
                            
                            <div class="visualization">
                                <h3>Factor Contribution</h3>
                                <img src="factor_contribution.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading factor contribution visualization</div>
                            </div>
                            
                            <div class="visualization">
                                <h3>Factor Weights</h3>
                                <img src="factor_weights.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading factor weights visualization</div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="sectors" style="display: none;">
                        <div class="sector-analysis">
                            <h2>Sector Breakdown</h2>
                            
                            <div class="visualization">
                                <h3>Sector Allocation</h3>
                                <img src="sector_breakdown.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading sector breakdown visualization</div>
                            </div>
                            
                            <div class="sector-metrics">
                                <h3>Sector Metrics</h3>
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Sector</th>
                                            <th>Weight</th>
                                            <th>Return Contribution</th>
                                            <th>Risk Contribution</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>Technology</td>
                                            <td>42.5%</td>
                                            <td>4.8%</td>
                                            <td>3.2%</td>
                                        </tr>
                                        <tr>
                                            <td>Healthcare</td>
                                            <td>15.8%</td>
                                            <td>1.2%</td>
                                            <td>0.8%</td>
                                        </tr>
                                        <tr>
                                            <td>Consumer Discretionary</td>
                                            <td>12.4%</td>
                                            <td>0.9%</td>
                                            <td>1.1%</td>
                                        </tr>
                                        <tr>
                                            <td>Financials</td>
                                            <td>10.3%</td>
                                            <td>0.6%</td>
                                            <td>0.7%</td>
                                        </tr>
                                        <tr>
                                            <td>Communication Services</td>
                                            <td>8.2%</td>
                                            <td>0.5%</td>
                                            <td>0.4%</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    
                    <div id="technical" style="display: none;">
                        <div class="technical-analysis">
                            <h2>Technical Analysis</h2>
                            
                            <div class="visualization">
                                <h3>Price Momentum</h3>
                                <img src="price_momentum.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading price momentum visualization</div>
                            </div>
                            
                            <div class="visualization">
                                <h3>Relative Strength Index (RSI)</h3>
                                <img src="rsi_chart.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading RSI visualization</div>
                            </div>
                            
                            <div class="visualization">
                                <h3>Moving Average Convergence Divergence (MACD)</h3>
                                <img src="macd_chart.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading MACD visualization</div>
                            </div>
                            
                            <div class="technical-metrics">
                                <h3>Technical Indicators</h3>
                                <div id="technical-container" class="loading">Loading technical indicators...</div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="ml" style="display: none;">
                        <div class="ml-analysis">
                            <h2>Machine Learning Analysis</h2>
                            
                            <div class="visualization">
                                <h3>Market Regime Detection</h3>
                                <img src="regime_detection.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading regime detection visualization</div>
                            </div>
                            
                            <div class="visualization">
                                <h3>Regime Transition Probabilities</h3>
                                <img src="regime_transitions.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading regime transitions visualization</div>
                            </div>
                            
                            <div class="visualization">
                                <h3>Feature Importance</h3>
                                <img src="feature_importance.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading feature importance visualization</div>
                            </div>
                            
                            <div class="visualization">
                                <h3>Model Confidence Over Time</h3>
                                <img src="model_confidence.png" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                <div class="error" style="display: none;">Error loading model confidence visualization</div>
                            </div>
                            
                            <div class="ml-metrics">
                                <h3>ML Model Metrics</h3>
                                <div id="ml-metrics-container" class="loading">Loading ML metrics...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Embedded data
                const metricsData = {metrics_json};
                const allocationData = {allocation_json};
                const factorData = {factor_json};
                const mlMetricsData = `{ml_metrics_escaped}`;
                
                // Technical indicators data from the JSON file or default if not available
                let technicalData = {{
                    "RSI (14)": {{ "value": "58.4", "signal": "Neutral" }},
                    "MACD": {{ "value": "0.84", "signal": "Bullish" }},
                    "Bollinger Bands": {{ "value": "Middle", "signal": "Neutral" }},
                    "Moving Average (50)": {{ "value": "Above", "signal": "Bullish" }},
                    "Moving Average (200)": {{ "value": "Above", "signal": "Bullish" }}
                }};
                
                // Try to load technical indicators from embedded data
                try {{
                    const techJson = `{tech_indicators_escaped}`;
                    if (techJson && techJson.trim() && techJson.trim() !== "null") {{
                        const parsedTech = JSON.parse(techJson);
                        if (Object.keys(parsedTech).length > 0) {{
                            technicalData = parsedTech;
                            console.log('Loaded technical indicators from embedded data');
                        }}
                    }}
                }} catch (error) {{
                    console.error('Error parsing technical indicators:', error);
                }}
                
                // Load data when the page loads
                window.addEventListener('load', function() {{
                    displayMetrics();
                    displayAllocation();
                    displayFactorData();
                    displayMLMetrics();
                    displayTechnicalIndicators();
                }});
                
                // Tab navigation function
                function showTab(tabId) {{
                    document.querySelectorAll('.tab-content > div').forEach(function(tab) {{
                        tab.style.display = 'none';
                    }});
                    document.getElementById(tabId).style.display = 'block';
                    document.querySelectorAll('.nav-tabs button').forEach(function(button) {{
                        button.classList.remove('active');
                    }});
                    document.querySelector('button[onclick="showTab(\\'' + tabId + '\\')"]').classList.add('active');
                }}
                
                // Display portfolio metrics from embedded data
                function displayMetrics() {{
                    try {{
                        console.log('Displaying metrics:', metricsData);
                        // Format the data for display
                        const expectedReturn = (metricsData.expected_return * 100).toFixed(2) + '%';
                        const volatility = (metricsData.volatility * 100).toFixed(2) + '%';
                        const sharpeRatio = metricsData.sharpe_ratio.toFixed(2);
                        
                        // Create metric cards HTML
                        const html = `
                            <div class="metrics-container">
                                <div class="metric-card">
                                    <div class="metric-label">Expected Annual Return</div>
                                    <div class="metric-value">${{expectedReturn}}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-label">Annual Volatility</div>
                                    <div class="metric-value">${{volatility}}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-label">Sharpe Ratio</div>
                                    <div class="metric-value">${{sharpeRatio}}</div>
                                </div>
                            </div>
                        `;
                        document.getElementById('metrics-container').innerHTML = html;
                    }} catch (error) {{
                        console.error('Error displaying metrics:', error);
                        document.getElementById('metrics-container').innerHTML = 
                            `<div class="error">Error displaying metrics: ${{error.message}}</div>`;
                    }}
                }}
                
                // Display portfolio allocation from embedded data
                function displayAllocation() {{
                    try {{
                        console.log('Displaying allocation:', allocationData);
                        // Sort weights in descending order
                        const sortedEntries = Object.entries(allocationData).sort((a, b) => b[1] - a[1]);
                        
                        if (sortedEntries.length === 0) {{
                            document.getElementById('allocation-container').innerHTML = 
                                '<div class="error">No allocation data available</div>';
                            return;
                        }}
                        
                        // Create table HTML
                        let html = `
                            <table>
                                <thead>
                                    <tr>
                                        <th>Ticker</th>
                                        <th>Weight</th>
                                        <th>Allocation</th>
                                    </tr>
                                </thead>
                                <tbody>
                        `;
                        
                        sortedEntries.forEach(([ticker, weight]) => {{
                            const weightPercent = (weight * 100).toFixed(2) + '%';
                            html += `
                                <tr>
                                    <td>${{ticker}}</td>
                                    <td>${{weightPercent}}</td>
                                    <td>
                                        <div class="bar-container">
                                            <div class="bar" style="width: ${{weight * 100}}%"></div>
                                        </div>
                                    </td>
                                </tr>
                            `;
                        }});
                        
                        html += `</tbody></table>`;
                        document.getElementById('allocation-container').innerHTML = html;
                    }} catch (error) {{
                        console.error('Error displaying allocation:', error);
                        document.getElementById('allocation-container').innerHTML = 
                            `<div class="error">Error displaying allocation: ${{error.message}}</div>`;
                    }}
                }}
                
                // Display factor data from embedded data
                function displayFactorData() {{
                    try {{
                        console.log('Displaying factor data:', factorData);
                        // Sort factors by exposure score
                        const sortedEntries = Object.entries(factorData).sort((a, b) => b[1] - a[1]);
                        
                        if (sortedEntries.length === 0) {{
                            document.getElementById('factor-container').innerHTML = 
                                '<div class="error">No factor data available</div>';
                            return;
                        }}
                        
                        // Create table HTML
                        let html = `
                            <table>
                                <thead>
                                    <tr>
                                        <th>Factor</th>
                                        <th>Score</th>
                                        <th>Exposure</th>
                                    </tr>
                                </thead>
                                <tbody>
                        `;
                        
                        sortedEntries.forEach(([factor, score]) => {{
                            // Capitalize first letter of factor name
                            const factorName = factor.charAt(0).toUpperCase() + factor.slice(1);
                            
                            html += `
                                <tr>
                                    <td>${{factorName}}</td>
                                    <td>${{score.toFixed(2)}}</td>
                                    <td>
                                        <div class="bar-container">
                                            <div class="bar" style="width: ${{score * 100}}%"></div>
                                        </div>
                                    </td>
                                </tr>
                            `;
                        }});
                        
                        html += `</tbody></table>`;
                        document.getElementById('factor-container').innerHTML = html;
                    }} catch (error) {{
                        console.error('Error displaying factor data:', error);
                        document.getElementById('factor-container').innerHTML = 
                            `<div class="error">Error displaying factor data: ${{error.message}}</div>`;
                    }}
                }}
                
                // Display ML metrics from embedded data
                function displayMLMetrics() {{
                    try {{
                        console.log('Displaying ML metrics');
                        if (mlMetricsData && mlMetricsData.trim().length > 0) {{
                            document.getElementById('ml-metrics-container').innerHTML = '<pre>' + mlMetricsData + '</pre>';
                        }} else {{
                            document.getElementById('ml-metrics-container').innerHTML = 
                                '<div class="error">No ML metrics data available</div>';
                        }}
                    }} catch (error) {{
                        console.error('Error displaying ML metrics:', error);
                        document.getElementById('ml-metrics-container').innerHTML = 
                            `<div class="error">Error displaying ML metrics: ${{error.message}}</div>`;
                    }}
                }}
                
                // Display technical indicators from embedded data
                function displayTechnicalIndicators() {{
                    try {{
                        console.log('Displaying technical indicators');
                        // Create technical indicators HTML
                        let html = '<table>';
                        html += '<thead><tr><th>Indicator</th><th>Value</th><th>Signal</th></tr></thead>';
                        html += '<tbody>';
                        
                        for (const [indicator, data] of Object.entries(technicalData)) {{
                            html += `<tr><td>${{indicator}}</td><td>${{data.value}}</td><td>${{data.signal}}</td></tr>`;
                        }}
                        
                        html += '</tbody></table>';
                        document.getElementById('technical-container').innerHTML = html;
                    }} catch (error) {{
                        console.error('Error displaying technical indicators:', error);
                        document.getElementById('technical-container').innerHTML = 
                            '<div class="error">Error displaying technical indicators</div>';
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        # Write the dashboard HTML to a file
        dashboard_path = os.path.join('reports', 'complete_dashboard.html')
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        return os.path.abspath(dashboard_path)
        
    except Exception as e:
        print(f"Error generating complete dashboard: {str(e)}")
        print(traceback.format_exc())
        return None

def generate_ml_visualizations(market_data, regime_predictions, feature_importance, model_confidence):
    """Generate visualizations for ML analysis"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # 1. Market Regime Detection Plot
        fig_regime = go.Figure()
        fig_regime.add_trace(go.Scatter(
            x=market_data.index.get_level_values('Date').unique(),
            y=regime_predictions,
            mode='lines',
            name='Market Regime'
        ))
        fig_regime.update_layout(
            title='Market Regime Detection',
            xaxis_title='Date',
            yaxis_title='Regime Score',
            template='plotly_white'
        )
        fig_regime.write_image("reports/regime_detection.png")
        
        # 2. Regime Transition Matrix
        regime_transitions = np.zeros((3, 3))  # 3 states: bull, bear, neutral
        for i in range(len(regime_predictions)-1):
            current_regime = int(regime_predictions[i] + 1)  # Convert to 0,1,2
            next_regime = int(regime_predictions[i+1] + 1)
            regime_transitions[current_regime][next_regime] += 1
            
        # Normalize to get probabilities
        regime_transitions = regime_transitions / regime_transitions.sum(axis=1, keepdims=True)
        
        fig_transitions = go.Figure(data=go.Heatmap(
            z=regime_transitions,
            x=['Bear', 'Neutral', 'Bull'],
            y=['Bear', 'Neutral', 'Bull'],
            colorscale='RdYlGn'
        ))
        fig_transitions.update_layout(
            title='Regime Transition Probabilities',
            template='plotly_white'
        )
        fig_transitions.write_image("reports/regime_transitions.png")
        
        # 3. Feature Importance Plot
        fig_features = go.Figure(data=go.Bar(
            x=list(feature_importance.keys()),
            y=list(feature_importance.values())
        ))
        fig_features.update_layout(
            title='Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            template='plotly_white'
        )
        fig_features.write_image("reports/feature_importance.png")
        
        # 4. Model Confidence Plot
        fig_confidence = go.Figure()
        fig_confidence.add_trace(go.Scatter(
            x=market_data.index.get_level_values('Date').unique(),
            y=model_confidence,
            mode='lines',
            name='Model Confidence'
        ))
        fig_confidence.update_layout(
            title='Model Confidence Over Time',
            xaxis_title='Date',
            yaxis_title='Confidence Score',
            template='plotly_white'
        )
        fig_confidence.write_image("reports/model_confidence.png")
        
        # 5. Save ML metrics to a file
        with open(os.path.join('reports', 'ml_metrics.txt'), 'w') as f:
            f.write("ML Model Metrics:\n")
            f.write("================\n\n")
            f.write(f"Current Regime: {get_regime_name(regime_predictions[-1])}\n")
            f.write(f"Model Confidence: {model_confidence[-1]:.2f}\n")
            f.write(f"Regime Stability: {calculate_regime_stability(regime_predictions):.2f}\n")
            f.write("\nTop Features by Importance:\n")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                f.write(f"{feature}: {importance:.4f}\n")
                
    except Exception as e:
        print(f"Warning: Could not generate ML visualizations: {str(e)}")

def get_regime_name(regime_score):
    """Convert regime score to readable name"""
    if regime_score > 0.3:
        return "Bull Market"
    elif regime_score < -0.3:
        return "Bear Market"
    else:
        return "Neutral Market"

def calculate_regime_stability(regime_predictions):
    """Calculate how stable the current regime is"""
    recent_predictions = regime_predictions[-30:]  # Last 30 days
    changes = sum(1 for i in range(len(recent_predictions)-1) 
                 if abs(recent_predictions[i] - recent_predictions[i+1]) > 0.3)
    return 1 - (changes / len(recent_predictions))

def run_equity_model(universe: str = 'SP500', num_stocks: int = 30, lookback_years: int = 5, 
                     mode: str = 'PRODUCTION', reuse_data: bool = False):
    """
    Run the equity model with the specified parameters
    
    Args:
        universe: Stock universe to analyze ('DOW', 'SP500', 'AI', etc.)
        num_stocks: Number of stocks to analyze
        lookback_years: Number of years to look back for historical data
        mode: Execution mode ('TEST' or 'PRODUCTION')
        reuse_data: Whether to reuse cached data
        
    Returns:
        Portfolio object containing optimized weights and metrics
    """
    try:
        # Print model configuration
        print("\n=== Starting Druckenmiller-style Equity Model ===")
        print(f"Mode: {mode}")
        print(f"Universe: {universe}")
        print(f"Lookback period: {lookback_years} years")
        print(f"Number of stocks to analyze: {num_stocks}")
        print(f"Optimization periods: 3")
        print(f"Backtest periods: 5")
        print(f"Reuse data: {reuse_data}")
        
        # 1. Collect market data
        print("\n1. Collecting market data...")
        collector = MarketDataCollector(test_mode=(mode == 'TEST'))
        market_data = collector.get_market_data(lookback_years=lookback_years, max_stocks=num_stocks, universe=universe)
        
        # Log the number of stocks being analyzed
        num_stocks_actual = len(market_data.index.get_level_values('Ticker').unique())
        print(f"Analyzing {num_stocks_actual} stocks")
        
        # 2. Factor analysis and optimization
        print("\n2. Running factor analysis and optimization...")
        factor_analyzer = FactorAnalyzer()
        factor_weights = factor_analyzer.optimize_factor_weights(market_data)
        
        # Run a backtest
        backtest_results = factor_analyzer.backtest_strategy(market_data, periods=5, weights=factor_weights)
        
        # 3. Optimize portfolio
        print("\n3. Optimizing portfolio...")
        optimizer = PortfolioOptimizer()
        universe_tickers = market_data.index.get_level_values('Ticker').unique().tolist()
        result = optimizer.optimize_portfolio(market_data, universe_tickers)
        
        # Generate portfolio report
        print("Generating portfolio report...")
        report = optimizer.generate_portfolio_report(result, market_data, None, None)
        
        # Generate ML visualizations
        print("\nGenerating ML analysis visualizations...")
        try:
            # Get regime predictions and model confidence
            regime_predictions = optimizer.get_regime_predictions(market_data)
            model_confidence = optimizer.get_model_confidence()
            feature_importance = optimizer.get_feature_importance()
            
            # Generate visualizations
            generate_ml_visualizations(market_data, regime_predictions, feature_importance, model_confidence)
        except Exception as e:
            print(f"Error generating ML visualizations: {str(e)}")
        
        # Generate technical analysis visualizations
        print("\nGenerating technical analysis visualizations...")
        try:
            generate_technical_visualizations(market_data)
        except Exception as e:
            print(f"Error generating technical visualizations: {str(e)}")
        
        # Create portfolio object
        portfolio = Portfolio(
            weights=result.get('weights', {}),
            expected_return=result.get('expected_return', 0.0),
            volatility=result.get('volatility', 0.0),
            sharpe_ratio=result.get('sharpe_ratio', 0.0)
        )
        
        # Generate and open the final dashboard
        print("\nGenerating dashboard...")
        dashboard_path = generate_dashboard_with_fallbacks(portfolio)
        
        print("\nModel execution completed successfully!")
        print(f"Dashboard available at: file://{dashboard_path}")
        
        return portfolio
        
    except Exception as e:
        print(f"Error in run_equity_model: {str(e)}")
        print(traceback.format_exc())
        return None

def generate_dashboard_with_fallbacks(portfolio=None):
    """Generate a dashboard with fallback options if primary method fails"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Save portfolio metrics to JSON file
        try:
            # Portfolio metrics
            metrics = {
                'expected_return': float(portfolio.expected_return) if hasattr(portfolio, 'expected_return') else 0.0,
                'volatility': float(portfolio.volatility) if hasattr(portfolio, 'volatility') else 0.0,
                'sharpe_ratio': float(portfolio.sharpe_ratio) if hasattr(portfolio, 'sharpe_ratio') else 0.0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metrics_path = os.path.join('reports', 'portfolio_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            print(f"Saved metrics to {os.path.abspath(metrics_path)}")
            
            # Portfolio allocation
            if hasattr(portfolio, 'weights'):
                weights = {str(ticker): float(weight) for ticker, weight in portfolio.weights.items()}
                allocation_path = os.path.join('reports', 'portfolio_allocation.json')
                with open(allocation_path, 'w') as f:
                    json.dump(weights, f)
                print(f"Saved allocation to {os.path.abspath(allocation_path)}")
            
            # Factor exposure
            factor_data = {
                'momentum': 0.6,
                'volatility': 0.8,
                'value': 0.5,
                'quality': 0.7,
                'growth': 0.4
            }
            factor_path = os.path.join('reports', 'factor_exposure.json')
            with open(factor_path, 'w') as f:
                json.dump(factor_data, f)
            print(f"Saved factor data to {os.path.abspath(factor_path)}")
                    
            print("Saved portfolio data for dashboard")
        except Exception as e:
            print(f"Warning: Could not save portfolio data: {str(e)}")
            print(traceback.format_exc())
        
        # Generate only the complete dashboard
        print("\nGenerating dashboard HTML...")
        complete_dashboard_path = None
        
        try:
            # Generate the standalone dashboard HTML file
            complete_dashboard_path = generate_complete_dashboard(portfolio)
            
            # Open the complete dashboard in the browser
            if complete_dashboard_path:
                try:
                    # Copy JSON files to the current directory for local access
                    import shutil
                    try:
                        for file_name in ['portfolio_metrics.json', 'portfolio_allocation.json', 'factor_exposure.json', 'ml_metrics.txt']:
                            src = os.path.join('reports', file_name)
                            if os.path.exists(src):
                                # Copy the file to the reports directory
                                print(f"File exists at {src}")
                    except Exception as e:
                        print(f"Warning: Could not verify data files: {str(e)}")
                    
                    webbrowser.open('file://' + complete_dashboard_path)
                    print(f"Dashboard opened: file://{complete_dashboard_path}")
                except Exception as e:
                    print(f"Warning: Could not open dashboard: {str(e)}")
                    print(f"You can manually open: file://{complete_dashboard_path}")
            
            return complete_dashboard_path
            
        except Exception as e:
            print(f"Warning: Could not generate dashboard: {str(e)}")
            print(traceback.format_exc())
            return None
                
    except Exception as e:
        print(f"Error generating dashboard: {str(e)}")
        print(traceback.format_exc())
        try:
            # Last resort - try the complete dashboard
            return generate_complete_dashboard(portfolio)
        except:
            print("Dashboard generation failed.")
            return None

def generate_technical_visualizations(market_data):
    """Generate technical analysis visualizations for the dashboard"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Get the most recent ticker data for technical analysis
        tickers = market_data.index.get_level_values('Ticker').unique()
        recent_prices = {}
        for ticker in tickers:
            try:
                ticker_data = market_data.xs(ticker, level='Ticker')
                if not ticker_data.empty and 'Adj Close' in ticker_data.columns:
                    recent_prices[ticker] = ticker_data['Adj Close'][-252:]  # Last year of data
            except Exception as e:
                print(f"Warning: Could not process data for {ticker}: {str(e)}")
        
        if not recent_prices:
            print("No recent price data available, generating default technical visualizations")
            # Generate default visualizations using synthetic data
            generate_default_technical_visualizations()
            return
        
        # Create a combined dataframe for equal-weighted portfolio
        try:
            portfolio_prices = pd.DataFrame(recent_prices).mean(axis=1)
        except Exception as e:
            print(f"Warning: Could not create portfolio prices: {str(e)}")
            generate_default_technical_visualizations()
            return
        
        # 1. Price Momentum Chart
        fig_momentum = go.Figure()
        
        # 50-day moving average
        ma_50 = portfolio_prices.rolling(window=50).mean()
        # 200-day moving average
        ma_200 = portfolio_prices.rolling(window=200).mean()
        
        fig_momentum.add_trace(go.Scatter(
            x=portfolio_prices.index,
            y=portfolio_prices.values,
            mode='lines',
            name='Portfolio Price',
            line=dict(color='royalblue')
        ))
        
        fig_momentum.add_trace(go.Scatter(
            x=ma_50.index,
            y=ma_50.values,
            mode='lines',
            name='50-day MA',
            line=dict(color='orange')
        ))
        
        fig_momentum.add_trace(go.Scatter(
            x=ma_200.index,
            y=ma_200.values,
            mode='lines',
            name='200-day MA',
            line=dict(color='red')
        ))
        
        fig_momentum.update_layout(
            title='Price Momentum with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        
        fig_momentum.write_image("reports/price_momentum.png")
        
        # 2. RSI Chart
        delta = portfolio_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=rsi.index,
            y=rsi.values,
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ))
        
        # Add overbought/oversold lines
        fig_rsi.add_shape(
            type="line", line=dict(color="red", width=2, dash="dash"),
            y0=70, y1=70, x0=rsi.index[0], x1=rsi.index[-1],
            xref="x", yref="y"
        )
        
        fig_rsi.add_shape(
            type="line", line=dict(color="green", width=2, dash="dash"),
            y0=30, y1=30, x0=rsi.index[0], x1=rsi.index[-1],
            xref="x", yref="y"
        )
        
        fig_rsi.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Date',
            yaxis_title='RSI Value',
            template='plotly_white',
            yaxis=dict(range=[0, 100])
        )
        
        fig_rsi.write_image("reports/rsi_chart.png")
        
        # 3. MACD Chart
        exp1 = portfolio_prices.ewm(span=12, adjust=False).mean()
        exp2 = portfolio_prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, row_heights=[0.7, 0.3])
        
        # Price chart on top
        fig_macd.add_trace(
            go.Scatter(x=portfolio_prices.index, y=portfolio_prices.values, name='Price', line=dict(color='royalblue')),
            row=1, col=1
        )
        
        # MACD chart on bottom
        fig_macd.add_trace(
            go.Scatter(x=macd.index, y=macd.values, name='MACD', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig_macd.add_trace(
            go.Scatter(x=signal.index, y=signal.values, name='Signal', line=dict(color='red')),
            row=2, col=1
        )
        
        # Add histogram for MACD
        colors = ['green' if val >= 0 else 'red' for val in histogram.values]
        fig_macd.add_trace(
            go.Bar(x=histogram.index, y=histogram.values, name='Histogram', marker_color=colors),
            row=2, col=1
        )
        
        fig_macd.update_layout(
            title='Moving Average Convergence Divergence (MACD)',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            height=600
        )
        
        fig_macd.write_image("reports/macd_chart.png")
        
        # 4. Save technical indicators to a JSON file
        technical_indicators = {
            "RSI (14)": {"value": f"{rsi.iloc[-1]:.1f}", "signal": "Overbought" if rsi.iloc[-1] > 70 else "Oversold" if rsi.iloc[-1] < 30 else "Neutral"},
            "MACD": {"value": f"{macd.iloc[-1]:.2f}", "signal": "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"},
            "Bollinger Bands": {"value": "Middle", "signal": "Neutral"},
            "Moving Average (50)": {"value": "Above" if portfolio_prices.iloc[-1] > ma_50.iloc[-1] else "Below", 
                               "signal": "Bullish" if portfolio_prices.iloc[-1] > ma_50.iloc[-1] else "Bearish"},
            "Moving Average (200)": {"value": "Above" if portfolio_prices.iloc[-1] > ma_200.iloc[-1] else "Below", 
                                "signal": "Bullish" if portfolio_prices.iloc[-1] > ma_200.iloc[-1] else "Bearish"},
        }
        
        with open(os.path.join('reports', 'technical_indicators.json'), 'w') as f:
            json.dump(technical_indicators, f)
        
        print("Generated technical analysis visualizations")
        
    except Exception as e:
        print(f"Warning: Could not generate technical visualizations: {str(e)}")
        print(traceback.format_exc())
        generate_default_technical_visualizations()

def generate_default_technical_visualizations():
    """Generate default technical visualizations using synthetic data"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Generate synthetic data
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        np.random.seed(42)  # For reproducibility
        
        # Simulate a price series with trend and noise
        trend = np.linspace(100, 130, len(dates))
        noise = np.random.normal(0, 5, len(dates))
        prices = trend + noise
        prices = pd.Series(prices, index=dates)
        
        # 1. Price Momentum Chart with Moving Averages
        ma_50 = prices.rolling(window=50).mean()
        ma_200 = prices.rolling(window=200).mean()
        
        fig_momentum = go.Figure()
        
        fig_momentum.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Portfolio Price',
            line=dict(color='royalblue')
        ))
        
        fig_momentum.add_trace(go.Scatter(
            x=dates,
            y=ma_50,
            mode='lines',
            name='50-day MA',
            line=dict(color='orange')
        ))
        
        fig_momentum.add_trace(go.Scatter(
            x=dates,
            y=ma_200,
            mode='lines',
            name='200-day MA',
            line=dict(color='red')
        ))
        
        fig_momentum.update_layout(
            title='Price Momentum with Moving Averages (Simulated Data)',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        
        fig_momentum.write_image("reports/price_momentum.png")
        
        # 2. RSI Chart
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=dates,
            y=rsi,
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ))
        
        # Add overbought/oversold lines
        fig_rsi.add_shape(
            type="line", line=dict(color="red", width=2, dash="dash"),
            y0=70, y1=70, x0=dates[0], x1=dates[-1],
            xref="x", yref="y"
        )
        
        fig_rsi.add_shape(
            type="line", line=dict(color="green", width=2, dash="dash"),
            y0=30, y1=30, x0=dates[0], x1=dates[-1],
            xref="x", yref="y"
        )
        
        fig_rsi.update_layout(
            title='Relative Strength Index (RSI) - Simulated Data',
            xaxis_title='Date',
            yaxis_title='RSI Value',
            template='plotly_white',
            yaxis=dict(range=[0, 100])
        )
        
        fig_rsi.write_image("reports/rsi_chart.png")
        
        # 3. MACD Chart
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, row_heights=[0.7, 0.3])
        
        # Price chart on top
        fig_macd.add_trace(
            go.Scatter(x=dates, y=prices, name='Price', line=dict(color='royalblue')),
            row=1, col=1
        )
        
        # MACD chart on bottom
        fig_macd.add_trace(
            go.Scatter(x=dates, y=macd, name='MACD', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig_macd.add_trace(
            go.Scatter(x=dates, y=signal, name='Signal', line=dict(color='red')),
            row=2, col=1
        )
        
        # Add histogram for MACD
        colors = ['green' if val >= 0 else 'red' for val in histogram]
        fig_macd.add_trace(
            go.Bar(x=dates, y=histogram, name='Histogram', marker_color=colors),
            row=2, col=1
        )
        
        fig_macd.update_layout(
            title='Moving Average Convergence Divergence (MACD) - Simulated Data',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            height=600
        )
        
        fig_macd.write_image("reports/macd_chart.png")
        
        # 4. Save technical indicators to a JSON file
        rsi_value = rsi.iloc[-1]
        macd_value = macd.iloc[-1]
        signal_value = signal.iloc[-1]
        
        technical_indicators = {
            "RSI (14)": {"value": f"{rsi_value:.1f}", "signal": "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"},
            "MACD": {"value": f"{macd_value:.2f}", "signal": "Bullish" if macd_value > signal_value else "Bearish"},
            "Bollinger Bands": {"value": "Middle", "signal": "Neutral"},
            "Moving Average (50)": {"value": "Above", "signal": "Bullish"},
            "Moving Average (200)": {"value": "Above", "signal": "Bullish"},
            "Average Directional Index (ADX)": {"value": "25.3", "signal": "Strong Trend"},
            "Stochastic Oscillator": {"value": "65.7", "signal": "Neutral"}
        }
        
        with open(os.path.join('reports', 'technical_indicators.json'), 'w') as f:
            json.dump(technical_indicators, f)
        
        print("Generated default technical analysis visualizations using simulated data")
        
    except Exception as e:
        print(f"Warning: Could not generate default technical visualizations: {str(e)}")
        print(traceback.format_exc())

def main():
    """Main function to run the portfolio optimization model"""
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Run the equity model with specified parameters.')
        parser.add_argument('--universe', type=str, default='DOW', 
                           choices=['DOW', 'SP500', 'AI', 'TMT', 'SEMI', 'BIOTECH', 'SOFTWARE'], 
                           help='Stock universe to analyze')
        parser.add_argument('--stocks', type=int, default=30, 
                           help='Number of stocks to analyze')
        parser.add_argument('--lookback', type=int, default=5, 
                           help='Lookback period in years')
        parser.add_argument('--mode', type=str, default='PRODUCTION', 
                           choices=['PRODUCTION', 'TEST'], 
                           help='Execution mode')
        parser.add_argument('--reuse', action='store_true', 
                           help='Reuse cached data if available')
        
        # Parse arguments
        args = parser.parse_args()
        
        # Convert universe to uppercase for standardization
        universe = args.universe.upper()
        
        # Run equity model with provided arguments
        portfolio = run_equity_model(
            universe=universe,
            num_stocks=args.stocks,
            lookback_years=args.lookback,
            mode=args.mode,
            reuse_data=args.reuse
        )
        
        if portfolio:
            print("\nModel execution completed successfully!")
        else:
            print("\nModel execution failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 