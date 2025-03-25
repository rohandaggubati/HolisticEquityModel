#!/usr/bin/env python3
import os
import sys
import http.server
import socketserver
import webbrowser
from threading import Timer
import subprocess
import socket
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# Ensure the current directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the equity model (this will be used to run the model if needed)
try:
    from main import run_equity_model
except ImportError:
    print("Warning: Could not import the equity model. Run functionality will be disabled.")
    run_equity_model = None

# Create the reports directory if it doesn't exist
os.makedirs('reports', exist_ok=True)

def run_model():
    """Run the equity model"""
    if run_equity_model:
        print("Running the equity model...")
        try:
            return run_equity_model(test_mode=False, lookback_years=5, num_stocks=100,
                              optimization_periods=3, backtest_periods=5)
        except Exception as e:
            print(f"Error running the equity model: {str(e)}")
            return None
    else:
        print("Run functionality is disabled. Please run the model manually using 'python main.py'")
        return None

def generate_dashboard():
    """Generate the HTML dashboard based on the reports"""
    # Check if reports exist
    if not os.path.exists('reports'):
        os.makedirs('reports', exist_ok=True)
        return get_empty_dashboard_html()
    
    files = os.listdir('reports')
    if not files:
        return get_empty_dashboard_html()
    
    # Read portfolio allocation
    portfolio_html = ""
    backtest_html = ""
    try:
        if os.path.exists('reports/portfolio_allocation.txt'):
            with open('reports/portfolio_allocation.txt', 'r') as f:
                portfolio_content = f.read()
                portfolio_html = f"<pre>{portfolio_content}</pre>"
        
        # Add backtest performance if available
        backtest_file = 'reports/backtest_performance.png'
        if os.path.exists(backtest_file):
            backtest_html += f"""
            <div class="chart-container">
                <h3>Backtest Performance</h3>
                <img src="{backtest_file}" alt="Backtest Performance" class="chart-img">
            </div>
            """
        
        # Add factor weights if available
        factor_weights_file = 'reports/factor_weights.png'
        if os.path.exists(factor_weights_file):
            backtest_html += f"""
            <div class="chart-container">
                <h3>Optimal Factor Weights</h3>
                <img src="{factor_weights_file}" alt="Factor Weights" class="chart-img">
            </div>
            """
        
        # Add factor returns if available
        factor_returns_file = 'reports/factor_returns.png'
        if os.path.exists(factor_returns_file):
            backtest_html += f"""
            <div class="chart-container">
                <h3>Factor Returns by Period</h3>
                <img src="{factor_returns_file}" alt="Factor Returns" class="chart-img">
            </div>
            """
    except Exception as e:
        portfolio_html = f"<p>Error loading portfolio data: {str(e)}</p>"
    
    # Read chart files
    chart_html = ""
    chart_files = [
        ('Efficient Frontier', 'reports/efficient_frontier.png'),
        ('Portfolio Weights', 'reports/portfolio_weights.png')
    ]
    
    for name, file_path in chart_files:
        if os.path.exists(file_path):
            chart_html += f"""
            <div class="chart-container">
                <h3>{name}</h3>
                <img src="{file_path}" alt="{name}" class="chart-img">
            </div>
            """
    
    # Check for factor scores CSV
    factor_html = ""
    try:
        if os.path.exists('reports/factor_scores.csv'):
            df = pd.read_csv('reports/factor_scores.csv')
            factor_html += f"""
            <h3>Factor Scores</h3>
            <div class="table-container">
                {df.to_html(classes='styled-table', index=False)}
            </div>
            """
        
        if os.path.exists('reports/weighted_scores.csv'):
            df = pd.read_csv('reports/weighted_scores.csv')
            factor_html += f"""
            <h3>Weighted Scores</h3>
            <div class="table-container">
                {df.to_html(classes='styled-table', index=False)}
            </div>
            """
    except Exception as e:
        factor_html = f"<p>Error loading factor data: {str(e)}</p>"
    
    # Check for interactive HTML
    interactive_html = ""
    interactive_files = glob.glob('reports/*.html')
    for file_path in interactive_files:
        file_name = os.path.basename(file_path)
        if file_name != 'index.html':  # Avoid self-reference
            interactive_html += f"""
            <div class="interactive-link">
                <h3>{file_name}</h3>
                <p><a href="{file_path}" target="_blank">Open {file_name}</a></p>
                <iframe src="{file_path}" width="100%" height="600px"></iframe>
            </div>
            """
    
    # Generate the dashboard HTML
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Equity Model Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                color: #333;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
            }}
            .container {{
                width: 90%;
                margin: 0 auto;
                padding: 20px;
            }}
            .tab {{
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                border-radius: 5px 5px 0 0;
            }}
            .tab button {{
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 17px;
            }}
            .tab button:hover {{
                background-color: #ddd;
            }}
            .tab button.active {{
                background-color: #2c3e50;
                color: white;
            }}
            .tabcontent {{
                display: none;
                padding: 20px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 0 0 5px 5px;
                background-color: white;
                animation: fadeEffect 1s;
            }}
            @keyframes fadeEffect {{
                from {{opacity: 0;}}
                to {{opacity: 1;}}
            }}
            .chart-container {{
                margin: 20px 0;
                padding: 15px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .chart-img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
            .table-container {{
                overflow-x: auto;
                margin: 20px 0;
            }}
            .styled-table {{
                border-collapse: collapse;
                width: 100%;
                margin: 0 auto;
                font-size: 0.9em;
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            .styled-table thead tr {{
                background-color: #2c3e50;
                color: white;
                text-align: left;
                font-weight: bold;
            }}
            .styled-table th,
            .styled-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #dddddd;
            }}
            .styled-table tbody tr:nth-of-type(even) {{
                background-color: #f3f3f3;
            }}
            .styled-table tbody tr:last-of-type {{
                border-bottom: 2px solid #2c3e50;
            }}
            pre {{
                background-color: #f8f8f8;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .run-button {{
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 20px;
                transition: background-color 0.3s;
            }}
            .run-button:hover {{
                background-color: #45a049;
            }}
            .footer {{
                background-color: #2c3e50;
                color: white;
                text-align: center;
                padding: 10px;
                position: fixed;
                bottom: 0;
                width: 100%;
            }}
            .interactive-link {{
                margin: 20px 0;
                padding: 15px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .status {{
                margin-top: 20px;
                padding: 10px;
                border-radius: 5px;
                background-color: #e0f7fa;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Druckenmiller-Style Equity Model Dashboard</h1>
            <p>Last updated: {timestamp}</p>
        </div>
        
        <div class="container">
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'Portfolio')">Portfolio</button>
                <button class="tablinks" onclick="openTab(event, 'Backtest')">Multi-Period Backtest</button>
                <button class="tablinks" onclick="openTab(event, 'Charts')">Charts</button>
                <button class="tablinks" onclick="openTab(event, 'Factors')">Factor Analysis</button>
                <button class="tablinks" onclick="openTab(event, 'Interactive')">Interactive</button>
                <button class="tablinks" onclick="openTab(event, 'RunModel')">Run Model</button>
            </div>
            
            <div id="Portfolio" class="tabcontent" style="display: block;">
                <h2>Portfolio Allocation</h2>
                <div class="chart-container">
                    {portfolio_html}
                </div>
            </div>
            
            <div id="Backtest" class="tabcontent">
                <h2>Multi-Period Backtest Analysis</h2>
                {backtest_html}
            </div>
            
            <div id="Charts" class="tabcontent">
                <h2>Portfolio Charts</h2>
                {chart_html}
            </div>
            
            <div id="Factors" class="tabcontent">
                <h2>Factor Analysis</h2>
                {factor_html}
            </div>
            
            <div id="Interactive" class="tabcontent">
                <h2>Interactive Visualizations</h2>
                {interactive_html}
            </div>
            
            <div id="RunModel" class="tabcontent">
                <h2>Run the Equity Model</h2>
                <p>Click the button below to run the equity model with default parameters (this may take a few minutes):</p>
                <button class="run-button" onclick="runModel()">Run Equity Model</button>
                <div id="status" class="status"></div>
            </div>
        </div>
        
        <div class="footer">
            <p>Druckenmiller-Style Equity Model © {datetime.now().year}</p>
        </div>
        
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }}
            
            function runModel() {{
                // Update status
                document.getElementById("status").innerHTML = "<p>Running the equity model... This may take several minutes. Please wait.</p>";
                
                // Make an AJAX request to run the model
                var xhr = new XMLHttpRequest();
                xhr.open("GET", "/run_model", true);
                xhr.onreadystatechange = function() {{
                    if (xhr.readyState === 4) {{
                        if (xhr.status === 200) {{
                            document.getElementById("status").innerHTML = "<p>Model executed successfully! Refreshing page in 3 seconds...</p>";
                            setTimeout(function() {{ window.location.reload(); }}, 3000);
                        }} else {{
                            document.getElementById("status").innerHTML = "<p>Error running the model. Please check the console for details.</p>";
                        }}
                    }}
                }};
                xhr.send();
            }}
        </script>
    </body>
    </html>
    """
    
    return html


def get_empty_dashboard_html():
    """Generate an empty dashboard with instructions to run the model"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Equity Model Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
                color: #333;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
            }}
            .container {{
                width: 80%;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                margin-top: 20px;
            }}
            .run-button {{
                background-color: #4CAF50;
                color: white;
                padding: 15px 25px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 18px;
                margin-top: 20px;
                transition: background-color 0.3s;
                display: block;
                margin: 20px auto;
            }}
            .run-button:hover {{
                background-color: #45a049;
            }}
            .footer {{
                background-color: #2c3e50;
                color: white;
                text-align: center;
                padding: 10px;
                position: fixed;
                bottom: 0;
                width: 100%;
            }}
            .status {{
                margin-top: 20px;
                padding: 10px;
                border-radius: 5px;
                background-color: #e0f7fa;
            }}
            .instructions {{
                background-color: #f8f8f8;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                border-left: 5px solid #2c3e50;
            }}
            .code {{
                background-color: #272822;
                color: #f8f8f2;
                padding: 10px;
                border-radius: 5px;
                font-family: 'Courier New', Courier, monospace;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Druckenmiller-Style Equity Model Dashboard</h1>
            <p>Last updated: {timestamp}</p>
        </div>
        
        <div class="container">
            <h2>Welcome to the Equity Model Dashboard</h2>
            <p>No report data found. You need to run the equity model to generate reports.</p>
            
            <div class="instructions">
                <h3>Instructions:</h3>
                <p>You can run the equity model in two ways:</p>
                <ol>
                    <li>Click the "Run Equity Model" button below (recommended)</li>
                    <li>Or run it manually from the terminal with:
                        <div class="code">python main.py</div>
                    </li>
                </ol>
                <p>The model will analyze stocks, perform factor analysis, and generate portfolio recommendations.</p>
                <p>This may take several minutes to complete.</p>
            </div>
            
            <button class="run-button" onclick="runModel()">Run Equity Model</button>
            <div id="status" class="status"></div>
        </div>
        
        <div class="footer">
            <p>Druckenmiller-Style Equity Model © {datetime.now().year}</p>
        </div>
        
        <script>
            function runModel() {{
                // Update status
                document.getElementById("status").innerHTML = "<p>Running the equity model... This may take several minutes. Please wait.</p>";
                
                // Make an AJAX request to run the model
                var xhr = new XMLHttpRequest();
                xhr.open("GET", "/run_model", true);
                xhr.onreadystatechange = function() {{
                    if (xhr.readyState === 4) {{
                        if (xhr.status === 200) {{
                            document.getElementById("status").innerHTML = "<p>Model executed successfully! Refreshing page in 3 seconds...</p>";
                            setTimeout(function() {{ window.location.reload(); }}, 3000);
                        }} else {{
                            document.getElementById("status").innerHTML = "<p>Error running the model. Please check the console for details.</p>";
                        }}
                    }}
                }};
                xhr.send();
            }}
        </script>
    </body>
    </html>
    """
    
    return html


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Generate and serve the dashboard HTML
            html = generate_dashboard()
            self.wfile.write(html.encode())
        
        elif self.path == '/run_model':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Run the model and return result
            result = run_model()
            success = result is not None
            
            response = json.dumps({'success': success})
            self.wfile.write(response.encode())
        
        else:
            # Serve static files
            return http.server.SimpleHTTPRequestHandler.do_GET(self)


def get_local_ip():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def open_browser(port=8000):
    """Open the web browser to the dashboard"""
    webbrowser.open(f'http://localhost:{port}')


if __name__ == "__main__":
    port = 8000
    
    print("Generating HTML dashboard...")
    html = generate_dashboard()
    print("HTML dashboard generated successfully.")
    
    # Start the server
    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        print(f"Server running at http://localhost:{port}")
        print(f"Open your browser and navigate to http://localhost:{port}")
        
        # Open the browser after a short delay
        Timer(1.0, open_browser).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            httpd.server_close() 