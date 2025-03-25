#!/usr/bin/env python3
import os
import webbrowser
import time
import socket
import http.server
import socketserver
import threading

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

def start_dashboard_server():
    """Start a simple HTTP server to serve the dashboard"""
    # Check if reports directory exists
    if not os.path.exists("reports"):
        os.makedirs("reports")
        print("Created reports directory")
    
    # Check if dashboard file exists
    dashboard_path = os.path.join("reports", "dashboard.html")
    if not os.path.exists(dashboard_path):
        print(f"Warning: Dashboard file not found at {dashboard_path}")
        print("Please run the main model first to generate the dashboard")
        return False
    
    # Find an available port
    port = find_available_port()
    if not port:
        print("Could not find an available port for the dashboard server")
        return False
    
    print(f"Starting dashboard server on port {port}...")
    
    # Custom handler to serve the dashboard
    class DashboardHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.getcwd(), **kwargs)
        
        def translate_path(self, path):
            # Fix path issues with /reports/reports/
            if path.startswith('/reports/reports/'):
                path = path.replace('/reports/reports/', '/reports/', 1)
            return super().translate_path(path)
    
    # Start server
    try:
        httpd = socketserver.TCPServer(("", port), DashboardHandler)
        
        # Dashboard URL
        dashboard_url = f"http://localhost:{port}/reports/dashboard.html"
        print(f"Dashboard available at: {dashboard_url}")
        
        # Try to open browser
        time.sleep(1)
        webbrowser.open(dashboard_url)
        
        # Start server
        print("Server running. Press Ctrl+C to stop.")
        httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        return False
        
    return True

if __name__ == "__main__":
    print("\n=== Equity Model Dashboard Launcher ===\n")
    start_dashboard_server() 