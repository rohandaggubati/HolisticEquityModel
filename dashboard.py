import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import glob
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import sys

# Add the current directory to the path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try to import the needed modules
try:
    # Run the equity model and generate reports
    from main import run_equity_model
    from portfolio_optimizer import PortfolioOptimizer
    print("Successfully imported model modules")
except Exception as e:
    print(f"Error importing model modules: {str(e)}")

def load_reports():
    """Load all report data"""
    reports = {}
    
    # Load portfolio allocation
    portfolio_allocation_path = "reports/portfolio_allocation.txt"
    if os.path.exists(portfolio_allocation_path):
        with open(portfolio_allocation_path, "r") as f:
            reports["portfolio_allocation"] = f.read()
    
    # Load factor analysis
    factor_analysis_path = "reports/factor_analysis.txt"
    if os.path.exists(factor_analysis_path):
        with open(factor_analysis_path, "r") as f:
            reports["factor_analysis"] = f.read()
    
    # Load management analysis if it exists
    management_analysis_path = "reports/management_analysis.txt"
    if os.path.exists(management_analysis_path):
        with open(management_analysis_path, "r") as f:
            reports["management_analysis"] = f.read()
    
    # Load HTML visualization
    portfolio_analysis_html_path = "reports/portfolio_analysis.html"
    if os.path.exists(portfolio_analysis_html_path):
        with open(portfolio_analysis_html_path, "r") as f:
            reports["portfolio_analysis_html"] = f.read()
    
    # Load any images
    reports["images"] = {}
    for img_path in glob.glob("reports/*.png"):
        img_name = os.path.basename(img_path)
        reports["images"][img_name] = img_path
    
    return reports

def parse_portfolio_allocation(text):
    """Parse the portfolio allocation text into structured data"""
    if not text:
        return None
    
    lines = text.strip().split('\n')
    data = {"allocation": {}, "metrics": {}}
    
    section = None
    for line in lines:
        if not line.strip():
            continue
            
        if line.startswith("Portfolio Allocation:"):
            section = "allocation"
            continue
        elif line.startswith("Portfolio Metrics:"):
            section = "metrics"
            continue
        elif line.startswith("Factor Exposure:"):
            section = "factor_exposure"
            data["factor_exposure"] = {}
            continue
        elif line.startswith("Sector Exposure:"):
            section = "sector_exposure"
            data["sector_exposure"] = {}
            continue
        elif line.startswith("Industry Exposure:"):
            section = "industry_exposure"
            data["industry_exposure"] = {}
            continue
            
        if section == "allocation" and ":" in line:
            ticker, weight = line.split(":")
            data["allocation"][ticker.strip()] = float(weight.strip().rstrip("%"))
        elif section == "metrics" and ":" in line:
            metric, value = line.split(":")
            data["metrics"][metric.strip()] = value.strip()
        elif section == "factor_exposure" and ":" in line:
            factor, exposure = line.split(":")
            data["factor_exposure"][factor.strip()] = float(exposure.strip())
        elif section == "sector_exposure" and ":" in line:
            sector, exposure = line.split(":")
            data["sector_exposure"][sector.strip()] = float(exposure.strip().rstrip("%"))
        elif section == "industry_exposure" and ":" in line:
            industry, exposure = line.split(":")
            data["industry_exposure"][industry.strip()] = float(exposure.strip().rstrip("%"))
    
    return data

def display_portfolio_allocation(parsed_data):
    """Display the portfolio allocation data"""
    if not parsed_data:
        st.warning("No portfolio allocation data available")
        return
    
    # Portfolio allocation
    st.subheader("Portfolio Allocation")
    
    # Create a dataframe for the allocation
    allocation_df = pd.DataFrame({
        "Ticker": list(parsed_data["allocation"].keys()),
        "Weight (%)": list(parsed_data["allocation"].values())
    }).sort_values(by="Weight (%)", ascending=False)
    
    # Display as a bar chart
    fig = px.bar(
        allocation_df,
        x="Ticker",
        y="Weight (%)",
        title="Portfolio Weights",
        color="Weight (%)",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_title="Stock", yaxis_title="Weight (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics in a clean format
    if "metrics" in parsed_data:
        col1, col2, col3 = st.columns(3)
        
        # Extract metrics
        metrics = parsed_data["metrics"]
        expected_return = metrics.get("Expected Return", "N/A")
        volatility = metrics.get("Volatility", "N/A")
        sharpe_ratio = metrics.get("Sharpe Ratio", "N/A")
        
        col1.metric("Expected Annual Return", expected_return)
        col2.metric("Annual Volatility", volatility)
        col3.metric("Sharpe Ratio", sharpe_ratio)
    
    # Display additional exposures if available
    if "sector_exposure" in parsed_data and parsed_data["sector_exposure"]:
        st.subheader("Sector Exposure")
        
        # Create pie chart for sector exposure
        sector_df = pd.DataFrame({
            "Sector": list(parsed_data["sector_exposure"].keys()),
            "Exposure (%)": list(parsed_data["sector_exposure"].values())
        })
        
        fig = px.pie(
            sector_df,
            values="Exposure (%)",
            names="Sector",
            title="Sector Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Industry exposure
    if "industry_exposure" in parsed_data and parsed_data["industry_exposure"]:
        st.subheader("Industry Exposure")
        
        # Create pie chart for industry exposure
        industry_df = pd.DataFrame({
            "Industry": list(parsed_data["industry_exposure"].keys()),
            "Exposure (%)": list(parsed_data["industry_exposure"].values())
        })
        
        fig = px.pie(
            industry_df,
            values="Exposure (%)",
            names="Industry",
            title="Industry Allocation",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Factor exposure
    if "factor_exposure" in parsed_data and parsed_data["factor_exposure"]:
        st.subheader("Factor Exposure")
        
        # Create a dataframe for the factor exposure
        factor_df = pd.DataFrame({
            "Factor": list(parsed_data["factor_exposure"].keys()),
            "Exposure": list(parsed_data["factor_exposure"].values())
        })
        
        # Display as a horizontal bar chart
        fig = px.bar(
            factor_df,
            y="Factor",
            x="Exposure",
            title="Factor Exposure",
            orientation="h",
            color="Exposure",
            color_continuous_scale="RdBu",
            range_color=[-1, 1]
        )
        fig.update_layout(yaxis_title="Factor", xaxis_title="Exposure")
        st.plotly_chart(fig, use_container_width=True)

def display_factor_analysis(text):
    """Display the factor analysis data"""
    if not text:
        st.warning("No factor analysis data available")
        return
    
    st.subheader("Factor Analysis")
    
    # Parse the factor analysis text
    factors = {}
    current_factor = None
    
    lines = text.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
            
        if line.endswith(":") and not line.startswith(" "):
            current_factor = line.rstrip(":")
            factors[current_factor] = {}
        elif current_factor and ":" in line:
            key, value = line.split(":")
            factors[current_factor][key.strip()] = value.strip()
    
    # Remove the Factor Analysis Report entry
    if "Factor Analysis Report" in factors:
        del factors["Factor Analysis Report"]
    
    # Display factor details in expandable sections
    for factor, details in factors.items():
        with st.expander(f"{factor.capitalize()} Factor"):
            col1, col2 = st.columns(2)
            
            # Display statistics
            col1.metric("Mean Score", details.get("Mean Score", "N/A"))
            col1.metric("Standard Deviation", details.get("Std Dev", "N/A"))
            
            # Display top and bottom stocks
            col2.markdown(f"**Top Stocks:** {details.get('Top 3 Stocks', 'N/A')}")
            col2.markdown(f"**Bottom Stocks:** {details.get('Bottom 3 Stocks', 'N/A')}")

def display_management_analysis(text):
    """Display the management analysis data"""
    if not text:
        st.warning("No management analysis data available")
        return
    
    st.subheader("Management Quality Analysis")
    
    # Parse the management analysis text
    management_scores = {}
    
    lines = text.strip().split('\n')
    for line in lines:
        if not line.strip() or ":" not in line:
            continue
            
        if line.startswith("Management Quality Analysis"):
            continue
            
        ticker, score = line.split(":")
        try:
            management_scores[ticker.strip()] = float(score.strip())
        except ValueError:
            management_scores[ticker.strip()] = 0
    
    # Create a dataframe for the management scores
    if management_scores:
        management_df = pd.DataFrame({
            "Ticker": list(management_scores.keys()),
            "Management Score": list(management_scores.values())
        }).sort_values(by="Management Score", ascending=False)
        
        # Display as a bar chart
        fig = px.bar(
            management_df,
            x="Ticker",
            y="Management Score",
            title="Management Quality Scores",
            color="Management Score",
            color_continuous_scale="Viridis",
            range_color=[0, 1]
        )
        fig.update_layout(xaxis_title="Stock", yaxis_title="Management Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Also display as a table
        st.dataframe(management_df, use_container_width=True)

def display_images(images):
    """Display any image files from the reports directory"""
    if not images:
        return
    
    st.subheader("Additional Visualizations")
    
    # Display efficient frontier if it exists
    if "efficient_frontier.png" in images:
        st.image(images["efficient_frontier.png"], caption="Efficient Frontier")
    
    # Display portfolio weights if it exists
    if "portfolio_weights.png" in images:
        st.image(images["portfolio_weights.png"], caption="Portfolio Weights")
    
    # Display any other images
    for name, path in images.items():
        if name not in ["efficient_frontier.png", "portfolio_weights.png"]:
            st.image(path, caption=name.replace(".png", "").replace("_", " ").title())

def display_interactive_html(html_content):
    """Display the interactive HTML content"""
    if not html_content:
        return
    
    st.subheader("Interactive Portfolio Analysis")
    st.components.v1.html(html_content, height=800, scrolling=True)

def run_model_interface():
    """Interface for running the model with parameters"""
    st.subheader("Run Equity Model")
    
    with st.form("run_model_form"):
        col1, col2 = st.columns(2)
        
        test_mode = col1.checkbox("Test Mode", value=True, 
                                help="Run with limited data for testing")
        lookback_years = col2.number_input("Lookback Period (Years)", 
                                         min_value=1, max_value=10, value=2,
                                         help="Number of years to look back for data")
        
        submitted = st.form_submit_button("Run Model")
        
        if submitted:
            with st.spinner("Running equity model... This may take a few minutes."):
                # Run the model with the specified parameters
                run_equity_model(test_mode=test_mode, lookback_years=lookback_years)
                st.success("Model execution completed! Refreshing dashboard...")
                st.experimental_rerun()

def main():
    """Main function for the dashboard"""
    st.set_page_config(
        page_title="Equity Model Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Druckenmiller-style Equity Model Dashboard")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Wall_Street_Sign_NYC.jpg/320px-Wall_Street_Sign_NYC.jpg", width=280)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Run Model", "About"])
    
    # Sidebar metadata
    st.sidebar.title("Model Info")
    st.sidebar.info(
        """
        **Model Type:** Multi-factor Equity Model  
        **Factors:** Momentum, Value, Quality, Growth, Volatility  
        **Optimization:** Mean-Variance (Max Sharpe)  
        """
    )
    
    if page == "Dashboard":
        # Load all report data
        reports = load_reports()
        
        # Check if reports exist
        if not reports or not reports.get("portfolio_allocation"):
            st.warning("No reports found. Please run the model first.")
            st.info("Go to the 'Run Model' tab to execute the equity model.")
            
            # Show a sample of what the dashboard will look like
            st.subheader("Dashboard Preview")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Global_stock_market.jpg/640px-Global_stock_market.jpg", 
                    caption="Sample dashboard visualization (actual data will appear after running the model)")
            return
        
        # Display summary metrics at the top
        portfolio_data = parse_portfolio_allocation(reports.get("portfolio_allocation", ""))
        
        if portfolio_data and "metrics" in portfolio_data:
            st.subheader("Portfolio Performance Summary")
            metrics = portfolio_data["metrics"]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Annual Return", metrics.get("Expected Return", "N/A"))
            col2.metric("Annual Volatility", metrics.get("Volatility", "N/A"))
            col3.metric("Sharpe Ratio", metrics.get("Sharpe Ratio", "N/A"))
        
        # Display tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Portfolio Allocation", 
            "Factor Analysis", 
            "Management Quality", 
            "Visualizations",
            "Interactive Charts"
        ])
        
        with tab1:
            display_portfolio_allocation(portfolio_data)
        
        with tab2:
            display_factor_analysis(reports.get("factor_analysis", ""))
        
        with tab3:
            display_management_analysis(reports.get("management_analysis", ""))
        
        with tab4:
            display_images(reports.get("images", {}))
        
        with tab5:
            display_interactive_html(reports.get("portfolio_analysis_html", ""))
    
    elif page == "Run Model":
        run_model_interface()
    
    elif page == "About":
        st.header("About the Equity Model")
        st.markdown("""
        ### Druckenmiller-style Equity Model
        
        This model implements a multi-factor approach to equity selection and portfolio optimization, 
        inspired by the investment philosophy of Stanley Druckenmiller.
        
        #### Key Components:
        
        1. **Factor Analysis**: Analyzes stocks across multiple factors:
           - Momentum: Captures price trends and relative strength
           - Value: Identifies undervalued stocks
           - Quality: Focuses on financial stability and profitability
           - Growth: Targets companies with strong growth metrics
           - Volatility: Measures risk and stability
        
        2. **Management Quality Analysis**: Evaluates company management through NLP analysis of 10-K filings.
        
        3. **Portfolio Optimization**: Uses mean-variance optimization to maximize the Sharpe ratio.
        
        #### Data Sources:
        - Market data from Yahoo Finance
        - Financial statements and SEC filings
        - S&P 500 constituents
        
        #### Model Parameters:
        - Test mode: Limits the number of stocks for faster processing
        - Lookback period: Number of years of historical data to analyze
        """)
        
        st.subheader("How to Use This Dashboard")
        st.markdown("""
        1. **Dashboard Tab**: View all reports and analyses from the most recent model run
        2. **Run Model Tab**: Configure and execute a new model run with custom parameters
        3. **About Tab**: Learn about the model methodology and dashboard features
        
        The visualization tabs provide different perspectives on the model output:
        - Portfolio allocation shows the optimal weights and performance metrics
        - Factor analysis breaks down the stock selections by factor
        - Management quality displays the NLP analysis of management effectiveness
        - Additional visualizations show the efficient frontier and other plots
        """)

if __name__ == "__main__":
    main() 