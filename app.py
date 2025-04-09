import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Ethereum Metrics Visualizer",
    page_icon="📊",
    layout="wide"
)

# Function to smooth data series
def smooth_series(df, column, window=7):
    """Apply smoothing to a time series"""
    if column in df.columns:
        return df[column].rolling(window=window, center=True, min_periods=1).mean()
    return None

# Function to create an interactive plot
def create_interactive_plot(df, selected_metrics, use_normalized=False, smoothing_window=7):
    """Create an interactive plot with selected metrics"""
    # Create a plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Apply smoothing to selected metrics
    for metric in selected_metrics:
        df[f"{metric}_smooth"] = smooth_series(df, metric, window=smoothing_window)
    
    # If normalized is selected, normalize all series to 0-1 scale
    if use_normalized:
        for metric in selected_metrics:
            smooth_col = f"{metric}_smooth"
            if smooth_col in df.columns:
                min_val = df[smooth_col].min()
                max_val = df[smooth_col].max()
                if max_val > min_val:  # Avoid division by zero
                    df[f"{metric}_norm"] = (df[smooth_col] - min_val) / (max_val - min_val)
    
    # Create color map for metrics
    colors = px.colors.qualitative.Plotly
    
    # Always include ETH price as the first metric on primary y-axis if it's selected
    eth_price_included = 'eth_usd_price' in selected_metrics
    
    if eth_price_included:
        if use_normalized:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['eth_usd_price_norm'],
                    name='ETH Price (normalized)',
                    line=dict(color=colors[0])
                ),
                secondary_y=False
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df['eth_usd_price_smooth'],
                    name='ETH Price ($)',
                    line=dict(color=colors[0])
                ),
                secondary_y=False
            )
    
    # Add other selected metrics on the secondary y-axis (or primary if normalized)
    for i, metric in enumerate(selected_metrics):
        if metric != 'eth_usd_price' or not eth_price_included:  # Skip ETH price if already added
            color_idx = i if not eth_price_included else i+1
            if use_normalized:
                norm_col = f"{metric}_norm"
                if norm_col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['date'], 
                            y=df[norm_col],
                            name=f"{metric} (normalized)",
                            line=dict(color=colors[color_idx % len(colors)])
                        ),
                        secondary_y=False
                    )
            else:
                smooth_col = f"{metric}_smooth"
                if smooth_col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['date'], 
                            y=df[smooth_col],
                            name=metric,
                            line=dict(color=colors[color_idx % len(colors)])
                        ),
                        secondary_y=True if eth_price_included else False
                    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Date")
    
    if use_normalized:
        fig.update_yaxes(title_text="Normalized Value (0-1)")
    else:
        if eth_price_included:
            fig.update_yaxes(title_text="ETH Price ($)", secondary_y=False)
            fig.update_yaxes(title_text="Value", secondary_y=True)
        else:
            fig.update_yaxes(title_text="Value")
    
    # Update layout with larger height
    fig.update_layout(
        title="Ethereum Metrics Comparison",
        hovermode="x unified",
        height=600,  # Increase height for better visibility
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Application title and description
st.title("Ethereum Metrics Visualizer")
st.write("An interactive dashboard to visualize and compare Ethereum blockchain metrics")

# Dictionary of metric descriptions
metric_descriptions = {
    "eth_usd_price": "The price of Ethereum in US Dollars.",
    "transaction_count": "The total number of transactions processed on the Ethereum blockchain in a given day.",
    "new_address_count": "The number of new addresses created on the Ethereum blockchain in a given day.",
    "gas_used": "The total amount of gas consumed by transactions in a given day.",
    "gas_utilization": "The percentage of the gas limit that was utilized in blocks (higher values indicate higher network congestion).",
    "avg_block_tx_count": "The average number of transactions included in each block.",
    "base_fee_gwei": "The base fee (in Gwei) required for transactions, which is burned after EIP-1559.",
    "eth_supply": "The total supply of ETH in circulation.",
    "eth2_staking": "The amount of ETH locked in the Ethereum 2.0 staking contract.",
    "burnt_fees": "The cumulative amount of ETH that has been burned through transaction fees.",
    "gas_price_low": "The low-end gas price in Gwei for transactions.",
    "gas_price_average": "The average gas price in Gwei for transactions.",
    "gas_price_high": "The high-end gas price in Gwei for transactions.",
    "gas_price_base_fee": "The base fee component of gas price in Gwei.",
    "gas_usage_ratio": "The ratio of gas used to the gas limit.",
    "total_nodes": "The total number of nodes participating in the Ethereum network.",
    "eth_btc_ratio": "The price ratio of ETH to BTC, indicating Ethereum's relative value compared to Bitcoin.",
    "Unique Address Total Count": "The total number of unique addresses active on the Ethereum network in a given day.",
    "Unique Address Receive Count": "The number of unique addresses that received ETH in a given day.",
    "Unique Address Sent Count": "The number of unique addresses that sent ETH in a given day."
}

# Sidebar for data input - file uploader or use sample data
st.sidebar.header("Data Input")
data_source = st.sidebar.radio("Select Data Source", ["Upload CSV File", "Use Sample Data"])

df = None

if data_source == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload Ethereum CSV data", type=["csv"])
    if uploaded_file is not None:
        # Load and process the data
        df = pd.read_csv(uploaded_file)
else:
    # Use the sample data provided in the prompt
    sample_data_str = """date,unixTimeStamp_x,transaction_count,unixTimeStamp_y,new_address_count,unixTimeStamp,eth_usd_price,gas_used,gas_utilization,avg_block_tx_count,base_fee_gwei,eth_supply,eth2_staking,burnt_fees,gas_price_low,gas_price_average,gas_price_high,gas_price_base_fee,gas_usage_ratio,total_nodes,eth_btc_ratio,Unique Address Total Count,Unique Address Receive Count,Unique Address Sent Count
01/01/2022,1640995200,1180989,1640995200,121458,1640995200,3766.74,29992112.0,99.97370666666666,321.0,7.094909389,109838249.77944385,1402299.7149588368,2319438.0453994297,7.094909389,7.141676756752922,7.855844424633652,7.094909389,0.07094909389,8354.0,0.0489656528371151,522711,389723,303859
01/01/2023,1672531200,742785,1672531200,170532,1672531200,1200.1,29873565.0,99.57855,44.0,13.511877887,113762666.2178,2326479.714958837,2921021.6070432654,13.511877887,13.60094385352998,14.961038224038647,13.511877887,0.13511877887,8638.0,0.0156006732532168,281764,230298,130374
01/01/2024,1704067200,1101465,1704067200,84381,1704067200,2352.65,18428822.0,61.429406666666665,116.0,11.058938485,117558666.2178,2603859.714958837,3651021.607043266,11.058938485,11.131835461512036,12.24501899551374,11.058938485,0.11058938485,8922.0,0.0305832213392055,483842,413772,198165
01/01/2025,1735689600,1040808,1735689600,96015,1735689600,3353.28,12790955.0,42.636516666666665,121.0,3.186303726,121365066.2178,2823459.714958837,4383021.607043265,3.186303726,3.2073068184929614,3.528037496841742,3.186303726,0.03186303726,9206.0,0.0435908887647253,396439,321486,212044"""
    df = pd.read_csv(io.StringIO(sample_data_str))
    st.sidebar.info("Using sample data with 4 data points from 2022-2025")

if df is not None:
    # Process date column
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'Date(UTC)' in df.columns:
        df['date'] = pd.to_datetime(df['Date(UTC)'], errors='coerce')
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'], errors='coerce')
    elif len(date_columns) > 0:
        df['date'] = pd.to_datetime(df[date_columns[0]], errors='coerce')
    else:
        st.error("No date column found in the data. Please ensure your CSV contains a date column.")
        st.stop()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Display data preview if requested
    with st.expander("Preview Data"):
        st.dataframe(df.head())
    
    # Visualization controls
    st.sidebar.header("Visualization Controls")
    
    # Option to normalize data
    use_normalized = st.sidebar.checkbox("Normalize Data (0-1 scale)", value=True, 
                                    help="Scale all metrics to a 0-1 range for better comparison")
    
    # Smoothing window control
    smoothing_window = st.sidebar.slider("Smoothing Window", min_value=1, max_value=30, value=7,
                                    help="Number of data points to use for smoothing")
    
    # Filter only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available_metrics = [col for col in numeric_cols if col != 'date' and 'unix' not in col.lower()]
    
    # Make sure eth_usd_price is the default selection if available
    default_metrics = []
    if 'eth_usd_price' in available_metrics:
        default_metrics.append('eth_usd_price')
    if 'transaction_count' in available_metrics:
        default_metrics.append('transaction_count')
    
    if not default_metrics and available_metrics:
        default_metrics = [available_metrics[0]]
    
    # User selects metrics
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Display",
        available_metrics,
        default=default_metrics
    )
    
            # Create and display the interactive plot
    if selected_metrics:
        # Main visualization area - use a full-width container
        with st.container():
            # Create a larger plot
            fig = create_interactive_plot(df, selected_metrics, use_normalized, smoothing_window)
            st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of the visualization
        if use_normalized:
            st.info("📊 **Normalized View**: All metrics are scaled to a range between 0 and 1 for easier comparison of trends, regardless of their original values.")
        else:
            st.info("📈 **Raw Values View**: Metrics are displayed with their original values. ETH price uses the left Y-axis, other metrics use the right Y-axis.")
        
        # Add a tip for interaction
        st.markdown("**Tip**: Click on items in the legend to hide/show metrics. Double-click to isolate a single metric.")
        
        # Add metric explanations
        st.subheader("Metric Explanations")
        st.write("Below are explanations of the metrics you've selected:")
        
        for metric in selected_metrics:
            if metric in metric_descriptions:
                st.markdown(f"**{metric}**: {metric_descriptions[metric]}")
            else:
                # For metrics not in our dictionary, provide a generic description
                st.markdown(f"**{metric}**: A metric from the Ethereum blockchain data.")
        
    else:
        st.warning("Please select at least one metric to display")
else:
    st.warning("Please select a data source to begin")