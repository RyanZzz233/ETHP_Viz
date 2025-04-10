import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Function to create an interactive correlation heatmap
def create_correlation_heatmap(df, correlation_metrics):
    """
    Create an interactive correlation heatmap using Plotly
    No filtering is done here - all selected metrics are shown
    """
    if len(correlation_metrics) < 2:
        return None
    
    # Calculate correlation matrix
    corr_df = df[correlation_metrics].corr()
    
    # Create mask for lower triangle - we only show the lower triangle without diagonal
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=0)  # k=0 excludes the diagonal
    corr_df_masked = corr_df.copy()
    
    # Set upper triangle (including diagonal) to NaN (will not be displayed)
    corr_df_masked.mask(mask, np.nan, inplace=True)
    
    # Round values for display
    z_text = corr_df_masked.round(2)
    
    # Create heatmap using Plotly
    fig = px.imshow(
        corr_df_masked,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect="equal",  # Make it square
        title="相关性热图 (Correlation Heatmap)"
    )
    
    # Update layout for better appearance and larger size
    fig.update_layout(
        height=900,
        width=900,
        title_font=dict(size=24),
        margin=dict(l=60, r=60, t=100, b=60),
        coloraxis_colorbar=dict(
            title="Correlation",
            title_font=dict(size=18),
            tickfont=dict(size=16),
            thicknessmode="pixels", 
            thickness=25,
            lenmode="pixels", 
            len=600,
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.05
        )
    )
    
    # Add text annotations with the correlation values - show ALL values in the lower triangle
    annotations = []
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            if i > j:  # Only for lower triangle excluding diagonal
                if not np.isnan(z_text.iloc[i, j]):
                    annotations.append(dict(
                        x=j,
                        y=i,
                        text=str(z_text.iloc[i, j]),
                        font=dict(size=16, color="black", family="Arial Bold"),
                        showarrow=False
                    ))
    
    fig.update_layout(annotations=annotations)
    
    # Improve axis labels
    fig.update_xaxes(
        tickfont=dict(size=14),
        tickangle=45,  # Angle the x-axis labels for better readability
        side="bottom"
    )
    
    fig.update_yaxes(
        tickfont=dict(size=14)
    )
    
    return fig

# Function to create a heatmap of correlations with ETH price
def create_eth_price_correlation_heatmap(df, correlation_metrics):
    """Create a heatmap specifically for correlations with ETH price"""
    if 'eth_usd_price' not in correlation_metrics or len(correlation_metrics) < 2:
        return None
        
    # Calculate correlations with ETH price
    eth_correlations = {}
    for metric in correlation_metrics:
        if metric != 'eth_usd_price':
            corr = df[metric].corr(df['eth_usd_price'])
            eth_correlations[metric] = corr
    
    # Sort by absolute correlation value
    sorted_metrics = sorted(eth_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Create dataframe for heatmap
    if not sorted_metrics:
        return None
    
    # Create a DataFrame with the correlated metrics - exclude eth_usd_price from x-axis
    metrics = [metric for metric, _ in sorted_metrics]
    
    # Create a matrix for the heatmap (1 x n)
    corr_matrix = np.zeros((1, len(metrics)))
    
    for i, (_, corr) in enumerate(sorted_metrics):
        corr_matrix[0, i] = corr
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',  # Using the same color scale as main heatmap
        zmin=-1,
        zmax=1,
        x=metrics,
        y=['eth_usd_price'],
        labels=dict(color="Correlation"),
        title="与以太坊价格 (ETH Price) 的相关性热图"
    )
    
    # Add text annotations
    annotations = []
    
    # Add correlations with other metrics
    for i, (_, corr) in enumerate(sorted_metrics):
        annotations.append(dict(
            x=i,
            y=0,
            text=f"{corr:.2f}",
            font=dict(size=16, color="black", family="Arial Bold"),
            showarrow=False
        ))
    
    fig.update_layout(
        height=300,
        width=max(400, len(metrics) * 80),  # Adjust width based on number of metrics
        annotations=annotations,
        margin=dict(l=60, r=60, t=80, b=60),
        coloraxis_colorbar=dict(
            title="Correlation",
            title_font=dict(size=18),
            tickfont=dict(size=16),
            thicknessmode="pixels", 
            thickness=25,
            lenmode="pixels", 
            len=200,
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.05
        )
    )
    
    # Improve axis labels
    fig.update_xaxes(
        tickfont=dict(size=12),
        tickangle=45  # Angle the x-axis labels for better readability
    )
    
    return fig

# Enhanced Chinese metric descriptions
metric_descriptions_zh = {
    "date": "数据的日历日期。用于比较日常活动和趋势。",
    "Unique Address Total Count": "当天活跃的钱包地址总数（发送或接收）。高数字表明以太坊用户参与度或兴趣强烈。",
    "Total Count": "当天的交易总数。表示整体网络使用情况；峰值可能意味着需求增加（DeFi、NFT等）。",
    "Unique Address Receive Count": "当天接收ETH的唯一钱包地址数量。帮助识别ETH是否被更广泛地分配。",
    "Unique Address Sent Count": "当天发送ETH的唯一钱包地址数量。帮助识别ETH是否被更多用户转移。",
    "unixTimeStamp": "日期的数字表示，主要用于计算目的。对时间序列分析有用。",
    "transaction_count": "处理的以太坊交易总数。显示网络的拥堵或繁忙程度。",
    "new_address_count": "在网络上创建的全新钱包地址数量。高数字表明新用户正在加入以太坊生态系统。",
    "eth_usd_price": "当天1 ETH的美元价格。与需求和投资者情绪相关 - 价格上涨可能会增加活动。",
    "gas_used": "当天以太坊上使用的总gas量（计算）。显示网络上完成了多少工作 - 高gas使用量通常意味着许多智能合约（DeFi、NFT）正在运行。",
    "gas_utilization": "以太坊区块的填充程度（占最大容量的百分比）。接近100%表示网络完全利用，可能拥堵。",
    "avg_block_tx_count": "每个区块中的平均交易数量。数字越高意味着区块中包含的交易越多；如果这个值下降而gas使用量保持高位，交易可能很复杂。",
    "base_fee_gwei": "处理交易所需的最低费用（以Gwei为单位，在EIP-1559中引入）。低基本费用意味着拥堵度低或对区块空间的需求低。",
    "eth_supply": "流通中的ETH总量。对货币政策和通胀跟踪很重要。",
    "eth2_staking": "在以太坊2.0（信标链）中质押的ETH数量。更多质押 = 对ETH 2.0转型的安全性和信心更强。",
    "burnt_fees": "通过EIP-1559永久销毁的ETH数量。有助于减少ETH供应 - 促使ETH随着时间推移成为通缩货币。",
    "gas_price_low": "当天支付的最低gas价格（以Gwei为单位）。",
    "gas_price_average": "平均gas价格。",
    "gas_price_high": "有人为了优先处理交易而支付的最高价格。",
    "gas_price_base_fee": "gas费用的强制部分（以Gwei为单位）。差异小意味着需求稳定；峰值表明拥堵或优先交易。",
    "gas_usage_ratio": "使用的gas量与gas限制的比率。表示网络效率 - 接近1意味着完全利用。",
    "total_nodes": "运行网络的以太坊节点（计算机）数量。更多节点 = 更强的去中心化和安全性。",
    "eth_btc_ratio": "ETH与BTC相比的价值。帮助衡量ETH相对于比特币的表现 - 比率上升表明ETH正在增强。"
}

# Application title and description
st.title("Ethereum Metrics Visualizer")
st.write("An interactive dashboard to visualize and analyze Ethereum blockchain metrics")

# Sidebar for data input - file uploader only
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Ethereum CSV data", type=["csv"])

df = None

if uploaded_file is not None:
    # Load and process the data
    df = pd.read_csv(uploaded_file)
else:
    # Placeholder message when no file is uploaded
    st.warning("Please upload a CSV file to begin")
    st.stop()

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

# Display the count of available metrics
st.sidebar.info(f"Found {len(available_metrics)} metrics in the dataset")

# Make sure eth_usd_price is the default selection if available
default_metrics = []
if 'eth_usd_price' in available_metrics:
    default_metrics.append('eth_usd_price')
if 'transaction_count' in available_metrics:
    default_metrics.append('transaction_count')

if not default_metrics and available_metrics:
    default_metrics = [available_metrics[0]]

# User selects metrics for time series visualization
selected_metrics = st.sidebar.multiselect(
    "Select Metrics for Time Series Visualization",
    available_metrics,
    default=default_metrics
)

# User selects metrics for correlation heatmap
st.sidebar.header("Correlation Analysis")

# Add correlation threshold slider
correlation_threshold = st.sidebar.slider(
    "Correlation Significance Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.3,  # Default to filter out weak correlations (-0.3 to 0.3)
    step=0.05,
    help="Only show metrics with correlation to ETH price stronger than this threshold (absolute value). Set to 0 to show all metrics."
)

# Filter metrics based on correlation with ETH price
if 'eth_usd_price' in available_metrics:
    # Calculate correlations with ETH price
    eth_correlations = {}
    for metric in available_metrics:
        if metric != 'eth_usd_price':
            if metric in df.columns and df[metric].dtype.kind in 'ifc':  # Only numeric columns
                corr = abs(df['eth_usd_price'].corr(df[metric]))
                eth_correlations[metric] = corr
    
    # Only keep metrics with correlation above threshold
    significant_metrics = ['eth_usd_price']
    for metric, corr in eth_correlations.items():
        if corr > correlation_threshold:
            significant_metrics.append(metric)
    
    # Display info about the filtered metrics
    if correlation_threshold > 0:
        st.sidebar.info(f"Found {len(significant_metrics)-1} metrics with correlation > {correlation_threshold} to ETH price")
    
    # Default to metrics with significant correlation to ETH price
    default_corr_metrics = significant_metrics
else:
    # If ETH price is not available, use all metrics
    default_corr_metrics = available_metrics[:min(10, len(available_metrics))]  # Limit to 10 by default
    st.sidebar.info(f"ETH price not found. Selecting first {len(default_corr_metrics)} metrics by default")

# User selects metrics for correlation heatmap
correlation_metrics = st.sidebar.multiselect(
    "Select Metrics for Correlation Heatmap",
    available_metrics,
    default=default_corr_metrics
)

# Create tabs for different analyses
tab1, tab2 = st.tabs(["Time Series Visualization", "Correlation Analysis"])

with tab1:
    # Main visualization area - use a full-width container
    if selected_metrics:
        # Create a larger plot
        fig = create_interactive_plot(df, selected_metrics, use_normalized, smoothing_window)
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of the visualization
        if use_normalized:
            st.info("📊 **标准化视图**: 所有指标均缩放至0到1范围，便于比较趋势，无论原始值如何。")
        else:
            st.info("📈 **原始值视图**: 显示指标的原始值。ETH价格使用左侧Y轴，其他指标使用右侧Y轴。")
        
        # Add a tip for interaction
        st.markdown("**提示**: 点击图例中的项目可隐藏/显示指标。双击可隔离单个指标进行查看。")
    else:
        st.warning("请至少选择一个指标进行显示")

with tab2:
    # Show the correlation heatmap and explanation
    st.subheader("相关性热图 (Correlation Heatmap)")
    
    # Display the current threshold setting
    if correlation_threshold > 0:
        st.info(f"基于与ETH价格相关系数绝对值 > {correlation_threshold} 筛选指标。")
    
    if len(correlation_metrics) > 1:
        # If eth_usd_price is included, show correlations with it specifically first
        if 'eth_usd_price' in correlation_metrics:
            st.subheader("与以太坊价格 (ETH Price) 的相关性")
            
            # Create ETH price correlation heatmap
            eth_corr_fig = create_eth_price_correlation_heatmap(df, correlation_metrics)
            if eth_corr_fig:
                st.plotly_chart(eth_corr_fig, use_container_width=True)
            else:
                st.info(f"没有指标与ETH价格有足够的相关性")
            
            # Also display as table for clarity
            eth_correlations = {}
            for metric in correlation_metrics:
                if metric != 'eth_usd_price':
                    corr = df[metric].corr(df['eth_usd_price'])
                    eth_correlations[metric] = corr
            
            # Sort by absolute correlation value
            sorted_correlations = sorted(eth_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Display in a cleaner table format
            eth_corr_data = []
            for metric, corr in sorted_correlations:
                eth_corr_data.append({"指标 (Metric)": metric, "相关性 (Correlation)": f"{corr:.2f}"})
            
            if eth_corr_data:
                st.table(pd.DataFrame(eth_corr_data))
            
            # Add a spacer
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Now show the general correlation heatmap
            st.subheader("指标间相关性热图 (Correlation Heatmap)")
        
        # Create the heatmap with selected metrics - no additional filtering
        corr_fig = create_correlation_heatmap(df, correlation_metrics)
        
        if corr_fig:
            # Create a centered container for the heatmap
            container = st.container()
            with container:
                # Create a centered column to hold the heatmap
                col1, col2, col3 = st.columns([1, 10, 1])
                with col2:
                    st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info(f"没有足够的指标来创建相关性热图")
        
        # Add a spacer
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Display Chinese explanations for selected metrics
        st.subheader("指标解释 (Metric Explanations)")
        
        if correlation_metrics:
            col1, col2 = st.columns(2)
            
            # Split metrics into two columns for better space usage
            metrics_left = correlation_metrics[:len(correlation_metrics)//2 + len(correlation_metrics)%2]
            metrics_right = correlation_metrics[len(correlation_metrics)//2 + len(correlation_metrics)%2:]
            
            with col1:
                for metric in metrics_left:
                    with st.expander(f"{metric}"):
                        if metric in metric_descriptions_zh:
                            st.write(metric_descriptions_zh[metric])
                        else:
                            st.write("以太坊区块链数据中的一个指标。")
            
            with col2:
                for metric in metrics_right:
                    with st.expander(f"{metric}"):
                        if metric in metric_descriptions_zh:
                            st.write(metric_descriptions_zh[metric])
                        else:
                            st.write("以太坊区块链数据中的一个指标。")
        else:
            st.info("没有选择任何指标。")
    else:
        st.warning("请至少选择两个指标用于相关性热图")