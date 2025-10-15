import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import random
import scipy.stats as stats 
import seaborn as sns
import matplotlib.pyplot as plt

# Use st.cache_data to run this expensive data simulation and feature 
# engineering function only once, ensuring fast dashboard interactions.
@st.cache_data
def load_and_engineer_data():
    """
    Simulates the Coca-Cola stock history and performs advanced feature engineering.
    Features are based on the analysis in Coca_Cola_Project.ipynb.
    """
    # 1. Simulate a long time-series dataset (60+ years)
    dates = pd.date_range(start='1962-01-02', periods=15311, freq='B') # Business days
    initial_value = 0.05
    time_index = np.arange(len(dates)) / len(dates) * 60 
    
    # Exponential growth trend + noise
    base_price = initial_value * np.exp(time_index * 0.1) 
    noise = np.random.randn(len(dates)).cumsum() * 0.5
    close_prices = base_price + noise
    
    # Simulate OHLC and Volume
    open_prices = close_prices * (1 + np.random.uniform(-0.005, 0.005, len(dates)))
    high_prices = np.maximum(close_prices, open_prices) * (1 + np.random.uniform(0.001, 0.005, len(dates)))
    low_prices = np.minimum(close_prices, open_prices) * (1 - np.random.uniform(0.001, 0.005, len(dates)))
    volume = np.random.randint(1_000_000, 30_000_000, len(dates)) * (1 + time_index/10)

    # Simulate Dividends and Stock Splits
    dividends = np.zeros(len(dates))
    stock_splits = np.zeros(len(dates), dtype=int)
    for i in random.sample(range(len(dates)), 200): 
        dividends[i] = round(random.uniform(0.05, 0.44), 2)
    for i in random.sample(range(len(dates)), 5):
        if dates[i].year > 1980:
            stock_splits[i] = random.choice([2, 3])

    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices.round(4),
        'High': high_prices.round(4),
        'Low': low_prices.round(4),
        'Close': close_prices.round(4),
        'Volume': volume.astype(int),
        'Dividends': dividends,
        'Stock Splits': stock_splits,
    })
    
    # 2. Feature Engineering 
    df['5_Day_SMA'] = df['Close'].rolling(window=5).mean().round(4)
    df['Daily_Range'] = (df['High'] - df['Low']).round(4)
    df['Daily_Returns'] = df['Close'].pct_change().round(6) 
    
    # Annualized Volatility
    df['Annualized_Volatility'] = df['Daily_Returns'].rolling(window=252).std() * np.sqrt(252)
    
    # Volume Price Trend (VPT)
    df['VPT'] = (df['Daily_Returns'] * df['Volume']).cumsum()

    # Drop NaNs created by rolling windows
    df.dropna(inplace=True) 
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def create_dashboard():
    # Streamlit configuration
    st.set_page_config(layout="wide")
    st.title("🥤 Coca-Cola (KO) Financial Dashboard")
    st.markdown("***An dashboard showcasing quantitative analysis, feature engineering, and predictive modeling inputs.***")

    # Load data from cache
    df = load_and_engineer_data()

    # --- 1. Sidebar for Interactivity (Date Range) ---
    st.sidebar.header("⚙️ Dashboard Controls")
    
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.sidebar.slider(
        "Date Range Selection",
        min_date,
        max_date,
        (min_date, max_date)  # Default to the full range
    )
    
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    if start_date > end_date:
        st.sidebar.error("Error: End Date must fall after Start Date.")
        return

    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # --- 2. Data Preview and KPIs ---
    
    # NEW: Data Preview Button
    if st.button("Preview Data (First 10 Rows)"):
        st.dataframe(df_filtered.head(10).style.format(precision=4), use_container_width=True)
    
    st.header("Key Investment & Risk Metrics")
    
    if not df_filtered.empty:
        col1, col2, col3, col4 = st.columns(4)

        initial_close = df_filtered.iloc[0]['Close']
        final_close = df_filtered.iloc[-1]['Close']
        price_change = (final_close - initial_close) / initial_close * 100
        
        col1.metric("Start Close Price", f"${initial_close:,.2f}")
        col2.metric("Period Return", f"{price_change:,.2f}%", f"{final_close - initial_close:,.2f}")
        col3.metric("Avg Daily Volume", f"{df_filtered['Volume'].mean():,.0f}")
        col4.metric("Current Annual Volatility", f"{df_filtered['Annualized_Volatility'].iloc[-1]*100:,.2f}%")

    st.markdown("---")

    # --- 3. Correlation Heatmap (NEW) ---
    st.subheader("1. Feature Correlation Analysis (Heatmap) 🌡️")
    st.caption("Visualizing collinearity between features, crucial for model selection (e.g., avoiding multicollinearity in linear models).")
    
    # Select relevant features for correlation analysis
    corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Returns', '5_Day_SMA', 'Daily_Range']
    corr_df = df_filtered[corr_cols]
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr().round(2)
    
    # Create the heatmap using Matplotlib and Seaborn for reliability
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5, 
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # --- 4. Price and Trend Charts (Retained) ---
    st.subheader("2. Price Action and Trend Analysis 📈")
    
    fig_price = go.Figure()
    
    fig_price.add_trace(go.Candlestick(
        x=df_filtered['Date'],
        open=df_filtered['Open'],
        high=df_filtered['High'],
        low=df_filtered['Low'],
        close=df_filtered['Close'],
        name='Price OHLC',
        increasing_line_color='#00CC96', 
        decreasing_line_color='#EF553B' 
    ))

    fig_price.add_trace(go.Scatter(
        x=df_filtered['Date'], 
        y=df_filtered['5_Day_SMA'], 
        mode='lines', 
        name='5-Day SMA',
        line=dict(color='#E50000', width=2) # Coca-Cola Red
    ))
    
    fig_price.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=450,
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig_price, use_container_width=True)
    
    st.markdown("---")
    
    col_left, col_mid, col_right = st.columns(3)
    
    # --- 5. Annualized Volatility (Risk Analysis) ---
    with col_left:
        st.subheader("3. Annualized Volatility (Risk)")
        
        fig_vol = px.area(
            df_filtered, 
            x='Date', 
            y='Annualized_Volatility', 
            title='252-Day Rolling Volatility',
            color_discrete_sequence=['#FF8C00'] # Orange for Risk
        )
        fig_vol.update_traces(fill='tozeroy', line_color='#FF8C00')
        fig_vol.update_layout(yaxis_title="Annualized Std Dev", height=350)
        st.plotly_chart(fig_vol, use_container_width=True)

    # --- 6. Volume Price Trend (VPT) ---
    with col_mid:
        st.subheader("4. Volume Price Trend (VPT)")
        
        fig_vpt = px.line(
            df_filtered, 
            x='Date', 
            y='VPT', 
            title='Volume Price Trend (VPT)',
            color_discrete_sequence=['#1E90FF'] # Blue for Momentum
        )
        fig_vpt.update_layout(yaxis_title="VPT Value", height=350)
        st.plotly_chart(fig_vpt, use_container_width=True)
        
    # --- 7. Distribution of Daily Returns ---
    with col_right:
        st.subheader("5. Distribution of Daily Returns")
        
        returns = df_filtered['Daily_Returns'].dropna()

        # 1. Create Histogram
        fig_dist = go.Figure(data=[go.Histogram(
            x=returns,
            histnorm='probability density',
            name='Returns Distribution',
            marker_color='#00CC96',
            nbinsx=50
        )])
        
        # 2. Manually calculate and add KDE
        if len(returns) > 1:
            x_range = np.linspace(returns.min(), returns.max(), 500)
            kde = stats.gaussian_kde(returns)
            y_kde = kde.evaluate(x_range)
            
            fig_dist.add_trace(go.Scatter(
                x=x_range,
                y=y_kde,
                mode='lines',
                name='KDE',
                line=dict(color='black', width=2)
            ))
        
        fig_dist.update_layout(
            yaxis_title="Probability Density", 
            xaxis_title="Daily Return (Change in Close)",
            height=350,
            showlegend=False,
            title='Histogram & KDE of Daily Returns'
        )
        st.plotly_chart(fig_dist, use_container_width=True)


    st.markdown("---")
    
    # --- 8. Data Science Pipeline Summary ---
    st.header("💡 Summary")
    st.info(f"""
    This dashboard demonstrates the full analytical lifecycle for a time-series project:
    
    1.  **Data Quality**: Shown via the **Preview Button** and initial **KPIs**.
    2.  **Feature Engineering**: Creation and visualization of features like **5-Day SMA**, **Daily Range**, and **Annualized Volatility**.
    3.  **Exploratory Data Analysis (EDA)**: Visualized through the **Correlation Heatmap** and **Daily Returns Distribution** (checking for normality/risk).
    4.  **Model Readiness**: Data is ready for **Data Splitting** (80% Train / 20% Test) and **Feature Scaling** (**StandardScaler**).
    """)

if __name__ == "__main__":
    create_dashboard()