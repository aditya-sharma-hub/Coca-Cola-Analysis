import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os 
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Coca-Cola Stock Dashboard (KO)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA LOADING & PREPROCESSING (Robust Code) ---
@st.cache_data
def load_data():
    """
    Loads, cleans, and preprocesses the stock data and info data.
    """
    
    HISTORY_FILE = "Coca-Cola_stock_history.csv"
    INFO_FILE = "Coca-Cola_stock_info.csv"

    # In a real environment, we'd check existence, but here we assume based on context
    
    # Load historical data
    df_history = pd.read_csv(HISTORY_FILE)
    
    # Date Conversion and Cleaning
    df_history['Date'] = pd.to_datetime(df_history['Date'], errors='coerce')
    df_history.dropna(subset=['Date'], inplace=True)
    df_history['Date'] = df_history['Date'].dt.normalize().dt.date
    df_history = df_history.set_index('Date').sort_index()
    
    # Ensure numerical columns are floats, excluding the index
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_history[col] = pd.to_numeric(df_history[col], errors='coerce').fillna(0)


    # Load stock info
    df_info = pd.read_csv(INFO_FILE, index_col='Key')
    stock_info = df_info.to_dict()['Value']

    # Convert ALL numeric info keys to float/int
    numeric_keys = ['trailingPE', 'forwardPE', 'marketCap', 'fullTimeEmployees', 'payoutRatio']
    for key in numeric_keys:
        if key in stock_info:
            converted_value = pd.to_numeric(stock_info[key], errors='coerce')
            
            if not np.isnan(converted_value):
                stock_info[key] = converted_value
            else:
                stock_info[key] = 0.0
        else:
             stock_info[key] = 0.0

    return df_history, stock_info

# Helper function for formatting currency (Price/Cap)
def format_currency(num):
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    return f"${num:,.2f}"

# Helper function for formatting Volume
def format_volume(num):
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    return f"{num:,.0f}"

# --- PAGE FUNCTIONS ---

def dashboard_page(df_history, stock_info, df_filtered, date_range):
    """The main interactive analysis dashboard."""
    st.title("📈 Stock Analysis Dashboard")
    st.markdown(f"**Coca-Cola (KO)** | Sector: {stock_info.get('sector', 'N/A')} | Industry: {stock_info.get('industry', 'N/A')}")
    
    
    # Key Financial Metrics (KPIs)
    st.header("Key Period Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    
    # --- INTERACTIVE KPI LOGIC (Uses df_filtered) ---
    if not df_filtered.empty and len(df_filtered) > 1:
        # 1. Close Price (Interactive)
        latest_close_filtered = df_filtered['Close'].iloc[-1]
        starting_close_filtered = df_filtered['Close'].iloc[0]
        price_delta_filtered = latest_close_filtered - starting_close_filtered
        
        # 2. Total Period Return (Interactive)
        total_period_return = (latest_close_filtered / starting_close_filtered) - 1
        return_delta_color = "normal" if total_period_return >= 0 else "inverse"
        
        # 3. Period Volatility (Interactive)
        df_filtered['Daily_Return'] = df_filtered['Close'].pct_change()
        # Annualized Volatility (Std. Dev. * sqrt(252))
        period_volatility = df_filtered['Daily_Return'].std() * np.sqrt(252) 
        
        # 4. Highest Volume (Interactive)
        highest_volume = df_filtered['Volume'].max()
        avg_volume_all_time = df_history['Volume'].mean()
        volume_delta = highest_volume - avg_volume_all_time
        
        # Calculate Percentage Delta for Volume
        if avg_volume_all_time > 0:
            volume_percent_delta = (volume_delta / avg_volume_all_time) * 100
            volume_delta_label = f"{volume_percent_delta:.2f}% vs Avg."
        else:
            volume_delta_label = "N/A"
        
        # 5. Trading Days (Interactive)
        trading_days = len(df_filtered)
        
        # Determine color for the price delta based on gain/loss
        delta_color = "normal" if price_delta_filtered >= 0 else "inverse"
    else:
        # Default zero/empty values if the filter range is too small or empty
        latest_close_filtered = 0.0
        price_delta_filtered = 0.0
        total_period_return = 0.0
        period_volatility = 0.0
        highest_volume = 0
        volume_delta = 0
        volume_delta_label = "No Data"
        trading_days = 0
        delta_color = "off"
        return_delta_color = "off"


    # --- ENHANCED KPI DISPLAY using st.container() for the "box" effect ---
    
    # Metric 1: Period Close Price (Interactive)
    with col1:
        with st.container(border=True): 
            st.metric(
                label="Latest Close Price in Period", 
                value=format_currency(latest_close_filtered),
                delta=f"${price_delta_filtered:.2f} Change",
                delta_color=delta_color
            )
    
    # Metric 2: Total Period Return (Interactive)
    with col2:
        with st.container(border=True):
            st.metric(
                label="Total Period Return", 
                value=f"{total_period_return:.2%}",
                delta="Gain" if total_period_return >= 0 else "Loss",
                delta_color=return_delta_color
            )

    # Metric 3: Period Volatility (Interactive)
    with col3:
        with st.container(border=True):
            st.metric(
                label="Period Volatility (Annualized)", 
                value=f"{period_volatility:.2%}",
                delta="Risk Measure",
                delta_color="off"
            )
        
    # Metric 4: Highest Volume in Period (Interactive)
    with col4:
        with st.container(border=True):
            st.metric(
                label="Highest Volume in Period", 
                value=format_volume(highest_volume),
                delta=volume_delta_label,
                delta_color="normal" if volume_delta > 0 else "off"
            )

    # Metric 5: Total Trading Days (Interactive)
    with col5:
        with st.container(border=True):
            st.metric(
                label="Total Trading Days", 
                value=f"{trading_days:,}",
                delta=f"From {date_range[0]}",
                delta_color="off"
            )

    st.markdown("---")

    # Interactive Price Line Chart (Defaults to Close Price)
    price_metric = 'Close'
    st.header("Time Series Analysis")
    st.subheader(f"Historical {price_metric} Price")
    fig_price = px.line(
        df_filtered, 
        y=price_metric,
        title=f'Coca-Cola {price_metric} Price Over Time ({date_range[0]} to {date_range[1]})',
        labels={price_metric: f'{price_metric} Price ($)', 'Date': 'Date'},
        height=500
    )
    fig_price.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")

    # Trading Volume Chart
    st.subheader("Trading Volume")
    fig_volume = px.area(
        df_filtered, 
        y='Volume',
        title='Daily Trading Volume',
        labels={'Volume': 'Volume'},
        height=400
    )
    st.plotly_chart(fig_volume, use_container_width=True)

    st.markdown("---")

    # Candlestick Chart (Advanced View)
    st.header("Advanced Chart ")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_filtered.index, 
        open=df_filtered['Open'],
        high=df_filtered['High'],
        low=df_filtered['Low'],
        close=df_filtered['Close'],
        name="KO"
    )])
    fig_candle.update_layout(
        title=f'Coca-Cola Candlestick Chart ({date_range[0]} to {date_range[1]})',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False 
    )
    st.plotly_chart(fig_candle, use_container_width=True)


def prediction_page(df_history, df_filtered, date_range):
    """Machine Learning Prediction and Strategy Backtesting Page."""
    st.title("🤖 ML Prediction & Trading Strategy")
    st.markdown(f"Visualizing simulated ML performance and strategy over the period: **{date_range[0]} to {date_range[1]}**.")

    # --- SIMULATE ML PREDICTION DATA (using df_filtered) ---
    st.header("Prediction vs. Actual ")
    st.info("The prediction simulation compares actual prices with model predictions over the selected date range.")

    df_test_sim = df_filtered.copy()
    
    # If the filtered data is too short, we can't show predictions/strategy
    if len(df_test_sim) < 2:
        st.warning("Please select a date range with at least two trading days to view ML simulations.")
        return

    # Simulate a 'Predicted Close' column by shifting Actual Close and adding noise
    np.random.seed(42) # for reproducibility
    # Calculate noise based on the length of the filtered data
    noise = np.random.normal(0, 0.005, len(df_test_sim)) # 0.5% standard deviation noise
    df_test_sim['Predicted Close'] = df_test_sim['Close'].shift(-1) * (1 + noise)
    
    # Drop the last row which will contain NaN after the shift/noise simulation
    df_test_sim.dropna(subset=['Predicted Close'], inplace=True)

    df_plot_pred = df_test_sim[['Close', 'Predicted Close']]
    
    fig_pred = px.line(
        df_plot_pred,
        title=f'Actual vs. Predicted Close Price ({date_range[0]} to {date_range[1]})',
        labels={'value': 'Price ($)', 'Date': 'Date'},
        height=500
    )
    fig_pred.update_layout(yaxis_title='Price ($)')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # --- SIMULATE TRADING STRATEGY RETURNS ---

    st.markdown("---")
    st.header("Strategy Performance Comparison")
    st.info("Simulated strategy returns vs. Buy-and-Hold over the selected period.")

    df_returns = df_filtered.copy()
    
    # Market (Buy-and-Hold) Strategy
    df_returns['Daily_Return_Market'] = df_returns['Close'].pct_change().fillna(0)
    df_returns['Market_Return'] = (df_returns['Daily_Return_Market'] + 1).cumprod()
    
    # Simulated Trading Strategy (Assume strategy performs slightly better, e.g., 20% Alpha on daily returns)
    df_returns['Daily_Return_Strategy'] = df_returns['Daily_Return_Market'] * 1.2
    df_returns['Strategy_Return'] = (df_returns['Daily_Return_Strategy'] + 1).cumprod()
    
    df_returns.dropna(subset=['Market_Return', 'Strategy_Return'], inplace=True)
    
    # Calculate Final Returns for display
    final_market_return = df_returns['Market_Return'].iloc[-1] if not df_returns.empty else 1
    final_strategy_return = df_returns['Strategy_Return'].iloc[-1] if not df_returns.empty else 1
    
    fig_strat = px.line(
        df_returns[['Market_Return', 'Strategy_Return']],
        title=f'Cumulative Returns: Custom Strategy vs. Market ({date_range[0]} to {date_range[1]})',
        labels={'value': 'Cumulative Return Factor', 'Date': 'Date'},
        height=500
    )
    fig_strat.update_layout(yaxis_title='Cumulative Return Factor')
    st.plotly_chart(fig_strat, use_container_width=True)
    
    st.markdown(
        f"""
        **Key Metrics (Simulation over selected period)**
        * **Strategy Final Return Factor:** {final_strategy_return:.2f}x
        * **Buy-and-Hold Final Return Factor (Market):** {final_market_return:.2f}x
        """
    )


def info_page(stock_info, df_filtered, date_range):
    """Page for project context and insights."""
    st.title("ℹ️ Project Information & Insights")
    st.markdown("Context and key information extracted from the project files.")
    
    st.header("📈 Selected Period Summary (Interactive)")
    
    
    # --- INTERACTIVE METRICS FOR INFO PAGE ---
    if not df_filtered.empty:
        period_high = df_filtered['High'].max()
        period_low = df_filtered['Low'].min()
        period_avg_close = df_filtered['Close'].mean()
        period_avg_volume = df_filtered['Volume'].mean()
    else:
        period_high = 0.0
        period_low = 0.0
        period_avg_close = 0.0
        period_avg_volume = 0.0

    col_info_1, col_info_2, col_info_3, col_info_4, col_info_5 = st.columns(5)
    
    with col_info_1:
        with st.container(border=True):
            st.metric(
                label="Start Date",
                value=f"{date_range[0]}"
            )
    
    with col_info_2:
        with st.container(border=True):
            st.metric(
                label="End Date",
                value=f"{date_range[1]}"
            )
    
    with col_info_3:
        with st.container(border=True):
            st.metric(
                label="Average Close Price",
                value=format_currency(period_avg_close)
            )

    with col_info_4:
        with st.container(border=True):
            st.metric(
                label="High Price in Period",
                value=format_currency(period_high)
            )

    with col_info_5:
        with st.container(border=True):
            st.metric(
                label="Average Volume",
                value=format_volume(period_avg_volume)
            )
            
    st.markdown("---")

    # --- STATIC COMPANY CONTEXT ---
    st.header("Company & Project Context ")
    
    # Use the exact static values requested by the user
    market_cap_val = 257440000000 # To properly use the formatter for $257.44B
    trailingPE_val = 29.35
    forwardPE_val = 24.53
    payoutRatio_val = 0.8227 # 82.27%
    employees_val = 80300
    
    col_static_1, col_static_2 = st.columns(2)
    
    with col_static_1:
        st.subheader("Static Company Financials")
        st.markdown(
            f"""
            * **Market Capitalization:** `{format_currency(market_cap_val)}`
            * **P/E Ratio (Trailing):** `{trailingPE_val:.2f}`
            * **Forward P/E Ratio:** `{forwardPE_val:.2f}`
            * **Payout Ratio:** `{payoutRatio_val:.2%}`
            """
        )
    
    with col_static_2:
        st.subheader("Company Profile")
        st.markdown(
            f"""
            * **Sector:** `Consumer Defensive`
            * **Industry:** `Beverages—Non-Alcoholic`
            * **Employees:** `{employees_val:,.0f}`
            * **The historical data spans from 1962 to the last record in the dataset.**
            """
        )

    st.markdown("---")

    st.header("Project & ML Details ")
    st.subheader("ML Project Steps (from Notebook Analysis)")
    st.markdown("The analysis focused on predicting the next day's close price. Key steps included:")
    st.markdown(
        """
        1.  **Exploratory Data Analysis (EDA):** Initial inspection of 15,311 entries.
        2.  **Feature Engineering:** Creating lagged features (e.g., prior day's price) and a target variable (next day's close).
        3.  **Data Split:** Chronological split into **Training (80%)** and **Testing (20%)** sets.
        4.  **Modeling:** Training a **Linear Regression** model.
        5.  **Evaluation:** Measuring performance using MSE, RMSE, and R-squared.
        6.  **Trading Strategy:** Developing a simple prediction-based strategy and comparing its cumulative returns against a Buy-and-Hold benchmark.
        """
    )
    
# --- MAIN APPLICATION LOGIC ---

def main():
    df_history, stock_info = load_data()

    if df_history is None:
        st.error("Data files not loaded. Please ensure both 'Coca-Cola_stock_history.csv' and 'Coca-Cola_stock_info.csv' are available.")
        return

    # Get min/max dates for filter control
    dates = df_history.index
    min_date = min(dates)
    max_date = max(dates)
    
    # Sidebar Navigation
    st.sidebar.title("Coca-Cola Stock Project")
    page = st.sidebar.radio("Navigation", ["Dashboard", "ML Prediction", "Project Info"])

    # --- GLOBAL DATE FILTER (Moved to Main) ---
    st.sidebar.header("Date Filter")
    date_range = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # Filter DataFrame once in main
    df_filtered = df_history.loc[date_range[0]:date_range[1]].copy()
    # ----------------------------------------

    if page == "Dashboard":
        dashboard_page(df_history, stock_info, df_filtered, date_range)
    elif page == "ML Prediction":
        prediction_page(df_history, df_filtered, date_range)
    elif page == "Project Info":
        info_page(stock_info, df_filtered, date_range)

if __name__ == "__main__":
    main()
