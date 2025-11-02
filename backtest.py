# -*- coding: utf-8 -*-
"""
SPUS Quant Analyzer - Backtesting Module (v1.0)

Implements Point 5 (Backtesting Module).
Provides a framework for backtesting a quantitative strategy.

- Backtests a simplified Momentum + Low Vol strategy as a proof-of-concept.
- Calculates standard performance metrics (CAGR, Sharpe, MDD).
- Uses yfinance for historical price data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner="Fetching historical prices...")
def get_historical_data(tickers, start_date, end_date, benchmark='SPUS'):
    """
    Fetches historical daily close prices for all tickers and the benchmark.
    """
    all_tickers = tickers + [benchmark]
    try:
        data = yf.download(all_tickers, start=start_date, end=end_date)
        if data.empty:
            st.error("Failed to download any yfinance data.")
            return None, None
            
        prices = data['Close']
        
        # Handle single ticker download (pandas returns Series)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=all_tickers[0])

        benchmark_prices = prices.pop(benchmark)
        
        # Clean data: drop tickers with no data
        prices = prices.dropna(axis=1, how='all')
        
        return prices, benchmark_prices
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None, None

def calculate_backtest_factors(prices_df):
    """
    Calculates simplified factors (Momentum, Volatility) for the backtest.
    A real implementation would need historical fundamental data.
    """
    
    # 1. Momentum (12M return)
    momentum = prices_df.pct_change(periods=252).iloc[-1] # 1 year approx
    
    # 2. Low Volatility (1Y Volatility)
    returns = prices_df.pct_change()
    low_vol = returns.rolling(window=252).std().iloc[-1] * np.sqrt(252)
    
    # Create a factor dataframe
    factors_df = pd.DataFrame({
        'Momentum': momentum,
        'Low_Vol': low_vol
    })
    
    # Create Z-scores
    # We want HIGH momentum and LOW volatility
    factors_df['Z_Momentum'] = (factors_df['Momentum'] - factors_df['Momentum'].mean()) / factors_df['Momentum'].std()
    factors_df['Z_Low_Vol'] = ((factors_df['Low_Vol'] - factors_df['Low_Vol'].mean()) / factors_df['Low_Vol'].std()) * -1 # Invert
    
    # Combine scores (simple average)
    factors_df['Final_Score'] = (factors_df['Z_Momentum'] + factors_df['Z_Low_Vol']) / 2
    
    factors_df = factors_df.dropna()
    return factors_df

def run_backtest(prices_df, benchmark_prices, top_n=10, rebalance_freq='M'):
    """
    Runs the main backtest loop.
    
    Strategy: At each rebalance date, calculate factors and buy the
    Top N stocks, holding them until the next rebalance date.
    """
    
    # Get all rebalancing dates (e.g., end of month)
    rebalance_dates = prices_df.resample(rebalance_freq).last().index
    
    # Ensure rebalance dates are within our price data
    rebalance_dates = rebalance_dates[rebalance_dates > prices_df.index[252]] # Need 1y lookback
    rebalance_dates = rebalance_dates[rebalance_dates < prices_df.index[-1]]
    
    portfolio_value = 100000  # Starting capital
    portfolio_history = []
    
    for i in range(len(rebalance_dates) - 1):
        
        # --- 1. Ranking Phase ---
        date_start = rebalance_dates[i]
        date_end = rebalance_dates[i+1]
        
        # Get data available *up to* the rebalance date
        prices_lookback = prices_df.loc[:date_start]
        
        # Calculate factors based on lookback data
        factor_scores = calculate_backtest_factors(prices_lookback)
        
        if factor_scores.empty:
            continue
            
        # Select top N stocks
        top_n_stocks = factor_scores.nlargest(top_n, 'Final_Score').index
        
        if len(top_n_stocks) == 0:
            continue
            
        # --- 2. Holding Phase ---
        
        # Get prices for the *holding* period
        holding_prices = prices_df.loc[date_start:date_end]
        
        # Calculate returns for our selected stocks
        # Equally weight the portfolio (Point 3: 10% max)
        weights = 1 / len(top_n_stocks)
        
        # Get returns for the stocks we hold
        holding_returns = holding_prices[top_n_stocks].pct_change().dropna()
        
        # Apply weights
        portfolio_returns = holding_returns.dot([weights] * len(top_n_stocks))
        
        # Calculate value
        period_value = (1 + portfolio_returns).cumprod() * portfolio_value
        portfolio_history.append(period_value)
        
        # Update portfolio value for next loop
        portfolio_value = period_value.iloc[-1]

    if not portfolio_history:
        st.warning("Backtest did not generate any trades.")
        return None, None
        
    # Combine all holding periods into one series
    portfolio_ts = pd.concat(portfolio_history)
    
    # Align benchmark to portfolio
    benchmark_ts = (benchmark_prices.loc[portfolio_ts.index[0]:] / benchmark_prices.loc[portfolio_ts.index[0]]) * 100000
    
    return portfolio_ts, benchmark_ts

def calculate_metrics(portfolio_ts, benchmark_ts, periods_per_year=252):
    """
    Calculates key performance metrics (Point 5).
    """
    if portfolio_ts is None or portfolio_ts.empty:
        return {}
        
    # --- Returns ---
    port_returns = portfolio_ts.pct_change().dropna()
    bench_returns = benchmark_ts.pct_change().dropna()
    
    # --- CAGR (Compound Annual Growth Rate) ---
    total_days = (portfolio_ts.index[-1] - portfolio_ts.index[0]).days
    years = total_days / 365.25
    cagr = (portfolio_ts.iloc[-1] / portfolio_ts.iloc[0])**(1/years) - 1
    
    # --- Volatility ---
    volatility = port_returns.std() * np.sqrt(periods_per_year)
    
    # --- Sharpe Ratio (assuming 0 risk-free rate) ---
    sharpe_ratio = (cagr / volatility) if volatility != 0 else 0
    
    # --- Max Drawdown ---
    cumulative = (1 + port_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # --- Win Rate (vs Benchmark) ---
    win_rate = (port_returns > bench_returns).mean()
    
    return {
        "CAGR": f"{cagr:.2%}",
        "Annual Volatility": f"{volatility:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Win Rate (vs Benchmark)": f"{win_rate:.2%}",
        "Final Portfolio Value": f"${portfolio_ts.iloc[-1]:,.2f}",
        "_drawdown_ts": drawdown # For plotting
    }

def run_simplified_backtest(tickers, start_date):
    """
    Main entry point called by Streamlit app.
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    prices_df, benchmark_prices = get_historical_data(tickers, start_date, end_date)
    
    if prices_df is None:
        return None, None, None
    
    portfolio_ts, benchmark_ts = run_backtest(prices_df, benchmark_prices, top_n=10, rebalance_freq='M')
    
    if portfolio_ts is None:
        return None, None, None
        
    metrics = calculate_metrics(portfolio_ts, benchmark_ts)
    
    return portfolio_ts, benchmark_ts, metrics
