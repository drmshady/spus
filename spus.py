# -*- coding: utf-8 -*-
"""
SPUS Quantitative Analyzer v14 (No File Cache)

- REMOVED all .feather and .json file I/O.
- Caching is now handled entirely by @st.cache_data in streamlit_app.py.
- This is a simpler and more robust solution.
"""

import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time
import os
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import openpyxl
from openpyxl.styles import Font
import io
import json
import requests.exceptions
import numpy as np

# --- ⭐️ REMOVED finvizfinance ---

# --- Define Base Directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Function to load external config file ---
def load_config(path='config.json'):
    config_path = os.path.join(BASE_DIR, path)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"FATAL: Configuration file '{config_path}' not found.")
        return None
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not decode JSON from '{config_path}'. Check for syntax errors.")
        return None

# --- Load CONFIG at module level ---
CONFIG = load_config('config.json')


# --- ⭐️ REMOVED get_sector_valuation_averages function ---


# --- FETCHER FUNCTION (Unchanged) ---
def fetch_spus_tickers():
    local_path = os.path.join(BASE_DIR, CONFIG['SPUS_HOLDINGS_CSV_PATH'])
    ticker_column_name = 'StockTicker' 
    if not os.path.exists(local_path):
        logging.error(f"Local SPUS holdings file not found at: {local_path}")
        return []
    try:
        holdings_df = pd.read_csv(local_path)
    except pd.errors.ParserError:
        logging.warning("Pandas ParserError. Trying again with 'skiprows'...")
        for i in range(1, 10):
            try:
                holdings_df = pd.read_csv(local_path, skiprows=i)
                if ticker_column_name in holdings_df.columns:
                    logging.info(f"Successfully parsed CSV by skipping {i} rows.")
                    break
            except Exception:
                continue
        else:
            logging.error(f"Failed to parse CSV from {local_path} even after skipping 9 rows.")
            return []
    except Exception as e:
        logging.error(f"An unexpected error occurred during local CSV read/parse: {e}")
        return []
    if ticker_column_name not in holdings_df.columns:
        logging.error(f"CSV from {local_path} downloaded, but '{ticker_column_name}' column not found.")
        return []
    try:
        holdings_df[ticker_column_name] = holdings_df[ticker_column_name].astype(str)
        first_nan_index = holdings_df[holdings_df[ticker_column_name].isna()].index[0]
        holdings_df = holdings_df.iloc[:first_nan_index]
        logging.info("Removed footer metadata from CSV.")
    except IndexError:
        pass
    except Exception as e:
         logging.warning(f"Error cleaning footer metadata: {e}")
         pass
    ticker_symbols = holdings_df[ticker_column_name].tolist()
    ticker_symbols = [s for s in ticker_symbols if isinstance(s, str) and s and s != ticker_column_name and 'CASH' not in s]
    logging.info(f"Successfully fetched {len(ticker_symbols)} ticker symbols for SPUS from local file.")
    return ticker_symbols

# --- Support/Resistance Function (Unchanged) ---
def calculate_support_resistance(hist_df):
    if hist_df is None or hist_df.empty:
        return None, None, None, None, None, None
    try:
        lookback_period = CONFIG.get('SR_LOOKBACK_PERIOD', 90)
        if len(hist_df) < lookback_period:
            recent_hist = hist_df
        else:
            recent_hist = hist_df.iloc[-lookback_period:]
        support_val = recent_hist['Low'].min()
        support_date = recent_hist['Low'].idxmin()
        resistance_val = recent_hist['High'].max()
        resistance_date = recent_hist['High'].idxmax()
        fib_61_8_level = None
        fib_161_8_level = None
        high_low_diff = resistance_val - support_val
        if high_low_diff > 0:
            fib_61_8_level = resistance_val - (high_low_diff * 0.618)
            fib_161_8_level = resistance_val + (high_low_diff * 0.618)
        return support_val, support_date, resistance_val, resistance_date, fib_61_8_level, fib_161_8_level
    except Exception as e:
        logging.warning(f"Error in calculate_support_resistance: {e}. Defaulting to long-term S/R.")
        support_val = hist_df['Low'].min()
        support_date = hist_df['Low'].idxmin()
        resistance_val = hist_df['High'].max()
        resistance_date = hist_df['High'].idxmax()
        return support_val, support_date, resistance_val, resistance_date, None, None

# --- ⭐️ UPDATED: Financials and Fair Price Function ---
def calculate_financials_and_fair_price(ticker_obj, last_price, ticker):
    """
    Calculates financials. Now *only* fetches yfinance data.
    """
    try:
        # --- ⭐️ REMOVED all file caching. yfinance handles this in-session.
        info = ticker_obj.info
        
        pe_ratio = info.get('forwardPE', None)
        pb_ratio = info.get('priceToBook', None)
        div_yield = info.get('dividendYield', None)
        debt_to_equity = info.get('debtToEquity', None)
        rev_growth = info.get('revenueGrowth', None)
        roe = info.get('returnOnEquity', None)
        ev_ebitda = info.get('enterpriseToEbitda', None)
        ps_ratio = info.get('priceToSalesTrailing12Months', None)
        high_52wk = info.get('fiftyTwoWeekHigh', None)
        low_52wk = info.get('fiftyTwoWeekLow', None)
        market_cap = info.get('marketCap', None)
        sector = info.get('sector', 'N/A')
        eps = info.get('trailingEps', None)
        bvps = None
        graham_number = None
        valuation_signal = "N/A"

        if eps is not None and pb_ratio is not None and eps > 0 and pb_ratio > 0 and last_price is not None:
            bvps = last_price / pb_ratio
            graham_number = (22.5 * eps * bvps) ** 0.5
            if last_price < graham_number:
                valuation_signal = "Undervalued (Graham)"
            else:
                valuation_signal = "Overvalued (Graham)"
        elif eps is not None and eps <= 0:
            valuation_signal = "Unprofitable (EPS < 0)"

        financial_dict = {
            'Forward P/E': pe_ratio,
            'P/B Ratio': pb_ratio,
            'Market Cap': market_cap,
            'Sector': sector,
            'Dividend Yield': div_yield * 100 if div_yield else None,
            'Debt/Equity': debt_to_equity,
            'Revenue Growth (QoQ)': rev_growth * 100 if rev_growth else None,
            'Return on Equity (ROE)': roe * 100 if roe else None,
            'EV/EBITDA': ev_ebitda,
            'Price/Sales (P/S)': ps_ratio,
            '52 Week High': high_52wk,
            '52 Week Low': low_52wk,
            'Graham Number': graham_number,
            'Valuation (Graham)': valuation_signal,
        }
        return financial_dict

    except Exception as e:
        logging.error(f"Error fetching/processing fundamental data for {ticker}: {e}")
        default_keys = ['Forward P/E', 'P/B Ratio', 'Market Cap', 'Sector', 'Dividend Yield', 
                        'Debt/Equity', 'Revenue Growth (QoQ)', 'Graham Number', 'Valuation (Graham)', 
                        'Return on Equity (ROE)', 'EV/EBITDA', 'Price/Sales (P/S)', '52 Week High', '52 Week Low']
        return {key: None for key in default_keys}
# --- ⭐️ END UPDATED FUNCTION ---


# --- ⭐️ UPDATED: Process Ticker Function ⭐️ ---
def process_ticker(ticker):
    """
    Simplified function. Fetches 2y history and info, calculates metrics, returns.
    No file I/O.
    """
    null_return = {
        'ticker': ticker, 'momentum_12_1': None, 'rsi': None, 'last_price': None,
        'support_resistance': None, 'trend': None, 'macd': None, 'signal_line': None,
        'hist_val': None, 'macd_signal': None, 'financial_dict': None, 'success': False,
        'recent_news': "N/A", 'latest_headline': "N/A", 'earnings_date': "N/A",
        'volatility_1y': None
    }

    if CONFIG is None:
        logging.error(f"process_ticker ({ticker}): CONFIG is None.")
        return null_return

    # --- ⭐️ REMOVED all .feather cache logic ---
    
    hist = pd.DataFrame()
    ticker_obj = None
    
    try:
        ticker_obj = yf.Ticker(ticker)
        # Always fetch 2y of data. This is the core fix.
        hist = ticker_obj.history(period="2y")
        if hist.empty:
             logging.warning(f"[{ticker}] No history data found (possibly delisted).")
             return null_return
    except Exception as e:
        logging.error(f"Error fetching new history for {ticker}: {e}")
        return null_return
        
    # (Data cleanup - Unchanged)
    if not hist.index.is_unique:
         duplicate_dates = hist.index[hist.index.duplicated(keep='first')]
         logging.warning(f"Removed duplicate dates for {ticker} before calculating TA: {duplicate_dates.tolist()}")
         hist = hist[~hist.index.duplicated(keep='first')]

    # --- Momentum & Volatility Calcs (Unchanged) ---
    momentum_12_1 = None
    volatility_1y = None
    try:
        monthly_hist = hist['Close'].resample('ME').last()
        
        if len(monthly_hist) >= 14:
            price_1m_ago = monthly_hist.iloc[-2]
            price_13m_ago = monthly_hist.iloc[-14]
            if price_13m_ago != 0:
                momentum_12_1 = ((price_1m_ago - price_13m_ago) / price_13m_ago) * 100
        else:
             logging.warning(f"[{ticker}] Not enough monthly data for 12-1 momentum (Need 14, have {len(monthly_hist)}).")

        daily_returns = hist['Close'].pct_change().dropna() 
        
        if len(daily_returns) >= 252:
            returns_1y = daily_returns.iloc[-252:]
            volatility_1y = returns_1y.std() * np.sqrt(252)
        else:
            logging.warning(f"[{ticker}] Not enough daily data for 1y volatility (Need 252, have {len(daily_returns)}).")
        
    except Exception as e:
        logging.warning(f"[{ticker}] Error calculating 12-1 Momentum or Volatility: {e}")
    # --- END CALCS ---

    rsi = None
    last_price = None
    support_resistance = None
    trend = 'Insufficient data for trend'

    try:
        hist.ta.rsi(length=CONFIG['RSI_WINDOW'], append=True)
        hist.ta.sma(length=CONFIG['SHORT_MA_WINDOW'], append=True)
        hist.ta.sma(length=CONFIG['LONG_MA_WINDOW'], append=True)
        hist.ta.macd(fast=CONFIG['MACD_SHORT_SPAN'], slow=CONFIG['MACD_LONG_SPAN'], signal=CONFIG['MACD_SIGNAL_SPAN'], append=True)
    except Exception as e:
        logging.warning(f"Error calculating TA for {ticker}: {e}. Skipping TA.")
        pass

    rsi_col = f'RSI_{CONFIG["RSI_WINDOW"]}'
    short_ma_col = f'SMA_{CONFIG["SHORT_MA_WINDOW"]}'
    long_ma_col = f'SMA_{CONFIG["LONG_MA_WINDOW"]}'
    macd_col = f'MACD_{CONFIG["MACD_SHORT_SPAN"]}_{CONFIG["MACD_LONG_SPAN"]}_{CONFIG["MACD_SIGNAL_SPAN"]}'
    macd_h_col = f'MACDh_{CONFIG["MACD_SHORT_SPAN"]}_{CONFIG["MACD_LONG_SPAN"]}_{CONFIG["MACD_SIGNAL_SPAN"]}'
    macd_s_col = f'MACDs_{CONFIG["MACD_SHORT_SPAN"]}_{CONFIG["MACD_LONG_SPAN"]}_{CONFIG["MACD_SIGNAL_SPAN"]}'

    if not hist.empty:
        last_price = hist['Close'].iloc[-1]
        rsi = hist[rsi_col].iloc[-1] if rsi_col in hist.columns else None
        last_short_ma = hist[short_ma_col].iloc[-1] if short_ma_col in hist.columns else None
        last_long_ma = hist[long_ma_col].iloc[-1] if long_ma_col in hist.columns else None
        macd = hist[macd_col].iloc[-1] if macd_col in hist.columns else None
        hist_val = hist[macd_h_col].iloc[-1] if macd_h_col in hist.columns else None
        signal_line = hist[macd_s_col].iloc[-1] if macd_s_col in hist.columns else None

        trend = 'No Clear Trend'
        if not pd.isna(last_short_ma) and not pd.isna(last_long_ma) and not pd.isna(last_price):
            if last_short_ma > last_long_ma:
                if last_price > last_short_ma:
                    trend = 'Confirmed Uptrend'
                else:
                    trend = 'Uptrend (Correction)'
            elif last_short_ma < last_long_ma:
                if last_price < last_short_ma:
                    trend = 'Confirmed Downtrend'
                else:
                    trend = 'Downtrend (Rebound)'

        macd_signal = "N/A"
        if macd_h_col in hist.columns and len(hist) >= 2 and not pd.isna(hist_val):
            prev_hist = hist[macd_h_col].iloc[-2]
            if not pd.isna(prev_hist):
                if hist_val > 0 and prev_hist <= 0:
                    macd_signal = "Bullish Crossover (Favorable)"
                elif hist_val < 0 and prev_hist >= 0:
                    macd_signal = "Bearish Crossover (Unfavorable)"
                elif hist_val > 0:
                    macd_signal = "Bullish (Favorable)"
                elif hist_val < 0:
                    macd_signal = "Bearish (Unfavorable)"

        support, support_date, resistance, resistance_date, fib_61_8, fib_161_8 = calculate_support_resistance(hist)
        support_date_str = support_date.strftime('%Y-%m-%d') if pd.notna(support_date) else "N/A"
        resistance_date_str = resistance_date.strftime('%Y-%m-%d') if pd.notna(resistance_date) else "N/A"

        if support is not None and resistance is not None:
            support_resistance = {
                'Support': support,
                'Support_Date': support_date_str,
                'Resistance': resistance,
                'Resistance_Date': resistance_date_str,
                'Fib_61_8': fib_61_8,
                'Fib_161_8': fib_161_8
            }

        financial_dict = calculate_financials_and_fair_price(ticker_obj, last_price, ticker)

        recent_news_flag = "No"
        latest_headline = "N/A"
        earnings_date = "N/A"
        try:
            news = ticker_obj.news
            if news:
                latest_headline = news[0].get('title', "N/Service")
                now_ts = datetime.now().timestamp()
                forty_eight_hours_ago_ts = now_ts - 172800
                for item in news:
                    if item.get('providerPublishTime', 0) > forty_eight_hours_ago_ts:
                        recent_news_flag = "Yes"
                        break
            calendar = ticker_obj.calendar
            if calendar and isinstance(calendar, dict) and 'Earnings Date' in calendar and calendar['Earnings Date']:
                date_val = calendar['Earnings Date'][0]
                if isinstance(date_val, str):
                    earnings_date = date_val
                elif hasattr(date_val, 'strftime'):
                    earnings_date = date_val.strftime('%Y-%m-%d')
                else:
                    earnings_date = str(date_val)
        except Exception as e:
            logging.warning(f"[{ticker}] Error fetching news or calendar: {e}")

        # --- ⭐️ REMOVED .to_feather() save ---

        return {
            'ticker': ticker,
            'momentum_12_1': momentum_12_1,
            'volatility_1y': volatility_1y,
            'rsi': rsi,
            'last_price': last_price,
            'support_resistance': support_resistance,
            'trend': trend,
            'macd': macd,
            'signal_line': signal_line,
            'hist_val': hist_val,
            'macd_signal': macd_signal,
            'financial_dict': financial_dict,
            'recent_news': recent_news_flag,
            'latest_headline': latest_headline,
            'earnings_date': earnings_date,
            'success': True
        }

    return null_return
# --- ⭐️ END UPDATED FUNCTION ---
