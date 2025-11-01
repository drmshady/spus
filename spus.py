# -*- coding: utf-8 -*-
"""
SPUS Quantitative Analyzer v12 (Fibonacci & Relative Valuation Model)

This script performs a multi-factor quantitative analysis (Value, Momentum, Quality, Size)
on SPUS holdings. It calculates dynamic Fibonacci levels and now integrates
relative valuation by comparing stock P/E and P/B ratios to their sector averages.
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

# --- ⭐️ NEW: Import finvizfinance ---
try:
    from finvizfinance.group import finvizfinance_group
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False
    logging.warning("finvizfinance library not found. Sector valuation will be skipped.")

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


# --- ⭐️ NEW: Function to get Sector Averages ---
def get_sector_valuation_averages(sector):
    """
    Fetches and caches average P/E and P/B for a given sector from Finviz.
    Cache duration is set in config.json (default: 24 hours).
    """
    if not FINVIZ_AVAILABLE or CONFIG is None:
        return None, None

    cache_dir = os.path.join(BASE_DIR, CONFIG['INFO_CACHE_DIR'])
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError as e:
            logging.warning(f"Could not create cache directory {cache_dir}: {e}")
            return None, None
            
    cache_path = os.path.join(cache_dir, "sector_valuation_cache.json")
    cache_duration_seconds = CONFIG.get('SECTOR_CACHE_DURATION_HOURS', 24) * 3600

    sector_data = {}
    try:
        # Load from cache if valid
        if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < cache_duration_seconds):
            with open(cache_path, 'r') as f:
                sector_data = json.load(f)
            logging.info("Loaded sector valuation data from cache.")
        
        # Fetch new data if cache is old or doesn't exist
        else:
            logging.info("Fetching new sector valuation data from Finviz...")
            f_group = finvizfinance_group()
            # 'valuation' contains P/E, P/B, etc.
            df = f_group.screener_view(group_by='Sector', order_by='Name', view='Valuation')
            df.set_index('Name', inplace=True)
            # Convert to dictionary for easy JSON serialization
            sector_data = df.to_dict(orient='index')
            with open(cache_path, 'w') as f:
                json.dump(sector_data, f, indent=4)

    except Exception as e:
        logging.error(f"Error fetching/caching sector data from Finviz: {e}")
        # Try to load expired cache as a last resort
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                sector_data = json.load(f)
            logging.warning("Using expired sector cache due to fetch error.")
        else:
            return None, None # No data available

    # Get data for the specific sector
    if sector in sector_data:
        try:
            sector_pe = pd.to_numeric(sector_data[sector].get('P/E'), errors='coerce')
            sector_pb = pd.to_numeric(sector_data[sector].get('P/B'), errors='coerce')
            return sector_pe, sector_pb
        except Exception as e:
            logging.warning(f"Error parsing sector data for {sector}: {e}")
            return None, None
    else:
        logging.warning(f"Sector '{sector}' not found in Finviz group data.")
        return None, None
# --- ⭐️ END NEW FUNCTION ---


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
# --- END FETCHER FUNCTION ---


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
# --- END Support/Resistance Function ---


# --- ⭐️ UPDATED: Financials and Fair Price Function ---
def calculate_financials_and_fair_price(ticker_obj, last_price, ticker):
    """
    Calculates financials. NOW includes relative sector valuation.
    Caches .info data and sector data.
    """
    info = {}
    cache_dir = os.path.join(BASE_DIR, CONFIG['INFO_CACHE_DIR'])
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError as e:
            logging.warning(f"Could not create cache directory {cache_dir}: {e}")
    cache_path = os.path.join(cache_dir, f"{ticker}.json")
    cache_duration_seconds = CONFIG['INFO_CACHE_DURATION_HOURS'] * 3600

    try:
        if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < cache_duration_seconds):
            with open(cache_path, 'r') as f:
                info = json.load(f)
            logging.info(f"[{ticker}] Loaded fundamental data from cache.")
        else:
            logging.info(f"[{ticker}] Fetching fundamental data from network...")
            info = ticker_obj.info
            with open(cache_path, 'w') as f:
                json.dump(info, f, indent=4)

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

        # --- ⭐️ NEW: Relative Valuation Logic ---
        sector_pe_avg, sector_pb_avg = None, None
        relative_pe_signal, relative_pb_signal = "N/A", "N/A"
        
        if sector != 'N/A' and FINVIZ_AVAILABLE:
            sector_pe_avg, sector_pb_avg = get_sector_valuation_averages(sector)
        
            if pe_ratio is not None and sector_pe_avg is not None and sector_pe_avg > 0:
                if pe_ratio < sector_pe_avg:
                    relative_pe_signal = "Undervalued (Sector)"
                else:
                    relative_pe_signal = "Overvalued (Sector)"
            
            if pb_ratio is not None and sector_pb_avg is not None and sector_pb_avg > 0:
                if pb_ratio < sector_pb_avg:
                    relative_pb_signal = "Undervalued (Sector)"
                else:
                    relative_pb_signal = "Overvalued (Sector)"
        # --- ⭐️ END NEW ---

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
            # --- ⭐️ NEW: Add to dict ---
            'Sector P/E': sector_pe_avg,
            'Relative P/E': relative_pe_signal,
            'Sector P/B': sector_pb_avg,
            'Relative P/B': relative_pb_signal
        }
        return financial_dict

    except Exception as e:
        logging.error(f"Error fetching/processing fundamental data for {ticker}: {e}")
        # Ensure all keys exist, even on failure
        default_keys = ['Forward P/E', 'P/B Ratio', 'Market Cap', 'Sector', 'Dividend Yield', 
                        'Debt/Equity', 'Revenue Growth (QoQ)', 'Graham Number', 'Valuation (Graham)', 
                        'Return on Equity (ROE)', 'EV/EBITDA', 'Price/Sales (P/S)', '52 Week High', '52 Week Low',
                        'Sector P/E', 'Relative P/E', 'Sector P/B', 'Relative P/B']
        return {key: None for key in default_keys}
# --- ⭐️ END UPDATED FUNCTION ---


# --- Process Ticker Function (Unchanged) ---
def process_ticker(ticker):
    null_return = {
        'ticker': ticker, 'momentum': None, 'rsi': None, 'last_price': None,
        'support_resistance': None, 'trend': None, 'macd': None, 'signal_line': None,
        'hist_val': None, 'macd_signal': None, 'financial_dict': None, 'success': False,
        'recent_news': "N/A", 'latest_headline': "N/A", 'earnings_date': "N/A"
    }

    if CONFIG is None:
        logging.error(f"process_ticker ({ticker}): CONFIG is None. Cannot find cache paths.")
        return null_return

    hist_cache_dir = os.path.join(BASE_DIR, CONFIG['HISTORICAL_DATA_DIR'])
    if not os.path.exists(hist_cache_dir):
        try:
            os.makedirs(hist_cache_dir)
        except OSError as e:
            logging.warning(f"Could not create cache directory {hist_cache_dir}: {e}")
    file_path = os.path.join(hist_cache_dir, f'{ticker}.feather')

    existing_hist_df = pd.DataFrame()
    start_date = None

    if os.path.exists(file_path):
        try:
            existing_hist_df = pd.read_feather(file_path)
            existing_hist_df['Date'] = pd.to_datetime(existing_hist_df['Date'])
            existing_hist_df.set_index('Date', inplace=True)

            if not existing_hist_df.empty:
                existing_hist_df.index = pd.to_datetime(existing_hist_df.index, utc=True)
                last_date = existing_hist_df.index.max()
                start_date = last_date + timedelta(days=1)
        except Exception as e:
            logging.warning(f"Error loading existing FEATHER data for {ticker}: {e}")
            existing_hist_df = pd.DataFrame()

    new_hist = pd.DataFrame()
    ticker_obj = None
    fetch_success = False

    today = datetime.now().date()
    should_fetch = True
    if start_date:
        if start_date.date() > today:
            logging.info(f"[{ticker}] History cache is already up-to-date (Last: {start_date.date() - timedelta(days=1)}). Skipping network fetch.")
            should_fetch = False

    if should_fetch:
        for attempt in range(3):
            try:
                ticker_obj = yf.Ticker(ticker)
                if start_date:
                    new_hist = ticker_obj.history(start=start_date.strftime('%Y-%m-%d'))
                else:
                    new_hist = ticker_obj.history(period=CONFIG['HISTORICAL_DATA_PERIOD'])
                fetch_success = True
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, TimeoutError) as e:
                logging.warning(f"[{ticker}] Network error on attempt {attempt+1}/3: {e}. Retrying in {5*(attempt+1)}s...")
                time.sleep(5 * (attempt+1))
            except Exception as e:
                logging.error(f"Error fetching new history for {ticker}: {e}")
                break

    if not fetch_success and existing_hist_df.empty:
        logging.error(f"[{ticker}] Failed to fetch history and no cache exists.")
        return null_return

    if not new_hist.empty:
        combined_hist = pd.concat([existing_hist_df, new_hist]).drop_duplicates(keep='last').sort_index()
    else:
        combined_hist = existing_hist_df

    if combined_hist.empty:
        logging.warning(f"[{ticker}] No history data found (possibly delisted).")
        return null_return

    if not combined_hist.index.is_unique:
         duplicate_dates = combined_hist.index[combined_hist.index.duplicated(keep='first')]
         logging.warning(f"Removed duplicate dates for {ticker} before calculating TA: {duplicate_dates.tolist()}")
         combined_hist = combined_hist[~combined_hist.index.duplicated(keep='first')]

    hist = combined_hist
    momentum = None
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
        min_data_points = CONFIG['MIN_DATA_POINTS_MOMENTUM']
        if len(hist) >= min_data_points:
            try:
                price_now = hist['Close'].iloc[-1]
                price_year_ago = hist['Close'].iloc[-min_data_points]
                momentum = ((price_now - price_year_ago) / price_year_ago) * 100
            except IndexError:
                pass

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

        if ticker_obj is None:
            ticker_obj = yf.Ticker(ticker)

        # This now calls the updated function
        financial_dict = calculate_financials_and_fair_price(ticker_obj, last_price, ticker)

        recent_news_flag = "No"
        latest_headline = "N/A"
        earnings_date = "N/A"
        try:
            news = ticker_obj.news
            if news:
                latest_headline = news[0].get('title', "N/A")
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

        try:
            if not new_hist.empty:
                combined_hist.reset_index().to_feather(file_path)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to save Feather cache: {e}")

        return {
            'ticker': ticker,
            'momentum': momentum,
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
# --- END Process Ticker Function ---
