# -*- coding: utf-8 -*-
"""
SPUS Quantitative Analyzer v19.12 (Force v1 API)

- Implements data fallbacks (Alpha Vantage) and validation.
- Fetches a wide range of metrics for 6-factor modeling.
- Includes robust data fetching for tickers and fundamentals.
- REWORKED: find_order_blocks for SMC (BOS, Mitigation, Validation).
- FIXED: Replaced pandas_ta.pivothigh/low with scipy.signal.argrelelextrema.
- FIXED: Corrected logic in find_order_blocks.
- FIXED: Hardened all type-casting in parse_ticker_data.
- FIXED: Replaced pd.NA with bool(False) for nullable boolean columns
  ('earnings_volatile', 'earnings_negative') to force a pure
  boolean column and fix all pyarrow.lib.ArrowInvalid conversion errors.
- FIXED: Removed 'hist_df' and flattened 'news_list' from the
  parsed dictionary to ensure the final DataFrame is flat.
- FIXED: Removed SyntaxError typo ('C') in risk management section.
- ADDED: 'entry_signal' filter based on proximity to validated OBs.
- MODIFIED: Risk logic to use dynamic 'Final Stop Loss' comparing
  ATR vs. 'Cut Loss' (last swing low).
- UPGRADED: find_order_blocks to use Break of Structure (BOS),
  Fair Value Gaps (FVG), and Volume confirmation.
- ADDED: pct_above_cutloss metric for filtering.
- FIXED: All failure-path return dictionaries to include default booleans,
  preventing pyarrow crashes in streamlit.
- FIXED: Changed deprecated yfinance .news to .get_news()
- REFACTORED: Now accepts CONFIG object in all major functions
  to support multi-market analysis.
- NEW: fetch_market_tickers() as a router for ticker sources.
- NEW: fetch_tickers_from_local_csv() for TASI and EGX.
- ✅ ADDED: 'shortName' (Company Name) to data fetching and parsing.
- ✅ FIXED: Added 'shortName' to all failure dictionaries.
- ✅ NEW (P1): check_market_regime() function added.
- ✅ MODIFIED (P2): parse_ticker_data() Risk Management section
     uses Bearish OB as a potential Take Profit target.
- ✅ ADDED (P2): MACD_EXIT_SIGNAL added for exit rule logic.
- ✅ MODIFIED (P4): Replaced Google Gemini with OpenAI (ChatGPT).
- ✅ MODIFIED (P4): AI function call moved to streamlit_app.py
  for on-demand (lazy) loading to reduce API costs.
- ✅ MODIFIED (P5): Added check for placeholder API key.
- ✅ MODIFIED (USER REQ): Changed default AI model to gpt-4o-mini
- ✅ ADDED (USER REQ): Retry logic for fetch_data_yfinance
- ✅ ADDED (USER REQ): Retry logic for fetch_data_alpha_vantage
- ✅ ADDED (USER REQ): Volatility gate (ATR % > X) in check_market_regime
- ✅ ADDED (USER REQ): Dynamic risk % (from equity) in parse_ticker_data
- ✅ MODIFIED (USER REQ): get_ai_stock_analysis now supports a 
  `position_assessment` type for the portfolio tab.
- ✅ BUG FIX: Fixed NameError 'parsed' is not defined in get_ai_stock_analysis
- ✅ NEW: Replaced all news fetching with Finnhub.io API.
- ✅ NEW: Added `fetch_data_finnhub_news` function.
- ✅ MODIFIED: `process_ticker` now calls Finnhub for news.
- ✅ MODIFIED: `parse_ticker_data` now parses Finnhub news format.
- ✅ REMOVED: `search_google_news` (fragile scraping method).
- ✅ REMOVED: News fetching from yfinance and Alpha Vantage.
- ✅ NEW: Added `get_ai_portfolio_summary` for holistic portfolio review.
- ✅ MODIFIED (USER REQ): `get_ai_portfolio_summary` prompt now
     analyzes detailed holdings for stock-specific recommendations.
- ✅ MODIFIED (USER REQ): `process_ticker` now has a `fetch_news`
     parameter to limit API calls.
- ✅ UPGRADE (Phase 1): Added 12-hour caching to `fetch_market_tickers`
     to make web scraping fallback more robust.
- ✅ UPGRADE (Phase 2): Added `smc_volume_missing` flag for UI trust.
- ✅ UPGRADE (Phase 2): Added `z_score_fallback` flag for small sectors.
- ✅ UPGRADE (Phase 3): Replaced `recent_news_count` with
     `news_sentiment_score` (float) calculated on-demand by AI.
- ✅ NEW (USER REQ): Added `get_ai_top20_summary` function.
"""

import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time
import os
import logging
from datetime import datetime, timedelta # <-- ✅ NEW IMPORT
import json
import numpy as np
from bs4 import BeautifulSoup
import random
from scipy.signal import argrelextrema 
import openai
from openai import OpenAI
import google.genai as genai 
from google.genai.errors import APIError as GeminiAPIError
import pickle # <-- ✅ ADDED (Phase 1)
# ❌ REMOVED: from urllib.parse import quote_plus

# --- Define Base Directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load CONFIG at module level (Used ONLY by scheduler) ---
# The Streamlit app will pass its own CONFIG object
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

# --- 1. DATA RELIABILITY & SOURCING ---

def fetch_spus_tickers_from_csv(local_path):
    """Helper to parse the local SPUS CSV file."""
    # --- MODIFIED: Use 'StockTicker' from the user's config.json ---
    ticker_column_name = 'StockTicker' 
    if not os.path.exists(local_path):
        logging.error(f"Local SPUS holdings file not found at: {local_path}")
        return None
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
            return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during local CSV read/parse: {e}")
        return None
    
    if ticker_column_name not in holdings_df.columns:
        logging.error(f"CSV from {local_path} found, but '{ticker_column_name}' column not found.")
        # --- FALLBACK: Try finding 'ticker' ---
        if 'ticker' in holdings_df.columns:
            logging.warning("Found 'ticker' column, using it as fallback.")
            ticker_column_name = 'ticker'
        else:
            return None
    
    return holdings_df[ticker_column_name].tolist()

def fetch_spus_tickers_from_web(url="https://www.sp-funds.com/spus/"):
    """Fallback to scrape SPUS holdings page if CSV fails."""
    logging.info(f"Attempting to scrape tickers from {url}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        holdings_table = soup.find('table', {'id': 'etf-holdings'})
        if not holdings_table:
            holdings_table = soup.find('table', class_='holdings-table')
            
        if not holdings_table:
            logging.error(f"Could not find holdings table on {url}. Web scrape failed.")
            return None
            
        tickers = []
        rows = holdings_table.find('tbody').find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) > 1:
                ticker = cells[1].get_text(strip=True)
                if ticker and ticker.isalpha() and ticker.upper() not in ["CASH", "OTHER"]:
                    tickers.append(ticker)
        
        if not tickers:
            logging.warning(f"Found table on {url}, but extracted no tickers.")
            return None
            
        logging.info(f"Successfully scraped {len(tickers)} tickers from {url}.")
        return tickers

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error parsing HTML from {url}: {e}")
        return None

# --- ✅ NEW FUNCTION ---
def fetch_tickers_from_local_csv(csv_path):
    """Generic function to load a ticker list from a local CSV file."""
    local_path = os.path.join(BASE_DIR, csv_path)
    if not os.path.exists(local_path):
        logging.error(f"Local ticker file not found at: {local_path}")
        return None
    try:
        df = pd.read_csv(local_path)
        if 'ticker' not in df.columns:
            logging.error(f"CSV from {local_path} found, but 'ticker' column not found.")
            return None
        return df['ticker'].dropna().unique().tolist()
    except Exception as e:
        logging.error(f"Error reading local CSV {local_path}: {e}")
        return None

# --- ✅ NEW ROUTER FUNCTION ---
def fetch_market_tickers(CONFIG):
    """
    Reads the CONFIG to determine *how* to fetch the list of tickers.
    """
    source = CONFIG.get("TICKER_SOURCE", "SPUS_CSV") # Default to old method
    
    if source == "SPUS_CSV":
        # --- 1. Try local SPUS CSV ---
        local_path = os.path.join(BASE_DIR, CONFIG['SPUS_HOLDINGS_CSV_PATH'])
        tickers = fetch_spus_tickers_from_csv(local_path)
        
        # --- 2. Try web scrape fallback (with Caching) ---
        if tickers is None:
            logging.warning("Local SPUS CSV failed, trying web scrape fallback...")
            
            # --- ✅ NEW: Caching Logic ---
            CACHE_DIR = os.path.join(BASE_DIR, "cache")
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(CACHE_DIR, "scraped_tickers.pkl")
            CACHE_TTL_SECONDS = 12 * 3600 # 12 hours

            try:
                if os.path.exists(cache_path):
                    cache_mod_time = os.path.getmtime(cache_path)
                    if (time.time() - cache_mod_time) < CACHE_TTL_SECONDS:
                        logging.info(f"Using cached ticker list from {cache_path} (less than 12h old).")
                        with open(cache_path, 'rb') as f:
                            tickers = pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load ticker cache, will re-scrape: {e}")
                tickers = None # Ensure tickers is None so scrape runs
            # --- ✅ END: Caching Logic ---

            if tickers is None: # If cache was old, missing, or failed
                logging.info("Cache empty or stale, running web scraper...")
                tickers = fetch_spus_tickers_from_web()
                
                # If scrape was successful, save to cache
                if tickers:
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(tickers, f)
                        logging.info(f"Successfully saved new ticker list to cache: {cache_path}")
                    except Exception as e:
                        logging.warning(f"Failed to save ticker cache: {e}")

    elif source == "LOCAL_CSV":
        csv_file = CONFIG.get("TICKER_LIST_FILE")
        if not csv_file:
            logging.error("TICKER_SOURCE is 'LOCAL_CSV' but 'TICKER_LIST_FILE' is missing in config.")
            return []
        tickers = fetch_tickers_from_local_csv(csv_file)
        
    else:
        logging.error(f"Unknown TICKER_SOURCE: {source}")
        return []

    if tickers is None:
        logging.critical(f"All ticker sources for {source} failed. Returning empty list.")
        return []

    # Clean list
    tickers = [s for s in tickers if isinstance(s, str) and s and 'CASH' not in s.upper() and 'OTHER' not in s.upper()]
    # Remove potential header artifacts
    tickers = [t for t in tickers if t != 'StockTicker' and len(t) < 10] # Increased length for .SR, .CA
    
    logging.info(f"Successfully fetched {len(tickers)} unique ticker symbols for market.")
    return list(set(tickers)) # Return unique list


# --- ✅ MODIFIED (USER REQ): Added Volatility Gate ---
def check_market_regime(CONFIG):
    """
    Checks the trend of the primary market index (e.g., S&P 500/TASI).
    Returns 'BULLISH', 'BEARISH', or 'TRANSITIONAL'.
    Includes a volatility gate to override trend in choppy markets.
    """
    try:
        # Get index ticker from config, default to ^GSPC
        index_ticker = CONFIG.get("MARKET_INDEX_TICKER", "^GSPC") 
        
        # Get MA windows from config
        tech_config = CONFIG.get('TECHNICALS', {})
        short_ma_window = tech_config.get('SHORT_MA_WINDOW', 50)
        long_ma_window = tech_config.get('LONG_MA_WINDOW', 200)
        atr_window = tech_config.get('ATR_WINDOW', 14)
        vol_threshold = tech_config.get('REGIME_VOLATILITY_THRESHOLD', 5.0) # e.g., 5.0%

        index_obj = yf.Ticker(index_ticker)
        # Fetch slightly more than 1y to ensure 200MA is calculated
        hist = index_obj.history(period="300d") 
        
        if hist.empty or len(hist) < long_ma_window:
            logging.warning(f"Could not fetch sufficient history for market index {index_ticker}")
            return "UNKNOWN" # Failsafe: allow run if index data is missing
        
        # --- 1. Volatility Gate Check ---
        hist.ta.atr(length=atr_window, append=True)
        atr_col = f'ATRr_{atr_window}'
        
        if atr_col in hist.columns:
            last_price = hist['Close'].iloc[-1]
            last_atr = hist[atr_col].iloc[-1]
            
            if pd.notna(last_price) and pd.notna(last_atr) and last_price > 0:
                volatility_pct = (last_atr / last_price) * 100
                if volatility_pct > vol_threshold:
                    logging.warning(f"Market Regime for {index_ticker} is VOLATILE (ATR {volatility_pct:.2f}% > {vol_threshold}%). Forcing TRANSITIONAL.")
                    return "TRANSITIONAL"
        
        # --- 2. Trend Check (if volatility is acceptable) ---
        hist[f'SMA_{short_ma_window}'] = hist['Close'].rolling(window=short_ma_window).mean()
        hist[f'SMA_{long_ma_window}'] = hist['Close'].rolling(window=long_ma_window).mean()

        last_price = hist['Close'].iloc[-1] # Already defined, but good for clarity
        last_short_ma = hist[f'SMA_{short_ma_window}'].iloc[-1]
        last_long_ma = hist[f'SMA_{long_ma_window}'].iloc[-1]

        if pd.isna(last_short_ma) or pd.isna(last_long_ma):
            logging.warning(f"Could not calculate MAs for index {index_ticker}")
            return "UNKNOWN"

        if last_price > last_short_ma and last_short_ma > last_long_ma:
            regime = "BULLISH"
        elif last_price < last_short_ma and last_short_ma < last_long_ma:
            regime = "BEARISH"
        else:
            regime = "TRANSITIONAL"
            
        logging.info(f"Market Regime for {index_ticker} is {regime} (Volatility OK)")
        return regime

    except Exception as e:
        logging.error(f"Error checking market regime for {index_ticker}: {e}")
        return "UNKNOWN" # Failsafe

# --- ✅ MODIFIED FUNCTION (Accepts CONFIG) ---
def find_order_blocks(hist_df_full, ticker, CONFIG):
    """
    Finds the most recent Bullish and Bearish Order Blocks based on
    Smart Money Concepts (SMC) including Break of Structure (BOS),
    Fair Value Gaps (FVG), and Volume confirmation.
    """
    
    # --- 1. Initialize & Load Config ---
    smc_config = CONFIG.get('TECHNICALS', {}).get('SMC_ORDER_BLOCKS', {})
    lookback = smc_config.get('LOOKBACK_PERIOD', 252)
    pivots_n = smc_config.get('PIVOT_BARS', 5) # How many bars on each side to confirm a pivot
    vol_lookback = smc_config.get('VOLUME_LOOKBACK', 50)
    vol_multiplier = smc_config.get('VOLUME_MULTIPLIER', 1.5)
    fvg_check = smc_config.get('CHECK_FOR_FVG', True)

    # Base return object - now includes FVG and Volume flags
    ob_data = {
        'bullish_ob_low': np.nan, 'bullish_ob_high': np.nan, 'bullish_ob_validated': bool(False),
        'bullish_ob_fvg': bool(False), 'bullish_ob_volume_ok': bool(False),
        'bearish_ob_low': np.nan, 'bearish_ob_high': np.nan, 'bearish_ob_validated': bool(False),
        'bearish_ob_fvg': bool(False), 'bearish_ob_volume_ok': bool(False),
        'last_swing_low': np.nan, 'last_swing_high': np.nan,
        'smc_volume_missing': bool(False) # <-- ✅ ADDED (Phase 2)
    }

    if len(hist_df_full) < lookback:
        logging.warning(f"[{ticker}] Not enough history ({len(hist_df_full)} days) for SMC analysis (needs {lookback}).")
        return ob_data

    try:
        hist_df = hist_df_full.iloc[-lookback:].copy()
        
        # --- 2. Add Volume & Pivot Indicators ---
        if 'Volume' not in hist_df.columns:
             logging.warning(f"[{ticker}] 'Volume' not in hist_df. Cannot perform volume confirmation.")
             hist_df['Volume'] = 0
             vol_multiplier = 999 # Effectively disables volume check
             ob_data['smc_volume_missing'] = bool(True) # <-- ✅ ADDED (Phase 2)
             
        hist_df['vol_sma'] = hist_df['Volume'].rolling(window=vol_lookback).mean()
        
        high_idx = argrelextrema(hist_df['High'].values, np.greater_equal, order=pivots_n)[0]
        low_idx = argrelextrema(hist_df['Low'].values, np.less_equal, order=pivots_n)[0]
        
        hist_df['sh'] = np.nan
        hist_df.iloc[high_idx, hist_df.columns.get_loc('sh')] = hist_df.iloc[high_idx]['High']
        hist_df['sl'] = np.nan
        hist_df.iloc[low_idx, hist_df.columns.get_loc('sl')] = hist_df.iloc[low_idx]['Low']

        swing_highs = hist_df[hist_df['sh'].notna()]
        swing_lows = hist_df[hist_df['sl'].notna()]

        if swing_highs.empty or swing_lows.empty:
            logging.warning(f"[{ticker}] No swing points found in the last {lookback} days.")
            return ob_data

        ob_data['last_swing_high'] = swing_highs.iloc[-1].sh
        ob_data['last_swing_low'] = swing_lows.iloc[-1].sl

        # --- 3. Find Most Recent Bullish OB (BOS Up) ---
        # Find the last time price broke *above* a swing high
        
        # Iterate backwards through swing highs
        for i in range(len(swing_highs) - 1, 0, -1):
            last_sh = swing_highs.iloc[i-1] # The SH we need to break
            hist_after_sh = hist_df.loc[last_sh.name:]
            
            # Find all candles that closed above that SH (BOS)
            bos_up_candles = hist_after_sh[hist_after_sh['Close'] > last_sh.sh]
            
            if not bos_up_candles.empty:
                first_bos_candle = bos_up_candles.iloc[0]
                
                # --- Find the OB ---
                # Look for the last down candle *before* the BOS candle
                candles_before_bos = hist_df.loc[:first_bos_candle.name].iloc[:-1]
                bearish_candles = candles_before_bos[candles_before_bos['Close'] < candles_before_bos['Open']]
                
                if not bearish_candles.empty:
                    ob_candle = bearish_candles.iloc[-1]
                    ob_data['bullish_ob_low'] = float(ob_candle['Low'])
                    ob_data['bullish_ob_high'] = float(ob_candle['High'])
                    
                    # --- Check Volume on BOS ---
                    if first_bos_candle['Volume'] > (first_bos_candle['vol_sma'] * vol_multiplier):
                        ob_data['bullish_ob_volume_ok'] = bool(True)
                    
                    # --- Check for FVG (Imbalance) ---
                    try:
                        # Find the FVG (Imbalance) created *after* the OB
                        candle_after_ob_idx = hist_df.index.get_loc(ob_candle.name) + 1
                        
                        # Find candle after that (for 3-bar FVG: OB, C1, C2)
                        if candle_after_ob_idx + 1 < len(hist_df):
                            candle_after_bos = hist_df.iloc[candle_after_ob_idx + 1]
                            
                            # FVG exists if High[OB] < Low[C2]
                            if fvg_check and ob_candle['High'] < candle_after_bos['Low']:
                                 ob_data['bullish_ob_fvg'] = bool(True)
                    except Exception:
                        pass # Index errors, etc.
                    
                    # --- Check Mitigation (Validation) ---
                    history_after_ob = hist_df.loc[ob_candle.name:].iloc[1:]
                    if not history_after_ob.empty:
                        candles_that_touched = history_after_ob[history_after_ob['Low'] <= ob_data['bullish_ob_high']]
                        
                        if not candles_that_touched.empty:
                            # Price returned to the OB
                            invalidated = (candles_that_touched['Close'] < ob_data['bullish_ob_low']).any()
                            if not invalidated:
                                ob_data['bullish_ob_validated'] = bool(True)
                                logging.info(f"[{ticker}] Bullish OB {ob_data['bullish_ob_low']:.2f}-{ob_data['bullish_ob_high']:.2f} was mitigated (validated).")
                            else:
                                 logging.info(f"[{ticker}] Bullish OB was invalidated.")
                        else:
                            # This is a "fresh" unmitigated OB
                            logging.info(f"[{ticker}] Found Fresh Bullish OB at {ob_candle.name.date()}. Zone: {ob_data['bullish_ob_low']:.2f}-{ob_data['bullish_ob_high']:.2f}")

                    break # We found the most recent one

        # --- 4. Find Most Recent Bearish OB (BOS Down) ---
        # Find the last time price broke *below* a swing low
        
        for i in range(len(swing_lows) - 1, 0, -1):
            last_sl = swing_lows.iloc[i-1] # The SL we need to break
            hist_after_sl = hist_df.loc[last_sl.name:]
            
            # Find all candles that closed below that SL (BOS)
            bos_down_candles = hist_after_sl[hist_after_sl['Close'] < last_sl.sl]
            
            if not bos_down_candles.empty:
                first_bos_candle = bos_down_candles.iloc[0]
                
                # --- Find the OB ---
                # Look for the last up candle *before* the BOS candle
                candles_before_bos = hist_df.loc[:first_bos_candle.name].iloc[:-1]
                bullish_candles = candles_before_bos[candles_before_bos['Close'] > candles_before_bos['Open']]
                
                if not bullish_candles.empty:
                    ob_candle = bullish_candles.iloc[-1]
                    ob_data['bearish_ob_low'] = float(ob_candle['Low'])
                    ob_data['bearish_ob_high'] = float(ob_candle['High'])
                    
                    # --- Check Volume on BOS ---
                    if first_bos_candle['Volume'] > (first_bos_candle['vol_sma'] * vol_multiplier):
                        ob_data['bearish_ob_volume_ok'] = bool(True)
                    
                    # --- Check for FVG (Imbalance) ---
                    try:
                        candle_after_ob_idx = hist_df.index.get_loc(ob_candle.name) + 1
                        
                        if candle_after_ob_idx + 1 < len(hist_df):
                            candle_after_bos = hist_df.iloc[candle_after_ob_idx + 1]
                            
                            # FVG exists if Low[OB] > High[C2]
                            if fvg_check and ob_candle['Low'] > candle_after_bos['High']:
                                 ob_data['bearish_ob_fvg'] = bool(True)
                    except Exception:
                        pass # Index errors
                    
                    # --- Check Mitigation (Validation) ---
                    history_after_ob = hist_df.loc[ob_candle.name:].iloc[1:]
                    if not history_after_ob.empty:
                        candles_that_touched = history_after_ob[history_after_ob['High'] >= ob_data['bearish_ob_low']]
                        
                        if not candles_that_touched.empty:
                            # Price returned to the OB
                            invalidated = (candles_that_touched['Close'] > ob_data['bearish_ob_high']).any()
                            if not invalidated:
                                ob_data['bearish_ob_validated'] = bool(True)
                                logging.info(f"[{ticker}] Bearish OB {ob_data['bearish_ob_low']:.2f}-{ob_data['bearish_ob_high']:.2f} was mitigated (validated).")
                            else:
                                logging.info(f"[{ticker}] Bearish OB was invalidated.")
                        else:
                            # This is a "fresh" unmitigated OB
                             logging.info(f"[{ticker}] Found Fresh Bearish OB at {ob_candle.name.date()}. Zone: {ob_data['bearish_ob_low']:.2f}-{ob_data['bearish_ob_high']:.2f}")

                    break # We found the most recent one
        
        return ob_data
                
    except Exception as e:
        logging.warning(f"[{ticker}] Error in find_order_blocks (v2): {e}", exc_info=True)
        # Return default structure on error
        ob_data_default = {
            'bullish_ob_low': np.nan, 'bullish_ob_high': np.nan, 'bullish_ob_validated': bool(False),
            'bullish_ob_fvg': bool(False), 'bullish_ob_volume_ok': bool(False),
            'bearish_ob_low': np.nan, 'bearish_ob_high': np.nan, 'bearish_ob_validated': bool(False),
            'bearish_ob_fvg': bool(False), 'bearish_ob_volume_ok': bool(False),
            'last_swing_low': np.nan, 'last_swing_high': np.nan,
            'smc_volume_missing': bool(False) # <-- ✅ ADDED (Phase 2)
        }
        return ob_data_default

# --- MODIFIED: Multi-API Fallback Function with Position Assessment ---
def get_ai_stock_analysis(ticker_symbol, company_name, news_headlines_str, parsed_data, CONFIG, analysis_type="deep_dive", position_data=None):
    """
    Tries Gemini API first, falls back to OpenAI API if the Gemini call fails.
    The prompt is customized based on the `analysis_type`.
    
    ✅ NOTE: News is now reliably provided by Finnhub, so the live search 
    fallback (`search_google_news`) has been removed.
    """
    gemini_api_key = CONFIG.get("DATA_PROVIDERS", {}).get("GEMINI_API_KEY") # NEW KEY
    openai_api_key = CONFIG.get("DATA_PROVIDERS", {}).get("OPENAI_API_KEY")
    
    # --- 1. Prepare Prompt & Context ---
    
    # 1. Check/Format the news string
    news_text = "No recent news found."
    if news_headlines_str and news_headlines_str != "N/A" and news_headlines_str != "No recent news found.":
        logging.info(f"[{ticker_symbol}] Using Finnhub news for AI summary.")
        # Finnhub news is already a clean string of headlines
        news_text = news_headlines_str.replace(", ", "\n- ")
        if not news_text.startswith("- "):
             news_text = "- " + news_text
    else:
        logging.info(f"[{ticker_symbol}] No news found in parsed_data for AI summary.")


    # --- ✅ BUG FIX: Define `parsed` *before* it is used ---
    # Convert pandas Series to dict *first* if needed
    if isinstance(parsed_data, pd.Series):
         parsed = parsed_data.to_dict()
    else:
         parsed = parsed_data.copy() # Avoid modifying the original dict
    # --- END OF BUG FIX ---

    # 2. Extract key quantitative data
    last_price = parsed.get('last_price', 'N/A')
    sector = parsed.get('Sector', 'N/A')
    valuation = parsed.get('grahamValuation', 'N/A')
    trend = parsed.get('Trend (50/200 Day MA)', 'N/A')
    macd = parsed.get('MACD_Signal', 'N/A')
    smc_signal = parsed.get('entry_signal', 'N/A')
    rr_ratio = parsed.get('Risk/Reward Ratio', 'N/A') # Now this works
    if isinstance(rr_ratio, float):
        rr_ratio = f"{rr_ratio:.2f}"


    # 3. Create the holistic prompt (customized by type)
    if analysis_type == "position_assessment":
        # Position Assessment Prompt (Portfolio Tab)
        position_details_str = "\n".join([f"- **{k}**: {v}" for k, v in position_data.items()])
        
        prompt = f"""
        Task: Act as a portfolio manager. Analyze the stock '{company_name}' (Ticker: {ticker_symbol}) in the context of the investor's current position and market data. Generate an **assessment and recommendation** in Arabic.

        Output Format (Use Arabic and Markdown - **Forced RTL**):
        1.  **تقييم الوضع الحالي:** (Analyze the stock's performance in the portfolio (P/L) versus its current market trend and Quant Score.)
        2.  **التحليل الأساسي والفني:** (Briefly summarize the key fundamental factors and current technical signals.)
        3.  **التوصية (Recommendation):** (Should the investor **Hold**, **Add** (Averaging down/up), or **Reduce/Exit**? Justify based on the position data and technical signals like stop loss/demand zone proximity.)

        Investor's Current Position:
        {position_details_str}

        Key Market Data for {ticker_symbol}:
        -   **Last Price:** {last_price}
        -   **MA Trend (50/200):** {trend}
        -   **SMC Entry Signal (New Trade):** {smc_signal}
        -   **Risk/Reward Ratio (New Trade):** {rr_ratio}

        Recent News Headlines:
        ---
        {news_text}
        ---
        
        Analysis (in Arabic):
        """
        
    else: # Default Deep Dive Prompt (Deep Dive Tab)
        prompt = f"""
        Task: Act as a stock market analyst. Analyze the following quantitative data and qualitative news headlines for the company '{company_name}' (Ticker: {ticker_symbol}). Generate a brief, holistic analysis for an investor.

        Output Format (Use Arabic and Markdown - **Forced RTL**):
        1.  **الملخص التنفيذي:** (A 1-2 sentence summary of the stock's current situation.)
        2.  **التحليل الفني (Technical Analysis):** (Analyze the trend, MACD, and SMC signal. Is it a good time to buy?)
        3.  **التحليل الأساسي (Fundamental Analysis):** (Analyze the valuation and sector.)
        4.  **تحليل المخاطر والأخبار:** (Summarize the key news and the Risk/Reward ratio.)

        Key Quantitative Data:
        -   **Last Price:** {last_price}
        -   **Sector:** {sector}
        -   **Valuation (Graham):** {valuation}
        -   **MA Trend (50/200):** {trend}
        -   **MACD Signal:** {macd}
        -   **SMC Entry Signal:** {smc_signal}
        -   **Risk/Reward Ratio:** {rr_ratio}
        -   **Full Data (for context):** {json.dumps(parsed, indent=2, default=str)}

        Recent News Headlines:
        ---
        {news_text}
        ---
        
        Analysis (in Arabic):
        """
    # --- End Prompt Prep ---


    # --- 2. Try Gemini (Primary) ---
    if gemini_api_key and gemini_api_key != "AIzaSy...YOUR_GEMINI_KEY":
        try:
            logging.info(f"[{ticker_symbol}] Attempting Gemini API (Primary)...")
            client = genai.Client(api_key=gemini_api_key)
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # Use a cost-effective, fast model
                contents=[
                    {"role": "user", "parts": [{"text": prompt}]}
                ],
                config={"system_instruction": "You are a helpful stock market analyst providing summaries in Arabic. The output must be formatted with Arabic markdown headings and bullet points."}
            )
            summary = response.text
            
            logging.info(f"[{ticker_symbol}] Successfully received summary from Gemini.")
            return str(summary)
            
        except GeminiAPIError as e:
            # Catch API errors specific to Gemini (e.g., rate limits, invalid key, model failure)
            logging.warning(f"[{ticker_symbol}] Gemini API failed: {e}. Falling back to OpenAI...")
        except Exception as e:
            # Catch other potential errors (e.g., connection issue)
            logging.warning(f"[{ticker_symbol}] Non-API error with Gemini: {e}. Falling back to OpenAI...")


    # --- 3. Fallback to OpenAI ---
    if openai_api_key and openai_api_key != "sk-YOUR_ACTUAL_API_KEY_GOES_HERE":
        try:
            logging.info(f"[{ticker_symbol}] Attempting OpenAI API (Fallback)...")
            client = OpenAI(api_key=openai_api_key)

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful stock market analyst providing summaries in Arabic. The output must be formatted with Arabic markdown headings and bullet points."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = completion.choices[0].message.content
            
            logging.info(f"[{ticker_symbol}] Successfully received summary from OpenAI (Fallback).")
            return str(summary)

        except Exception as e:
            logging.error(f"[{ticker_symbol}] OpenAI API failed: {e}")
            return f"N/A (AI Summary Error: OpenAI API failed.)"
            
    else:
        logging.warning(f"[{ticker_symbol}] Gemini failed and OpenAI key is missing or invalid.")
        return "N/A (AI Summary Disabled: API Keys Missing or Invalid)"

# --- ✅ NEW FUNCTION: AI Portfolio Summary ---
def get_ai_portfolio_summary(portfolio_data, CONFIG):
    """
    Analyzes a dictionary of portfolio metrics and generates a holistic summary.
    """
    gemini_api_key = CONFIG.get("DATA_PROVIDERS", {}).get("GEMINI_API_KEY")
    openai_api_key = CONFIG.get("DATA_PROVIDERS", {}).get("OPENAI_API_KEY")
    
    # --- 1. Prepare Prompt ---
    try:
        # Convert pandas DataFrames/Series to JSON strings for the prompt
        data_str = json.dumps(portfolio_data, indent=2, default=str)
    except Exception as e:
        logging.error(f"Failed to serialize portfolio data for AI: {e}")
        return "Error: Could not format portfolio data for analysis."

    prompt = f"""
    Task: Act as a professional portfolio manager. The user has provided a snapshot of their stock portfolio, including high-level metrics and a detailed list of every stock they own.
    1. Analyze the high-level data (P/L, Allocation, Factor Exposure).
    2. **Crucially, analyze the detailed 'Holdings Details' list.**
    3. Combine these analyses to provide a holistic assessment and actionable, stock-specific recommendations in Arabic.

    Output Format (Use Arabic and Markdown - **Forced RTL**):
    1.  **الملخص التنفيذي (Executive Summary):** (A 1-2 sentence summary of the portfolio's current state, main risk, and primary strength.)
    2.  **تحليل الأداء والتوزيع (Performance & Allocation):** (Analyze the P/L, sector concentration, and cash vs. stock allocation.)
    3.  **تحليل العوامل (Factor Analysis):** (Analyze the 'Weighted Factor Exposure'.)
    4.  **⭐ توصيات الأسهم الفردية (Stock-Specific Recommendations):**
        (This is the most important section. Look at 'Holdings Details' for each stock.
        -   **Identify Top Buys/Adds:** Which stocks have a 'Buy' signal ('entry_signal') and strong 'Final Quant Score'?
        -   **Identify Top Sells/Trims:** Which stocks have high 'Unrealized P/L' (good time to trim profit) OR very weak 'Final Quant Score' (good time to cut)?
        -   **Identify Top Holds:** Which stocks are performing well and should be left alone?
        -   **Give 3-5 specific recommendations** like: "توصية: **شراء/إضافة** سهم [Stock Name]... (السبب: إشارة شراء قوية ودرجة كمية عالية)" or "توصية: **تخفيف/بيع** سهم [Stock Name]... (ال..."السبب: ربح مرتفع / درجة كمية ضعيفة)")

    Portfolio Data Snapshot:
    ---
    {data_str}
    ---
    
    Analysis (in Arabic):
    """
    
    # --- 2. Try Gemini (Primary) ---
    if gemini_api_key and gemini_api_key != "AIzaSy...YOUR_GEMINI_KEY":
        try:
            logging.info("[Portfolio] Attempting Gemini API for portfolio summary...")
            client = genai.Client(api_key=gemini_api_key)
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {"role": "user", "parts": [{"text": prompt}]}
                ],
                config={"system_instruction": "You are a helpful portfolio analyst providing summaries in Arabic. The output must be formatted with Arabic markdown headings and bullet points."}
            )
            summary = response.text
            logging.info("[Portfolio] Successfully received summary from Gemini.")
            return str(summary)
            
        except GeminiAPIError as e:
            logging.warning(f"[Portfolio] Gemini API failed: {e}. Falling back to OpenAI...")
        except Exception as e:
            logging.warning(f"[Portfolio] Non-API error with Gemini: {e}. Falling back to OpenAI...")

    # --- 3. Fallback to OpenAI ---
    if openai_api_key and openai_api_key != "sk-YOUR_ACTUAL_API_KEY_GOES_HERE":
        try:
            logging.info("[Portfolio] Attempting OpenAI API (Fallback)...")
            client = OpenAI(api_key=openai_api_key)

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful portfolio analyst providing summaries in Arabic. The output must be formatted with Arabic markdown headings and bullet points."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = completion.choices[0].message.content
            logging.info("[Portfolio] Successfully received summary from OpenAI (Fallback).")
            return str(summary)

        except Exception as e:
            logging.error(f"[Portfolio] OpenAI API failed: {e}")
            return f"N/A (AI Summary Error: OpenAI API failed.)"
            
    else:
        logging.warning("[Portfolio] Gemini failed and OpenAI key is missing or invalid.")
        return "N/A (AI Summary Disabled: API Keys Missing or Invalid)"

# --- ✅ NEW FUNCTION (Phase 3): AI News Sentiment ---
def get_ai_news_sentiment(headlines_list, CONFIG):
    """
    Analyzes a list of news headlines and returns a sentiment score from -1.0 to 1.0.
    """
    gemini_api_key = CONFIG.get("DATA_PROVIDERS", {}).get("GEMINI_API_KEY")
    openai_api_key = CONFIG.get("DATA_PROVIDERS", {}).get("OPENAI_API_KEY")
    
    if not headlines_list:
        return 0.0

    # Join headlines into a single string for the prompt
    headlines_str = "\n- ".join(headlines_list)
    
    prompt = f"""
    Task: Act as a financial news sentiment analyst. I will provide you with a list of recent news headlines for a stock.
    Analyze the *overall sentiment* of these headlines.
    Respond with a single JSON object with one key, "sentiment_score", which must be a single float between -1.0 (very bearish) and 1.0 (very bullish).
    
    Example:
    Headlines:
    - "XYZ Corp beats earnings estimates"
    - "XYZ Corp raises guidance"
    Response: {{"sentiment_score": 0.8}}
    
    Example:
    Headlines:
    - "ABC Corp misses revenue targets"
    - "CEO of ABC Corp under investigation"
    Response: {{"sentiment_score": -0.9}}
    
    Example:
    Headlines:
    - "Analysts await Fed decision"
    - "Market is flat"
    Response: {{"sentiment_score": 0.0}}

    Headlines to Analyze:
    - {headlines_str}
    
    Response (JSON only):
    """
    
    raw_response = None

    # --- 1. Try Gemini (Primary) ---
    if gemini_api_key and gemini_api_key != "AIzaSy...YOUR_GEMINI_KEY":
        try:
            logging.info(f"[AI News] Attempting Gemini API for sentiment...")
            client = genai.Client(api_key=gemini_api_key)
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {"role": "user", "parts": [{"text": prompt}]}
                ],
                config={"system_instruction": "You are a helpful JSON-only sentiment analyst."}
            )
            raw_response = response.text
            
        except Exception as e:
            logging.warning(f"[AI News] Gemini API failed: {e}. Falling back to OpenAI...")

    # --- 2. Fallback to OpenAI ---
    if raw_response is None and openai_api_key and openai_api_key != "sk-YOUR_ACTUAL_API_KEY_GOES_HERE":
        try:
            logging.info(f"[AI News] Attempting OpenAI API (Fallback)...")
            client = OpenAI(api_key=openai_api_key)

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful JSON-only sentiment analyst. Respond only with the JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_response = completion.choices[0].message.content
        except Exception as e:
            logging.error(f"[AI News] OpenAI API failed: {e}")
            return 0.0
            
    # --- 3. Parse the response ---
    if raw_response:
        try:
            # Clean the response (LLMs sometimes add markdown)
            clean_json_str = raw_response.strip().replace("```json\n", "").replace("\n```", "")
            sentiment_data = json.loads(clean_json_str)
            score = float(sentiment_data.get("sentiment_score", 0.0))
            logging.info(f"[AI News] Successfully parsed sentiment score: {score}")
            return score
        except Exception as e:
            logging.error(f"[AI News] Failed to parse sentiment JSON: {e}. Response was: {raw_response}")
            return 0.0
            
    logging.warning("[AI News] All AI providers failed or were disabled.")
    return 0.0

# --- ✅ NEW FUNCTION (USER REQ): AI Top 20 Summary ---
def get_ai_top20_summary(top_20_data_json, CONFIG):
    """
    Analyzes a JSON string of the Top 20 stocks and returns recommendations.
    """
    gemini_api_key = CONFIG.get("DATA_PROVIDERS", {}).get("GEMINI_API_KEY")
    openai_api_key = CONFIG.get("DATA_PROVIDERS", {}).get("OPENAI_API_KEY")
    
    prompt = f"""
    Task: Act as a professional quantitative analyst. The user has provided a JSON list of the Top 20 stocks that matched their filters, ranked by 'Final Quant Score'.
    Your job is to analyze this list and identify the 2-3 *strongest* "Buy" or "Add" opportunities.
    
    A strong opportunity is a combination of:
    1.  A high 'Final Quant Score' (already ranked).
    2.  A 'entry_signal' of "Buy near Bullish OB".
    3.  A good 'Risk/Reward Ratio' (e.g., > 1.5).
    4.  Strong underlying factor scores (e.g., 'Z_Value' > 0, 'Z_Quality' > 0).

    Provide a brief summary and then list your top recommendations in Arabic.

    Output Format (Use Arabic and Markdown - **Forced RTL**):
    1.  **ملخص القائمة:** (A 1-2 sentence summary of the overall list. Are there many 'Buy' signals? Is it mostly 'Value' or 'Momentum' stocks?)
    2.  **⭐ أفضل الفرص (Top Recommendations):**
        (List 2-3 specific stocks. For each, state the Ticker, Name, Quant Score, and a *brief* justification using the criteria above.)
        -   **[Ticker] - [Name]** (الدرجة: [Score]): [Justification in Arabic]
        -   **[Ticker] - [Name]** (الدرجة: [Score]): [Justification in Arabic]

    Top 20 Stocks Data (JSON):
    ---
    {top_20_data_json}
    ---
    
    Analysis (in Arabic):
    """
    
    # --- 1. Try Gemini (Primary) ---
    if gemini_api_key and gemini_api_key != "AIzaSy...YOUR_GEMINI_KEY":
        try:
            logging.info("[AI Top 20] Attempting Gemini API...")
            client = genai.Client(api_key=gemini_api_key)
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {"role": "user", "parts": [{"text": prompt}]}
                ],
                config={"system_instruction": "You are a helpful quantitative analyst providing summaries in Arabic."}
            )
            summary = response.text
            logging.info("[AI Top 20] Successfully received summary from Gemini.")
            return str(summary)
            
        except GeminiAPIError as e:
            logging.warning(f"[AI Top 20] Gemini API failed: {e}. Falling back to OpenAI...")
        except Exception as e:
            logging.warning(f"[AI Top 20] Non-API error with Gemini: {e}. Falling back to OpenAI...")

    # --- 2. Fallback to OpenAI ---
    if openai_api_key and openai_api_key != "sk-YOUR_ACTUAL_API_KEY_GOES_HERE":
        try:
            logging.info("[AI Top 20] Attempting OpenAI API (Fallback)...")
            client = OpenAI(api_key=openai_api_key)

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful quantitative analyst providing summaries in Arabic."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = completion.choices[0].message.content
            logging.info("[AI Top 20] Successfully received summary from OpenAI (Fallback).")
            return str(summary)

        except Exception as e:
            logging.error(f"[AI Top 20] OpenAI API failed: {e}")
            return f"N/A (AI Summary Error: OpenAI API failed.)"
            
    else:
        logging.warning("[AI Top 20] Gemini failed and OpenAI key is missing or invalid.")
        return "N/A (AI Summary Disabled: API Keys Missing or Invalid)"

# --- ✅ MODIFIED: Removed News from yfinance fetch ---
def fetch_data_yfinance(ticker_obj, CONFIG):
    """Fetches history and info from yfinance with retry logic."""
    retries = 3
    for attempt in range(retries):
        try:
            hist = ticker_obj.history(period=CONFIG.get("HISTORICAL_DATA_PERIOD", "5y"))
            info = ticker_obj.info
            
            # Add earnings data
            earnings = {
                "earnings": ticker_obj.income_stmt,
                "quarterly_earnings": ticker_obj.quarterly_income_stmt,
                "calendar": ticker_obj.calendar,
                # ❌ REMOVED: "news": ticker_obj.get_news()
            }
            
            if hist.empty or info is None:
                logging.warning(f"[{ticker_obj.ticker}] yfinance returned empty history or info (Attempt {attempt + 1}).")
                # Don't retry on empty data, yfinance responded successfully
                return None 
                
            return {"hist": hist, "info": info, "earnings_data": earnings, "source": "yfinance"}
        
        except Exception as e:
            logging.error(f"[{ticker_obj.ticker}] yfinance data fetch error (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(1) # 1 second backoff
            else:
                return None # Failed all retries
    return None

# --- ✅ MODIFIED: Removed News from Alpha Vantage fetch ---
def fetch_data_alpha_vantage(ticker, api_key, CONFIG):
    """Fallback data provider: Alpha Vantage with retry logic."""
    if not api_key or api_key == "YOUR_API_KEY_1": # Check against a default placeholder
        logging.warning(f"[{ticker}] Alpha Vantage API key ('{api_key}') is not set. Skipping fallback.")
        return None

    logging.info(f"[{ticker}] Using Alpha Vantage fallback with a rotated key.")
    base_url = "https://www.alphavantage.co/query"
    hist_data = None
    info_data = None
    # ❌ REMOVED: news_list = []

    retries = 3
    for attempt in range(retries):
        try:
            # 1. Fetch History
            hist_params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": "full",
                "apikey": api_key
            }
            response_hist = requests.get(base_url, params=hist_params, timeout=10)
            response_hist.raise_for_status()
            hist_json = response_hist.json()
            
            if "Time Series (Daily)" in hist_json:
                hist_data = pd.DataFrame.from_dict(hist_json["Time Series (Daily)"], orient='index')
                hist_data.index = pd.to_datetime(hist_data.index)
                hist_data = hist_data.astype(float)
                hist_data.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. adjusted close': 'Adj Close',
                    '6. volume': 'Volume'
                }, inplace=True)
                hist_data = hist_data.sort_index()
                hist_data['Dividends'] = 0.0
                hist_data['Stock Splits'] = 0.0
            else:
                logging.warning(f"[{ticker}] AV History fetch warning: {hist_json.get('Note') or hist_json.get('Error Message')}")
                if 'Note' in hist_json: # API limit
                    time.sleep(15) # Longer sleep for API limits
                    continue # Retry

            # 2. Fetch Info
            info_params = {"function": "OVERVIEW", "symbol": ticker, "apikey": api_key}
            response_info = requests.get(base_url, params=info_params, timeout=10)
            response_info.raise_for_status()
            info_data = response_info.json()
            if not info_data or "Symbol" not in info_data:
                 logging.warning(f"[{ticker}] AV Info fetch warning: {info_data.get('Note') or info_data.get('Error Message')}")
                 if 'Note' in info_data: # API limit
                     time.sleep(15) # Longer sleep for API limits
                     continue # Retry
                 info_data = None

            # ❌ REMOVED: 3. Fetch News & Sentiment block

            # If we got here, all requests succeeded
            break # Exit retry loop

        except requests.exceptions.RequestException as e:
            logging.error(f"[{ticker}] Alpha Vantage request error (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(1) # 1 second backoff
            else:
                return None # Failed all retries
        except Exception as e:
            logging.error(f"[{ticker}] Alpha Vantage processing error (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return None # Failed all retries

    if hist_data is not None and info_data is not None:
        return {
            "hist": hist_data, 
            "info": info_data, 
            "earnings_data": {}, # <-- ✅ MODIFIED: Return empty dict
            "source": "alpha_vantage"
        }
    else:
        return None

# --- ✅ NEW FUNCTION: Fetch News from Finnhub ---
def fetch_data_finnhub_news(ticker, api_key):
    """
    Fetches company news from Finnhub.io for the past 3 days.
    """
    if not api_key or api_key == "YOUR_NEW_FINNHUB_KEY_GOES_HERE":
        logging.warning(f"[{ticker}] Finnhub API key is not set. Skipping news fetch.")
        return []
    
    try:
        # Get dates for the last 3 days
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        base_url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": api_key
        }
        
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        news_data = response.json()
        
        if isinstance(news_data, list) and len(news_data) > 0:
            logging.info(f"[{ticker}] Successfully fetched {len(news_data)} news items from Finnhub.")
            return news_data # Returns a list of news dicts
        else:
            logging.info(f"[{ticker}] Finnhub returned no news.")
            return []
            
    except requests.exceptions.RequestException as e:
        logging.error(f"[{ticker}] Finnhub news request error: {e}")
        return []
    except Exception as e:
        logging.error(f"[{ticker}] Error processing Finnhub news: {e}")
        return []


def is_data_valid(data, source="yfinance"):
    """
    Validation layer to check for critical missing data.
    """
    if data is None:
        return False
        
    info = data.get("info", {})
    if not isinstance(info, dict):
        logging.warning(f"Info object is not a dictionary. Data is invalid.")
        return False
    
    if source == "yfinance":
        key_fields = ['sector', 'marketCap', 'trailingEps', 'priceToBook', 'returnOnEquity', 'forwardPE']
    else: # Alpha Vantage
        key_fields = ['Sector', 'MarketCapitalization', 'EPS', 'BookValue', 'ReturnOnEquityTTM', 'ForwardPE']
    
    missing_fields = [f for f in key_fields if info.get(f) is None or info.get(f) == 0 or info.get(f) == "None"]
    
    # Loosen validation for non-US markets (allow 3 missing fields)
    if len(missing_fields) > 3:
        logging.warning(f"[{info.get('symbol', 'TICKER')}] Data from {source} failed validation. Missing: {missing_fields}")
        return False
        
    return True

# --- ✅ MODIFIED FUNCTION (Accepts CONFIG) ---
def parse_ticker_data(data, ticker_symbol, CONFIG):
    """
    Parses data from either yfinance or Alpha Vantage into a common format.
    Also calculates all TA metrics.
    --- MODIFIED with explicit type casting to prevent pyarrow errors ---
    """
    hist = data['hist']
    info = data['info']
    earnings_data = data['earnings_data']
    source = data['source']
    
    parsed = {'ticker': str(ticker_symbol), 'success': bool(True), 'source': str(source)} # Force types
    parsed['data_warning'] = None
    
    try:
        # --- 0. Clean History & Get Last Price ---
        if not hist.index.is_unique:
             hist = hist[~hist.index.duplicated(keep='first')]
        last_price = hist['Close'].iloc[-1]
        parsed['last_price'] = float(last_price)
    
        # --- 1. Value Factors (WITH TYPE CASTING) ---
        if source == "yfinance":
            try:
                parsed['forwardPE'] = float(info.get('forwardPE'))
            except (TypeError, ValueError):
                parsed['forwardPE'] = np.nan
            try:
                parsed['priceToBook'] = float(info.get('priceToBook'))
            except (TypeError, ValueError):
                parsed['priceToBook'] = np.nan
            try:
                parsed['marketCap'] = float(info.get('marketCap'))
            except (TypeError, ValueError):
                parsed['marketCap'] = np.nan
            
            parsed['Sector'] = str(info.get('sector', 'Unknown')) # Force string
            
            # --- ✅ NEW: Add shortName ---
            parsed['shortName'] = str(info.get('shortName', ticker_symbol))
            
            try:
                parsed['enterpriseToEbitda'] = float(info.get('enterpriseToEbitda'))
            except (TypeError, ValueError):
                parsed['enterpriseToEbitda'] = np.nan
            try:
                parsed['freeCashflow'] = float(info.get('freeCashflow'))
            except (TypeError, ValueError):
                parsed['freeCashflow'] = np.nan
            try:
                parsed['trailingEps'] = float(info.get('trailingEps'))
            except (TypeError, ValueError):
                parsed['trailingEps'] = np.nan
                
        else: # Alpha Vantage mapping
            parsed['forwardPE'] = float(info.get('ForwardPE', 'nan'))
            parsed['priceToBook'] = float(info.get('PriceToBookRatio', 'nan'))
            parsed['marketCap'] = float(info.get('MarketCapitalization', 'nan'))
            parsed['Sector'] = str(info.get('Sector', 'Unknown')) # Force string
            
            # --- ✅ NEW: Add shortName ---
            parsed['shortName'] = str(info.get('Name', ticker_symbol))
            
            parsed['enterpriseToEbitda'] = float(info.get('EVToEBITDA', 'nan'))
            parsed['freeCashflow'] = np.nan
            parsed['trailingEps'] = float(info.get('EPS', 'nan'))
        
        # Derived Value Metrics
        parsed['P/FCF'] = (parsed['marketCap'] / parsed['freeCashflow']) if pd.notna(parsed['marketCap']) and pd.notna(parsed['freeCashflow']) and parsed['freeCashflow'] != 0 else np.nan
        
        bvps = None
        if pd.notna(parsed['priceToBook']) and parsed['priceToBook'] != 0 and pd.notna(last_price):
            bvps = last_price / parsed['priceToBook']
        
        graham_str = "N/A (Missing Data)"
        parsed['grahamNumber'] = np.nan
        if pd.notna(parsed['trailingEps']) and pd.notna(bvps):
            if parsed['trailingEps'] > 0 and bvps > 0:
                parsed['grahamNumber'] = (22.5 * parsed['trailingEps'] * bvps) ** 0.5
                if pd.notna(parsed['grahamNumber']):
                    graham_str = "Undervalued (Graham)" if last_price < parsed['grahamNumber'] else "Overvalued (Graham)"
            else:
                graham_str = "N/A (Unprofitable)"
        parsed['grahamValuation'] = str(graham_str) # Force string


        # --- 2. Momentum Factors ---
        monthly_hist = hist['Close'].resample('ME').last()
        if len(monthly_hist) >= 13:
            price_1m_ago = monthly_hist.iloc[-2]
            price_13m_ago = monthly_hist.iloc[-13]
            parsed['momentum_12m'] = ((price_1m_ago - price_13m_ago) / price_13m_ago) * 100 if price_13m_ago else np.nan
        else:
            parsed['momentum_12m'] = np.nan
            
        if len(monthly_hist) >= 4:
            price_1m_ago = monthly_hist.iloc[-2]
            price_4m_ago = monthly_hist.iloc[-4]
            parsed['momentum_3m'] = ((price_1m_ago - price_4m_ago) / price_4m_ago) * 100 if price_4m_ago else np.nan
        else:
            parsed['momentum_3m'] = np.nan

        daily_returns = hist['Close'].pct_change().dropna()
        if len(daily_returns) >= 252:
            returns_1y = daily_returns.iloc[-252:]
            parsed['volatility_1y'] = returns_1y.std() * np.sqrt(252)
        else:
            parsed['volatility_1y'] = np.nan

        parsed['risk_adjusted_momentum'] = (parsed['momentum_12m'] / parsed['volatility_1y']) if pd.notna(parsed['momentum_12m']) and pd.notna(parsed['volatility_1y']) and parsed['volatility_1y'] != 0 else np.nan

        # --- 3. Quality Factors (WITH TYPE CASTING) ---
        if source == "yfinance":
            try:
                parsed['returnOnEquity'] = float(info.get('returnOnEquity'))
            except (TypeError, ValueError):
                parsed['returnOnEquity'] = np.nan
            try:
                parsed['profitMargins'] = float(info.get('profitMargins'))
            except (TypeError, ValueError):
                parsed['profitMargins'] = np.nan
            try:
                parsed['returnOnAssets'] = float(info.get('returnOnAssets'))
            except (TypeError, ValueError):
                parsed['returnOnAssets'] = np.nan
            try:
                parsed['debtToEquity'] = float(info.get('debtToEquity'))
            except (TypeError, ValueError):
                parsed['debtToEquity'] = np.nan
        else: # Alpha Vantage mapping
            parsed['returnOnEquity'] = float(info.get('ReturnOnEquityTTM', 'nan'))
            parsed['profitMargins'] = float(info.get('ProfitMargin', 'nan'))
            parsed['returnOnAssets'] = float(info.get('ReturnOnAssetsTTM', 'nan'))
            parsed['debtToEquity'] = float(info.get('DebtToEquityRatio', 'nan'))

        # --- 3.5. Quality Booleans (FIXED with bool(False) default) ---
        eps_val_for_bool = parsed.get('trailingEps')
        if pd.notna(eps_val_for_bool):
            parsed['earnings_negative'] = bool(eps_val_for_bool < 0)
        else:
            parsed['earnings_negative'] = bool(False) # Force default bool
            
        if source == "yfinance" and earnings_data.get("quarterly_earnings") is not None and not earnings_data["quarterly_earnings"].empty:
            quarterly_data = earnings_data["quarterly_earnings"]
            
            q_eps = pd.Series([np.nan]) # Default
            if 'Net Income' in quarterly_data.index:
                q_eps = quarterly_data.loc['Net Income']
            elif 'Earnings' in quarterly_data.columns:
                 q_eps = quarterly_data['Earnings']
            else:
                 logging.warning(f"[{ticker_symbol}] Could not find 'Net Income' (index) or 'Earnings' (column) in quarterly data.")
                 
            q_eps = pd.to_numeric(q_eps, errors='coerce').dropna()

            if not q_eps.empty and q_eps.abs().mean() != 0:
                cv = (q_eps.std() / q_eps.abs().mean())
                parsed['earnings_volatile'] = bool(cv > 0.5) if pd.notna(cv) else bool(False)
            else:
                parsed['earnings_volatile'] = bool(False)
        else:
            parsed['earnings_volatile'] = bool(False)
            
        # --- 4. Size Factors (WITH TYPE CASTING) ---
        if source == "yfinance":
            try:
                parsed['floatShares'] = float(info.get('floatShares'))
            except (TypeError, ValueError):
                parsed['floatShares'] = np.nan
            try:
                parsed['averageVolume'] = float(info.get('averageVolume'))
            except (TypeError, ValueError):
                parsed['averageVolume'] = np.nan
        else: # Alpha Vantage mapping
            parsed['floatShares'] = float(info.get('SharesOutstanding', 'nan')) # Proxy
            parsed['averageVolume'] = float(info.get('50DayMovingAverage', 'nan')) # Bad proxy
        
        parsed['floatAdjustedMarketCap'] = (parsed['floatShares'] * last_price) if pd.notna(parsed['floatShares']) and pd.notna(last_price) else parsed['marketCap']

        # --- 5. Low Volatility Factors (WITH TYPE CASTING) ---
        if source == "yfinance":
            try:
                parsed['beta'] = float(info.get('beta'))
            except (TypeError, ValueError):
                parsed['beta'] = np.nan
        else: # Alpha Vantage mapping
            parsed['beta'] = float(info.get('Beta', 'nan'))

        # --- 6. Technical Factors ---
        cfg = CONFIG['TECHNICALS']
        
        min_hist_len = max(cfg.get('LONG_MA_WINDOW', 200), 252) 
        if len(hist) < min_hist_len:
            warning_msg = f"Short history ({len(hist)} days). TA/Risk metrics may be N/A or unreliable."
            parsed['data_warning'] = str(warning_msg) # Force string
            logging.warning(f"[{ticker_symbol}] {warning_msg}")
        
        hist.ta.rsi(length=cfg['RSI_WINDOW'], append=True)
        hist.ta.sma(length=cfg['SHORT_MA_WINDOW'], append=True)
        hist.ta.sma(length=cfg['LONG_MA_WINDOW'], append=True)
        hist.ta.macd(fast=cfg['MACD_SHORT_SPAN'], slow=cfg['MACD_LONG_SPAN'], signal=cfg['MACD_SIGNAL_SPAN'], append=True)
        hist.ta.adx(length=cfg['ADX_WINDOW'], append=True)
        
        atr_col = f'ATR_{cfg["ATR_WINDOW"]}'
        atr_series = hist.ta.atr(length=cfg['ATR_WINDOW'])
        if atr_series is not None:
            hist[atr_col] = atr_series

        rsi_col = f'RSI_{cfg["RSI_WINDOW"]}'
        short_ma_col = f'SMA_{cfg["SHORT_MA_WINDOW"]}'
        long_ma_col = f'SMA_{cfg["LONG_MA_WINDOW"]}'
        macd_h_col = f'MACDh_{cfg["MACD_SHORT_SPAN"]}_{cfg["MACD_LONG_SPAN"]}_{cfg["MACD_SIGNAL_SPAN"]}'
        adx_col = f'ADX_{cfg["ADX_WINDOW"]}'

        parsed['RSI'] = hist[rsi_col].iloc[-1] if rsi_col in hist.columns and not hist[rsi_col].isnull().all() else np.nan
        parsed['ATR'] = hist[atr_col].iloc[-1] if atr_col in hist.columns and not hist[atr_col].isnull().all() else np.nan
        parsed['ADX'] = hist[adx_col].iloc[-1] if adx_col in hist.columns and not hist[adx_col].isnull().all() else np.nan

        last_short_ma = hist[short_ma_col].iloc[-1] if short_ma_col in hist.columns and not hist[short_ma_col].isnull().all() else np.nan
        last_long_ma = hist[long_ma_col].iloc[-1] if long_ma_col in hist.columns and not hist[long_ma_col].isnull().all() else np.nan
        
        parsed['Price_vs_SMA50'] = (last_price / last_short_ma) if last_short_ma and last_short_ma != 0 else np.nan
        parsed['Price_vs_SMA200'] = (last_price / last_long_ma) if last_long_ma and last_long_ma != 0 else np.nan
        
        hist_val = hist[macd_h_col].iloc[-1] if macd_h_col in hist.columns else np.nan
        trend_str = "N/A"
        if not pd.isna(last_short_ma) and not pd.isna(last_long_ma):
            if last_short_ma > last_long_ma:
                trend_str = 'Confirmed Uptrend' if last_price > last_short_ma else 'Uptrend (Correction)'
            else:
                trend_str = 'Confirmed Downtrend' if last_price < last_short_ma else 'Downtrend (Rebound)'
        parsed['Trend (50/200 Day MA)'] = str(trend_str) # Force string
            
        macd_str = "N/A"
        if macd_h_col in hist.columns and len(hist) >= 2 and not pd.isna(hist_val):
            prev_hist = hist[macd_h_col].iloc[-2]
            if not pd.isna(prev_hist):
                if hist_val > 0 and prev_hist <= 0: macd_str = "Bullish Crossover"
                elif hist_val < 0 and prev_hist >= 0: macd_str = "Bearish Crossover"
                elif hist_val > 0: macd_str = "Bullish"
                else: macd_str = "Bearish"
        parsed['MACD_Signal'] = str(macd_str) # Force string

        # --- 7. Other Info ---
        
        lookback = CONFIG.get('SR_LOOKBACK_PERIOD', 90)
        recent_hist = hist.iloc[-lookback:] if len(hist) >= lookback else hist
        parsed['Support_90d'] = float(recent_hist['Low'].min())
        parsed['Resistance_90d'] = float(recent_hist['High'].max())
        
        support_90d = parsed.get('Support_90d')
        if pd.notna(support_90d) and pd.notna(last_price) and last_price > 0:
            parsed['pct_above_support'] = ((last_price - support_90d) / last_price) * 100
        else:
            parsed['pct_above_support'] = np.nan
        
        # --- ✅ MODIFIED: Pass CONFIG ---
        ob_data = find_order_blocks(hist, ticker_symbol, CONFIG)
        parsed.update(ob_data) # keys are static and correctly typed
        
        # --- Entry Signal Logic ---
        smc_config = CONFIG.get('TECHNICALS', {}).get('SMC_ORDER_BLOCKS', {})
        proximity_pct = smc_config.get('ENTRY_PROXIMITY_PERCENT', 2.0) / 100.0
        entry_signal = "No Trade"
        
        bullish_ob_low = parsed.get('bullish_ob_low', np.nan)
        bullish_ob_high = parsed.get('bullish_ob_high', np.nan)
        is_bullish_ob_active = pd.notna(bullish_ob_low)
        
        bearish_ob_low = parsed.get('bearish_ob_low', np.nan)
        bearish_ob_high = parsed.get('bearish_ob_high', np.nan)
        is_bearish_ob_active = pd.notna(bearish_ob_low)

        if is_bullish_ob_active:
            entry_zone_top = bullish_ob_high * (1 + proximity_pct)
            if last_price >= bullish_ob_low and last_price <= entry_zone_top:
                entry_signal = "Buy near Bullish OB"
        elif is_bearish_ob_active:
            entry_zone_bottom = bearish_ob_low * (1 - proximity_pct)
            if last_price <= bearish_ob_high and last_price >= entry_zone_bottom:
                entry_signal = "Sell near Bearish OB"
        
        parsed['entry_signal'] = str(entry_signal) # Force string

        # --- ✅ NEW: News & Earnings Date (Consolidated Finnhub logic) ---
        try:
            # 1. Parse Finnhub News
            news = data.get('finnhub_news', []) # Get news from process_ticker
            news_list_str = "N/A"
            
            if news and isinstance(news, list):
                # Get top 5 headlines
                news_titles = [str(item.get('headline', 'N/A')) for item in news[:5]]
                news_list_str = ", ".join(news_titles) # Flatten list to string
            
            parsed['news_list'] = str(news_list_str) # Force string
            
            # --- ✅ MODIFIED (Phase 3): Get AI Sentiment Score ---
            # This score is now passed from process_ticker
            parsed['news_sentiment_score'] = float(data.get('news_sentiment_score', 0.0))
            # --- End of change ---

            # 2. Parse Earnings/Dividends (from yfinance info/earnings_data)
            last_div_date_ts = info.get('lastDividendDate')
            last_div_value = info.get('lastDividendValue')
            
            div_date_str = "N/A"
            div_val_float = np.nan

            if last_div_date_ts and pd.notna(last_div_date_ts) and last_div_value:
                div_date_str = pd.to_datetime(last_div_date_ts, unit='s').strftime('%Y-%m-%d')
                div_val_float = float(last_div_value)
            else:
                divs = hist[hist['Dividends'] > 0]
                if not divs.empty:
                    div_date_str = divs.index[-1].strftime('%Y-%m-%d')
                    div_val_float = float(divs['Dividends'].iloc[-1])
            
            parsed['last_dividend_date'] = str(div_date_str)
            parsed['last_dividend_value'] = float(div_val_float)

            calendar = earnings_data.get('calendar', {})
            date_val = "N/A"
            if calendar and 'Earnings Date' in calendar and calendar['Earnings Date']:
                raw_date = calendar['Earnings Date'][0]
                date_val = pd.to_datetime(raw_date).strftime('%Y-%m-%d') if pd.notna(raw_date) else "N/A"
            parsed['next_earnings_date'] = str(date_val) # Force string
            
            # Get Next Ex-Dividend Date
            next_ex_div_ts = info.get('exDividendDate')
            ex_div_date_str = "N/A"
            if next_ex_div_ts and pd.notna(next_ex_div_ts):
                if next_ex_div_ts > datetime.now().timestamp():
                    ex_div_date_str = pd.to_datetime(next_ex_div_ts, unit='s').strftime('%Y-%m-%d')
            parsed['next_ex_dividend_date'] = str(ex_div_date_str)

        except Exception as e:
             logging.warning(f"[{ticker_symbol}] Error parsing news/calendar: {e}")
             parsed['news_list'] = "N/A"
             parsed['news_sentiment_score'] = 0.0 # <-- ✅ MODIFIED (Phase 3)
             parsed['next_earnings_date'] = "N/A"
             parsed['last_dividend_date'] = "N/A"
             parsed['last_dividend_value'] = np.nan
             parsed['next_ex_dividend_date'] = "N/A"
        
        # ❌ REMOVED: Old news parsing blocks for yfinance and Alpha Vantage
        
        # --- 8. Risk Management ---
        
        rm_config = CONFIG.get('RISK_MANAGEMENT', {})
        atr_sl_mult = rm_config.get('ATR_STOP_LOSS_MULTIPLIER', 1.5)
        fib_target_mult = rm_config.get('ATR_TAKE_PROFIT_MULTIPLIER', 1.618) 
        use_cut_loss_filter = rm_config.get('USE_CUT_LOSS_FILTER', True)

        # --- MODIFIED (USER REQ): Dynamic Risk Calculation ---
        # The keys here were corrected to match the config keys
        total_equity = rm_config.get('TOTAL_EQUITY', None)
        risk_percent = rm_config.get('RISK_PERCENT_PER_TRADE', 0.005) # 0.5% default
        default_risk_amount = rm_config.get('RISK_PER_TRADE_AMOUNT', 50) # Fallback to $50

        risk_per_trade_usd = np.nan
        if total_equity and total_equity > 0:
            risk_per_trade_usd = total_equity * risk_percent
        else:
            # Use the explicit fallback amount ($50)
            risk_per_trade_usd = default_risk_amount 
        # --- END OF MODIFICATION ---
        
        atr = parsed.get('ATR')
        last_price = parsed.get('last_price')
        support_90d = parsed.get('Support_90d')
        last_swing_low = parsed.get('last_swing_low', np.nan)
        
        risk_per_share = np.nan
        final_stop_loss_price = np.nan
        stop_loss_price_atr = np.nan
        stop_loss_price_cutloss = np.nan
        sl_method_str = "N/A"
        
        if pd.notna(atr) and atr > 0:
            stop_loss_price_atr = last_price - (atr * atr_sl_mult)
            parsed['Stop Loss (ATR)'] = float(stop_loss_price_atr)
        
        if pd.notna(last_swing_low) and last_swing_low < last_price:
            stop_loss_price_cutloss = last_swing_low
            parsed['Stop Loss (Cut Loss)'] = float(stop_loss_price_cutloss)
            
        if use_cut_loss_filter and pd.notna(stop_loss_price_atr) and pd.notna(stop_loss_price_cutloss):
            final_stop_loss_price = max(stop_loss_price_atr, stop_loss_price_cutloss) # Tighter stop
            sl_method_str = "Cut-Loss" if final_stop_loss_price == stop_loss_price_cutloss else "ATR"
        elif pd.notna(stop_loss_price_atr):
            final_stop_loss_price = stop_loss_price_atr
            sl_method_str = "ATR"
        elif pd.notna(stop_loss_price_cutloss):
            final_stop_loss_price = stop_loss_price_cutloss
            sl_method_str = "Cut-Loss"
        elif pd.notna(support_90d) and support_90d < last_price:
            final_stop_loss_price = support_90d
            sl_method_str = "90-Day Low"
        else:
            if pd.notna(last_price):
                final_stop_loss_price = last_price * 0.90 # 10% hard stop
                sl_method_str = "10% Fallback"
        
        parsed['SL_Method'] = str(sl_method_str) # Force string

        if pd.notna(final_stop_loss_price) and final_stop_loss_price < last_price:
            risk_per_share = last_price - final_stop_loss_price
        else:
            risk_per_share = np.nan 

        # --- MODIFICATION START: Exit Rule Logic ---
        
        # 1. Get default ATR/Fib-based Take Profit
        take_profit_price_atr = np.nan
        if pd.notna(risk_per_share) and risk_per_share > 0:
            take_profit_price_atr = last_price + (risk_per_share * fib_target_mult)
        
        # 2. Get SMC-based Take Profit (Opposing Supply Zone)
        # We target the *bottom* of the Bearish OB
        take_profit_price_smc = parsed.get('bearish_ob_low', np.nan)
        
        # 3. Choose the *tighter* (more conservative) target
        take_profit_price = np.nan
        if pd.notna(take_profit_price_atr) and pd.notna(take_profit_price_smc):
            # Use the SMC target ONLY if it's a tighter, valid target
            if take_profit_price_smc < take_profit_price_atr and take_profit_price_smc > last_price:
                take_profit_price = take_profit_price_smc
            else:
                take_profit_price = take_profit_price_atr
        elif pd.notna(take_profit_price_atr):
            take_profit_price = take_profit_price_atr
        elif pd.notna(take_profit_price_smc) and take_profit_price_smc > last_price:
            # Use SMC as fallback if ATR failed
            take_profit_price = take_profit_price_smc
        else:
            take_profit_price = np.nan # No valid target found
        
        # 4. Add MACD Exit Signal
        macd_signal = parsed.get('MACD_Signal', 'N/A')
        if macd_signal == "Bearish Crossover":
            parsed['MACD_EXIT_SIGNAL'] = str("SELL (Crossover)")
        else:
            parsed['MACD_EXIT_SIGNAL'] = str("HOLD")
            
        # --- MODIFICATION END ---
        
        if pd.notna(risk_per_share) and risk_per_share > 0 and pd.notna(take_profit_price):
            # Now, use the FINAL take_profit_price
            reward_per_share = take_profit_price - last_price
            
            position_size_shares = risk_per_trade_usd / risk_per_share
            position_size_usd = position_size_shares * last_price
            
            parsed['Stop Loss Price'] = float(final_stop_loss_price)
            parsed['Final Stop Loss'] = float(final_stop_loss_price)
            parsed['Take Profit Price'] = float(take_profit_price) # <-- Updated
            # Recalculate R/R based on the new, potentially tighter, target
            parsed['Risk/Reward Ratio'] = float(reward_per_share / risk_per_share) if risk_per_share != 0 else np.nan # <-- Updated
            parsed['Risk % (to Stop)'] = float((risk_per_share / last_price) * 100) if last_price != 0 else np.nan
            parsed['Position Size (Shares)'] = float(position_size_shares)
            parsed['Position Size (USD)'] = float(position_size_usd)
            parsed['Risk Per Trade (USD)'] = float(risk_per_trade_usd) # <-- Now dynamic
        else:
            parsed['Stop Loss Price'] = np.nan
            parsed['Final Stop Loss'] = np.nan
            parsed['Take Profit Price'] = np.nan
            parsed['Risk/Reward Ratio'] = np.nan
            parsed['Risk % (to Stop)'] = np.nan
            parsed['Position Size (Shares)'] = np.nan
            parsed['Position Size (USD)'] = np.nan
            parsed['Risk Per Trade (USD)'] = float(risk_per_trade_usd) # <-- Now dynamic
        
        if 'Stop Loss (ATR)' not in parsed:
            parsed['Stop Loss (ATR)'] = np.nan
        if 'Stop Loss (Cut Loss)' not in parsed:
            parsed['Stop Loss (Cut Loss)'] = np.nan
            
        sl_cutloss = parsed.get('Stop Loss (Cut Loss)')
        if pd.notna(sl_cutloss) and pd.notna(last_price) and last_price > 0 and sl_cutloss < last_price:
             parsed['pct_above_cutloss'] = ((last_price - sl_cutloss) / last_price) * 100
        else:
             parsed['pct_above_cutloss'] = np.nan
            
        if 'hist_df' in parsed:
            del parsed['hist_df'] 

        # --- ✅ MODIFIED (P4) - AI CALL REMOVED ---
        # Set placeholder to None. AI call now happens on-demand in streamlit_app.py
        parsed['ai_holistic_analysis'] = None
        # --- END MODIFICATION ---

        return parsed
        
    except Exception as e:
        logging.error(f"[{ticker_symbol}] Fatal error in parse_ticker_data: {e}", exc_info=True)
        # ✅ FIX: Standardized error dict
        return {
            'ticker': str(ticker_symbol), 
            'success': bool(False), 
            'error': str(e),
            'bullish_ob_validated': bool(False),
            'bearish_ob_validated': bool(False),
            'earnings_negative': bool(False),
            'earnings_volatile': bool(False),
            'bullish_ob_fvg': bool(False),
            'bullish_ob_volume_ok': bool(False),
            'bearish_ob_fvg': bool(False),
            'bearish_ob_volume_ok': bool(False),
            'smc_volume_missing': bool(False),
            'z_score_fallback': bool(False),
            'next_ex_dividend_date': 'N/A', 
            'shortName': 'N/A', 
            'ai_holistic_analysis': None,
            'news_sentiment_score': 0.0,
            'news_list': 'N/A'
        }


# --- ✅ MODIFIED FUNCTION (Accepts CONFIG, Fetches News) ---
def process_ticker(ticker, CONFIG, fetch_news=True):
    """
    Main ticker processing function.
    Attempts yfinance, validates, falls back to Alpha Vantage, validates,
    fetches news from Finnhub, then parses all data.
    """
    # ✅ FIX: Standardized error dict
    error_dict = {
        'ticker': str(ticker), 'success': bool(False), 'error': 'Unknown error',
        'bullish_ob_validated': bool(False), 'bearish_ob_validated': bool(False),
        'earnings_negative': bool(False), 'earnings_volatile': bool(False),
        'bullish_ob_fvg': bool(False), 'bullish_ob_volume_ok': bool(False),
        'bearish_ob_fvg': bool(False), 'bearish_ob_volume_ok': bool(False),
        'smc_volume_missing': bool(False), 'z_score_fallback': bool(False),
        'next_ex_dividend_date': 'N/A', 'shortName': 'N/A', 
        'ai_holistic_analysis': None, 'news_sentiment_score': 0.0,
        'news_list': 'N/A'
    }

    if CONFIG is None:
        logging.error(f"process_ticker ({ticker}): CONFIG is None.")
        error_dict['error'] = 'Config not loaded'
        return error_dict
        
    # 1. Attempt yfinance
    ticker_obj = yf.Ticker(ticker)
    yf_data = fetch_data_yfinance(ticker_obj, CONFIG) # <-- Now has retries
    
    hist_df_for_storage = yf_data.get('hist') if yf_data else None
    
    data_to_parse = None
    
    if is_data_valid(yf_data, source="yfinance"):
        data_to_parse = yf_data
    else:
        # 2. Attempt Alpha Vantage Fallback
        logging.warning(f"[{ticker}] yfinance data invalid. Trying Alpha Vantage fallback.")
        
        # --- MODIFIED: Use correct key from user's config ---
        av_keys_list = CONFIG.get('DATA_PROVIDERS', {}).get('ALPHA_VANTAGE_API_KEYS', []) 
        
        if not av_keys_list:
            logging.error(f"[{ticker}] Alpha Vantage fallback failed: No API keys found in config.json.")
            av_data = None
        else:
            selected_av_key = random.choice(av_keys_list)
            av_data = fetch_data_alpha_vantage(ticker, selected_av_key, CONFIG) # <-- Now has retries
        
        if is_data_valid(av_data, source="alpha_vantage"):
            data_to_parse = av_data
            if hist_df_for_storage is None:
                 hist_df_for_storage = av_data.get('hist')
        else:
            logging.error(f"[{ticker}] All data providers failed or returned invalid data.")
            error_dict['error'] = 'All data providers failed'
            return error_dict
            
    # --- ✅ NEW: 3. Fetch News from Finnhub (Conditional) ---
    if fetch_news:
        finnhub_key = CONFIG.get("DATA_PROVIDERS", {}).get("FINNHUB_API_KEY")
        if finnhub_key and finnhub_key != "YOUR_NEW_FINNHUB_KEY_GOES_HERE":
            news_list = fetch_data_finnhub_news(ticker, finnhub_key)
            data_to_parse['finnhub_news'] = news_list # Add news to the data dict
            
            # --- ✅ NEW (Phase 3): Get AI Sentiment Score ---
            if news_list:
                headlines = [item.get('headline', '') for item in news_list]
                sentiment_score = get_ai_news_sentiment(headlines, CONFIG)
                data_to_parse['news_sentiment_score'] = sentiment_score
            else:
                data_to_parse['news_sentiment_score'] = 0.0
            # --- End of change ---
            
        else:
            logging.warning(f"[{ticker}] FINNHUB_API_KEY not found or is placeholder. News will be missing.")
            data_to_parse['finnhub_news'] = []
            data_to_parse['news_sentiment_score'] = 0.0
    else:
        # Don't fetch news to save API calls
        data_to_parse['finnhub_news'] = []
        data_to_parse['news_sentiment_score'] = 0.0 # Default for non-deep dive
            
    # 4. Parse and Calculate
    try:
        parsed_data = parse_ticker_data(data_to_parse, ticker, CONFIG)
        
        if hist_df_for_storage is not None:
             parsed_data['hist_df'] = hist_df_for_storage
            
        return parsed_data
        
    except Exception as e:
        logging.critical(f"[{ticker}] Unhandled exception in parse_ticker_data: {e}", exc_info=True)
        error_dict['error'] = f'Parsing error: {e}'
        return error_dict


# --- DEPRECATED FUNCTIONS (Kept for compatibility) ---
# (These are unchanged but now require CONFIG)

def calculate_support_resistance(hist_df, CONFIG):
    """DEPRECATED: Logic is now inside parse_ticker_data"""
    logging.warning("Called deprecated function: calculate_support_resistance")
    if hist_df is None or hist_df.empty:
        return None, None, None, None, None, None
    try:
        lookback_period = CONFIG.get('SR_LOOKBACK_PERIOD', 90)
        recent_hist = hist_df.iloc[-lookback_period:] if len(hist_df) >= lookback_period else hist_df
        
        support_val = recent_hist['Low'].min()
        support_date = recent_hist['Low'].idxmin()
        resistance_val = recent_hist['High'].max()
        resistance_date = recent_hist['High'].idxmax()
        
        high_low_diff = resistance_val - support_val
        fib_161_8_level = resistance_val + (high_low_diff * 0.618) if high_low_diff > 0 else None
        fib_61_8_level = resistance_val - (high_low_diff * 0.618) if high_low_diff > 0 else None
        return support_val, support_date, resistance_val, resistance_date, fib_161_8_level, fib_161_8_level
    except Exception as e:
        logging.error(f"Error in deprecated calculate_support_resistance: {e}")
        return None, None, None, None, None, None

def calculate_financials_and_fair_price(ticker_obj, last_price, ticker, CONFIG):
    """DEPRECATED: Logic is now inside parse_ticker_data/fetch_data_yfinance"""
    logging.warning("Called deprecated function: calculate_financials_and_fair_price")
    try:
        info = ticker_obj.info
        pe_ratio = info.get('forwardPE')
        pb_ratio = info.get('priceToBook')
        market_cap = info.get('marketCap')
        sector = info.get('sector', 'Unknown')
        eps = info.get('trailingEps')
        
        graham_number = None
        valuation_signal = "N/A"

        if eps and pb_ratio and eps > 0 and pb_ratio > 0 and last_price:
            bvps = last_price / pb_ratio
            graham_number = (22.5 * eps * bvps) ** 0.5
            valuation_signal = "Undervalued (Graham)" if last_price < graham_number else "Overvalued (Graham)"
        elif eps and eps <= 0:
            valuation_signal = "Unprofitable (EPS < 0)"

        return {
            'Forward P/E': pe_ratio,
            'P/B Ratio': pb_ratio,
            'Market Cap': market_cap,
            'Sector': sector,
            'Graham Number': graham_number,
            'Valuation (Graham)': valuation_signal,
        }
    except Exception as e:
        logging.error(f"Error in deprecated calculate_financials_and_fair_price for {ticker}: {e}")
        return {'Sector': 'Error', 'Valuation (Graham)': 'Error'}
