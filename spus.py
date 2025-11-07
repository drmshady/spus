# -*- coding: utf-8 -*-
"""
SPUS Quantitative Analyzer v19.10 (Force v1 API)

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
- ✅ MODIFIED (P4): parse_ticker_data now sets 'ai_holistic_analysis' to None.
"""

import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time
import os
import logging
from datetime import datetime
import json
import numpy as np
from bs4 import BeautifulSoup
import random
from scipy.signal import argrelextrema 
import openai
from openai import OpenAI

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
        
        # --- 2. Try web scrape fallback ---
        if tickers is None:
            logging.warning("Local SPUS CSV failed, trying web scrape fallback...")
            tickers = fetch_spus_tickers_from_web()

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


# --- ✅ NEW (P1): Market Regime Filter ---
def check_market_regime(CONFIG):
    """
    Checks the trend of the primary market index (e.g., S&P 500/TASI).
    Returns 'BULLISH', 'BEARISH', or 'TRANSITIONAL'.
    """
    try:
        # Get index ticker from config, default to ^GSPC
        index_ticker = CONFIG.get("MARKET_INDEX_TICKER", "^GSPC") 
        
        # Get MA windows from config
        short_ma_window = CONFIG.get('TECHNICALS', {}).get('SHORT_MA_WINDOW', 50)
        long_ma_window = CONFIG.get('TECHNICALS', {}).get('LONG_MA_WINDOW', 200)

        index_obj = yf.Ticker(index_ticker)
        # Fetch slightly more than 1y to ensure 200MA is calculated
        hist = index_obj.history(period="300d") 
        
        if hist.empty or len(hist) < long_ma_window:
            logging.warning(f"Could not fetch sufficient history for market index {index_ticker}")
            return "UNKNOWN" # Failsafe: allow run if index data is missing

        hist[f'SMA_{short_ma_window}'] = hist['Close'].rolling(window=short_ma_window).mean()
        hist[f'SMA_{long_ma_window}'] = hist['Close'].rolling(window=long_ma_window).mean()

        last_price = hist['Close'].iloc[-1]
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
            
        logging.info(f"Market Regime for {index_ticker} is {regime}")
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
        'last_swing_low': np.nan, 'last_swing_high': np.nan
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
            'last_swing_low': np.nan, 'last_swing_high': np.nan
        }
        return ob_data_default

# --- ✅ MODIFIED (P4): Replaced Gemini with OpenAI ---
def get_ai_stock_analysis(ticker_symbol, company_name, news_headlines_str, parsed_data, CONFIG):
    """
    Takes all parsed data and news, sending them to OpenAI API for a holistic summary.
    This function is now called ON-DEMAND from streamlit_app.py.
    """
    api_key = CONFIG.get("DATA_PROVIDERS", {}).get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-YOUR_ACTUAL_API_KEY"):
        logging.warning(f"[{ticker_symbol}] OpenAI API Key not configured. Skipping AI summary.")
        return "N/A (AI Summary Disabled: API Key Missing)"

    try:
        client = OpenAI(api_key=api_key)

        # 1. Check/Format the news string
        news_text = "No recent news found."
        if news_headlines_str and news_headlines_str != "N/A":
            # Replace the ", " separator from the original list with newlines
            news_text = news_headlines_str.replace(", ", "\n- ")
            if not news_text.startswith("- "):
                 news_text = "- " + news_text
        else:
            logging.info(f"[{ticker_symbol}] No news headlines found in parsed_data.")

        # 2. Extract key quantitative data
        last_price = parsed_data.get('last_price', 'N/A')
        sector = parsed_data.get('Sector', 'N/A')
        valuation = parsed_data.get('grahamValuation', 'N/A')
        trend = parsed_data.get('Trend (50/200 Day MA)', 'N/A')
        macd = parsed_data.get('MACD_Signal', 'N/A')
        smc_signal = parsed_data.get('entry_signal', 'N/A')
        rr_ratio = parsed_data.get('Risk/Reward Ratio', 'N/A')
        if isinstance(rr_ratio, float):
            rr_ratio = f"{rr_ratio:.2f}"
        if isinstance(parsed_data, pd.Series):
             # Convert pandas Series to dict for easier prompting
             parsed_data = parsed_data.to_dict()

        # 3. Create the holistic prompt
        prompt = f"""
        Task: Act as a stock market analyst. Analyze the following quantitative data and qualitative news headlines for the company '{company_name}' (Ticker: {ticker_symbol}). Generate a brief, holistic analysis for an investor.

        Output Format (Use Arabic and Markdown):
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
        -   **Full Data (for context):** {json.dumps(parsed_data, indent=2, default=str)}

        Recent News Headlines:
        ---
        {news_text}
        ---
        
        Analysis (in Arabic):
        """

        # 4. Call the API
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # You can change this to gpt-4 if preferred
            messages=[
                {"role": "system", "content": "You are a helpful stock market analyst providing summaries in Arabic."},
                {"role": "user", "content": prompt}
            ]
        )
        
        summary = completion.choices[0].message.content
        
        # Clean up the response
        summary = summary.replace('•', '-') # Clean bullets
        return str(summary)

    except Exception as e:
        logging.error(f"[{ticker_symbol}] Error calling OpenAI API: {e}")
        return f"N/A (AI Summary Error: {e})"

# --- ✅ MODIFIED FUNCTION (Accepts CONFIG) ---
def fetch_data_yfinance(ticker_obj, CONFIG):
    """Fetches history and info from yfinance."""
    try:
        hist = ticker_obj.history(period=CONFIG.get("HISTORICAL_DATA_PERIOD", "5y"))
        info = ticker_obj.info
        
        # Add earnings data
        earnings = {
            "earnings": ticker_obj.income_stmt,
            "quarterly_earnings": ticker_obj.quarterly_income_stmt,
            "calendar": ticker_obj.calendar,
            "news": ticker_obj.get_news() # <-- ✅ NEWS FIX
        }
        
        if hist.empty or info is None:
            logging.warning(f"[{ticker_obj.ticker}] yfinance returned empty history or info.")
            return None
            
        return {"hist": hist, "info": info, "earnings_data": earnings, "source": "yfinance"}
    except Exception as e:
        logging.error(f"[{ticker_obj.ticker}] yfinance data fetch error: {e}")
        return None

# --- ✅ MODIFIED FUNCTION (Accepts CONFIG) ---
def fetch_data_alpha_vantage(ticker, api_key, CONFIG):
    """Fallback data provider: Alpha Vantage."""
    if not api_key or api_key == "YOUR_API_KEY_1": # Check against a default placeholder
        logging.warning(f"[{ticker}] Alpha Vantage API key ('{api_key}') is not set. Skipping fallback.")
        return None

    logging.info(f"[{ticker}] Using Alpha Vantage fallback with a rotated key.")
    base_url = "https://www.alphavantage.co/query"
    hist_data = None
    info_data = None

    try:
        # 1. Fetch History
        hist_params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": api_key
        }
        response = requests.get(base_url, params=hist_params, timeout=10)
        response.raise_for_status()
        hist_json = response.json()
        
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

        # 2. Fetch Info
        info_params = {"function": "OVERVIEW", "symbol": ticker, "apikey": api_key}
        response = requests.get(base_url, params=info_params, timeout=10)
        response.raise_for_status()
        info_data = response.json()
        if not info_data or "Symbol" not in info_data:
             logging.warning(f"[{ticker}] AV Info fetch warning: {info_data.get('Note') or info_data.get('Error Message')}")
             info_data = None

        # 3. Fetch News & Sentiment
        news_list = [] # Default
        try:
            news_params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": 10, # Get 10 recent articles
                "apikey": api_key
            }
            response = requests.get(base_url, params=news_params, timeout=10)
            response.raise_for_status()
            news_data = response.json()
            
            if "feed" in news_data:
                for item in news_data["feed"]:
                    try:
                        # Convert AV time format 'YYYYMMDDTHHMMSS' to a timestamp
                        publish_time = datetime.strptime(item.get('time_published'), '%Y%m%dT%H%M%S')
                        publish_timestamp = int(publish_time.timestamp())
                    except Exception:
                        publish_timestamp = 0
                        
                    news_list.append({
                        'title': item.get('title'),
                        'providerPublishTime': publish_timestamp 
                    })
            else:
                logging.warning(f"[{ticker}] AV news fetch warning: {news_data.get('Note') or news_data.get('Error Message')}")

        except Exception as e:
            logging.warning(f"[{ticker}] Failed to fetch Alpha Vantage news: {e}")
            # Don't fail the whole function, just return no news

    except requests.exceptions.RequestException as e:
        logging.error(f"[{ticker}] Alpha Vantage request error: {e}")
        return None
    except Exception as e:
        logging.error(f"[{ticker}] Alpha Vantage processing error: {e}")
        return None

    if hist_data is not None and info_data is not None:
        return {
            "hist": hist_data, 
            "info": info_data, 
            "earnings_data": {"news": news_list}, # <-- ADD THE NEWS HERE
            "source": "alpha_vantage"
        }
    else:
        return None

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

        # News & Earnings Date
        if source == "yfinance":
            try:
                # --- THIS IS THE ORIGINAL NEWS CODE ---
                news = earnings_data.get('news', [])
                news_str = "No"
                news_list_str = "N/A"
                
                if news and isinstance(news, list):
                    news_titles = [str(item.get('title', 'N/A')) for item in news[:5]]
                    news_list_str = ", ".join(news_titles) # Flatten list to string
                    
                    now_ts = datetime.now().timestamp()
                    recent_news_ts = now_ts - (CONFIG.get('NEWS_LOOKBACK_HOURS', 48) * 3600)
                    if any(item.get('providerPublishTime', 0) > recent_news_ts for item in news):
                        news_str = "Yes"
                
                parsed['news_list'] = str(news_list_str) # Force string
                parsed['recent_news'] = str(news_str) # Force string

                # --- AI Summary Call moved to the end of the function ---

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
                 parsed['recent_news'] = "N/A"
                 parsed['next_earnings_date'] = "N/A"
                 parsed['last_dividend_date'] = "N/A"
                 parsed['last_dividend_value'] = np.nan
                 parsed['next_ex_dividend_date'] = "N/A"
        
        else: # Alpha Vantage
             # --- MODIFIED: Use the news we fetched ---
             news = earnings_data.get('news', [])
             news_str = "No"
             news_list_str = "N/A"
             
             if news and isinstance(news, list):
                news_titles = [str(item.get('title', 'N/A')) for item in news[:5]]
                news_list_str = ", ".join(news_titles)
                
                now_ts = datetime.now().timestamp()
                recent_news_ts = now_ts - (CONFIG.get('NEWS_LOOKBACK_HOURS', 48) * 3600)
                if any(item.get('providerPublishTime', 0) > recent_news_ts for item in news):
                    news_str = "Yes"
            
             parsed['news_list'] = str(news_list_str) # Force string
             parsed['recent_news'] = str(news_str) # Force string
             # --- END MODIFICATION ---
             
             parsed['next_earnings_date'] = str(info.get('DividendDate', 'N/A (AV)'))
             parsed['last_dividend_date'] = str(info.get('DividendDate', 'N/A (AV)'))
             parsed['last_dividend_value'] = float(info.get('DividendPerShare', 'nan'))
             parsed['next_ex_dividend_date'] = str(info.get('ExDividendDate', 'N/A (AV)'))

        # --- 8. Risk Management ---
        # --- ✅ MODIFIED (P2): Exit Rule Logic ---
        
        rm_config = CONFIG.get('RISK_MANAGEMENT', {})
        atr_sl_mult = rm_config.get('ATR_STOP_LOSS_MULTIPLIER', 1.5)
        # --- Use Fib 1.618 as a fallback if ATR_TAKE_PROFIT_MULTIPLIER is not in config ---
        fib_target_mult = rm_config.get('ATR_TAKE_PROFIT_MULTIPLIER', 1.618) 
        risk_per_trade_usd = rm_config.get('RISK_PER_TRADE_AMOUNT', 500)
        use_cut_loss_filter = rm_config.get('USE_CUT_LOSS_FILTER', True)
        
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
            parsed['Risk Per Trade (USD)'] = float(risk_per_trade_usd)
        else:
            parsed['Stop Loss Price'] = np.nan
            parsed['Final Stop Loss'] = np.nan
            parsed['Take Profit Price'] = np.nan
            parsed['Risk/Reward Ratio'] = np.nan
            parsed['Risk % (to Stop)'] = np.nan
            parsed['Position Size (Shares)'] = np.nan
            parsed['Position Size (USD)'] = np.nan
            parsed['Risk Per Trade (USD)'] = float(risk_per_trade_usd)
        
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
        # ✅ FIX: Return a flat dict with default bools to prevent Arrow error
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
            'next_ex_dividend_date': 'N/A',
            'shortName': 'N/A', # <-- ✅ ADD THIS
            'ai_holistic_analysis': None # <-- MODIFIED (P4)
        }


# --- ✅ MODIFIED FUNCTION (Accepts CONFIG) ---
def process_ticker(ticker, CONFIG):
    """
    Main ticker processing function.
    Attempts yfinance, validates, falls back to Alpha Vantage, validates,
    then parses all data and calculates metrics.
    """
    if CONFIG is None:
        logging.error(f"process_ticker ({ticker}): CONFIG is None.")
        return {
            'ticker': str(ticker), 'success': bool(False), 'error': 'Config not loaded',
            'bullish_ob_validated': bool(False), 'bearish_ob_validated': bool(False),
            'earnings_negative': bool(False), 'earnings_volatile': bool(False),
            'bullish_ob_fvg': bool(False), 'bullish_ob_volume_ok': bool(False),
            'bearish_ob_fvg': bool(False), 'bearish_ob_volume_ok': bool(False),
            'next_ex_dividend_date': 'N/A',
            'shortName': 'N/A', # <-- ✅ ADD THIS
            'ai_holistic_analysis': None # <-- MODIFIED (P4)
        }
        
    # 1. Attempt yfinance
    ticker_obj = yf.Ticker(ticker)
    yf_data = fetch_data_yfinance(ticker_obj, CONFIG)
    
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
            av_data = fetch_data_alpha_vantage(ticker, selected_av_key, CONFIG)
        
        if is_data_valid(av_data, source="alpha_vantage"):
            data_to_parse = av_data
            if hist_df_for_storage is None:
                 hist_df_for_storage = av_data.get('hist')
        else:
            logging.error(f"[{ticker}] All data providers failed or returned invalid data.")
            return {
                'ticker': str(ticker), 'success': bool(False), 'error': 'All data providers failed',
                'bullish_ob_validated': bool(False), 'bearish_ob_validated': bool(False),
                'earnings_negative': bool(False), 'earnings_volatile': bool(False),
                'bullish_ob_fvg': bool(False), 'bullish_ob_volume_ok': bool(False),
                'bearish_ob_fvg': bool(False), 'bearish_ob_volume_ok': bool(False),
                'next_ex_dividend_date': 'N/A',
                'shortName': 'N/A', # <-- ✅ ADD THIS
                'ai_holistic_analysis': None # <-- MODIFIED (P4)
            }
            
    # 3. Parse and Calculate
    try:
        parsed_data = parse_ticker_data(data_to_parse, ticker, CONFIG)
        
        if hist_df_for_storage is not None:
             parsed_data['hist_df'] = hist_df_for_storage
            
        return parsed_data
        
    except Exception as e:
        logging.critical(f"[{ticker}] Unhandled exception in parse_ticker_data: {e}", exc_info=True)
        return {
            'ticker': str(ticker), 'success': bool(False), 'error': f'Parsing error: {e}',
            'bullish_ob_validated': bool(False), 'bearish_ob_validated': bool(False),
            'earnings_negative': bool(False), 'earnings_volatile': bool(False),
            'bullish_ob_fvg': bool(False), 'bullish_ob_volume_ok': bool(False),
            'bearish_ob_fvg': bool(False), 'bearish_ob_volume_ok': bool(False),
            'next_ex_dividend_date': 'N/A',
            'shortName': 'N/A', # <-- ✅ ADD THIS
            'ai_holistic_analysis': None # <-- MODIFIED (P4)
        }


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
