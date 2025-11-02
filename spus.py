# -*- coding: utf-8 -*-
"""
SPUS Quantitative Analyzer v15 (Research Grade)

- Implements Point 1 (Data Reliability) with a DataFetcher class.
- Implements Point 2 (Factor Model) with new factor calculations.
- Implements Point 3 (Risk Management) by adding ATR calculation.
- Implements Point 9 (Code Quality) with docstrings and modularity.
"""

import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time
import os
import logging
from datetime import datetime, timedelta
import json
import numpy as np

# --- Define Base Directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Function to load external config file ---
def load_config(path='config.json'):
    config_path = os.path.join(BASE_DIR, path)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Successfully loaded configuration from {config_path}")
        
        # --- ⭐️ NEW: Add API Key checks (Point 1) ---
        if 'FMP_API_KEY' not in config or not config['FMP_API_KEY']:
            logging.warning("FMP_API_KEY not found in config.json. Fallback provider will be disabled.")
            config['FMP_API_KEY'] = None
        
        return config
    except FileNotFoundError:
        logging.error(f"FATAL: Configuration file '{config_path}' not found.")
        return None
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not decode JSON from '{config_path}'. Check for syntax errors.")
        return None

# --- Load CONFIG at module level ---
CONFIG = load_config('config.json')


# --- ⭐️ NEW: DataFetcher Class (Point 1: Data Reliability) ⭐️ ---
class DataFetcher:
    """
    Handles fetching data from primary (yfinance) and secondary (FMP) providers.
    Manages API keys and provides a unified data validation layer.
    """
    def __init__(self, fmp_api_key=None):
        self.fmp_api_key = fmp_api_key
        logging.info("DataFetcher initialized.")

    def _fetch_yfinance_data(self, ticker):
        """Fetches data from yfinance."""
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # 1. Fetch history (2 years for TA)
            hist_2y = ticker_obj.history(period="2y")
            if hist_2y.empty:
                logging.warning(f"[{ticker}] yfinance: No history data found.")
                return None, None
            
            # 2. Fetch info (fundamentals)
            info = ticker_obj.info
            if not info or info.get('trailingEps') is None:
                logging.warning(f"[{ticker}] yfinance: 'info' object is empty or missing key data (EPS).")
                # Don't fail yet, secondary provider might have it
            
            # 3. Fetch financials (for ROIC, FCF)
            financials = ticker_obj.financials
            balance_sheet = ticker_obj.balance_sheet
            cash_flow = ticker_obj.cashflow
            
            return hist_2y, info, financials, balance_sheet, cash_flow

        except Exception as e:
            logging.error(f"[{ticker}] yfinance: Unhandled exception: {e}")
            return None, None

    def _fetch_fmp_data(self, ticker):
        """
        Placeholder for fetching data from Financial Modeling Prep (FMP) as a fallback.
        This would be used if yfinance fails or returns incomplete data.
        """
        if not self.fmp_api_key:
            return None # FMP is disabled

        logging.info(f"[{ticker}] FMP: Attempting fallback fetch...")
        
        # --- This is a placeholder implementation ---
        # In a real scenario, you would make API calls to FMP endpoints:
        # 1. /v3/profile/{ticker} (for sector, market cap, beta)
        # 2. /v3/key-metrics/{ticker} (for P/E, P/B, EV/EBITDA, P/FCF)
        # 3. /v3/ratios/{ticker} (for ROIC, ROE, Profit Margin)
        #
        # For demonstration, we'll just return None.
        #
        # Example (pseudo-code):
        #
        # try:
        #     base_url = "https://financialmodelingprep.com/api"
        #     key = self.fmp_api_key
        #
        #     profile_url = f"{base_url}/v3/profile/{ticker}?apikey={key}"
        #     profile = requests.get(profile_url).json()
        #
        #     metrics_url = f"{base_url}/v3/key-metrics-ttm/{ticker}?apikey={key}"
        #     metrics = requests.get(metrics_url).json()
        #
        #     return {'profile': profile[0], 'metrics': metrics[0]}
        #
        # except Exception as e:
        #     logging.error(f"[{ticker}] FMP: Fallback failed: {e}")
        #     return None
        
        logging.warning(f"[{ticker}] FMP: Fallback logic is not implemented.")
        return None

    def _validate_and_merge_data(self, yf_data, fmp_data, ticker):
        """
        Merges data from yfinance and FMP, prioritizing yfinance
        but filling NaNs with FMP data.
        
        Applies validation layer (Point 1).
        """
        
        # Unpack yfinance data
        hist, info, financials, balance_sheet, cash_flow = yf_data
        
        # --- 1. History (from yfinance only) ---
        if hist is None or hist.empty:
            logging.error(f"[{ticker}] Validation: No history data available. Cannot proceed.")
            return None
        
        # --- 2. Fundamentals (merged) ---
        data = {}
        
        # Priority: yfinance info
        data['marketCap'] = info.get('marketCap')
        data['sector'] = info.get('sector')
        data['beta'] = info.get('beta')
        data['trailingEps'] = info.get('trailingEps')
        data['forwardPE'] = info.get('forwardPE')
        data['priceToBook'] = info.get('priceToBook')
        data['returnOnEquity'] = info.get('returnOnEquity')
        data['profitMargin'] = info.get('profitMargins') # yfinance uses 'profitMargins'
        data['dividendYield'] = info.get('dividendYield')
        data['debtToEquity'] = info.get('debtToEquity')
        data['floatShares'] = info.get('floatShares') # Point 2 (Size)
        
        # yfinance financials (more complex)
        try:
            # Point 2 (Value, Quality)
            ebit = financials.loc['EBIT'].iloc[0]
            op_income = financials.loc['Operating Income'].iloc[0]
            enterprise_val = info.get('enterpriseValue')
            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
            total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            tax = financials.loc['Tax Provision'].iloc[0]
            free_cash_flow = cash_flow.loc['Free Cash Flow'].iloc[0]
            
            data['enterpriseValue'] = enterprise_val
            data['evToEbitda'] = info.get('enterpriseToEbitda') # Use info if available
            
            if not data['evToEbitda'] and enterprise_val and ebit:
                data['evToEbit'] = enterprise_val / ebit # Fallback
            
            if free_cash_flow and data['marketCap']:
                data['priceToFreeCashFlow'] = data['marketCap'] / free_cash_flow
            
            # ROIC (Return on Invested Capital) = NOPAT / (Total Debt + Total Equity)
            # NOPAT (Net Operating Profit After Tax) = Operating Income * (1 - Tax Rate)
            tax_rate = tax / op_income if op_income and tax and op_income > tax else 0.35 # Assume 35% if missing
            nopat = op_income * (1 - tax_rate)
            invested_capital = total_debt + total_equity
            if nopat and invested_capital:
                data['roic'] = nopat / invested_capital

        except Exception as e:
            logging.warning(f"[{ticker}] yfinance: Could not calculate ROIC/FCF: {e}")
            data['priceToFreeCashFlow'] = None
            data['roic'] = None
            data['evToEbit'] = None

        # --- 3. FMP Fallback (placeholder) ---
        if fmp_data:
            # Example:
            # data['sector'] = data['sector'] or fmp_data['profile'].get('sector')
            # data['evToEbitda'] = data['evToEbitda'] or fmp_data['metrics'].get('enterpriseValueOverEBITDATTM')
            pass

        # --- 4. Validation Layer (Point 1) ---
        
        # Fill missing sector
        if not data.get('sector') or pd.isna(data.get('sector')):
            data['sector'] = "Unknown"
        
        # Check for critical missing data
        critical_fields = ['marketCap', 'trailingEps', 'priceToBook', 'returnOnEquity']
        for field in critical_fields:
            if data.get(field) is None or pd.isna(data.get(field)) or data.get(field) == 0:
                logging.warning(f"[{ticker}] Validation: Critical field '{field}' is missing, 0, or NaN. Factor scores may be weak.")
                # We don't drop the stock, but we'll penalize it in scoring
                if data.get(field) is None:
                     data[field] = np.nan # Ensure it's NaN for scoring
        
        return hist, data

    def get_validated_data(self, ticker):
        """
        Main public method. Fetches, merges, and validates data for a single ticker.
        """
        # 1. Try Primary (yfinance)
        yf_data = self._fetch_yfinance_data(ticker)
        
        # 2. Try Secondary (FMP)
        fmp_data = self._fetch_fmp_data(ticker) # This is a placeholder

        # 3. Validate and Merge
        if yf_data[0] is None:
             logging.error(f"[{ticker}] No data from any provider. Skipping.")
             return None, None
             
        return self._validate_and_merge_data(yf_data, fmp_data, ticker)


# --- FETCHER FUNCTION (Unchanged) ---
# This remains the same as it's just reading a local CSV
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

# --- ⭐️ REVISED: Factor Calculation Function (Point 2: Factor Model) ⭐️ ---
def calculate_all_factors(hist_df, fundamental_data):
    """
    Calculates all technical and fundamental factors from the raw data.
    This replaces the old 'calculate_financials_and_fair_price' and 'calculate_support_resistance'.
    """
    
    factors = {}
    
    # --- 1. Basic Info ---
    last_price = hist_df['Close'].iloc[-1]
    factors['last_price'] = last_price
    
    # --- 2. Fundamental Factors (from fundamental_data dict) ---
    factors.update(fundamental_data) # Add all raw data
    
    # Value Factors
    factors['P/E'] = fundamental_data.get('forwardPE')
    factors['P/B'] = fundamental_data.get('priceToBook')
    factors['EV/EBITDA'] = fundamental_data.get('evToEbitda') or fundamental_data.get('evToEbit')
    factors['P/FCF'] = fundamental_data.get('priceToFreeCashFlow')
    
    # Quality Factors
    factors['ROE'] = fundamental_data.get('returnOnEquity')
    factors['ROIC'] = fundamental_data.get('roic')
    factors['Profit Margin'] = fundamental_data.get('profitMargin')
    factors['D/E'] = fundamental_data.get('debtToEquity')
    # Penalize negative/volatile earnings (if EPS is NaN or < 0)
    if not fundamental_data.get('trailingEps') or fundamental_data.get('trailingEps') <= 0:
        factors['Quality_Penalty'] = -1 # Apply penalty in scoring
    else:
        factors['Quality_Penalty'] = 0

    # Size Factor
    factors['Float-Adj Market Cap'] = fundamental_data.get('floatShares') # Use float if available
    if not factors['Float-Adj Market Cap']:
        factors['Float-Adj Market Cap'] = fundamental_data.get('marketCap') # Fallback to market cap
    
    # Low Volatility Factor
    factors['Beta'] = fundamental_data.get('beta')

    # --- 3. Technical & Momentum Factors (from hist_df) ---
    try:
        # --- Technicals (Point 2) ---
        hist_df.ta.rsi(length=14, append=True)
        hist_df.ta.adx(length=14, append=True)
        hist_df.ta.atr(length=14, append=True)
        hist_df.ta.sma(length=50, append=True)
        hist_df.ta.sma(length=200, append=True)
        hist_df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        factors['RSI'] = hist_df['RSI_14'].iloc[-1]
        factors['ADX'] = hist_df['ADX_14'].iloc[-1]
        factors['ATR'] = hist_df['ATRr_14'].iloc[-1] # For Point 3 (Risk)
        
        sma_50 = hist_df['SMA_50'].iloc[-1]
        sma_200 = hist_df['SMA_200'].iloc[-1]
        
        factors['Price_vs_SMA50'] = (last_price / sma_50) - 1
        factors['Price_vs_SMA200'] = (last_price / sma_200) - 1
        
        # MACD Signal
        hist_val = hist_df['MACDh_12_26_9'].iloc[-1]
        prev_hist = hist_df['MACDh_12_26_9'].iloc[-2]
        if hist_val > 0 and prev_hist <= 0:
            factors['MACD_Signal'] = "Bullish Crossover"
        elif hist_val < 0 and prev_hist >= 0:
            factors['MACD_Signal'] = "Bearish Crossover"
        elif hist_val > 0:
            factors['MACD_Signal'] = "Bullish"
        else:
            factors['MACD_Signal'] = "Bearish"

        # --- Volatility (Point 2) ---
        daily_returns = hist_df['Close'].pct_change().dropna()
        factors['Volatility_1Y'] = daily_returns.iloc[-252:].std() * np.sqrt(252)

        # --- Momentum (Point 2) ---
        monthly_hist = hist_df['Close'].resample('ME').last()
        if len(monthly_hist) >= 13:
            ret_12m = (monthly_hist.iloc[-1] / monthly_hist.iloc[-13]) - 1
            factors['Momentum_12M'] = ret_12m
        else:
            factors['Momentum_12M'] = np.nan
            
        if len(monthly_hist) >= 4:
            ret_3m = (monthly_hist.iloc[-1] / monthly_hist.iloc[-4]) - 1
            factors['Momentum_3M'] = ret_3m
        else:
            factors['Momentum_3M'] = np.nan

        # Risk-Adjusted Momentum (12M Return / 1Y Volatility)
        if factors['Volatility_1Y'] and factors['Volatility_1Y'] > 0:
            factors['Risk_Adj_Momentum_12M'] = factors['Momentum_12M'] / factors['Volatility_1Y']
        else:
            factors['Risk_Adj_Momentum_12M'] = np.nan
            
        # --- Support/Resistance & Risk (Point 3) ---
        recent_hist = hist_df.iloc[-90:] # 90-day lookback
        support = recent_hist['Low'].min()
        resistance = recent_hist['High'].max()
        
        factors['Support'] = support
        factors['Resistance'] = resistance
        
        # ATR-based Risk
        atr_val = factors['ATR']
        if atr_val:
            factors['Stop_Loss_ATR'] = last_price - (1.5 * atr_val)
            factors['Take_Profit_ATR'] = last_price + (3.0 * atr_val)
        else:
            factors['Stop_Loss_ATR'] = np.nan
            factors['Take_Profit_ATR'] = np.nan

    except Exception as e:
        logging.warning(f"Error calculating TA factors: {e}")

    return factors


# --- ⭐️ REVISED: Process Ticker Function (Points 1, 2, 9) ⭐️ ---
def process_ticker(ticker, data_fetcher):
    """
    Processes a single ticker.
    1. Fetches data using DataFetcher (Point 1)
    2. Calculates all factors (Point 2)
    3. Returns a unified dictionary.
    """
    
    # --- 1. Fetch and Validate Data (Point 1) ---
    try:
        hist_df, fundamental_data = data_fetcher.get_validated_data(ticker)
        
        if hist_df is None or fundamental_data is None:
            logging.error(f"[{ticker}] Failed to get validated data. Skipping.")
            return {'ticker': ticker, 'success': False}
    
    except Exception as e:
        logging.error(f"[{ticker}] Error in data fetch/validation step: {e}")
        return {'ticker': ticker, 'success': False}

    # --- 2. Calculate All Factors (Point 2, 3) ---
    try:
        all_factors = calculate_all_factors(hist_df, fundamental_data)
        
        all_factors['ticker'] = ticker
        all_factors['success'] = True
        
        # --- 3. Add News/Earnings (from old function) ---
        try:
            ticker_obj = yf.Ticker(ticker) # Re-init for news
            news = ticker_obj.news
            if news:
                all_factors['Latest_Headline'] = news[0].get('title', "N/A")
                now_ts = datetime.now().timestamp()
                forty_eight_hours_ago_ts = now_ts - 172800
                all_factors['Recent_News_48h'] = "No"
                for item in news:
                    if item.get('providerPublishTime', 0) > forty_eight_hours_ago_ts:
                        all_factors['Recent_News_48h'] = "Yes"
                        break

            calendar = ticker_obj.calendar
            if calendar and 'Earnings Date' in calendar and calendar['Earnings Date']:
                all_factors['Next_Earnings_Date'] = str(calendar['Earnings Date'][0])
            else:
                all_factors['Next_Earnings_Date'] = "N/A"
        except:
            all_factors['Latest_Headline'] = "N/A"
            all_factors['Recent_News_48h'] = "N/A"
            all_factors['Next_Earnings_Date'] = "N/A"
        
        return all_factors
        
    except Exception as e:
        logging.error(f"[{ticker}] Error in factor calculation step: {e}")
        return {'ticker': ticker, 'success': False}
