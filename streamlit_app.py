import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import sys
import glob
import numpy as np
import streamlit.components.v1 as components
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import openpyxl
from openpyxl.styles import Font
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats.mstats import winsorize
import pytz 
import pickle 
from collections import deque # For FIFO

# --- ⭐️ 1. Set Page Configuration FIRST ⭐️ ---
st.set_page_config(
    page_title="Multi-Market Quant Analyzer",
    page_icon="https://www.sp-funds.com/wp-content/uploads/2019/07/favicon-32x32.png", 
    layout="wide"
)

# --- DEFINE TIMEZONE ---
SAUDI_TZ = pytz.timezone('Asia/Riyadh')

# --- Path Fix & Import ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    # --- ✅ MODIFIED: Import new AI portfolio summary function ---
    from spus import (
        load_config,
        fetch_market_tickers,
        process_ticker,
        check_market_regime,
        get_ai_stock_analysis,
        get_ai_portfolio_summary, # <-- ✅ NEW
        get_ai_top20_summary # <-- ✅ NEW (USER REQ)
    )
except ImportError as e:
    st.error(f"Error: Failed to import 'spus.py'. Details: {e}")
    st.stop()

# --- ReportLab Import (Optional) ---
try:
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("Module 'reportlab' not found. PDF report generation will be disabled.")

# --- ⭐️ 2. Custom CSS (Corrected v2) ⭐️ ---
def load_css():
    """Injects custom CSS for a modern, minimal, card-based theme."""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
            font-family: 'Inter', sans-serif;
        }}
        h1 {{ font-weight: 700; }}
        h2 {{ font-weight: 600; }}
        h3 {{ font-weight: 600; margin-top: 20px; margin-bottom: 0px; }}
        .main .block-container {{
            padding-top: 2rem; padding-bottom: 2rem;
            padding-left: 2.5rem; padding-right: 2.5rem;
        }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid var(--gray-800); padding: 1.5rem;
        }}
        [data-testid="stSidebar"] h2 {{ font-size: 1.5rem; font-weight: 700; }}
        [data-testid="stSidebar"] .stButton > button, [data-testid="stSidebar"] .stDownloadButton > button {{
            width: 100%; border-radius: 8px; font-weight: 600;
        }}
        
        /* --- ⭐️ CORRECTED: Radio-to-Tabs styling v2 ⭐️ --- */
        [data-testid="stRadio"] > label[data-baseweb="radio"] {{
            display: none; /* Hides the "Navigation:" label */
        }}
        [data-testid="stRadio"] > div[role="radiogroup"] {{
            display: flex;
            flex-direction: row;
            justify-content: stretch; 
            border-bottom: 2px solid var(--gray-800);
            margin-bottom: 1.5rem;
            width: 100%;
        }}
        [data-testid="stRadio"] input[type="radio"] {{
            display: none; /* Hide the actual <input> element */
        }}
        
        /* --- THIS IS THE FIX --- */
        /* 1. Hide the visual radio button circle */
        [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{
            display: none;
        }}
        
        /* 2. Style the text container (which is now the last-child) */
        [data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child {{
            padding: 10px 15px;
            font-weight: 500;
            cursor: pointer;
            border: 2px solid transparent;
            border-bottom: none;
            margin-bottom: -2px; 
            transition: all 0.2s ease;
            width: auto;      
            flex-grow: 1;     
            text-align: center;
        }}
        /* --- END OF FIX --- */
        
        /* The selected "tab" */
        [data-testid="stRadio"] input[type="radio"]:checked + div:last-child {{
            border-color: var(--gray-800);
            border-bottom-color: var(--secondary-background-color); 
            border-radius: 8px 8px 0 0;
            background-color: var(--secondary-background-color);
            color: var(--primary);
            font-weight: 600;
        }}
        /* Hover effect */
        [data-testid="stRadio"] input[type="radio"]:not(:checked) + div:last-child:hover {{
            background-color: var(--gray-900);
            border-radius: 8px 8px 0 0;
        }}
        
        [data-testid="stMetric"] {{
            background-color: var(--background-color);
            border: 1px solid var(--gray-800); border-radius: 8px;
            padding: 1rem 1.25rem;
        }}
        
        /* --- NEW: Style for Entry Signal Delta --- */
        [data-testid="stMetricDelta"] > div:first-child {{
            font-weight: 600;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
        }}

        /* --- ✅ NEW: RTL Formatting for Arabic Content (AI Summary) --- */
        .rtl-container {{
            direction: rtl; 
            text-align: right; 
        }}
        .rtl-container h2, .rtl-container h3, .rtl-container p, .rtl-container li, .rtl-container span, .rtl-container strong {{
            text-align: right !important;
        }}

    </style>
    """, unsafe_allow_html=True)

# --- ⭐️ 3. Core Analysis Logic (Helper Functions) ⭐️ ---
# (These are now defined *before* the main app function)

def calculate_robust_zscore_grouped(group_series):
    """Applies robust Z-score (MAD) to a pandas group."""
    series = pd.to_numeric(group_series, errors='coerce')
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=group_series.index)
        mean = series.mean()
        return (series - mean) / std
    z_score = (series - median) / (1.4826 * mad)
    return z_score

# --- ✅ MODIFIED (P3): Added Factor Interaction Logic ---
def calculate_all_z_scores(df, config):
    """
    Calculates sector-relative Z-scores for all factor components.
    Implements statistical robustness checks.
    """
    logging.info("Calculating Z-Scores...")
    df_analysis = df.copy()
    
    factor_defs = config.get('FACTOR_DEFINITIONS', {})
    stat_config = config.get('STATISTICAL', {})
    win_limit = stat_config.get('WINSORIZE_LIMIT', 0.05)
    min_sector_size = stat_config.get('MIN_SECTOR_SIZE_FOR_MEDIAN', 5)

    sector_counts = df_analysis['Sector'].value_counts()
    small_sectors = sector_counts[sector_counts < min_sector_size].index
    logging.info(f"Small sectors (<{min_sector_size} stocks) found: {list(small_sectors)}. Global medians will be used.")

    # --- ✅ NEW (Phase 2): Add and set the z_score_fallback flag ---
    df_analysis['z_score_fallback'] = bool(False)
    df_analysis.loc[df_analysis['Sector'].isin(small_sectors), 'z_score_fallback'] = bool(True)
    # --- End of change ---

    all_components = []
    for factor in factor_defs.keys():
        all_components.extend(factor_defs[factor]['components'])
    
    for comp in all_components:
        col = comp['name']
        if col not in df_analysis.columns:
            logging.warning(f"Factor component '{col}' not found in data. Skipping.")
            continue
            
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        
        # --- ✅ FIX (Phase 1): Use scipy.stats.winsorize for consistent outlier clipping ---
        # This replaces the manual quantile().clip() method
        df_analysis[col] = winsorize(df_analysis[col].fillna(df_analysis[col].median()), limits=(win_limit, win_limit))
        # We fillna with the median *before* winsorizing to handle NaNs robustly
        
        global_median = df_analysis[col].median()
        if global_median == 0: global_median = 1e-6 
        
        sector_medians = df_analysis.groupby('Sector')[col].median()
        sector_medians.loc[small_sectors] = global_median
        sector_medians = sector_medians.fillna(global_median)
        sector_medians[sector_medians == 0] = global_median
        
        df_analysis[f"{col}_Sector_Median"] = df_analysis['Sector'].map(sector_medians)
        df_analysis[f"{col}_Rel_Ratio"] = df_analysis[col] / df_analysis[f"{col}_Sector_Median"]
        
        z_col_name = f"Z_{col}"
        df_analysis[z_col_name] = df_analysis.groupby('Sector')[f"{col}_Rel_Ratio"].transform(calculate_robust_zscore_grouped)
        
        if not comp['high_is_good']:
            df_analysis[z_col_name] = df_analysis[z_col_name] * -1.0
            
        df_analysis[z_col_name] = df_analysis[z_col_name].fillna(0)

    logging.info("Combining components into final Z-Scores...")
    for factor, details in factor_defs.items():
        z_cols_to_average = [f"Z_{c['name']}" for c in details['components'] if f"Z_{c['name']}" in df_analysis.columns]
        if z_cols_to_average:
            df_analysis[f"Z_{factor}"] = df_analysis[z_cols_to_average].mean(axis=1)
        else:
            df_analysis[f"Z_{factor}"] = 0.0
            
    # --- MODIFICATION START (P3): Add Interaction Factors ---
    logging.info("Calculating Factor Interaction Scores (QxM)...")
    
    # 1. Quality x Momentum (QxM)
    z_quality_col = 'Z_Quality'
    z_momentum_col = 'Z_Momentum'
    qxm_raw_col = 'QxM_raw'
    z_qxm_col = 'Z_QxM' # This is the new factor name

    if z_quality_col in df_analysis.columns and z_momentum_col in df_analysis.columns:
        # Create the raw interaction score
        df_analysis[qxm_raw_col] = df_analysis[z_quality_col] * df_analysis[z_momentum_col]
        
        # Normalize the interaction score itself using the same robust z-score logic
        # This treats the raw QxM score as its own "component"
        df_analysis[z_qxm_col] = df_analysis.groupby('Sector')[qxm_raw_col].transform(calculate_robust_zscore_grouped)
        df_analysis[z_qxm_col] = df_analysis[z_qxm_col].fillna(0.0)
    else:
        df_analysis[z_qxm_col] = 0.0
        
    # --- MODIFICATION END (P3) ---
            
    return df_analysis

# --- ✅ MODIFIED (P1): Added Market Regime Check ---
def generate_quant_report(CONFIG, progress_callback=None):
    """
    Core logic, decoupled from Streamlit.
    Fetches data, runs analysis, calculates Z-scores, and generates reports.
    
    --- ✅ MODIFIED: Passes CONFIG to child functions ---
    """
    
    def report_progress(percent, text):
        if progress_callback:
            progress_callback(percent, text)
        logging.info(f"Progress: {percent*100:.0f}% - {text}")

    report_progress(0.01, "Starting analysis...")

    # --- 1. Check Market Regime (NEW STEP - P1) ---
    report_progress(0.02, "(1/8) Checking market regime...")
    market_regime = check_market_regime(CONFIG)
    report_progress(0.04, f"(1/8) Market Regime: {market_regime}")

    # Store in session state to display on UI
    st.session_state.market_regime = market_regime 
    
    if market_regime == "BEARISH" and CONFIG.get('HALT_IN_BEAR_MARKET', True):
        report_progress(1.0, f"Analysis Halted: Market Regime is {market_regime}.")
        # Return empty but valid data structures
        return pd.DataFrame(), {}, {}, "BEARISH" # Return regime status
        
    # --- 2. Fetch Tickers (WAS STEP 1) ---
    report_progress(0.05, "(2/8) Fetching market ticker list...")
    ticker_symbols = fetch_market_tickers(CONFIG) # <-- ✅ MODIFIED
    if not ticker_symbols:
        report_progress(1.0, "Error: No ticker symbols found. Analysis cancelled.")
        return None, None, None, market_regime
        
    exclude_tickers = CONFIG.get('EXCLUDE_TICKERS', [])
    ticker_symbols = [t for t in ticker_symbols if t not in exclude_tickers]
    
    limit = CONFIG.get('TICKER_LIMIT', 0)
    if limit > 0:
        ticker_symbols = ticker_symbols[:limit]
        report_progress(0.07, f"(2/8) Analysis limited to {limit} tickers.")
    
    # --- 3. Process Tickers Concurrently (WAS STEP 2) ---
    MAX_WORKERS = CONFIG.get('MAX_CONCURRENT_WORKERS', 10)
    report_progress(0.1, f"(3/8) Checking cache for {len(ticker_symbols)} tickers...")

    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    CACHE_TTL_SECONDS = 6 * 3600 # 6 hours
    
    results_list = []
    all_histories = {}
    tickers_to_fetch = []
    current_time = time.time()
    
    for ticker in ticker_symbols:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
        
        if os.path.exists(cache_path):
            try:
                cache_mod_time = os.path.getmtime(cache_path)
                if (current_time - cache_mod_time) < CACHE_TTL_SECONDS:
                    with open(cache_path, 'rb') as f:
                        result = pickle.load(f)
                    
                    if result.get('success', False):
                        if 'hist_df' in result:
                            all_histories[ticker] = result.pop('hist_df')
                        results_list.append(result)
                        continue 
            except Exception as e:
                logging.warning(f"Failed to load cache for {ticker}, will re-fetch: {e}")
                
        tickers_to_fetch.append(ticker)
    
    cached_count = len(results_list)
    report_progress(0.15, f"(3/8) Loaded {cached_count} tickers from cache. Fetching {len(tickers_to_fetch)} new tickers...")
    
    processed_count = 0
    total_to_fetch = len(tickers_to_fetch)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # --- ✅ MODIFIED: Pass CONFIG and fetch_news=False ---
        future_to_ticker = {executor.submit(process_ticker, ticker, CONFIG, fetch_news=False): ticker for ticker in tickers_to_fetch}
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result(timeout=60) 
                
                if result.get('success', False):
                    cache_path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(result, f)
                    except Exception as e:
                        logging.warning(f"Failed to save cache for {ticker}: {e}")
                    
                    if 'hist_df' in result:
                        all_histories[ticker] = result.pop('hist_df') 
                    results_list.append(result)
                
                else:
                    logging.error(f"Failed to process {ticker}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logging.error(f"Error processing {ticker} in main loop: {e}", exc_info=True)
            
            processed_count += 1
            if total_to_fetch > 0:
                percent_done = 0.15 + (0.55 * (processed_count / total_to_fetch)) 
                report_progress(percent_done, f"(3/8) Processing: {ticker} ({processed_count}/{total_to_fetch})")

    end_time = time.time()
    report_progress(0.7, f"(4/8) Data fetch complete. Time taken: {end_time - start_time:.2f}s")

    if not results_list:
        report_progress(1.0, "Error: No data successfully processed. Analysis cancelled.")
        return None, None, None, market_regime
        
    results_df = pd.DataFrame(results_list)
    results_df.set_index('ticker', inplace=True)
    
    report_progress(0.75, "(5/8) Risk metrics calculated in spus.py.")
    
    # --- 6. Factor Z-Score Calculation (WAS STEP 4) ---
    report_progress(0.8, "(6/8) Calculating robust Z-Scores...")
    results_df = calculate_all_z_scores(results_df, CONFIG)
    
    # --- 7. Save Reports (Excel, PDF, CSV) (WAS STEP 5) ---
    report_progress(0.9, "(7/8) Generating reports...")
    
    results_df.sort_values(by='Z_Value', ascending=False, inplace=True)

    results_df_display = results_df.rename(columns={
        'last_price': 'Last Price', 'Sector': 'Sector', 'marketCap': 'Market Cap',
        'forwardPE': 'Forward P/E', 'priceToBook': 'P/B Ratio', 'grahamValuation': 'Valuation (Graham)',
        'momentum_12m': 'Momentum (12M %)', 'volatility_1y': 'Volatility (1Y)',
        'returnOnEquity': 'ROE (%)', 'debtToEquity': 'Debt/Equity', 'profitMargins': 'Profit Margin (%)',
        'beta': 'Beta', 'RSI': 'RSI (14)', 'ADX': 'ADX (14)',
        'Final Stop Loss': 'Stop Loss',
        'shortName': 'Name' # <-- ✅ ADDED
    })
    
    pct_cols = ['ROE (%)', 'Profit Margin (%)', 'Momentum (12M %)', 'Risk % (to Stop)']
    for col in pct_cols:
        if col in results_df_display.columns:
            results_df_display[col] = results_df_display[col] * 100

    data_sheets = {
        'Top 20 (By Value)': results_df_display.sort_values(by='Z_Value', ascending=False).head(20),
        'Top 20 (By Momentum)': results_df_display.sort_values(by='Z_Momentum', ascending=False).head(20),
        'Top 20 (By Quality)': results_df_display.sort_values(by='Z_Quality', ascending=False).head(20),
        'Top Bullish Technicals': results_df_display.sort_values(by='Z_Technical', ascending=False).head(20),
        'Top Undervalued (Graham)': results_df_display[results_df_display['Valuation (Graham)'] == 'Undervalued (Graham)'].sort_values(by='Z_Value', ascending=False).head(20),
        'All Results (Raw)': results_df
    }

    excel_file_path = os.path.join(BASE_DIR, CONFIG['LOGGING']['EXCEL_FILE_PATH'])
    try:
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            for sheet_name, df_sheet in data_sheets.items():
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=True)
        report_progress(0.92, f"Excel report saved: {excel_file_path}")
    except Exception as e:
        logging.error(f"Failed to save Excel file: {e}")

    if REPORTLAB_AVAILABLE:
        try:
            timestamp = datetime.now(SAUDI_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
            base_pdf_path = os.path.splitext(excel_file_path)[0]
            pdf_file_path = f"{base_pdf_path}_{datetime.now(SAUDI_TZ).strftime('%Y%m%d_%H%M%S')}.pdf"
            
            doc = SimpleDocTemplate(pdf_file_path, pagesize=landscape(letter))
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='Left', alignment=TA_LEFT))
            
            elements = [Paragraph(f"Quant Report - {timestamp}", styles['h1'])]
            
            pdf_cols = ['Name', 'Last Price', 'Z_Value', 'Z_Momentum', 'Z_Quality', 'Risk/Reward Ratio']
            
            for sheet_name, df_sheet in data_sheets.items():
                if sheet_name == 'All Results (Raw)': continue 
                
                elements.append(Paragraph(sheet_name, styles['h2']))
                
                cols_to_show = [col for col in pdf_cols if col in df_sheet.columns]
                
                # --- ✅ MODIFIED: Add Name to PDF ---
                df_pdf = df_sheet.head(15).reset_index()[['ticker', 'Name'] + cols_to_show]
                
                df_pdf = df_pdf.fillna('N/A')
                for col in cols_to_show:
                    if col in df_pdf.select_dtypes(include=[np.number]).columns:
                        df_pdf[col] = df_pdf[col].round(2)
                
                data = [df_pdf.columns.tolist()] + df_pdf.values.tolist()
                
                col_widths = [1.2*inch, 2.0*inch] + [1*inch] * len(cols_to_show) # Added width for Name
                table = Table(data, hAlign='LEFT', colWidths=col_widths)
                
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.green),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 0.25*inch))
                
            doc.build(elements)
            report_progress(0.95, f"PDF report saved: {pdf_file_path}")
        except Exception as e:
            logging.error(f"Failed to create PDF report: {e}")
    
    try:
        results_dir = os.path.join(BASE_DIR, CONFIG.get('LOGGING', {}).get('RESULTS_DIR', 'results_history'))
        os.makedirs(results_dir, exist_ok=True)
        timestamp_csv = datetime.now(SAUDI_TZ).strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(results_dir, f"quant_results_{timestamp_csv}.csv")
        results_df.to_csv(csv_path)
        report_progress(0.98, f"Timestamped CSV saved: {csv_path}")
    except Exception as e:
        logging.error(f"Failed to save timestamped CSV: {e}")

    report_progress(1.0, "Analysis complete.")
    
    return results_df, all_histories, data_sheets, market_regime

# --- ⭐️ 4. Streamlit UI Functions ⭐️ ---

# --- ✅ MODIFIED: Caching now uses config_file_name as key ---
@st.cache_data(show_spinner=False, ttl=3600) 
def load_analysis_data(config_file_name, run_timestamp):
    """
    Streamlit cache wrapper for the core analysis function.
    """
    progress_bar = st.progress(0, text="Starting analysis...")
    status_text = st.empty()
    
    def st_progress_callback(percent, text):
        progress_bar.progress(percent, text=text)
        status_text.info(text)
        
    logging.info(f"Cache miss or manual run. Running full analysis for {config_file_name}... (Timestamp: {run_timestamp})")
    
    # --- ✅ MODIFIED (P5): Config is now loaded from session state ---
    # The config is already loaded and modified in run_market_analyzer_app
    _CONFIG = st.session_state.get('CONFIG')
    if _CONFIG is None:
        st.error(f"Failed to load CONFIG from session state in load_analysis_data.")
        return None, None, None, None, None
    
    df, histories, sheets, market_regime = generate_quant_report(_CONFIG, st_progress_callback)
    
    progress_bar.empty()
    status_text.empty()
    
    if df is None:
        st.error("Analysis failed. Check logs.")
        return None, None, None, None, None
    
    # Handle the case where analysis was halted
    if market_regime == "BEARISH" and df.empty:
        st.warning(f"Analysis Halted: Market Regime is {market_regime}.")
        
    return df, histories, sheets, datetime.now(SAUDI_TZ).timestamp(), market_regime

def get_latest_reports(excel_base_path):
    """Gets paths for the latest Excel and PDF reports."""
    base_dir = os.path.dirname(excel_base_path)
    excel_name_no_ext = os.path.splitext(os.path.basename(excel_base_path))[0]
    
    latest_pdf = None
    pdf_pattern = os.path.join(base_dir, f"{excel_name_no_ext}_*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    if pdf_files:
        latest_pdf = max(pdf_files, key=os.path.getmtime)
        
    excel_path = excel_base_path if os.path.exists(excel_base_path) else None
    return excel_path, latest_pdf

# --- ✅ MODIFIED: Now accepts CONFIG ---
def create_price_chart(hist_df, ticker_data, CONFIG):
    """Creates an interactive Plotly Price Chart with SMAs, MACD, and OB Zones."""
    
    ticker = ticker_data.name 
    cfg = CONFIG['TECHNICALS'] # Use passed CONFIG
    short_ma_col = f'SMA_{cfg["SHORT_MA_WINDOW"]}'
    long_ma_col = f'SMA_{cfg["LONG_MA_WINDOW"]}'
    macd_col = f'MACD_{cfg["MACD_SHORT_SPAN"]}_{cfg["MACD_LONG_SPAN"]}_{cfg["MACD_SIGNAL_SPAN"]}'
    macd_h_col = f'MACDh_{cfg["MACD_SHORT_SPAN"]}_{cfg["MACD_LONG_SPAN"]}_{cfg["MACD_SIGNAL_SPAN"]}'
    macd_s_col = f'MACDs_{cfg["MACD_SHORT_SPAN"]}_{cfg["MACD_LONG_SPAN"]}_{cfg["MACD_SIGNAL_SPAN"]}'

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'MACD'), 
                        row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=hist_df.index,
                                open=hist_df['Open'],
                                high=hist_df['High'],
                                low=hist_df['Low'],
                                close=hist_df['Close'],
                                name='Price'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df[short_ma_col], 
                             line=dict(color='orange', width=1), name=f'SMA {cfg["SHORT_MA_WINDOW"]}'),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df[long_ma_col], 
                             line=dict(color='blue', width=1), name=f'SMA {cfg["LONG_MA_WINDOW"]}'),
                  row=1, col=1)

    fig.add_trace(go.Bar(x=hist_df.index, y=hist_df[macd_h_col], 
                         name='Histogram',
                         marker_color=np.where(hist_df[macd_h_col] < 0, 'red', 'green')),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df[macd_col], 
                             line=dict(color='blue', width=1), name='MACD'),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df[macd_s_col], 
                             line=dict(color='orange', width=1), name='Signal'),
                  row=2, col=1)

    # --- Add Key Price Zones ---
    
    b_ob_low = ticker_data.get('bullish_ob_low', np.nan)
    b_ob_high = ticker_data.get('bullish_ob_high', np.nan)
    be_ob_low = ticker_data.get('bearish_ob_low', np.nan)
    be_ob_high = ticker_data.get('bearish_ob_high', np.nan)
    last_sl = ticker_data.get('last_swing_low', np.nan)
    last_sh = ticker_data.get('last_swing_high', np.nan)
    
    chart_start_date = hist_df.index.min()
    chart_end_date = hist_df.index.max()

    if pd.notna(b_ob_low) and pd.notna(b_ob_high):
        fig.add_shape(
            type="rect",
            x0=chart_start_date, y0=b_ob_low,
            x1=chart_end_date, y1=b_ob_high,
            line=dict(width=0),
            fillcolor="rgba(0, 255, 0, 0.2)",
            layer="below",
            row=1, col=1
        )
    if pd.notna(be_ob_low) and pd.notna(be_ob_high):
        fig.add_shape(
            type="rect",
            x0=chart_start_date, y0=be_ob_low,
            x1=chart_end_date, y1=be_ob_high,
            line=dict(width=0),
            fillcolor="rgba(255, 0, 0, 0.2)",
            layer="below",
            row=1, col=1
        )
    if pd.notna(last_sl):
        fig.add_hline(
            y=last_sl,
            line=dict(color="blue", width=1, dash="dot"),
            annotation_text="Swing Low",
            annotation_position="bottom right",
            row=1, col=1
        )
    if pd.notna(last_sh):
        fig.add_hline(
            y=last_sh,
            line=dict(color="red", width=1, dash="dot"),
            annotation_text="Swing High",
            annotation_position="top right",
            row=1, col=1
        )
    
    fig.update_layout(
        title_text=f"{ticker} Technical Chart",
        xaxis_rangeslider_visible=False,
        height=500,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    
    return fig

def create_radar_chart(ticker_data, factor_cols):
    """Creates a Plotly Radar Chart for factor explainability."""
    
    values = ticker_data[factor_cols].values.flatten().tolist()
    theta = [col.replace('Z_', '') for col in factor_cols]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], # Close the loop
        theta=theta + [theta[0]], # Close the loop
        fill='toself',
        name='Factor Z-Score'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(-2, min(values)-0.5), max(2, max(values)+0.5)] 
            )
        ),
        title=f"Factor Profile for {ticker_data.name}",
        height=400
    )
    return fig

# --- ✅ MODIFIED (USER REQ): Now uses 'open_positions' df ---
def create_portfolio_treemap(open_positions_df):
    """
    Creates a Plotly Treemap to visualize portfolio allocation and performance.
    """
    if open_positions_df.empty:
        return go.Figure().update_layout(title_text="Portfolio Treemap (No data)")

    if 'Sector' not in open_positions_df.columns:
        open_positions_df['Sector'] = "Unknown"
        
    # Create 'P/L (%)' for color
    open_positions_df['P/L (%)'] = (open_positions_df['Unrealized P/L'] / open_positions_df['Total Cost']) * 100
        
    fig = px.treemap(
        open_positions_df,
        path=[px.Constant("My Portfolio"), 'Sector', 'Name'],  # Hierarchy
        values='Market Value',
        color='P/L (%)',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title="Portfolio Allocation by Market Value (Color by P/L %)",
    )
    
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        height=500
    )
    fig.update_traces(
        textinfo="label+value+text",
        texttemplate="%{label}<br>$%{value:,.0f}<br>%{customdata[0]:.2f}%",
        customdata=open_positions_df[['P/L (%)']]
    )
    return fig

# --- ✅ MODIFIED (Phase 2 & 3): Dynamic Checklist ---
def display_buy_signal_checklist(ticker_data):
    """
    Displays a 5-step checklist on the Ticker Deep Dive tab.
    Thresholds are now DYNAMIC based on market regime.
    """
    
    # --- ✅ NEW (Phase 3): Dynamic Thresholds ---
    market_regime = st.session_state.get('market_regime', 'UNKNOWN')
    
    if market_regime == "BEARISH":
        SCORE_THRESHOLD = 1.2
        FACTOR_Z_THRESHOLD = 0.75
        RR_RATIO_THRESHOLD = 2.0
        RSI_OVERBOUGHT = 60.0 # Be more sensitive
        TREND_GOOD = ["Confirmed Uptrend"] # Be more strict
        MACD_GOOD = ["Bullish Crossover"]
    elif market_regime == "TRANSITIONAL":
        SCORE_THRESHOLD = 1.0
        FACTOR_Z_THRESHOLD = 0.5
        RR_RATIO_THRESHOLD = 1.5
        RSI_OVERBOUGHT = 70.0
        TREND_GOOD = ["Confirmed Uptrend", "Uptrend (Correction)"]
        MACD_GOOD = ["Bullish Crossover", "Bullish"]
    else: # BULLISH
        SCORE_THRESHOLD = 0.8 # Be less strict
        FACTOR_Z_THRESHOLD = 0.25
        RR_RATIO_THRESHOLD = 1.5
        RSI_OVERBOUGHT = 80.0 # Allow for overbought
        TREND_GOOD = ["Confirmed Uptrend", "Uptrend (Correction)"]
        MACD_GOOD = ["Bullish Crossover", "Bullish"]
    # --- End of change ---

    # Step 1: Quant Score
    step1_met = False
    step1_text = f"**1. Quant Score > {SCORE_THRESHOLD}**"
    score = ticker_data.get('Final Quant Score', 0)
    if pd.notna(score) and score > SCORE_THRESHOLD:
        step1_met = True
    step1_details = f"Score is {score:.2f}"

    # Step 2: Factor Profile (Value & Quality)
    step2_met = False
    step2_text = f"**2. Value & Quality > {FACTOR_Z_THRESHOLD}**"
    z_value = ticker_data.get('Z_Value', -99)
    z_quality = ticker_data.get('Z_Quality', -99)
    if pd.notna(z_value) and pd.notna(z_quality) and (z_value > FACTOR_Z_THRESHOLD) and (z_quality > FACTOR_Z_THRESHOLD):
        step2_met = True
    step2_details = f"Value: {z_value:.2f}, Quality: {z_quality:.2f}"

    # Step 3: Favorable Technicals
    step3_met = False
    step3_text = "**3. Favorable Technicals**"
    
    rsi_val = ticker_data.get('RSI', 99)
    trend_val = ticker_data.get('Trend (50/200 Day MA)', 'N/A')
    macd_val = ticker_data.get('MACD_Signal', 'N/A')

    is_rsi_ok = pd.notna(rsi_val) and rsi_val < RSI_OVERBOUGHT
    is_trend_ok = trend_val in TREND_GOOD
    is_macd_ok = macd_val in MACD_GOOD

    if is_rsi_ok and is_trend_ok and is_macd_ok:
        step3_met = True
        
    rsi_icon = "✅" if is_rsi_ok else "❌"
    trend_icon = "✅" if is_trend_ok else "❌"
    macd_icon = "✅" if is_macd_ok else "❌"
    
    step3_details = f"{trend_icon} Trend: {trend_val}<br>{macd_icon} MACD: {macd_val}<br>{rsi_icon} RSI: {rsi_val:.1f} (<{RSI_OVERBOUGHT})"

    # Step 4: SMC Entry Signal
    step4_met = False
    step4_text = "**4. SMC Entry Signal**"
    entry_signal = ticker_data.get('entry_signal', 'No Trade')
    has_fvg = ticker_data.get('bullish_ob_fvg', False)
    has_vol = ticker_data.get('bullish_ob_volume_ok', False)
    vol_missing_data = ticker_data.get('smc_volume_missing', False) # <-- ✅ NEW (Phase 2): Check for the flag
    
    details = []
    if entry_signal == 'Buy near Bullish OB':
        step4_met = True
        details.append(f"Signal: {entry_signal}")
    elif entry_signal == 'Sell near Bearish OB':
        details.append(f"Signal: {entry_signal}")
    else:
        details.append("Signal: No Trade")

    details.append(f"FVG: {'✅' if has_fvg else '❌'}")
    
    # --- ✅ NEW (Phase 2): Add warning if volume data was missing ---
    vol_icon = '✅' if has_vol else '❌'
    vol_warning = " (Data N/A)" if vol_missing_data else ""
    details.append(f"Vol: {vol_icon}{vol_warning}")
    # --- End of change ---
    
    step4_details = ", ".join(details)
        
    # Step 5: Risk/Reward
    step5_met = False
    step5_text = f"**5. R/R Ratio > {RR_RATIO_THRESHOLD}**"
    rr_ratio = ticker_data.get('Risk/Reward Ratio', 0)
    if pd.notna(rr_ratio) and rr_ratio > RR_RATIO_THRESHOLD:
        step5_met = True
    step5_details = f"Ratio is {rr_ratio:.2f}"
    
    st.subheader(f"Buy Signal Checklist (Mode: {market_regime})")
    cols = st.columns(5)
    
    criteria = [
        (step1_met, step1_text, step1_details),
        (step2_met, step2_text, step2_details),
       (step3_met, step3_text, step3_details),
        (step4_met, step4_text, step4_details),
        (step5_met, step5_text, step5_details)
    ]
    
    for i, (met, text, details) in enumerate(criteria):
        with cols[i]:
            icon = "✅" if met else "❌"
            st.markdown(f"**{icon} {text}**")
            st.markdown(details, unsafe_allow_html=True)


# --- ⭐️ 5. NEW: Main App Logic (as a function) ⭐️ ---

def run_market_analyzer_app(config_file_name):

    # --- Load Config & CSS ---
    # --- ✅ MODIFIED (P5): Inject API key from st.secrets ---
    if 'CONFIG' not in st.session_state:
        # 1. Load the base config from the file
        config_data = load_config(config_file_name)
        
        if config_data is None:
            st.error(f"FATAL: {config_file_name} not found or corrupted. App cannot start.")
            st.stop()
        
        # 2. Inject API keys from Streamlit secrets
        try:
            if "DATA_PROVIDERS" not in config_data:
                config_data["DATA_PROVIDERS"] = {}
            
            # Inject Gemini API Key (Primary)
            gemini_api_key = st.secrets.get("GEMINI_API_KEY")
            if gemini_api_key:
                config_data["DATA_PROVIDERS"]["GEMINI_API_KEY"] = gemini_api_key
                logging.info("Successfully injected GEMINI API key from st.secrets.")
            else:
                logging.warning("GEMINI_API_KEY not found in Streamlit secrets. Gemini features will fail.")

            # Inject OpenAI API Key (Fallback)
            openai_api_key = st.secrets.get("OPENAI_API_KEY")
            if openai_api_key:
                config_data["DATA_PROVIDERS"]["OPENAI_API_KEY"] = openai_api_key
                logging.info("Successfully injected OpenAI API key from st.secrets.")
            else:
                logging.warning("OPENAI_API_KEY not found in Streamlit secrets. OpenAI features will fail.")

            # --- ✅ NEW: Inject Finnhub API Key ---
            finnhub_api_key = st.secrets.get("FINNHUB_API_KEY")
            if finnhub_api_key:
                config_data["DATA_PROVIDERS"]["FINNHUB_API_KEY"] = finnhub_api_key
                logging.info("Successfully injected FINNHUB API key from st.secrets.")
            else:
                logging.warning("FINNHUB_API_KEY not found in Streamlit secrets. News features will fail.")
            
        except Exception as e:
            # This handles cases where st.secrets might not be available (e.g., local run without secrets file)
            logging.warning(f"Could not access Streamlit secrets. AI/News features may fail. Error: {e}")

        # 4. Store the modified config in session state
        st.session_state.CONFIG = config_data

    CONFIG = st.session_state.CONFIG
    if CONFIG is None: 
        st.error(f"FATAL: Config is None even after loading. App cannot start.")
        st.stop()
    # --- END OF MODIFICATION ---
    
    load_css()
    
    # --- Setup Logger ---
    log_file_path = os.path.join(BASE_DIR, CONFIG.get('LOGGING', {}).get('LOG_FILE_PATH', 'spus_analysis.log'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()
        ]
    )

    # --- Initialize Session State (scoped to this app) ---
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    if 'run_timestamp' not in st.session_state:
        st.session_state.run_timestamp = time.time() 
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "🏆 Quant Rankings"
    if 'market_regime' not in st.session_state:
        st.session_state.market_regime = "UNKNOWN"
    
    # --- ✅ NEW (USER REQ): Portfolio Transaction Log ---
    if 'transactions' not in st.session_state:
        st.session_state.transactions = [] # List of transaction dicts
    if 'cash' not in st.session_state:
        st.session_state.cash = 100000.0 # Default cash
    if 'prefill_transaction' not in st.session_state:
        st.session_state.prefill_transaction = None # For "Add to Portfolio" button
    
    # --- ⭐️ START OF MOVED BLOCK ⭐️ ---
    # This block is moved from the bottom to here to fix the UnboundLocalError
    # --- ✅ MODIFIED: Load Data using config_file_name as key ---
    base_raw_df, base_histories, base_sheets, base_last_run_time, base_market_regime = load_analysis_data(config_file_name, st.session_state.run_timestamp)
    
    # Check for cache invalidation
    if 'raw_df' not in st.session_state or st.session_state.get('base_run_timestamp') != st.session_state.run_timestamp:
        if base_raw_df is None:
            st.error("Analysis failed to produce base data. App cannot continue.")
            st.stop()
        
        st.session_state.raw_df = base_raw_df.copy()
        st.session_state.all_histories = base_histories.copy()
        st.session_state.data_sheets = base_sheets
        st.session_state.last_run_time = base_last_run_time
        st.session_state.market_regime = base_market_regime # Store regime from the run
        st.session_state.base_run_timestamp = st.session_state.run_timestamp
    
    raw_df = st.session_state.raw_df
    all_histories = st.session_state.all_histories
    last_run_time = st.session_state.last_run_time
    
    # This handles the "BEARISH" halt case
    if raw_df is None or raw_df.empty:
        if st.session_state.market_regime == "BEARISH":
            st.warning(f"Analysis Halted: Market Regime is BEARISH. No buy signals will be generated.")
            st.stop()
        else:
            st.error("No data available in session state.")
            st.stop()
    # --- ⭐️ END OF MOVED BLOCK ⭐️ ---


    # --- Sidebar ---
    with st.sidebar:
        try:
            st.image("logo.jpg", width=200)
        except Exception as e:
            st.warning(f"Could not load logo.jpg: {e}")
        
        st.title("Quant Analyzer")
        # --- ✅ MODIFIED (P4): Hard-coded market name ---
        st.markdown(f"Market: **US Market (SPUS)**")
        st.divider()

        st.subheader("Controls")
        if st.button("🔄 Run Full Analysis", type="primary"):
            st.session_state.selected_ticker = None
            st.session_state.run_timestamp = time.time() 
            st.session_state.active_tab = "🏆 Quant Rankings"
            if 'raw_df' in st.session_state:
                del st.session_state['raw_df']
            # --- ✅ MODIFIED (P4): Clear AI cache on full run
            for key in st.session_state.keys():
                if key.startswith("ai_summary_"):
                    del st.session_state[key]
            st.rerun()
        
        st.divider()
        st.subheader("Stock Analyzer")
        new_ticker = st.text_input("Analyze Single Ticker:", placeholder="e.g., MSFT or 1120.SR").upper().strip()
        
        if st.button("Analyze and Deep Dive"):
            if new_ticker:
                if 'raw_df' not in st.session_state:
                    st.warning("Priming data... please click 'Analyze' again.")
                    st.rerun() 
                elif new_ticker in st.session_state.raw_df.index:
                    st.success(f"'{new_ticker}' is already loaded.")
                    st.session_state.selected_ticker = new_ticker
                    st.session_state.active_tab = "🔬 Ticker Deep Dive"
                    st.rerun()
                else:
                    with st.spinner(f"Processing data for {new_ticker}..."):
                        try:
                            # --- ✅ MODIFIED: Pass CONFIG and fetch_news=True ---
                            result = process_ticker(new_ticker, CONFIG, fetch_news=True)
                            
                            if result and result.get('success', False):
                                new_hist_df = result.pop('hist_df', None)
                                if new_hist_df is not None:
                                    st.session_state.all_histories[new_ticker] = new_hist_df
                                
                                new_ticker_df = pd.DataFrame([result])
                                new_ticker_df.set_index('ticker', inplace=True)
                                
                                st.session_state.raw_df = pd.concat([st.session_state.raw_df, new_ticker_df])
                                
                                st.info(f"Re-calculating Z-Scores for {len(st.session_state.raw_df)} stocks...")
                                # --- ✅ MODIFIED: Pass CONFIG ---
                                st.session_state.raw_df = calculate_all_z_scores(st.session_state.raw_df, CONFIG)
                                
                                st.success(f"Successfully added '{new_ticker}'.")
                                st.session_state.selected_ticker = new_ticker
                                st.session_state.active_tab = "🔬 Ticker Deep Dive"
                                st.rerun()
                                
                            else:
                                st.error(f"Failed to fetch data for {new_ticker}. Error: {result.get('error', 'Unknown')}")
                        except Exception as e:
                            st.error(f"An exception occurred while processing {new_ticker}: {e}")
            else:
                st.warning("Please enter a ticker symbol.")
        
        st.divider()

        # --- ✅ MODIFIED (USER REQ): Factor Weight Presets ---
        st.subheader("Factor Weights")
        
        # Define presets
        factor_presets = {
            "Default": CONFIG.get('DEFAULT_FACTOR_WEIGHTS', {}),
            "Bullish (Momentum Focus)": {
                "Value": 0.15, "Momentum": 0.25, "Quality": 0.15,
                "Size": 0.05, "LowVolatility": 0.05, "Technical": 0.20, "QxM": 0.15
            },
            "Bearish (Quality Focus)": {
                "Value": 0.20, "Momentum": 0.05, "Quality": 0.25,
                "Size": 0.05, "LowVolatility": 0.25, "Technical": 0.10, "QxM": 0.10
            },
            "Value Focus": {
                "Value": 0.40, "Momentum": 0.10, "Quality": 0.20,
                "Size": 0.05, "LowVolatility": 0.10, "Technical": 0.05, "QxM": 0.10
            }
        }
        
        def apply_preset(preset_name):
            weights = factor_presets.get(preset_name, factor_presets["Default"])
            for factor, weight in weights.items():
                st.session_state[f"weight_{factor}"] = weight

        preset_selection = st.selectbox(
            "Load Weight Preset:", 
            options=factor_presets.keys(), 
            index=0, 
            key="preset_selector"
        )

        if st.button("Apply Preset"):
            apply_preset(preset_selection)
            st.rerun()

        st.info("Adjust weights to re-rank stocks. Weights will be normalized.")
        
        weights = {}
        default_weights = factor_presets["Default"] # Base defaults
        
        for factor, default in default_weights.items():
            if f"Z_{factor}" in raw_df.columns:
                # Use st.session_state to hold the current value, or default if not set
                current_weight = st.session_state.get(f"weight_{factor}", default)
                weights[factor] = st.slider(factor, 0.0, 1.0, current_weight, 0.05, key=f"weight_{factor}")
            else:
                if factor == "QxM" and 'raw_df' not in st.session_state:
                    st.info("Run analysis to enable 'QxM' factor.")
                else:
                    logging.warning(f"Factor {factor} defined in weights but not found in data. Skipping slider.")

            
        total_weight = sum(weights.values())
        norm_weights = {f: (w / total_weight) if total_weight > 0 else 0 for f, w in weights.items()}
        
        with st.expander("Normalized Weights"):
            for factor, weight in norm_weights.items():
                st.write(f"{factor}: {weight*100:.1f}%")
        
        st.divider()

        st.subheader("Downloads")
        excel_path, pdf_path = get_latest_reports(os.path.join(BASE_DIR, CONFIG['LOGGING']['EXCEL_FILE_PATH']))
        
        if excel_path:
            with open(excel_path, "rb") as file:
                st.download_button(
                    label="📥 Download Excel Report",
                    data=file,
                    file_name=os.path.basename(excel_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.info("Run analysis to generate reports.")

        if pdf_path:
            with open(pdf_path, "rb") as file:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=file,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                )
        
        st.divider()
        st.info("Analysis data is cached for 1 hour. Click 'Run' for fresh data.")


    # --- Main Page ---
    
    # --- ✅ MODIFIED (P1): Display Market Regime ---
    regime_status = st.session_state.get('market_regime', 'N/A')
    regime_color = "red" if regime_status == "BEARISH" else "green" if regime_status == "BULLISH" else "orange"
    
    st.title(f"Quantitative Dashboard")
    st.markdown(f"**Market Regime Status:** <span style='color:{regime_color}; font-weight: 600;'>{regime_status}</span>", unsafe_allow_html=True)
    
    
    # --- This block was moved to the top ---
    # base_raw_df, base_histories, ...
    # ...
    
    st.success(f"Data loaded from analysis run at: {datetime.fromtimestamp(last_run_time, SAUDI_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # --- UI: Dynamic Score Calculation & Filtering ---
    df = raw_df.copy()
    
    if df.empty:
        st.error("No stock data was successfully loaded. Check logs and data sources.")
        st.stop()

    df['Final Quant Score'] = 0.0
    factor_z_cols = []
    
    # --- ✅ MODIFIED (P3): norm_weights now includes 'QxM'
    for factor, weight in norm_weights.items():
        z_col = f"Z_{factor}"
        if z_col in df.columns:
            factor_z_cols.append(z_col) # Only add if it exists
            df[f"Weighted_{z_col}"] = df[z_col] * weight
            df['Final Quant Score'] += df[f"Weighted_{z_col}"]
        else:
            # This will now correctly log if Z_QxM is missing on first run
            logging.warning(f"Z-Score column {z_col} not found in dataframe.")

    st.subheader("Filters")
    
    filt_col1, filt_col2 = st.columns(2)
    
    all_sectors = sorted(df['Sector'].unique())
    selected_sectors = filt_col1.multiselect("Filter by Sector:", all_sectors, default=all_sectors)
    
    if df.empty or 'marketCap' not in df.columns or df['marketCap'].isnull().all():
        filt_col2.info("No Market Cap data to filter.")
        cap_range = (0.0, 0.0) 
    else:
        min_cap_val = float(df['marketCap'].min())
        max_cap_val = float(df['marketCap'].max())

        if min_cap_val == max_cap_val:
            min_cap = (min_cap_val / 1e9) * 0.9 
            max_cap = (max_cap_val / 1e9) * 1.1 
            if min_cap < 0: min_cap = 0.0
        else:
            min_cap = min_cap_val / 1e9
            max_cap = max_cap_val / 1e9
        
        if min_cap >= max_cap:
            min_cap = max_cap - 1.0
            if min_cap < 0: min_cap = 0.0

        cap_range = filt_col2.slider(
            "Filter by Market Cap (Billions):",
            min_value=min_cap,
            max_value=max_cap,
            value=(min_cap, max_cap),
            format="%.1f B"
        )
        
    filt_col3, filt_col4 = st.columns(2)

    all_trends = sorted(df['Trend (50/200 Day MA)'].unique())
    selected_trends = filt_col3.multiselect("Filter by MA Trend:", all_trends, default=all_trends)
    
    all_signals = sorted(df['entry_signal'].unique())
    selected_signals = filt_col4.multiselect("Filter by Entry Signal:", all_signals, default=all_signals)
    
    filt_col5, _ = st.columns(2)

    if 'pct_above_cutloss' in df.columns and not df['pct_above_cutloss'].isnull().all():
        valid_pct_data = df['pct_above_cutloss'].dropna()
        if not valid_pct_data.empty:
            min_pct = float(valid_pct_data.min())
            max_pct = float(valid_pct_data.max())
            default_min_pct = max(0.0, min_pct)
            if min_pct >= max_pct: max_pct = min_pct + 10.0
            
            support_range = filt_col5.slider(
                "Filter by % Above Cut Loss (Swing Low):",
                min_value=min_pct, 
                max_value=max_pct,
                value=(default_min_pct, max_pct),
                format="%.1f%%"
            )
        else:
            filt_col5.info("No Cut Loss % data to filter.")
            support_range = (0.0, 0.0)
    else:
        filt_col5.info("No Cut Loss % data to filter.")
        support_range = (0.0, 0.0)
    
    base_filters = (
        (df['Sector'].isin(selected_sectors)) &
        (df['Trend (50/200 Day MA)'].isin(selected_trends)) &
        (df['entry_signal'].isin(selected_signals))
    )

    if cap_range != (0.0, 0.0) and 'marketCap' in df.columns and not df['marketCap'].isnull().all():
        cap_filter = (
            (df['marketCap'].ge(cap_range[0] * 1e9)) &
            (df['marketCap'].le(cap_range[1] * 1e9))
        )
        base_filters &= cap_filter

    if support_range != (0.0, 0.0) and 'pct_above_cutloss' in df.columns and not df['pct_above_cutloss'].isnull().all():
        support_filter = (
            (df['pct_above_cutloss'].ge(support_range[0])) &
            (df['pct_above_cutloss'].le(support_range[1]))
        )
        base_filters &= support_filter

    filtered_df = df[base_filters].copy()
    
    filtered_df.sort_values(by='Final Quant Score', ascending=False, inplace=True)
    
    st.markdown(f"Displaying **{len(filtered_df)}** of **{len(df)}** total stocks matching filters.")
    st.divider()

    tab_list = ["🏆 Quant Rankings", "🔬 Ticker Deep Dive", "📈 Portfolio Analytics", "💼 My Portfolio"]
    
    try:
        default_idx = tab_list.index(st.session_state.active_tab)
    except ValueError:
        default_idx = 0

    selected_tab = st.radio(
        "Navigation:",
        tab_list,
        index=default_idx,
        horizontal=True
    )
    
    st.session_state.active_tab = selected_tab
    
    # --- Tab 1: Quant Rankings ---
    if selected_tab == "🏆 Quant Rankings":
        st.header("🏆 Top Stocks by Final Quant Score")
        st.info("Click a ticker to select it and automatically move to the 'Ticker Deep Dive' tab.")
        
        # --- ✅ NEW (USER REQ): AI Top 20 Summary ---
        st.subheader("🤖 AI Top 20 Summary")
        cache_key = "ai_summary_top20" 
        
        if cache_key in st.session_state:
            st.markdown(f'<div class="rtl-container">{st.session_state[cache_key]}</div>', unsafe_allow_html=True)
            if st.button("🔄 Regenerate Top 20 Summary", key="regen_ai_top20"):
                del st.session_state[cache_key]
                st.rerun()
        else:
            if st.button("🤖 Generate AI Summary for Top 20", type="secondary", key="gen_ai_top20"):
                with st.spinner("Analyzing Top 20 stocks... This may take a moment."):
                    try:
                        # Prepare data for the AI
                        top_20_df = filtered_df.head(20)
                        # Select key columns for the AI prompt
                        ai_cols = [
                            'shortName', 'Final Quant Score', 'entry_signal', 
                            'Risk/Reward Ratio', 'Z_Value', 'Z_Quality', 'Z_Momentum'
                        ]
                        existing_cols = [col for col in ai_cols if col in top_20_df.columns]
                        top_20_json = top_20_df[existing_cols].to_json(orient="records")
                        
                        summary = get_ai_top20_summary(top_20_json, CONFIG)
                        st.session_state[cache_key] = summary
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to generate AI summary: {e}")
                        st.session_state[cache_key] = "AI Summary generation failed."
        
        st.divider()
        # --- End of new block ---

        with st.expander("How to Find a Good Buy Signal (5-Step Guide)", expanded=False):
            st.markdown(f"""
                This 5-step method helps you use the app to find suitable buying opportunities.
                
                ### 1. Check the Final Quant Score (The "What")
                This is your primary signal. Look for stocks with a **high positive score** (e.g., > {st.session_state.get('dynamic_thresholds', {}).get('SCORE_THRESHOLD', 1.0)}) 
                in the ranked list below. 
                
                ### 2. Check the Factor Profile (The "Why")
                Click a stock and go to the **"🔬 Ticker Deep Dive"** tab. Look at the 
                **"Factor Profile"** radar chart. This tells you *why* the score is high. 
                Is it high on `Value` (it's cheap) and `Quality` (it's a good company)? 
                
                ### 3. Check the Technicals (The "Momentum")
                On the **"Deep Dive"** tab, look at the **"Buy Signal Checklist"**.
                * **Technicals:** Are the `Trend`, `MACD`, and `RSI` all favorable (✅)?
                
                ### 4. Check the SMC Signal (The "When")
                This is your high-precision entry signal.
                * **Entry Signal:** Does it show **"Buy near Bullish OB"**?
                * **FVG / Vol:** Does the checklist show a `✅` for **FVG** (Fair Value Gap) and **Vol** (BOS Volume)? This confirms a high-quality signal.

                ### 5. Check the Risk & Sizing (The "How")
                In the **"Risk & Position Sizing"** section, check the:
                * **Risk/Reward Ratio:** Is it favorable (e.g., > {st.session_state.get('dynamic_thresholds', {}).get('RR_RATIO_THRESHOLD', 1.5)})?
                * **Final Stop Loss:** Is this exit price (based on ATR or Cut-Loss) acceptable?
            """)
        
        rank_col1, rank_col2 = st.columns([1, 2])
        
        with rank_col1:
            st.subheader(f"Ranked List ({len(filtered_df)})")
            
            with st.container(height=800):
                if filtered_df.empty:
                    st.warning("No stocks match the current filters.")
                else:
                    for ticker in filtered_df.index:
                        data = filtered_df.loc[ticker]
                        score = data['Final Quant Score']
                        # --- ✅ MODIFIED: Add Name to label ---
                        name = data.get('shortName', ticker)
                        if len(name) > 25:
                            name = name[:25] + "..."
                        label = f"**{ticker}** - {name} (Score: {score:.3f})"
                        
                        is_selected = (st.session_state.selected_ticker == ticker)
                        button_type = "primary" if is_selected else "secondary"
                        
                        if st.button(label, key=f"rank_{ticker}", use_container_width=True, type=button_type):
                            st.session_state.selected_ticker = ticker
                            st.session_state.active_tab = "🔬 Ticker Deep Dive"
                            st.rerun()
        
        with rank_col2:
            st.subheader("Top 20 Overview")
            
            # --- ✅ MODIFIED: Add 'shortName' ---
            # --- ✅ MODIFIED (P3): Added 'Z_QxM' ---
            display_cols = [
                'shortName', 'Last Price', 'Sector', 
                'entry_signal',
                'Final Quant Score', 
                'Z_Value', 'Z_Momentum', 'Z_Quality', 'Z_QxM',
                'Z_Size', 'Z_LowVolatility', 'Z_Technical',
                'Risk/Reward Ratio',
                'Position Size (USD)',
                'pct_above_cutloss'
            ]
            display_cols = [c for c in display_cols if c in filtered_df.columns]
            
            filtered_df_display = filtered_df.copy()
            if 'marketCap' in filtered_df_display.columns:
                filtered_df_display['Market Cap'] = filtered_df_display['marketCap'] / 1e9
            
            st.dataframe(
                filtered_df_display.head(20)[display_cols],
                column_config={
                    "shortName": st.column_config.TextColumn("Name", width="medium"), # <-- ✅ ADDED
                    "Last Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Market Cap": st.column_config.NumberColumn(format="%.1f B", help="Market Cap in Billions"),
                    "entry_signal": st.column_config.TextColumn("SMC Signal"),
                    "Final Quant Score": st.column_config.NumberColumn(format="%.3f"),
                    "Z_Value": st.column_config.NumberColumn(format="%.2f"),
                    "Z_Momentum": st.column_config.NumberColumn(format="%.2f"),
                    "Z_Quality": st.column_config.NumberColumn(format="%.2f"),
                    "Z_QxM": st.column_config.NumberColumn(format="%.2f", help="Quality x Momentum Interaction"), # <-- ✅ ADDED
                    "Z_Size": st.column_config.NumberColumn(format="%.2f"),
                    "Z_LowVolatility": st.column_config.NumberColumn(format="%.2f"),
                    "Z_Technical": st.column_config.NumberColumn(format="%.2f"),
                    "Risk/Reward Ratio": st.column_config.NumberColumn(format="%.2f"),
                    "Position Size (USD)": st.column_config.NumberColumn(format="$%,.0f"),
                    "pct_above_cutloss": st.column_config.NumberColumn(format="%.1f%%", help="% Above Cut Loss (Swing Low)"),
                },
                use_container_width=True,
                height=700
            )

    # --- Tab 2: Ticker Deep Dive ---
    elif selected_tab == "🔬 Ticker Deep Dive":
        st.header("🔬 Ticker Deep Dive")
        
        selected_ticker = st.session_state.selected_ticker
        
        if selected_ticker is None:
            st.info("Go to the 'Quant Rankings' tab and click a ticker to see details.")
        elif filtered_df.empty:
            st.info("Go to the 'Quant Rankings' tab and click a ticker to see details.")
        elif selected_ticker not in filtered_df.index:
            try:
                ticker_data = st.session_state.raw_df.loc[selected_ticker]
                hist_data = all_histories.get(selected_ticker)
                st.warning(f"'{selected_ticker}' is not in the currently filtered list, but analysis is available.")
                display_deep_dive_details(ticker_data, hist_data, all_histories, factor_z_cols, norm_weights, filtered_df, CONFIG)
            except KeyError:
                st.error(f"Ticker '{selected_ticker}' not found in any data. Try the 'Stock Analyzer'.")
            
        else:
            ticker_data = filtered_df.loc[selected_ticker]
            hist_data = all_histories.get(selected_ticker)
            display_deep_dive_details(ticker_data, hist_data, all_histories, factor_z_cols, norm_weights, filtered_df, CONFIG)

    # --- Tab 3: Portfolio Analytics ---
    elif selected_tab == "📈 Portfolio Analytics":
        st.header("📈 Portfolio-Level Analytics")
        
        if filtered_df.empty:
            st.warning("No data to display. Adjust filters.")
        else:
            port_col1, port_col2 = st.columns(2)
            
            with port_col1:
                st.subheader("Factor Correlation Heatmap")
                st.info("This shows if factors are redundant (highly correlated). Aim for low values.")
                
                # --- ✅ MODIFIED (P3): 'factor_z_cols' now dynamically includes 'Z_QxM'
                corr_matrix = filtered_df[factor_z_cols].corr()
                corr_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, zmax=1,
                    title="Factor Z-Score Correlation Matrix"
                )
                st.plotly_chart(corr_heatmap, use_container_width=True)
                
            with port_col2:
                st.subheader("Sector Median Factor Strength")
                st.info("This shows which factors are strongest/weakest for each sector.")
                
                # --- ✅ MODIFIED (P3): 'factor_z_cols' now dynamically includes 'Z_QxM'
                sector_median_factors = filtered_df.groupby('Sector')[factor_z_cols].median()
                sector_heatmap = px.imshow(
                    sector_median_factors,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    title="Median Factor Z-Score by Sector"
                )
                st.plotly_chart(sector_heatmap, use_container_width=True)
    
    # --- ✅ NEW (USER REQ): Tab 4: My Portfolio (Transaction Log) ---
    elif selected_tab == "💼 My Portfolio":
        display_portfolio_tab_v2(df, all_histories, factor_z_cols, CONFIG)
            
# --- ⭐️ NEW HELPER FUNCTION ⭐️ ---
def display_deep_dive_details(ticker_data, hist_data, all_histories, factor_z_cols, norm_weights, filtered_df, CONFIG):
    """
    Helper function to display the full Ticker Deep Dive page.
    --- ✅ MODIFIED (USER REQ): AI Summary is now on-demand ---
    """
    selected_ticker = ticker_data.name
    
    # --- ✅ MODIFIED: Add Name to Subheader ---
    st.subheader(f"Analysis for: {selected_ticker} - {ticker_data.get('shortName', '')}")

    # --- ✅ NEW (USER REQ): Add to Portfolio Button & Fix Navigation Bug ---
    add_col1, add_col2, add_col3 = st.columns([2,2,1])
    
    # Add Previous/Next Buttons
    try:
        ticker_list = filtered_df.index.tolist()
        current_index = ticker_list.index(selected_ticker)
        
        is_first = (current_index == 0)
        # --- ✅ BUG FIX: Added dynamic key ---
        if add_col1.button("⬅️ Previous", use_container_width=True, disabled=is_first, key=f"prev_ticker_{selected_ticker}"):
            st.session_state.selected_ticker = ticker_list[current_index - 1]
            st.session_state.active_tab = "🔬 Ticker Deep Dive" # <-- FIX: Explicitly keep tab
            st.rerun()
            
        is_last = (current_index == len(ticker_list) - 1)
        # --- ✅ BUG FIX: Added dynamic key ---
        if add_col2.button("Next ➡️", use_container_width=True, disabled=is_last, key=f"next_ticker_{selected_ticker}"):
            st.session_state.selected_ticker = ticker_list[current_index + 1]
            st.session_state.active_tab = "🔬 Ticker Deep Dive" # <-- FIX: Explicitly keep tab
            st.rerun()

    except ValueError:
        st.info("Previous/Next navigation is only available for stocks in the filtered list.")

    if add_col3.button("➕ Add to Portfolio", use_container_width=True, type="primary"):
        st.session_state.prefill_transaction = {
            "ticker": selected_ticker,
            "price": ticker_data.get('last_price', 0.0)
        }
        st.session_state.active_tab = "💼 My Portfolio"
        st.rerun()
    # --- END OF NEW BLOCK ---

    display_buy_signal_checklist(ticker_data)
    st.divider()

    if pd.notna(ticker_data.get('data_warning')):
        st.warning(f"⚠️ **Data Warning:** {ticker_data['data_warning']}")
    
    st.markdown(f"**Sector:** {ticker_data['Sector']} | **Data Source:** `{ticker_data['source']}`")
    
    kpi_cols = st.columns(6) 
    
    entry_signal = ticker_data.get('entry_signal', 'No Trade')
    delta_text = "-"
    delta_color = "off"
    if entry_signal == 'Buy near Bullish OB':
        delta_text = "BUY"
        delta_color = "normal" 
    elif entry_signal == 'Sell near Bearish OB':
        delta_text = "SELL"
        delta_color = "inverse"
        
    kpi_cols[0].metric("SMC Entry Signal", entry_signal, delta=delta_text, delta_color=delta_color)

    # --- ✅ NEW (Phase 2): Add help text if z-score fallback was used ---
    fallback_help = None
    if ticker_data.get('z_score_fallback', False):
        fallback_help = "Note: Z-Scores for this stock use global medians, as its sector is too small for peer comparison."
    
    kpi_cols[1].metric("Final Quant Score", f"{ticker_data['Final Quant Score']:.3f}", help=fallback_help)
    # --- End of change ---

    kpi_cols[2].metric("Last Price", f"${ticker_data['last_price']:.2f}")
    kpi_cols[3].metric("Market Cap", f"${ticker_data['marketCap']/1e9:.1f} B")
    kpi_cols[4].metric("Trend (50/200 MA)", ticker_data['Trend (50/200 Day MA)'])
    
    last_div_val = ticker_data.get('last_dividend_value', np.nan)
    last_div_date = ticker_data.get('last_dividend_date', 'N/A')
    div_display = f"${last_div_val:.2f}" if pd.notna(last_div_val) else "N/A"
    div_help = f"Paid on: {last_div_date}"
    kpi_cols[5].metric("Last Dividend", div_display, help=div_help)
    
    st.divider()

    # --- ✅ MODIFIED (USER REQ): On-Demand AI Summary with RTL ---
    st.subheader("🤖 AI-Powered Deep Dive")
    cache_key = f"ai_summary_{selected_ticker}_deep_dive" # Unique key for deep dive
    
    # Check if we already generated this summary
    if cache_key in st.session_state:
        # Wrap in a div with RTL class
        st.markdown(f'<div class="rtl-container">{st.session_state[cache_key]}</div>', unsafe_allow_html=True)
    else:
        # If not, show the button
        # --- ✅ BUG FIX: Added dynamic key ---
        if st.button(f"🤖 Click to Generate AI Summary for {selected_ticker}", type="secondary", key=f"gen_ai_deep_dive_{selected_ticker}"):
            with st.spinner(f"Generating AI analysis for {selected_ticker}... This may take a moment."):
                try:
                    # Get the data needed for the prompt
                    company_name = ticker_data.get('shortName', selected_ticker)
                    news_headlines = ticker_data.get('news_list', 'No recent news found.')
                    
                    # Call the (new) AI function ON-DEMAND
                    summary = get_ai_stock_analysis(
                        ticker_symbol=selected_ticker,
                        company_name=company_name,
                        news_headlines_str=news_headlines,
                        parsed_data=ticker_data, # Pass the whole row
                        CONFIG=CONFIG, # Pass the config for the API key
                        analysis_type="deep_dive" # New argument for prompt tailoring
                    )
                    st.session_state[cache_key] = summary
                    st.rerun() # Rerun to display the summary
                except Exception as e:
                    st.error(f"Failed to generate AI summary: {e}")
                    st.session_state[cache_key] = "AI Summary generation failed."
    # --- END OF MODIFICATION ---
    
    # --- Raw News Headlines ---
    st.subheader("Latest News")
    
    # --- ✅ NEW (Phase 3): Display AI News Sentiment Score ---
    news_sentiment = ticker_data.get('news_sentiment_score', 0.0)
    if news_sentiment > 0.3:
        sentiment_text = "Positive"
        delta_color = "normal"
    elif news_sentiment < -0.3:
        sentiment_text = "Negative"
        delta_color = "inverse"
    else:
        sentiment_text = "Neutral"
        delta_color = "off"
        
    st.metric("AI News Sentiment", f"{sentiment_text} ({news_sentiment:.2f})", delta=f"{news_sentiment:.2f}", delta_color=delta_color)
    # --- End of change ---

    news_list_str = ticker_data.get('news_list', 'N/A')
    
    if news_list_str == "N/A" or not news_list_str:
        st.info("No raw news headlines found.")
    else:
        with st.expander("View Raw Headlines", expanded=False):
            news_list = news_list_str.split(", ")
            for i, headline in enumerate(news_list):
                st.markdown(f"- {headline}")
    st.divider() 
    
    chart_col1, chart_col2 = st.columns([2, 1])
    with chart_col1:
        st.subheader("Price Chart & Technicals")
        if hist_data is not None:
            # --- ✅ MODIFIED: Pass CONFIG ---
            price_chart = create_price_chart(hist_data, ticker_data, CONFIG)
            st.plotly_chart(price_chart, use_container_width=True)
        else:
            st.error("Historical data not found for this ticker.")
            
    with chart_col2:
        st.subheader("Factor Profile")
        # --- ✅ MODIFIED (P3): 'factor_z_cols' now dynamically includes 'Z_QxM'
        radar_chart = create_radar_chart(ticker_data, factor_z_cols)
        st.plotly_chart(radar_chart, use_container_width=True)
        
        with st.expander("Factor Contribution Breakdown", expanded=False):
            # --- ✅ MODIFIED (P3): 'norm_weights' now dynamically includes 'QxM'
            for factor in norm_weights.keys():
                z_col = f"Z_{factor}"
                w_z_col = f"Weighted_{z_col}"
                st.metric(
                    label=f"{factor} (Z-Score: {ticker_data[z_col]:.2f})",
                    value=f"Contrib: {ticker_data[w_z_col]:.3f}",
                    help=f"Weight: {norm_weights[factor]*100:.1f}%"
                )

    st.divider()
    
    st.subheader("Risk & Position Sizing")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    with risk_col1:
        sl_atr = ticker_data.get('Stop Loss (ATR)', np.nan)
        sl_cut = ticker_data.get('Stop Loss (Cut Loss)', np.nan)
        sl_final = ticker_data.get('Final Stop Loss', np.nan)
        sl_method = ticker_data.get('SL_Method', 'N/A')
        risk_pct = ticker_data.get('Risk % (to Stop)', np.nan)
        risk_display = f"Risk %: {risk_pct:.1f}%" if pd.notna(risk_pct) else "N/A"

        st.metric("Stop Loss (ATR)", f"${sl_atr:.2f}" if pd.notna(sl_atr) else "N/A")
        st.metric("Stop Loss (Cut Loss)", f"${sl_cut:.2f}" if pd.notna(sl_cut) else "N/A", help="Based on last swing low")
        st.metric(f"Final Stop ({sl_method})", f"${sl_final:.2f}" if pd.notna(sl_final) else "N/A", help=risk_display)

    with risk_col2:
        # --- ✅ MODIFIED (P2): This value is now the (better) SMC-aware target
        tp_price = ticker_data.get('Take Profit Price', np.nan)
        rr_ratio = ticker_data.get('Risk/Reward Ratio', np.nan)
        tp_display = f"${tp_price:.2f}" if pd.notna(tp_price) else "N/A"
        rr_display = f"{rr_ratio:.2f}" if pd.notna(rr_ratio) else "N/A"

        st.metric("Take Profit (SMC/Fib)", tp_display, help="Tighter of (Bearish OB) or (ATR/Fib)")
        st.metric("Risk/Reward Ratio", rr_display)

    with risk_col3:
        pos_shares = ticker_data.get('Position Size (Shares)', np.nan)
        pos_display = f"{pos_shares:.0f} Shares" if pd.notna(pos_shares) else "N/A"
        risk_usd = ticker_data.get('Risk Per Trade (USD)', 50)
        st.metric("Position Size (Shares)", pos_display, help=f"Based on ${risk_usd:,.0f} risk")
    
    with risk_col4:
        pos_usd = ticker_data.get('Position Size (USD)', np.nan)
        pos_usd_display = f"${pos_usd:,.0f}" if pd.notna(pos_usd) else "N/A"
        st.metric("Position Size (USD)", pos_usd_display, help="Shares * Last Price")

    st.divider() 
    
    st.subheader("Key Price Zones")
    zone_cols = st.columns(4)
    
    b_ob_low = ticker_data.get('bullish_ob_low', np.nan)
    b_ob_high = ticker_data.get('bullish_ob_high', np.nan)
    b_ob_validated = ticker_data.get('bullish_ob_validated', False)
    b_ob_fvg = ticker_data.get('bullish_ob_fvg', False)
    b_ob_vol = ticker_data.get('bullish_ob_volume_ok', False)
    
    b_ob_label = f"{'✅ Mitigated' if b_ob_validated else 'Fresh'} Bullish OB"
    b_ob_display = f"${b_ob_low:.2f} - ${b_ob_high:.2f}" if pd.notna(b_ob_low) else "N/A"
    b_ob_help = f"FVG: {'Yes' if b_ob_fvg else 'No'} | BOS Vol: {'High' if b_ob_vol else 'Low'}"
    zone_cols[0].metric(b_ob_label, b_ob_display, help=b_ob_help)
    
    be_ob_low = ticker_data.get('bearish_ob_low', np.nan)
    be_ob_high = ticker_data.get('bearish_ob_high', np.nan)
    be_ob_validated = ticker_data.get('bearish_ob_validated', False)
    be_ob_fvg = ticker_data.get('bearish_ob_fvg', False)
    be_ob_vol = ticker_data.get('bearish_ob_volume_ok', False)
    
    be_ob_label = f"{'✅ Mitigated' if be_ob_validated else 'Fresh'} Bearish OB"
    be_ob_display = f"${be_ob_high:.2f} - ${be_ob_low:.2f}" if pd.notna(be_ob_low) else "N/A"
    be_ob_help = f"FVG: {'Yes' if b_ob_fvg else 'No'} | BOS Vol: {'High' if b_ob_vol else 'Low'}"
    zone_cols[1].metric(be_ob_label, be_ob_display, help=be_ob_help)
    
    support = ticker_data.get('last_swing_low', np.nan)
    support_display = f"${support:.2f}" if pd.notna(support) else "N/A"
    zone_cols[2].metric("Last Swing Low", support_display)

    resistance = ticker_data.get('last_swing_high', np.nan)
    resistance_display = f"${resistance:.2f}" if pd.notna(resistance) else "N/A"
    zone_cols[3].metric("Last Swing High", resistance_display)
    
    st.divider() 
    
    st.subheader("Valuation & Key Dates")
    val_col1, val_col2, val_col3 = st.columns(3)
    
    val_col1.metric("Valuation (Graham)", ticker_data['grahamValuation'])
    
    next_earnings = ticker_data.get('next_earnings_date', 'N/A')
    val_col2.metric("Next Earnings Date", next_earnings)
    
    next_dividend = ticker_data.get('next_ex_dividend_date', 'N/A')
    val_col3.metric("Next Ex-Dividend Date", next_dividend, help="تاريخ الأحقية القادم لتوزيع الأرباح")
    
    with st.expander("View All Raw Data for " + selected_ticker):
        st.dataframe(ticker_data.astype(str))
# --- ⭐️ END OF DEEP DIVE HELPER ⭐️ ---


# --- ✅ NEW (USER REQ): Portfolio v2 (Transaction Log) ---
def process_transactions(transactions, all_data_df):
# ... (unchanged)
    """
    Processes a list of transactions to calculate open positions and realized P/L using FIFO.
    """
    open_positions = {}
    realized_pl = 0.0
    
    # Sort transactions by date to ensure correct FIFO processing
    transactions.sort(key=lambda x: x['date'])
    
    # Use a dictionary to hold a deque (FIFO queue) for each ticker's buy lots
    buy_lots = {} 

    for tx in transactions:
        ticker = tx['ticker']
        
        if tx['type'] == 'Buy':
            # Add new buy lot to the queue
            if ticker not in buy_lots:
                buy_lots[ticker] = deque()
            buy_lots[ticker].append({'shares': tx['shares'], 'price': tx['price']})
            
        elif tx['type'] == 'Sell':
            shares_to_sell = tx['shares']
            sell_price = tx['price']
            
            if ticker not in buy_lots or not buy_lots[ticker]:
                st.error(f"Error: Trying to sell {ticker} but no buy lots found.")
                continue
                
            while shares_to_sell > 0:
                if not buy_lots[ticker]:
                    st.error(f"Error: Sold more {ticker} than owned.")
                    break
                    
                first_lot = buy_lots[ticker][0]
                
                if first_lot['shares'] > shares_to_sell:
                    # Sell part of the first lot
                    realized_pl += (sell_price - first_lot['price']) * shares_to_sell
                    first_lot['shares'] -= shares_to_sell
                    shares_to_sell = 0
                else:
                    # Sell the entire first lot (or what's left)
                    realized_pl += (sell_price - first_lot['price']) * first_lot['shares']
                    shares_to_sell -= first_lot['shares']
                    buy_lots[ticker].popleft() # Remove the empty lot

    # --- Calculate Final Open Positions ---
    open_positions_list = []
    for ticker, lots in buy_lots.items():
        if not lots:
            continue
            
        total_shares = sum(lot['shares'] for lot in lots)
        total_cost = sum(lot['shares'] * lot['price'] for lot in lots)
        
        if total_shares > 0:
            avg_cost = total_cost / total_shares
            
            # Get current data from raw_df
            if ticker in all_data_df.index:
                ticker_data = all_data_df.loc[ticker]
                current_price = ticker_data['last_price']
                name = ticker_data.get('shortName', ticker)
                sector = ticker_data.get('Sector', 'Unknown')
                entry_signal = ticker_data.get('entry_signal', 'N/A') # <-- ✅ ADDED
                # Get factor scores
                factor_scores = {col: ticker_data.get(col, 0) for col in all_data_df.columns if col.startswith('Z_')}
            else:
                current_price = np.nan
                name = ticker
                sector = "Unknown"
                entry_signal = "N/A" # <-- ✅ ADDED
                factor_scores = {}
            
            market_value = total_shares * current_price
            unrealized_pl = market_value - total_cost
            
            pos_dict = {
                'Ticker': ticker,
                'Name': name,
                'Sector': sector,
                'entry_signal': entry_signal, # <-- ✅ ADDED
                'Shares': total_shares,
                'Avg Cost': avg_cost,
                'Total Cost': total_cost,
                'Current Price': current_price,
                'Market Value': market_value,
                'Unrealized P/L': unrealized_pl
            }
            pos_dict.update(factor_scores) # Add all Z_scores
            open_positions_list.append(pos_dict)

    open_positions_df = pd.DataFrame(open_positions_list)
    if not open_positions_df.empty:
        open_positions_df = open_positions_df.set_index('Ticker')
        
    return open_positions_df, realized_pl


def display_portfolio_tab_v2(all_data_df, all_histories, factor_z_cols, CONFIG):
    
    st.header("💼 My Portfolio (Transaction-Based)")
    
    # --- 1. Process Data ---
    transactions = st.session_state.get('transactions', [])
    open_positions, realized_pl = process_transactions(transactions, all_data_df)
    
    # --- 2. Portfolio KPIs ---
    st.subheader("Portfolio Summary")
    
    cash = st.session_state.get('cash', 0.0)
    market_value = open_positions['Market Value'].sum() if not open_positions.empty else 0.0
    total_portfolio_value = cash + market_value
    unrealized_pl = open_positions['Unrealized P/L'].sum() if not open_positions.empty else 0.0
    
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}", help="Cash + Market Value of Stocks")
    kpi_cols[1].metric("Cash", f"${cash:,.2f}")
    kpi_cols[2].metric("Unrealized P/L", f"${unrealized_pl:,.2f}", delta_color="normal" if unrealized_pl >= 0 else "inverse")
    kpi_cols[3].metric("Realized P/L (FIFO)", f"${realized_pl:,.2f}", delta_color="normal" if realized_pl >= 0 else "inverse")
    
    # --- ✅ NEW: AI Portfolio Assessment ---
    st.divider()
    st.subheader("🤖 AI Portfolio Assessment")
    
    cache_key = "ai_summary_portfolio"
    
    if cache_key in st.session_state:
        st.markdown(f'<div class="rtl-container">{st.session_state[cache_key]}</div>', unsafe_allow_html=True)
        if st.button("🔄 Regenerate Assessment", key="regen_ai_portfolio"):
            del st.session_state[cache_key]
            st.rerun()
    else:
        if st.button("🤖 Click to Generate AI Portfolio Assessment", type="secondary", key="gen_ai_portfolio"):
            with st.spinner("Analyzing your portfolio... This may take a moment."):
                
                # 1. Prepare data for the AI
                portfolio_metrics = {
                    "Total Value": f"${total_portfolio_value:,.2f}",
                    "Cash": f"${cash:,.2f}",
                    "Stock Market Value": f"${market_value:,.2f}",
                    "Unrealized P/L": f"${unrealized_pl:,.2f}",
                    "Realized P/L": f"${realized_pl:,.2f}",
                    "Number of Holdings": len(open_positions)
                }
                
                if not open_positions.empty:
                    # Calculate Sector Allocation
                    sector_alloc = open_positions.groupby('Sector')['Market Value'].sum()
                    sector_alloc_pct = (sector_alloc / market_value * 100).round(1).to_dict()
                    portfolio_metrics["Sector Allocation (%)"] = sector_alloc_pct
                    
                    # Calculate Weighted Factor Exposure
                    open_positions['Weight'] = open_positions['Market Value'] / market_value
                    weighted_factors = {}
                    for col in factor_z_cols:
                        if col in open_positions.columns:
                            weighted_score = (open_positions[col] * open_positions['Weight']).sum()
                            weighted_factors[col.replace('Z_', '')] = f"{weighted_score:.2f}"
                    portfolio_metrics["Weighted Factor Exposure"] = weighted_factors
                    
                    # Get Top 5 Holdings
                    top_5 = open_positions.sort_values(by='Market Value', ascending=False).head(5)
                    top_5_pct = (top_5['Market Value'].sum() / market_value) * 100
                    portfolio_metrics["Top 5 Holdings %"] = f"{top_5_pct:.1f}%"
                    portfolio_metrics["Top 5 Holdings"] = top_5[['Name', 'Weight']].to_dict('index')

                    # --- ✅ NEW: Add detailed holdings data for the AI ---
                    # Join with all_data_df to get the Final Quant Score
                    detailed_holdings_df = open_positions.join(all_data_df[['Final Quant Score']])
                    
                    # Select key columns for the AI to analyze
                    ai_columns = [
                        'Name', 'Sector', 'entry_signal', 'Unrealized P/L', 'Weight',
                        'Final Quant Score', 'Z_Value', 'Z_Momentum', 'Z_Quality'
                    ]
                    # Filter columns that actually exist
                    existing_ai_columns = [col for col in ai_columns if col in detailed_holdings_df.columns]
                    
                    # Convert to a list of dicts for clean JSON
                    portfolio_metrics["Holdings Details"] = detailed_holdings_df[existing_ai_columns].to_dict('records')
                    # --- END OF NEW BLOCK ---

                # 2. Call the new AI function from spus.py
                try:
                    summary = get_ai_portfolio_summary(portfolio_metrics, CONFIG)
                    st.session_state[cache_key] = summary
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate AI portfolio summary: {e}")
                    st.session_state[cache_key] = "AI Summary generation failed."
    
    # --- END OF NEW SECTION ---
    
    st.divider()

    # --- 3. Portfolio Management (Add/Load/Save) ---
    st.subheader("Portfolio Management")
    
    # --- Cash Management ---
    new_cash = st.number_input(
        "Set Your Cash Balance:", 
        min_value=0.0, 
        value=float(st.session_state.cash),  # <-- ✅ BUG FIX: Cast value to float
        step=1000.0, 
        format="%.2f",
        key="cash_input"
    )
    if new_cash != st.session_state.cash:
        st.session_state.cash = new_cash
        st.rerun()

    # --- Add New Transaction ---
    with st.expander("Add New Transaction"):
        # Check for pre-fill from Deep Dive tab
        prefill = st.session_state.prefill_transaction
        default_ticker = prefill['ticker'] if prefill else ""
        default_price = prefill['price'] if prefill else None 
        
        with st.form("add_transaction_form"):
            form_col1, form_col2, form_col3, form_col4, form_col5 = st.columns(5)
            
            tx_date = form_col1.date_input("Date", value=datetime.now(SAUDI_TZ).date())
            tx_ticker = form_col2.text_input("Ticker Symbol", value=default_ticker).upper().strip()
            tx_type = form_col3.selectbox("Type", ["Buy", "Sell"])
            tx_shares = form_col4.number_input("Shares", min_value=0.0, step=1.0)
            tx_price = form_col5.number_input("Price", min_value=0.01, value=default_price, format="%.2f", placeholder="0.00")
            tx_notes = st.text_input("Notes (e.g., 'SMC Entry', 'Averaging down')")
            
            submitted = st.form_submit_button("Add Transaction")
            
            if submitted:
                # --- ✅ BUG FIX 2: Check for None or 0 ---
                if not tx_ticker or tx_shares == 0 or (tx_price is None or tx_price <= 0):
                    st.error("Please fill out all fields (Ticker, Shares, Price).")
                elif tx_ticker not in all_data_df.index:
                    st.error(f"Ticker '{tx_ticker}' not found. Run a 'Deep Dive' for it first from the sidebar.")
                else:
                    tx_cost = tx_shares * tx_price
                    if tx_type == 'Buy' and st.session_state.cash < tx_cost:
                        st.error(f"Not enough cash! Transaction cost is ${tx_cost:,.2f}, but you only have ${st.session_state.cash:,.2f}.")
                    else:
                        new_tx = {
                            "id": f"{tx_date}_{tx_ticker}_{time.time()}", # Unique ID
                            "date": str(tx_date),
                            "ticker": tx_ticker,
                            "type": tx_type,
                            "shares": tx_shares,
                            "price": tx_price,
                            "notes": tx_notes
                        }
                        st.session_state.transactions.append(new_tx)
                        
                        # Adjust cash
                        if tx_type == 'Buy':
                            st.session_state.cash -= tx_cost
                        else: # Sell
                            st.session_state.cash += tx_cost
                            
                        st.success(f"Added {tx_type} of {tx_shares} shares of {tx_ticker}!")
                        
                        # Clear prefill
                        if st.session_state.prefill_transaction:
                            st.session_state.prefill_transaction = None
                        st.rerun()

    # --- 4. Load/Save Transactions ---
    file_col1, file_col2 = st.columns([1, 3])
    
    with file_col1:
        # Save both transactions and cash
        portfolio_data = {
            "cash": st.session_state.cash,
            "transactions": st.session_state.transactions
        }
        portfolio_json = json.dumps(portfolio_data, indent=4)
        
        st.download_button(
            label="💾 Save Portfolio (JSON)",
            data=portfolio_json,
            file_name="my_portfolio_transactions.json",
            mime="application/json",
            use_container_width=True
        )

    with file_col2:
        uploaded_file = st.file_uploader("📂 Load Portfolio (JSON)", type="json")
        if uploaded_file is not None:
            try:
                # --- ✅ BUG FIX: Read file content as string, then use json.loads ---
                file_content_bytes = uploaded_file.getvalue()
                file_content_string = file_content_bytes.decode("utf-8")
                new_portfolio_data = json.loads(file_content_string)
                # --- End of Fix ---

                if isinstance(new_portfolio_data, dict) and "transactions" in new_portfolio_data and "cash" in new_portfolio_data:
                    st.session_state.transactions = new_portfolio_data["transactions"]
                    st.session_state.cash = new_portfolio_data["cash"]
                    st.success(f"Successfully loaded {len(st.session_state.transactions)} transactions and cash balance!")
                    st.rerun()
                else:
                    st.error("Invalid portfolio file format. Expected a JSON with 'cash' and 'transactions' keys.")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                
    st.divider()
    
    if open_positions.empty and not transactions:
        st.info("Your portfolio is empty. Add a new transaction above.")
        return

    # --- 5. Open Positions & Analytics ---
    st.subheader("Open Positions")
    if open_positions.empty:
        st.info("No open positions.")
    else:
        st.dataframe(open_positions, use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "entry_signal": st.column_config.TextColumn("SMC Signal"), # <-- ✅ ADDED
                "Avg Cost": st.column_config.NumberColumn(format="$%.2f"),
                "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                "Total Cost": st.column_config.NumberColumn(format="$%.2f"),
                "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
            }
        )
    
    st.divider()
    st.subheader("Portfolio Analytics")

    if not open_positions.empty:
        # --- 6. Allocation & Factor Analysis (USER REQ) ---
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.subheader("Performance Attribution (Unrealized P/L)")
            # Add P/L %
            open_positions['P/L (%)'] = (open_positions['Unrealized P/L'] / open_positions['Total Cost']) * 100
            
            pl_by_ticker = open_positions[['Unrealized P/L', 'Name']].sort_values(by="Unrealized P/L")
            fig = px.bar(
                pl_by_ticker, 
                x='Unrealized P/L', 
                y='Name', 
                orientation='h', 
                title="P/L Contribution by Stock",
                color='Unrealized P/L',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

        with analysis_col2:
            st.subheader("Sector Allocation")
            sector_alloc = open_positions.groupby('Sector')['Market Value'].sum().reset_index()
            fig = px.pie(
                sector_alloc, 
                names='Sector', 
                values='Market Value', 
                title="Allocation by Sector",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- 7. Factor & Concentration Analysis (USER REQ) ---
        st.subheader("Factor & Concentration Risk")
        factor_col1, factor_col2 = st.columns(2)

        with factor_col1:
            st.markdown("**Weighted Factor Exposure**")
            open_positions['Weight'] = open_positions['Market Value'] / market_value
            
            weighted_factors = []
            for col in factor_z_cols: # Use the list from main app
                if col in open_positions.columns:
                    weighted_score = (open_positions[col] * open_positions['Weight']).sum()
                    weighted_factors.append({'Factor': col.replace('Z_', ''), 'Weighted Z-Score': weighted_score})
            
            if weighted_factors:
                factors_df = pd.DataFrame(weighted_factors)
                fig = px.bar(
                    factors_df, 
                    x='Factor', 
                    y='Weighted Z-Score', 
                    title="Portfolio Weighted Factor Z-Scores",
                    color='Weighted Z-Score',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)

        with factor_col2:
            st.markdown("**Concentration Risk**")
            top_5 = open_positions.sort_values(by='Market Value', ascending=False).head(5)
            top_5_pct = (top_5['Market Value'].sum() / market_value) * 100
            st.metric(f"Top 5 Holdings %", f"{top_5_pct:.1f}%")
            
            st.dataframe(top_5[['Name', 'Market Value', 'Weight']],
                column_config={
                    "Weight": st.column_config.ProgressColumn(
                        "Weight",
                        format="%.1f%%",
                        min_value=0,
                        max_value=max(50.0, top_5['Weight'].max() * 100), # Dynamic max
                    ),
                    "Market Value": st.column_config.NumberColumn(format="$%,.0f")
                }
            )

        # --- 8. Portfolio Correlation Matrix (USER REQ) ---
        st.subheader("Portfolio Correlation Matrix")
        open_tickers = open_positions.index.tolist()
        if len(open_tickers) > 1:
            hist_list = []
            for ticker in open_tickers:
                if ticker in all_histories:
                    hist_list.append(all_histories[ticker]['Close'].rename(ticker))
            
            if hist_list:
                combined_hist = pd.concat(hist_list, axis=1).fillna(method='ffill')
                returns_corr = combined_hist.pct_change().corr()
                
                fig = px.imshow(
                    returns_corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, zmax=1,
                    title="Stock Returns Correlation"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not fetch historical data for correlation.")
        else:
            st.info("You need at least two open positions to calculate correlation.")


    # --- 9. Position Analyzer ---
    st.divider()
    st.subheader("Position Analyzer")
    
    if not open_positions.empty:
        # --- ✅ BUG FIX: Removed static key ---
        selected_ticker_port = st.selectbox("Select a position to analyze:", options=open_positions.index)
        
        if selected_ticker_port:
            position_data = open_positions.loc[selected_ticker_port]
            ticker_data = all_data_df.loc[selected_ticker_port]
            
            if st.button(f"🔬 Go to Deep Dive for {selected_ticker_port}", key=f"deep_dive_port_{selected_ticker_port}"):
                st.session_state.selected_ticker = selected_ticker_port
                st.session_state.active_tab = "🔬 Ticker Deep Dive"
                st.rerun()

            display_position_analysis_v2(position_data, ticker_data, CONFIG)
    
    # --- 10. Transaction Log ---
    st.divider()
    st.subheader("Transaction Log")
    if not transactions:
        st.info("No transactions recorded.")
    else:
        tx_df = pd.DataFrame(transactions)
        # Add a column for deleting
        tx_df['Delete'] = False
        
        with st.form("edit_transactions_form"):
            edited_df = st.data_editor(
                tx_df,
                column_config={
                    "id": None, # Hide the ID column
                    "Delete": st.column_config.CheckboxColumn("Delete?", default=False)
                },
                use_container_width=True,
                num_rows="dynamic",
                key="transaction_editor"
            )
            
            if st.form_submit_button("Save Changes"):
                # Filter out rows marked for deletion
                new_transactions = edited_df[edited_df['Delete'] == False].to_dict('records')
                # Remove the 'Delete' key before saving
                for tx in new_transactions:
                    del tx['Delete']
                st.session_state.transactions = new_transactions
                st.success("Transactions updated!")
                st.rerun()

# --- ✅ NEW (USER REQ): Position Analysis v2 (with Trailing Stop) ---
def display_position_analysis_v2(position_data, ticker_data, CONFIG):
# ... (unchanged)
    
    selected_ticker = position_data.name # Get ticker from position data index
    
    pa_col1, pa_col2, pa_col3 = st.columns(3)
    
    # --- Data Prep ---
    cost_basis = position_data['Avg Cost']
    current_price = position_data['Current Price']
    
    # Get original trade setup from quant data
    initial_stop_loss = ticker_data.get('Final Stop Loss', np.nan)
    initial_target = ticker_data.get('Take Profit Price', np.nan)
    initial_risk_per_share = cost_basis - initial_stop_loss if pd.notna(initial_stop_loss) else np.nan
    
    current_rr = np.nan
    if pd.notna(initial_stop_loss) and pd.notna(initial_target):
        current_risk = current_price - initial_stop_loss
        current_reward = initial_target - current_price
        if current_risk > 0:
            current_rr = current_reward / current_risk

    with pa_col1:
        st.subheader("Current P/L & R/R")
        st.metric("Your Avg Cost Basis", f"${cost_basis:,.2f}")
        st.metric("Current Price", f"${current_price:,.2f}", delta=f"${current_price - cost_basis:,.2f}")
        
        if pd.notna(current_rr):
            st.metric("Current R/R Ratio", f"{current_rr:.2f} R", help="R/R based on current price and *original* SL/TP.")
            if current_rr < 1.0:
                st.warning("Warning: Current R/R is unfavorable (< 1.0).")
        
    with pa_col2:
        st.subheader("Trailing Stop Logic (Suggestion)")
        
        if pd.notna(initial_risk_per_share) and initial_risk_per_share > 0:
            one_r_target = cost_basis + initial_risk_per_share
            st.metric("1R Profit Target", f"${one_r_target:,.2f}", help="Your cost basis + initial risk per share.")
            
            if current_price >= one_r_target:
                # --- Trailing Stop Logic ---
                atr = ticker_data.get('ATR', np.nan)
                k_atr = 2.0 # Multiplier (can be from config)
                
                if pd.notna(atr):
                    suggested_trail_stop = current_price - (atr * k_atr)
                    # Stop should not go down. Max of (new stop) or (cost basis)
                    final_trail_stop = max(cost_basis, suggested_trail_stop)
                    
                    st.success("✅ Price is above 1R target.")
                    st.metric("Suggested Trailing Stop", f"${final_trail_stop:,.2f}", help=f"Max of (Cost Basis) or (Current Price - {k_atr} * ATR)")
                else:
                    st.metric("Suggested Trailing Stop", f"${cost_basis:,.2f}", help="Move stop to Breakeven (ATR data missing).")
            else:
                st.info("Price has not hit +1R target. Maintain original stop.")
                st.metric("Original Stop Loss", f"${initial_stop_loss:,.2f}")
        else:
            st.warning("Cannot calculate 1R target (missing initial risk data).")

    with pa_col3:
        st.subheader("Where to Add? (Averaging)")
        
        b_ob_low = ticker_data.get('bullish_ob_low', np.nan)
        b_ob_high = ticker_data.get('bullish_ob_high', np.nan)

        if pd.notna(b_ob_low):
            st.metric("Bullish OB (Demand Zone)", f"${b_ob_low:,.2f} - ${b_ob_high:,.2f}")
            if cost_basis > b_ob_high:
                st.info("The current Demand Zone is *below* your cost basis. This could be a good area to average down if the signal is confirmed.")
            else:
                st.success("Your cost basis is already at or below the current Demand Zone. This is a strong position.")
        else:
            st.warning("No clear Demand Zone (Bullish OB) found.")

    st.divider()
    
    # --- ✅ NEW: On-Demand AI Summary for Position ---
    st.subheader("🤖 Position-Specific AI Assessment")
    cache_key = f"ai_summary_{selected_ticker}_position" 
    
    # Check if we already generated this summary
    if cache_key in st.session_state:
        # Wrap in a div with RTL class
        st.markdown(f'<div class="rtl-container">{st.session_state[cache_key]}</div>', unsafe_allow_html=True)
    else:
        # If not, show the button
        # --- ✅ BUG FIX: Added dynamic key ---
        if st.button(f"🤖 Click to Generate Position Assessment for {selected_ticker}", type="secondary", key=f"gen_ai_position_{selected_ticker}"):
            with st.spinner(f"Generating AI position assessment for {selected_ticker}... This may take a moment."):
                try:
                    # Prepare extra position-specific data for the prompt
                    position_details = {
                        "Shares": position_data['Shares'],
                        "Average Cost": f"${cost_basis:,.2f}",
                        "Current Price": f"${current_price:,.2f}",
                        "Unrealized P/L": f"${position_data['Unrealized P/L']:,.2f}",
                        "P/L Percent": f"{(position_data['Unrealized P/L'] / position_data['Total Cost']) * 100:.2f}%",
                    }
                    
                    summary = get_ai_stock_analysis(
                        ticker_symbol=selected_ticker,
                        company_name=ticker_data.get('shortName', selected_ticker),
                        news_headlines_str=ticker_data.get('news_list', 'No recent news found.'),
                        parsed_data=ticker_data,
                        CONFIG=CONFIG,
                        analysis_type="position_assessment", # New argument
                        position_data=position_details # New argument
                    )
                    st.session_state[cache_key] = summary
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate AI position assessment: {e}")
                    st.session_state[cache_key] = "AI Assessment generation failed."
    # --- END OF NEW BLOCK ---


# --- ⭐️ 6. Scheduler Entry Point ---

def run_analysis_for_scheduler():
# ... (unchanged)
    """
    Function to be called by an external scheduler (e.g., cron).
    --- ✅ MODIFIED: Accepts config file as argument ---
    """
    print("--- [SPUS SCHEDULER] ---")
    
    if len(sys.argv) < 3:
        print("FATAL: Missing config file argument.")
        print("Usage: python streamlit_app.py --run-scheduler config_file.json")
        return
        
    config_file_name = sys.argv[2]
    print(f"Starting scheduled analysis for {config_file_name} at {datetime.now(SAUDI_TZ)}...")
    
    def print_progress_callback(percent, text):
        print(f"[{percent*100:.0f}%] {text}")
    
    # --- ✅ MODIFIED (P5): Inject secrets for scheduled run ---
    # Note: Scheduled runs on Streamlit Cloud must have secrets set in the environment.
    CONFIG = load_config(config_file_name) 
    if CONFIG is None:
        print(f"FATAL: Could not load {config_file_name}. Exiting.")
        return
        
    try:
        # Scheduled runs on Streamlit Cloud can't use st.secrets.
        # You must set OPENAI_API_KEY as an environment variable.
        # Check both Gemini and OpenAI keys
        gemini_api_key_env = os.environ.get("GEMINI_API_KEY")
        openai_api_key_env = os.environ.get("OPENAI_API_KEY")
        finnhub_api_key_env = os.environ.get("FINNHUB_API_KEY") # <-- ✅ NEW
        
        if "DATA_PROVIDERS" not in CONFIG:
            CONFIG["DATA_PROVIDERS"] = {}

        if gemini_api_key_env:
            CONFIG["DATA_PROVIDERS"]["GEMINI_API_KEY"] = gemini_api_key_env
            print("Successfully injected GEMINI API key from environment variable.")
        else:
            print("Warning: GEMINI_API_KEY environment variable not set.")
            
        if openai_api_key_env:
            CONFIG["DATA_PROVIDERS"]["OPENAI_API_KEY"] = openai_api_key_env
            print("Successfully injected OpenAI API key from environment variable.")
        else:
            print("Warning: OPENAI_API_KEY environment variable not set.")

        # --- ✅ NEW: Inject Finnhub key for scheduler ---
        if finnhub_api_key_env:
            CONFIG["DATA_PROVIDERS"]["FINNHUB_API_KEY"] = finnhub_api_key_env
            print("Successfully injected FINNHUB API key from environment variable.")
        else:
            print("Warning: FINNHUB_API_KEY environment variable not set. News will be missing.")
            
    except Exception as e:
        print(f"Error injecting env var key: {e}")
    # --- END OF MODIFICATION ---
        
    log_file_path = os.path.join(BASE_DIR, CONFIG.get('LOGGING', {}).get('LOG_FILE_PATH', 'spus_analysis.log'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        df, _, _, _ = generate_quant_report(CONFIG, print_progress_callback) # Added _ for market_regime
        if df is not None:
            print(f"Successfully generated report for {len(df)} tickers.")
        else:
            print("Analysis failed to produce data.")
            
    except Exception as e:
        logging.error(f"[SPUS SCHEDULER] Fatal error during scheduled run: {e}", exc_info=True)
        print(f"Error: Analysis failed. Check log file for details: {log_file_path}")

# --- ⭐️ 7. Main App Entry Point (Router) ⭐️ ---

# --- ✅ MODIFIED (P4): Replaced main() with direct app call ---
def main():
    """
    Main entry point. Runs the SPUS analyzer (config.json) directly.
    The market selection landing page has been removed.
    """
    try:
        # Hard-code the config.json file
        run_market_analyzer_app("config.json")
    except Exception as e:
        # Catch errors if state is mismatched
        st.error(f"An application error occurred: {e}")
        st.warning("Clearing session state and restarting. Please wait.")
        # Clear all session state keys to reset the app
        for key in st.session_state.keys():
            del st.session_state[key]
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    if "--run-scheduler" in sys.argv:
        run_analysis_for_scheduler()
    else:
        main()
