import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import sys
import glob
import numpy as np
import streamlit.components.v1 as components

# --- ⭐️ NEW: Imports for enhanced UI (Point 6 & 7) ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats.mstats import winsorize # Point 4

# --- 1. Set Page Configuration FIRST ---
st.set_page_config(
    page_title="SPUS Quant Analyzer (Research)",
    page_icon="https://www.sp-funds.com/wp-content/uploads/2019/07/favicon-32x32.png",
    layout="wide"
)

# --- Import Path Fix (Unchanged) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# --- ⭐️ REVISED: Import New `spus` functions (Point 1, 2) ---
try:
    from spus import (
        load_config,
        fetch_spus_tickers,
        process_ticker,
        DataFetcher # Import the new class
    )
except ImportError as e:
    st.error("خطأ: فشل استيراد 'spus.py'. تأكد من أن الملف المحدث موجود.")
    st.error(f"تفاصيل الخطأ: {e}")
    st.stop()
except Exception as e:
    st.error(f"خطأ غير متوقع أثناء استيراد spus.py: {e}")
    st.stop()

# --- ⭐️ NEW: Import Backtest Module (Point 5) ---
try:
    import backtest
except ImportError as e:
    st.error("خطأ: فشل استيراد 'backtest.py'. تأكد من أن الملف الجديد موجود.")
    st.error(f"تفاصيل الخطأ: {e}")
    st.stop()


# --- Import PDF/Logging Libraries (Unchanged) ---
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
# ... (Reportlab imports remain the same) ...


# --- ⭐️ REVISED: Custom CSS (Unchanged from original) ⭐️ ---
# The CSS from your file is good, so we'll keep it.
def load_css():
    st.markdown(f"""
    <style>
        /* ... (All your custom CSS remains here) ... */
    </style>
    """, unsafe_allow_html=True)


# --- Helper Functions (Excel/PDF) ---
# ... (get_latest_reports, create_pdf_table, etc. remain the same) ...
# ... (We will modify `run_full_analysis` instead) ...


# --- ⭐️ REVISED: Robust Z-Score (Point 4: No Global) ⭐️ ---
def calculate_robust_zscore_v2(group_series, global_median=None, global_mad=None):
    """
    Calculates robust Z-score.
    Uses global median/mad if group is too small (< 5).
    """
    if len(group_series.dropna()) < 5:
        # Use global stats passed as arguments
        median = global_median
        mad = global_mad
    else:
        # Use local group stats
        median = group_series.median()
        mad = (group_series - median).abs().median()

    # Safety checks for invalid or missing stats
    if mad is None or pd.isna(mad) or mad == 0:
        return 0
    if median is None or pd.isna(median):
         # Can't calculate z-score without a median
         return 0
         
    z_score = (group_series - median) / (1.4826 * mad)
    return z_score

# --- ⭐️ REVISED: Main Analysis Function (Points 1, 2, 3, 4, 8, 9) ⭐️ ---
@st.cache_data(show_spinner=False)
def run_full_analysis(CONFIG, factor_weights):
    """
    This function is heavily refactored to use the new data,
    calculate new factors, and apply robust statistics.
    """
    
    # --- ⭐️ NEW: Modularize (Point 9) ---
    def fetch_data_concurrently(ticker_symbols, data_fetcher):
        """ Part 1: Fetch data using ThreadPool """
        processed_data = []
        with ThreadPoolExecutor(max_workers=CONFIG['MAX_CONCURRENT_WORKERS']) as executor:
            future_to_ticker = {
                executor.submit(process_ticker, ticker, data_fetcher): ticker
                for ticker in ticker_symbols
            }
            
            total_tickers = len(ticker_symbols)
            for i, future in enumerate(as_completed(future_to_ticker)):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=60)
                    if result['success']:
                        processed_data.append(result)
                except Exception as e:
                    logging.error(f"Error processing {ticker} in main loop: {e}")
                
                progress_bar.progress((i+1)/total_tickers, text=f"Processing: {ticker} ({i+1}/{total_tickers})")
        
        return pd.DataFrame(processed_data)

    # --- ⭐️ REVISED: Modularize (Point 9) ---
    def calculate_factors_and_scores(df, weights):
        """ Part 2: Calculate all Z-Scores and final Quant Score """
        
        # --- ⭐️ NEW: Define Factor Columns (Point 2) ---
        # Format: { 'FactorName': ('ColumnName', Z_Score_Direction) }
        # Direction: 1 = High is Good, -1 = Low is Good
        FACTOR_MAP = {
            'Value': [
                ('P/E', -1), ('P/B', -1), ('EV/EBITDA', -1), ('P/FCF', -1)
            ],
            'Momentum': [
                ('Risk_Adj_Momentum_12M', 1), ('Momentum_3M', 1)
            ],
            'Quality': [
                ('ROE', 1), ('ROIC', 1), ('Profit Margin', 1), ('D/E', -1), ('Quality_Penalty', 1) # Penalty is already negative
            ],
            'Size': [
                # We use float cap, and 'Size' factor favors *smaller* companies
                ('Float-Adj Market Cap', -1) 
            ],
            'Low_Vol': [
                ('Volatility_1Y', -1), ('Beta', -1)
            ],
            'Technical': [
                ('RSI', 1), ('ADX', 1), ('Price_vs_SMA50', 1), ('Price_vs_SMA200', 1)
            ]
        }
        
        # --- ⭐️ NEW: Winsorization (Point 4) ---
        all_factor_cols = [col for factor_list in FACTOR_MAP.values() for col, _ in factor_list]
        
        for col in all_factor_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Replace infs
                df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Winsorize: Clip top/bottom 5% of outliers
                if df[col].dropna().empty: continue
                # Fill NaNs with median before winsorizing to avoid errors
                df[col] = winsorize(df[col].fillna(df[col].median()), limits=[0.05, 0.05])
        
        
        # --- ⭐️ REVISED: Local Stats for Small Sectors (Point 4) ---
        # Create a *local* dictionary, not a global one
        local_global_stats = {}
        for col in all_factor_cols:
            if col in df.columns:
                local_global_stats[col] = {
                    'median': df[col].median(),
                    'mad': (df[col] - df[col].median()).abs().median()
                }
            else:
                # Add a placeholder for safety, though it shouldn't be used
                local_global_stats[col] = {'median': np.nan, 'mad': np.nan}


        # --- Calculate Z-Scores ---
        z_score_cols = []
        
        for factor_group, factor_list in FACTOR_MAP.items():
            group_z_cols = []
            for col, direction in factor_list:
                if col in df.columns:
                    z_col_name = f"Z_{col}"
                    group_z_cols.append(z_col_name)
                    
                    # Get the pre-calculated global stats for this column
                    col_stats = local_global_stats[col]
                    
                    # --- ⭐️ THE FIX ⭐️ ---
                    # Pass the stats as keyword arguments to transform
                    df[z_col_name] = df.groupby('sector')[col].transform(
                        calculate_robust_zscore_v2,
                        global_median=col_stats['median'],
                        global_mad=col_stats['mad']
                    ) * direction
                    
                    df[z_col_name] = df[z_col_name].fillna(0) # Fill NaNs with neutral 0
            
            # Combine Z-scores for the factor group (e.g., Z_Value)
            if group_z_cols: # Only calculate if there are columns
                df[f"Z_{factor_group}"] = df[group_z_cols].mean(axis=1)
                z_score_cols.append(f"Z_{factor_group}")
            else:
                df[f"Z_{factor_group}"] = 0 # Set to 0 if no data

        # --- Final Quant Score (Point 7: Dynamic Weights) ---
        df['Final Quant Score'] = 0
        for factor_group, weight in weights.items():
            if f"Z_{factor_group}" in df.columns: # Check if factor was calculated
                df['Final Quant Score'] += df[f"Z_{factor_group}"] * weight
        
        # --- ⭐️ REVISED: Risk/Reward (Point 3) ---
        df['Risk_Pct_ATR'] = (df['last_price'] - df['Stop_Loss_ATR']) / df['last_price']
        df['Reward_Pct_ATR'] = (df['Take_Profit_ATR'] - df['last_price']) / df['last_price']
        
        df['Risk/Reward Ratio'] = df['Reward_Pct_ATR'] / df['Risk_Pct_ATR']
        df['Risk/Reward Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        # Cap at 0 (e.g., if risk is positive but reward is negative)
        df['Risk/Reward Ratio'] = df['Risk/Reward Ratio'].apply(lambda x: max(0, x) if pd.notna(x) else np.nan) 

        # --- ⭐️ NEW: Position Sizing (Point 3) ---
        account_risk = 1000 
        df['Risk_Per_Share_ATR'] = df['last_price'] - df['Stop_Loss_ATR']
        df['Position_Size_Shares'] = (account_risk / df['Risk_Per_Share_ATR']).replace([np.inf, -np.inf], 0)
        
        max_position_dollars = 10000 
        df['Position_Size_Shares'] = df.apply(
            lambda row: min(row['Position_Size_Shares'], max_position_dollars / row['last_price']) if pd.notna(row['last_price']) and row['last_price'] > 0 and pd.notna(row['Position_Size_Shares']) else 0,
            axis=1
        )
        
        df.sort_values(by='Final Quant Score', ascending=False, inplace=True)
        df.set_index('ticker', inplace=True)
        return df

    # --- ⭐️ NEW: Modularize (Point 9) ---
    def create_data_sheets(df):
        """ Part 3: Create the final Excel/PDF sheets """
        
        # --- ⭐️ REVISED: New Sheets based on new factors ---
        sheets = {
            'Top 20 Final Quant Score': df.head(20),
            'Top Quant & High R-R': df[df['Risk/Reward Ratio'] > 1.5].head(20).sort_values(by='Risk/Reward Ratio', ascending=False),
            'Top 10 Value (P/E, P/FCF)': df.nlargest(10, 'Z_Value'),
            'Top 10 Momentum (Risk-Adj)': df.nlargest(10, 'Z_Momentum'),
            'Top 10 Quality (ROIC, ROE)': df.nlargest(10, 'Z_Quality'),
            'Top 10 Low Volatility': df.nlargest(10, 'Z_Low_Vol'),
            'New Bullish Crossovers (MACD)': df[df['MACD_Signal'] == 'Bullish Crossover'].head(10),
            'All Results': df
        }
        return sheets
    
    # --- ⭐️ NEW: Modularize (Point 9) ---
    def save_reports(data_sheets, excel_path, pdf_base_path):
        """ Part 4: Save Excel and PDF reports """
        
        # --- ⭐️ NEW: Historical Logging (Point 8) ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save a historical CSV of all results
        try:
            hist_dir = os.path.join(BASE_DIR, "results_history")
            os.makedirs(hist_dir, exist_ok=True)
            hist_path = os.path.join(hist_dir, f"spus_results_{timestamp}.csv")
            data_sheets['All Results'].to_csv(hist_path)
            logging.info(f"Saved historical results to {hist_path}")
        except Exception as e:
            st.warning(f"Failed to save historical CSV: {e}")
        
        # Save Excel (Unchanged logic, just new path)
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for sheet_name, df in data_sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
            status_text.info(f"تم حفظ تقرير الإكسل بنجاح: {excel_path}")
        except Exception as e:
            st.error(f"فشل حفظ ملف الإكسل: {e}")
            return

        # Save PDF (Unchanged logic, just new path)
        # ... (Your PDF generation code would go here) ...
        # ... (It's omitted for brevity but would use the new `data_sheets`) ...
        status_text.info("تم تخطي إنشاء PDF (الرمز غير مُدرج للاختصار).")

    # --- Main Execution of run_full_analysis ---
    
    progress_bar = st.progress(0, text="Starting analysis...")
    status_text = st.empty()
    status_text.info("يتم الآن بدء التحليل...")
    
    # --- ⭐️ NEW: Init DataFetcher (Point 1) ---
    data_fetcher = DataFetcher(fmp_api_key=CONFIG.get('FMP_API_KEY'))
    
    status_text.info("... (1/4) جارٍ جلب قائمة الرموز (Tickers)...")
    ticker_symbols = fetch_spus_tickers()
    if not ticker_symbols:
        status_text.warning("لم يتم العثور على رموز. تم إلغاء التحليل.")
        return None, None
    
    # ... (Exclude/Limit Tickers logic remains the same) ...
    if CONFIG['TICKER_LIMIT'] > 0:
        ticker_symbols = ticker_symbols[:CONFIG['TICKER_LIMIT']]

    status_text.info(f"... (2/4) جارٍ جلب البيانات (Concurrent) لـ {len(ticker_symbols)} شركة...")
    results_df = fetch_data_concurrently(ticker_symbols, data_fetcher)
    
    if results_df.empty:
        status_text.error("فشل جلب البيانات لجميع الشركات. تحقق من الاتصال أو 'spus.py'.")
        return None, None
        
    status_text.info("... (3/4) جارٍ حساب Z-Scores وتصنيف العوامل...")
    scored_df = calculate_factors_and_scores(results_df, factor_weights)
    
    status_text.info("... (4/4) جارٍ إنشاء التقارير...")
    final_data_sheets = create_data_sheets(scored_df)
    
    excel_file_path = os.path.join(BASE_DIR, CONFIG['EXCEL_FILE_PATH'])
    pdf_base_path = os.path.splitext(excel_file_path)[0]
    
    save_reports(final_data_sheets, excel_file_path, pdf_base_path)
    
    progress_bar.progress(1.0, text="اكتمل التحليل!")
    status_text.success("اكتمل التحليل بنجاح!")
    
    return final_data_sheets, datetime.now().timestamp()


# --- ⭐️ REVISED: Streamlit Main UI Function ⭐️ ---
def main():
    
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    if 'scroll_to_detail' not in st.session_state:
        st.session_state.scroll_to_detail = False

    load_css()
    CONFIG = load_config('config.json')
    # ... (Config load error check) ...

    ABS_EXCEL_PATH = os.path.join(BASE_DIR, CONFIG.get('EXCEL_FILE_PATH', './spus_analysis_results.xlsx'))

    # --- ⭐️ REVISED: Sidebar (Point 7: Factor Weights) ⭐️ ---
    with st.sidebar:
        st.image("https://www.sp-funds.com/wp-content/uploads/2022/02/SP-Funds-Logo-Primary-Wht-1.svg", width=200)
        st.title("SPUS Quant Analyzer")
        st.markdown("Research-Grade 6-Factor Model")
        
        st.divider()

        st.subheader("Controls")
        if st.button("🔄 Run Full Analysis", type="primary"):
            st.cache_data.clear()
            st.session_state.selected_ticker = None
            st.session_state.scroll_to_detail = False
            st.rerun()
        
        st.divider()
        
        # --- ⭐️ NEW: Factor Weight Sliders (Point 7) ---
        st.subheader("Factor Weights")
        with st.expander("Adjust Model Weights"):
            # Use session state to store weights
            if 'factor_weights' not in st.session_state:
                st.session_state.factor_weights = {
                    'Value': 0.25, 'Momentum': 0.15, 'Quality': 0.20,
                    'Size': 0.10, 'Low_Vol': 0.15, 'Technical': 0.15
                }
            
            weights = st.session_state.factor_weights
            weights['Value'] = st.slider("Value", 0.0, 1.0, weights['Value'], 0.05)
            weights['Momentum'] = st.slider("Momentum", 0.0, 1.0, weights['Momentum'], 0.05)
            weights['Quality'] = st.slider("Quality", 0.0, 1.0, weights['Quality'], 0.05)
            weights['Size'] = st.slider("Size", 0.0, 1.0, weights['Size'], 0.05)
            weights['Low_Vol'] = st.slider("Low Volatility", 0.0, 1.0, weights['Low_Vol'], 0.05)
            weights['Technical'] = st.slider("Technical", 0.0, 1.0, weights['Technical'], 0.05)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if st.button("Normalize Weights to 1.0"):
                if total_weight > 0: # Avoid division by zero
                    for k in weights:
                        weights[k] = weights[k] / total_weight
                    st.success(f"Weights normalized (Total: {sum(weights.values()):.2f})")
                    st.rerun() # Rerun to apply new weights
                else:
                    st.warning("Total weight is 0. Cannot normalize.")

            st.info(f"**Current Total Weight:** {total_weight:.2f}")

        # Get weights for the analysis run
        factor_weights = st.session_state.factor_weights
        
        st.divider()
        
        # --- Downloads (Unchanged) ---
        st.subheader("Downloads")
        # ... (Your download button logic remains here) ...
        
        st.divider()
        # --- Glossary (Needs updating for new factors) ---
        with st.expander("Glossary & Abbreviations"):
            st.markdown("""
            * **P/FCF**: Price-to-Free Cash Flow
            * **EV/EBITDA**: Enterprise Value to EBITDA
            * **ROIC**: Return on Invested Capital
            * **Risk-Adj Mom**: Risk-Adjusted Momentum
            * **ATR**: Average True Range (Volatility)
            * **Beta**: Market Volatility Correlation
            * **ADX**: Average Directional Index (Trend)
            """)
        
    # --- Main Page Content ---
    st.title("SPUS Quantitative Dashboard")
    st.markdown(f"Using a dynamically weighted 6-factor model. Current weights: **Value** ({factor_weights['Value']:.0%}), **Momentum** ({factor_weights['Momentum']:.0%}), **Quality** ({factor_weights['Quality']:.0%}), **Size** ({factor_weights['Size']:.0%}), **Low Vol** ({factor_weights['Low_Vol']:.0%}), **Technical** ({factor_weights['Technical']:.0%}).")

    with st.spinner("Running full analysis... This may take several minutes on first run..."):
        # Pass the dynamic weights to the analysis function
        data_sheets, mod_time = run_full_analysis(CONFIG, factor_weights)
    
    if data_sheets is None:
        st.warning("Analysis failed. Please check logs or config.")
    else:
        st.success(f"يتم الآن عرض البيانات من آخر تحليل (بتاريخ: {datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')})")

        # --- ⭐️ NEW: Add Backtester Tab (Point 5) ---
        tab_titles = list(data_sheets.keys())
        tab_titles.insert(0, "📈 Backtester") # Add as first tab
        
        # ... (Tab ordering for 'All Results' remains) ...
        if "All Results" in tab_titles:
            tab_titles.remove("All Results")
            tab_titles.append("All Results")

        def set_ticker(ticker_symbol):
            st.session_state.selected_ticker = ticker_symbol
            st.session_state.scroll_to_detail = True

        tabs = st.tabs(tab_titles)
        
        # --- ⭐️ RENDER BACKTESTER TAB (Point 5) ---
        with tabs[0]:
            st.header("Proof-of-Concept Backtester")
            st.markdown("This backtests a simplified **Momentum + Low Volatility** strategy on the *entire* SPUS universe (not just today's top 10). It demonstrates the rebalancing engine.")
            st.info("This is a simplified model. A full factor backtest would require historical fundamental data, which is a complex data engineering task.")
            
            backtest_start_date = st.date_input("Backtest Start Date", datetime(2020, 1, 1))
            
            if st.button("Run Simplified Backtest"):
                with st.spinner("Running backtest... This may take a minute..."):
                    all_tickers = fetch_spus_tickers() # Get all tickers
                    
                    portfolio_ts, benchmark_ts, metrics = backtest.run_simplified_backtest(
                        all_tickers, 
                        start_date=backtest_start_date.strftime('%Y-%m-%d')
                    )
                
                if metrics:
                    st.subheader("Backtest Performance")
                    
                    # Plot Performance
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=portfolio_ts.index, y=portfolio_ts, mode='lines', name='Strategy'))
                    fig.add_trace(go.Scatter(x=benchmark_ts.index, y=benchmark_ts, mode='lines', name='SPUS Benchmark'))
                    fig.update_layout(title='Portfolio Value Over Time', yaxis_type="log")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot Drawdown
                    drawdown_ts = metrics.pop('_drawdown_ts', None)
                    if drawdown_ts is not None:
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(x=drawdown_ts.index, y=drawdown_ts, fill='tozeroy', name='Drawdown'))
                        fig_dd.update_layout(title='Portfolio Drawdown')
                        st.plotly_chart(fig_dd, use_container_width=True)

                    # Show Metrics
                    st.subheader("Key Metrics")
                    cols = st.columns(3)
                    cols[0].metric("CAGR", metrics.get("CAGR", "N/A"))
                    cols[1].metric("Sharpe Ratio", metrics.get("Sharpe Ratio", "N/A"))
                    cols[2].metric("Max Drawdown", metrics.get("Max Drawdown", "N/A"))
                    cols[0].metric("Annual Volatility", metrics.get("Annual Volatility", "N/A"))
                    cols[1].metric("Win Rate vs. SPUS", metrics.get("Win Rate (vs Benchmark)", "N/A"))
                    cols[2].metric("Final Value", metrics.get("Final Portfolio Value", "N/A"))
                    
                else:
                    st.error("Backtest failed to run or produced no results.")

        # --- RENDER DATA TABS ---
        for i, sheet_name in enumerate(tab_titles):
            if i == 0: continue # Skip backtester tab

            with tabs[i]:
                df_to_show = data_sheets[sheet_name]

                # --- ⭐️ NEW: Add Filters (Point 7) ---
                col_filter_1, col_filter_2 = st.columns([1, 2])
                with col_filter_1:
                    sectors = sorted(df_to_show['sector'].unique())
                    selected_sectors = st.multiselect(
                        "Filter by Sector",
                        options=sectors,
                        default=sectors,
                        key=f"filter_{sheet_name}"
                    )
                
                df_filtered = df_to_show[df_to_show['sector'].isin(selected_sectors)]
                
                # --- Master-Detail Layout (Unchanged) ---
                col1, col2 = st.columns([1, 2])

                # --- Column 1: Ticker List (Filtered) ---
                with col1:
                    st.subheader(f"Ticker List ({len(df_filtered)})")
                    with st.container(height=600):
                        for ticker in df_filtered.index:
                            score = df_filtered.loc[ticker, 'Final Quant Score']
                            label = f"{ticker} (Score: {score:.3f})"
                            is_selected = (st.session_state.selected_ticker == ticker)
                            button_type = "primary" if is_selected else "secondary"
                            st.button(
                                label, 
                                key=f"{sheet_name}_{ticker}", 
                                on_click=set_ticker, 
                                args=(ticker,), 
                                use_container_width=True,
                                type=button_type
                            )
                
                # --- Column 2: Ticker Details (Upgraded) ---
                with col2:
                    selected_ticker = st.session_state.selected_ticker
                    
                    if selected_ticker is None:
                        st.info("Click a ticker on the left to see its details.")
                    elif selected_ticker not in data_sheets['All Results'].index:
                         st.warning(f"Ticker '{selected_ticker}' not found in the latest data.")
                         st.session_state.selected_ticker = None
                    else:
                        # Get data from "All Results"
                        ticker_data = data_sheets['All Results'].loc[selected_ticker]
                        
                        st.markdown('<a id="detail-view-anchor"></a>', unsafe_allow_html=True)
                        st.header(f"Details for: {selected_ticker}")
                        st.markdown(f"**Sector:** {ticker_data['sector']}")
                        st.divider()
                        
                        st.subheader("Key Metrics")
                        kpi_cols = st.columns(3)
                        kpi_cols[0].metric("Final Quant Score", f"{ticker_data['Final Quant Score']:.3f}")
                        kpi_cols[1].metric("Last Price", f"${ticker_data['last_price']:.2f}")
                        kpi_cols[2].metric("MACD Signal", f"{ticker_data['MACD_Signal']}")

                        st.subheader("Risk Management (ATR-Based)")
                        lvl_cols = st.columns(3)
                        lvl_cols[0].metric("Stop Loss (1.5x ATR)", f"${ticker_data['Stop_Loss_ATR']:.2f}")
                        lvl_cols[1].metric("Take Profit (3.0x ATR)", f"${ticker_data['Take_Profit_ATR']:.2f}")
                        lvl_cols[2].metric("Risk/Reward Ratio", f"{ticker_data['Risk/Reward Ratio']:.2f}")
                        
                        # --- ⭐️ NEW: Explainability (Point 6) ---
                        with st.expander("Factor Profile (Z-Scores)", expanded=True):
                            
                            radar_col, breakdown_col = st.columns(2)
                            
                            with radar_col:
                                # Radar Chart
                                factors = ['Value', 'Momentum', 'Quality', 'Size', 'Low_Vol', 'Technical']
                                z_scores = [
                                    ticker_data['Z_Value'], ticker_data['Z_Momentum'],
                                    ticker_data['Z_Quality'], ticker_data['Z_Size'],
                                    ticker_data['Z_Low_Vol'], ticker_data['Z_Technical']
                                ]
                                # Close the loop
                                z_scores_loop = z_scores + [z_scores[0]]
                                factors_loop = factors + [factors[0]]
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatterpolar(
                                      r=z_scores_loop,
                                      theta=factors_loop,
                                      fill='toself',
                                      name='Z-Score'
                                ))
                                fig.update_layout(
                                  polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
                                  showlegend=False,
                                  title="6-Factor Z-Score Profile",
                                  margin=dict(l=40, r=40, t=40, b=40)
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with breakdown_col:
                                # Factor Contribution Breakdown
                                st.subheader("Score Contribution")
                                for f in factors:
                                    z = ticker_data[f"Z_{f}"]
                                    w = factor_weights[f]
                                    st.metric(
                                        f"{f} (Z: {z:.2f})",
                                        f"Score: {z * w:.3f}",
                                        delta=f"Weight: {w:.0%}",
                                        delta_color="off"
                                    )

                        # --- ⭐️ NEW: Interactive Chart (Point 7) ---
                        with st.expander("Interactive Price Chart"):
                            try:
                                hist = yf.Ticker(selected_ticker).history('1y')
                                
                                if hist.empty:
                                    st.warning("Could not download price history for chart.")
                                else:
                                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                                        vertical_spacing=0.03,
                                                        row_heights=[0.7, 0.3])
                                    
                                    # Candlestick
                                    fig.add_trace(go.Candlestick(x=hist.index,
                                                    open=hist['Open'], high=hist['High'],
                                                    low=hist['Low'], close=hist['Close'],
                                                    name='Price'), row=1, col=1)
                                    
                                    # SMAs
                                    hist.ta.sma(length=50, append=True)
                                    hist.ta.sma(length=200, append=True)
                                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], mode='lines', name='SMA 50'), row=1, col=1)
                                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], mode='lines', name='SMA 200'), row=1, col=1)
                                    
                                    # MACD
                                    hist.ta.macd(append=True)
                                    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_12_26_9'], mode='lines', name='MACD'), row=2, col=1)
                                    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACDs_12_26_9'], mode='lines', name='Signal'), row=2, col=1)
                                    fig.add_bar(x=hist.index, y=hist['MACDh_12_26_9'], name='Histogram', row=2, col=1)
                                    
                                    fig.update_layout(
                                        title='1-Year Price Chart w/ Technicals',
                                        xaxis_rangeslider_visible=False,
                                        height=600
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.warning(f"Could not load chart: {e}")

                        # --- Fundamental Analysis (Expanded) ---
                        with st.expander("Fundamental & Factor Details"):
                            st.dataframe(ticker_data)
                
    # --- JS Scroll (Unchanged) ---
    if st.session_state.get('scroll_to_detail', False):
        components.html(f"""
        <script>
            setTimeout(function() {{
                if (window.innerWidth < 768) {{
                    var anchor = window.parent.document.getElementById('detail-view-anchor');
                    if (anchor) {{ anchor.scrollIntoView({{ behavior: 'smooth', block: 'start' }}); }}
                }
            }}, 300);
        </script>
        """, height=0)
        st.session_state.scroll_to_detail = False


if __name__ == "__main__":
    main()
