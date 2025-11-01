import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import sys
import glob

# --- إصلاح مسار الاستيراد (Import Path Fix) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# --- نهاية الإصلاح ---


# --- استيراد الدوال من ملف spus.py الخاص بك ---
try:
    from spus import (
        load_config,
        fetch_spus_tickers,
        process_ticker,
        calculate_support_resistance,
        calculate_financials_and_fair_price,
        get_sector_valuation_averages
    )
except ImportError as e:
    st.error("خطأ: فشل استيراد 'spus.py'.")
    st.error(f"تفاصيل الخطأ: {e}")
    st.stop()
except Exception as e:
    st.error(f"خطأ غير متوقع أثناء استيراد spus.py: {e}")
    st.stop()

# --- استيراد المكتبات اللازمة لوظيفة التحليل الرئيسية ---
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import openpyxl
from openpyxl.styles import Font

try:
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("مكتبة 'reportlab' غير موجودة. لن يتم إنشاء تقارير PDF.")


# --- دالة لقراءة بيانات الإكسل مع Caching (Unchanged) ---
@st.cache_data
def load_excel_data(excel_path):
    abs_excel_path = os.path.join(BASE_DIR, excel_path)
    if not os.path.exists(abs_excel_path):
        return None, None
    try:
        mod_time = os.path.getmtime(abs_excel_path)
        xls = pd.ExcelFile(abs_excel_path)
        sheet_names = xls.sheet_names
        data_sheets = {}
        for sheet in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, index_col=0)
            data_sheets[sheet] = df
        return data_sheets, mod_time
    except Exception as e:
        st.error(f"خطأ أثناء قراءة ملف الإكسل: {e}")
        return None, None


# --- دالة التنسيق الشرطي المتقدم للجداول (Unchanged) ---
def apply_comprehensive_styling(df):
    """
    Applies comprehensive styling, now including Volatility and new Momentum.
    """
    RELEVANT_COLUMNS = [
        'Ticker', 'Sector', 'Last Price', 
        'Final Quant Score', # <-- Moved to front
        'Valuation (Graham)', 'Relative P/E', 'Relative P/B',
        'MACD_Signal', 'Trend (50/200 Day MA)', 'Price vs. Levels',
        'Risk/Reward Ratio', 
        '1-Year Momentum (12-1) (%)', # <-- UPDATED
        'Volatility (1Y)', # <-- NEW
        'Return on Equity (ROE)', # For reference
        'Debt/Equity', # For reference
        'Dividend Yield (%)', # For reference
        'Forward P/E', 'Sector P/E',
        'Cut Loss Level (Support)', 'Fib 161.8% Target', 'Next Earnings Date'
    ]

    cols_to_show = [col for col in RELEVANT_COLUMNS if col in df.columns]
    df_display = df[cols_to_show].copy()

    text_style_cols = [col for col in 
                       ['Valuation (Graham)', 'MACD_Signal', 'Price vs. Levels', 'Relative P/E', 'Relative P/B'] 
                       if col in cols_to_show]

    def highlight_text(val):
        val_str = str(val).lower()
        if 'undervalued' in val_str or 'bullish' in val_str: # More general 'bullish'
            return 'color: #00A600' # أخضر
        elif 'overvalued' in val_str or 'bearish' in val_str:
            return 'color: #D30000' # أحمر
        elif 'near support' in val_str:
            return 'color: #004FB0' # أزرق
        return ''

    styler = df_display.style.apply(lambda x: x.map(highlight_text), subset=text_style_cols)

    numeric_gradient_cols = [
        'Final Quant Score', 'Risk/Reward Ratio', 
        '1-Year Momentum (12-1) (%)', 
        'Volatility (1Y)', # <-- NEW
        'Risk % (to Support)', 'Forward P/E'
    ]
    
    for col in numeric_gradient_cols:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce')

    # Apply gradients
    if 'Final Quant Score' in df_display.columns:
        styler = styler.background_gradient(cmap='RdYlGn', subset=['Final Quant Score'], vmin=-2, vmax=2)
    if 'Risk/Reward Ratio' in df_display.columns:
        styler = styler.background_gradient(cmap='RdYlGn', subset=['Risk/Reward Ratio'], vmin=0, vmax=5)
    if '1-Year Momentum (12-1) (%)' in df_display.columns:
        styler = styler.background_gradient(cmap='RdYlGn', subset=['1-Year Momentum (12-1) (%)'], vmin=-20, vmax=50)
    if 'Risk % (to Support)' in df_display.columns:
        styler = styler.background_gradient(cmap='RdYlGn_r', subset=['Risk % (to Support)'], vmin=0, vmax=15)
    if 'Forward P/E' in df_display.columns:
        styler = styler.background_gradient(cmap='RdYlGn_r', subset=['Forward P/E'], vmin=0, vmax=40)
    if 'Volatility (1Y)' in df_display.columns:
        styler = styler.background_gradient(cmap='RdYlGn_r', subset=['Volatility (1Y)'], vmin=0.1, vmax=0.6)

    format_dict = {
        'Sector P/E': '{:.2f}', 'Sector P/B': '{:.2f}', 'Forward P/E': '{:.2f}',
        'Final Quant Score': '{:.3f}',
        'Volatility (1Y)': '{:.3f}',
        '1-Year Momentum (12-1) (%)': '{:.2f}%',
        'Return on Equity (ROE)': '{:.2f}%',
        'Debt/Equity': '{:.2f}',
        'Dividend Yield (%)': '{:.2f}%',
    }
    styler = styler.format(format_dict, na_rep="N/A", subset=[col for col in format_dict if col in df_display.columns])

    return styler

# --- دالة العثور على أحدث التقارير (Unchanged) ---
def get_latest_reports(excel_base_path):
    base_dir = os.path.dirname(excel_base_path)
    excel_name_no_ext = os.path.splitext(os.path.basename(excel_base_path))[0]
    latest_pdf = None
    pdf_pattern = os.path.join(base_dir, f"{excel_name_no_ext}_*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    if pdf_files:
        latest_pdf = max(pdf_files, key=os.path.getmtime)
    excel_path = excel_base_path if os.path.exists(excel_base_path) else None
    return excel_path, latest_pdf

# --- Robust Z-Score Function (Unchanged) ---
def calculate_robust_zscore(series):
    series = pd.to_numeric(series, errors='coerce')
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return 0
    z_score = (series - median) / (1.4826 * mad)
    return z_score

# --- ⭐️ UPDATED: دالة تشغيل التحليل ⭐️ ---
def run_full_analysis(CONFIG):
    progress_bar = st.progress(0, text="Starting analysis...")
    status_text = st.empty()
    status_text.info("يتم الآن بدء التحليل...")

    MAX_RISK_USD = 50

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(BASE_DIR, CONFIG['LOG_FILE_PATH'])),
            logging.StreamHandler()
        ]
    )

    status_text.info("... (1/7) جارٍ جلب قائمة الرموز (Tickers)...")
    ticker_symbols = fetch_spus_tickers() 

    if not ticker_symbols:
        status_text.warning("لم يتم العثور على رموز. تم إلغاء التحليل.")
        return False

    exclude_tickers = CONFIG['EXCLUDE_TICKERS']
    ticker_symbols = [ticker for ticker in ticker_symbols if ticker not in exclude_tickers]

    if CONFIG['TICKER_LIMIT'] > 0:
        ticker_symbols = ticker_symbols[:CONFIG['TICKER_LIMIT']]
        status_text.info(f"التحليل يقتصر على أول {CONFIG['TICKER_LIMIT']} شركة فقط.")

    historical_data_dir = os.path.join(BASE_DIR, CONFIG['HISTORICAL_DATA_DIR'])
    if not os.path.exists(historical_data_dir): os.makedirs(historical_data_dir)
    info_cache_dir = os.path.join(BASE_DIR, CONFIG['INFO_CACHE_DIR'])
    if not os.path.exists(info_cache_dir): os.makedirs(info_cache_dir)

    if CONFIG.get('PREFETCH_SECTOR_DATA', True):
        status_text.info("... (Pre-fetch) جارٍ جلب بيانات القطاعات (Sectors)...")
        try:
            get_sector_valuation_averages("Technology") 
            status_text.info("... (Pre-fetch) اكتمل جلب بيانات القطاعات.")
        except Exception as e:
            status_text.warning(f"فشل جلب بيانات القطاعات مسبقاً: {e}")

    # (Dictionaries for data collection)
    momentum_data = {}
    volatility_data = {} 
    rsi_data = {}
    last_prices = {}
    support_resistance_levels = {}
    trend_data = {}
    macd_data = {}
    financial_data = {}
    processed_tickers = set()
    news_data = {}
    headline_data = {}
    calendar_data = {}

    MAX_WORKERS = CONFIG['MAX_CONCURRENT_WORKERS']
    status_text.info(f"... (2/7) جارٍ جلب البيانات (Concurrent) باستخدام {MAX_WORKERS} عامل...")
    start_time = time.time()
    processed_count = 0
    total_tickers = len(ticker_symbols)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(process_ticker, ticker): ticker
            for ticker in ticker_symbols
        }

        for i, future in enumerate(as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            try:
                result = future.result(timeout=60)
                if result['success']:
                    ticker = result['ticker']
                    processed_tickers.add(ticker)
                    if result['momentum_12_1'] is not None: momentum_data[ticker] = result['momentum_12_1']
                    if result['volatility_1y'] is not None: volatility_data[ticker] = result['volatility_1y']
                    if result['rsi'] is not None: rsi_data[ticker] = result['rsi']
                    if result['last_price'] is not None: last_prices[ticker] = result['last_price']
                    if result['support_resistance'] is not None: support_resistance_levels[ticker] = result['support_resistance']
                    trend_data[ticker] = result['trend']
                    macd_data[ticker] = macd_data.get(ticker, {})
                    if result['macd'] is not None: macd_data[ticker]['MACD'] = result['macd']
                    if result['signal_line'] is not None: macd_data[ticker]['Signal_Line'] = result['signal_line']
                    if result['hist_val'] is not None: macd_data[ticker]['Histogram'] = result['hist_val']
                    if result['macd_signal'] is not None: macd_data[ticker]['Signal'] = result['macd_signal']
                    financial_data[ticker] = result['financial_dict']
                    news_data[ticker] = result['recent_news']
                    headline_data[ticker] = result['latest_headline']
                    calendar_data[ticker] = result['earnings_date']

            except Exception as e:
                logging.error(f"Error processing {ticker} in main loop: {e}")

            processed_count = i + 1
            progress_percentage = processed_count / total_tickers
            progress_bar.progress(progress_percentage, text=f"Processing: {ticker} ({processed_count}/{total_tickers})")

    end_time = time.time()
    status_text.info(f"... (3/7) انتهى جلب البيانات. الوقت المستغرق: {end_time - start_time:.2f} ثانية.")
    status_text.info("... (4/7) جارٍ حساب المخاطر/العوائد (R/R)...")
    progress_bar.progress(0.9, text="Calculating Risk/Reward...")

    # (Risk/Reward calculation - Unchanged)
    threshold_percentage = CONFIG['PRICE_THRESHOLD_PERCENT']
    comparison_results = {}
    risk_percentages = {}
    reward_percentages = {}
    risk_reward_ratios = {}
    for ticker in last_prices.keys():
        last_price = last_prices.get(ticker)
        levels = support_resistance_levels.get(ticker)
        comparison_results[ticker] = 'Price or S/R levels not available'
        risk_percentages[ticker] = "N/A"
        reward_percentages[ticker] = "N/A"
        risk_reward_ratios[ticker] = "N/A"
        if last_price is not None and levels is not None and last_price > 0:
            support = levels.get('Support')
            resistance = levels.get('Resistance')
            if support is not None and resistance is not None and resistance > support:
                support_diff = last_price - support
                resistance_diff = resistance - last_price
                risk_pct = (support_diff / last_price) * 100
                reward_pct = (resistance_diff / last_price) * 100
                risk_percentages[ticker] = risk_pct
                reward_percentages[ticker] = reward_pct
                if risk_pct > 0:
                    risk_reward_ratios[ticker] = reward_pct / risk_pct
                else:
                    risk_reward_ratios[ticker] = "N/A (Price below Support)"
                support_diff_percentage = ((last_price - support) / support) * 100 if support != 0 else float('inf')
                if abs(support_diff_percentage) <= threshold_percentage:
                    comparison_results[ticker] = 'Near Support'
                elif abs(((last_price - resistance) / resistance) * 100) <= threshold_percentage:
                     comparison_results[ticker] = 'Near Resistance'
                elif last_price > resistance:
                    comparison_results[ticker] = 'Above Resistance'
                elif last_price < support:
                    comparison_results[ticker] = 'Below Support'
                else:
                    comparison_results[ticker] = 'Between Support and Resistance'

    status_text.info("... (5/7) جارٍ تجميع النتائج وحساب Z-Scores...")
    progress_bar.progress(0.95, text="Aggregating and Scoring...")

    tickers_to_report = list(last_prices.keys()) 
    if not tickers_to_report:
        status_text.warning("لا توجد بيانات كافية لإنشاء التقرير.")
        return False

    results_list = []
    for ticker in tickers_to_report:
        fin_info = financial_data.get(ticker, {})
        support_resistance = support_resistance_levels.get(ticker, {})

        # (Shares to buy calculation - Unchanged)
        shares_to_buy_str = "N/A"
        try:
            last_price_num = pd.to_numeric(last_prices.get(ticker), errors='coerce')
            support_price_num = pd.to_numeric(support_resistance.get('Support'), errors='coerce')
            if pd.notna(last_price_num) and pd.notna(support_price_num):
                risk_per_share = last_price_num - support_price_num
                if risk_per_share > 0:
                    shares_to_buy = MAX_RISK_USD / risk_per_share
                    shares_to_buy_str = f"{shares_to_buy:.2f}"
                elif risk_per_share <= 0:
                    shares_to_buy_str = "N/A (Price below Support)"
        except Exception:
            pass

        result_data = {
            'Ticker': ticker,
            'Last Price': last_prices.get(ticker, pd.NA),
            'Sector': fin_info.get('Sector'),
            'Market Cap': fin_info.get('Market Cap'),
            'Valuation (Graham)': fin_info.get('Valuation (Graham)'),
            'Fair Price (Graham)': fin_info.get('Graham Number'),
            'Forward P/E': fin_info.get('Forward P/E'),
            'Sector P/E': fin_info.get('Sector P/E'),
            'Relative P/E': fin_info.get('Relative P/E'),
            'P/B Ratio': fin_info.get('P/B Ratio'),
            'Sector P/B': fin_info.get('Sector P/B'),
            'Relative P/B': fin_info.get('Relative P/B'),
            'MACD_Signal': macd_data.get(ticker, {}).get('Signal'),
            'Trend (50/200 Day MA)': trend_data.get(ticker, "N/A"),
            'Price vs. Levels': comparison_results.get(ticker, "N/A"),
            'Cut Loss Level (Support)': support_resistance.get('Support'),
            'Risk % (to Support)': risk_percentages.get(ticker, "N/A"),
            'Fib 161.8% Target': support_resistance.get('Fib_161_8'),
            'Risk/Reward Ratio': risk_reward_ratios.get(ticker, pd.NA),
            'Shares to Buy ($50 Risk)': shares_to_buy_str,
            'Recent News (48h)': news_data.get(ticker, "N/A"),
            'Next Earnings Date': calendar_data.get(ticker, "N/A"),
            'Latest Headline': headline_data.get(ticker, "N/A"),
            'Dividend Yield (%)': fin_info.get('Dividend Yield'),
            'Return on Equity (ROE)': fin_info.get('Return on Equity (ROE)'),
            'Debt/Equity': fin_info.get('Debt/Equity'), 
            '1-Year Momentum (12-1) (%)': momentum_data.get(ticker, pd.NA),
            'Volatility (1Y)': volatility_data.get(ticker, pd.NA),
        }
        results_list.append(result_data)

    results_df = pd.DataFrame(results_list)

    # --- 6-FACTOR MODEL LOGIC (Unchanged) ---
    FACTOR_WEIGHTS = {
        'VALUE': 0.25, 
        'MOMENTUM': 0.15, 
        'QUALITY': 0.20, 
        'SIZE': 0.10, 
        'LOW_VOL': 0.15,
        'TECHNICAL': 0.15
    }

    graham_price = pd.to_numeric(results_df['Fair Price (Graham)'], errors='coerce')
    last_price_pd = pd.to_numeric(results_df['Last Price'], errors='coerce')
    last_price_safe = last_price_pd.replace(0, pd.NA)
    results_df['Value_Discount'] = graham_price / last_price_safe
    stock_pe = pd.to_numeric(results_df['Forward P/E'], errors='coerce')
    sector_pe = pd.to_numeric(results_df['Sector P/E'], errors='coerce')
    results_df['Value_Discount_PE'] = sector_pe / stock_pe
    
    results_df['Z_Value_Graham'] = results_df.groupby('Sector')['Value_Discount'].transform(calculate_robust_zscore).fillna(0)
    results_df['Z_Value_Rel_PE'] = results_df.groupby('Sector')['Value_Discount_PE'].transform(calculate_robust_zscore).fillna(0)
    results_df['Z_Value'] = (results_df['Z_Value_Graham'] + results_df['Z_Value_Rel_PE']) / 2
    
    results_df['Z_Momentum'] = results_df.groupby('Sector')['1-Year Momentum (12-1) (%)'].transform(calculate_robust_zscore).fillna(0)
    
    results_df['Z_Profitability'] = results_df.groupby('Sector')['Return on Equity (ROE)'].transform(calculate_robust_zscore).fillna(0)
    results_df['Z_Leverage'] = results_df.groupby('Sector')['Debt/Equity'].transform(calculate_robust_zscore).fillna(0) * -1 
    results_df['Z_Payout'] = results_df.groupby('Sector')['Dividend Yield (%)'].transform(calculate_robust_zscore).fillna(0)
    results_df['Z_Quality'] = (results_df['Z_Profitability'] + results_df['Z_Leverage'] + results_df['Z_Payout']) / 3

    results_df['Market Cap'] = pd.to_numeric(results_df['Market Cap'], errors='coerce')
    results_df['Z_Size'] = results_df.groupby('Sector')['Market Cap'].transform(calculate_robust_zscore).fillna(0) * -1 

    results_df['Z_Low_Volatility'] = results_df.groupby('Sector')['Volatility (1Y)'].transform(calculate_robust_zscore).fillna(0) * -1
    
    def get_technical_score(row):
        score = 0
        if str(row['MACD_Signal']).startswith('Bullish'):
            score += 1
        if str(row['Trend (50/200 Day MA)']) == 'Confirmed Uptrend':
            score += 1
        if str(row['Price vs. Levels']) == 'Near Support':
            score += 0.5
        return score
    results_df['Technical_Score'] = results_df.apply(get_technical_score, axis=1)
    results_df['Z_Technical'] = results_df.groupby('Sector')['Technical_Score'].transform(calculate_robust_zscore).fillna(0)

    results_df['Final Quant Score'] = (
        (results_df['Z_Value'] * FACTOR_WEIGHTS['VALUE']) +
        (results_df['Z_Momentum'] * FACTOR_WEIGHTS['MOMENTUM']) +
        (results_df['Z_Quality'] * FACTOR_WEIGHTS['QUALITY']) +
        (results_df['Z_Size'] * FACTOR_WEIGHTS['SIZE']) +
        (results_df['Z_Low_Volatility'] * FACTOR_WEIGHTS['LOW_VOL']) +
        (results_df['Z_Technical'] * FACTOR_WEIGHTS['TECHNICAL'])
    )
    # --- END 6-FACTOR MODEL ---

    results_df['Risk/Reward Ratio'] = pd.to_numeric(results_df['Risk/Reward Ratio'], errors='coerce')
    results_df['Risk % (to Support)'] = pd.to_numeric(results_df['Risk % (to Support)'], errors='coerce')
    results_df['Final Quant Score'] = pd.to_numeric(results_df['Final Quant Score'], errors='coerce')
    results_df.sort_values(by='Final Quant Score', ascending=False, inplace=True)
    results_df.set_index('Ticker', inplace=True)

    # (Create result sets - Unchanged)
    top_10_market_cap = results_df.sort_values(by='Market Cap', ascending=False).head(10)
    top_20_quant = results_df.head(20)
    top_10_undervalued = results_df[
        (results_df['Valuation (Graham)'] == 'Undervalued (Graham)') |
        (results_df['Relative P/E'] == 'Undervalued (Sector)')
    ].sort_values(by='Final Quant Score', ascending=False).head(10)
    new_crossovers = results_df[results_df['MACD_Signal'] == 'Bullish Crossover (Favorable)'].sort_values(by='Final Quant Score', ascending=False).head(10)
    near_support = results_df[results_df['Price vs. Levels'] == 'Near Support'].sort_values(by='Final Quant Score', ascending=False).head(10)
    top_quant_high_rr = top_20_quant[pd.to_numeric(top_20_quant['Risk/Reward Ratio'], errors='coerce') > 1].sort_values(by='Risk/Reward Ratio', ascending=False)

    excel_file_path = os.path.join(BASE_DIR, CONFIG['EXCEL_FILE_PATH'])
    try:
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            format_cols = ['Last Price', 'Fair Price (Graham)', 'Cut Loss Level (Support)',
                           'Fib 161.8% Target', 'Final Quant Score', 'Risk/Reward Ratio',
                           'Risk % (to Support)', 'Dividend Yield (%)', 
                           '1-Year Momentum (12-1) (%)',
                           'Volatility (1Y)',
                           'Return on Equity (ROE)', 'Debt/Equity',
                           'Forward P/E', 'Sector P/E', 'P/B Ratio', 'Sector P/B']

            def format_for_excel(df):
                df_copy = df.copy()
                for col in format_cols:
                    if col in df_copy.columns:
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                return df_copy

            format_for_excel(results_df).to_excel(writer, sheet_name='All Results', index=True)
            format_for_excel(top_10_market_cap).to_excel(writer, sheet_name='Top 10 by Market Cap (SPUS)', index=True)
            format_for_excel(top_20_quant).to_excel(writer, sheet_name='Top 20 Final Quant Score', index=True)
            format_for_excel(top_quant_high_rr).to_excel(writer, sheet_name='Top Quant & High R-R', index=True)
            format_for_excel(top_10_undervalued).to_excel(writer, sheet_name='Top 10 Undervalued (Rel & Graham)', index=True)
            format_for_excel(new_crossovers).to_excel(writer, sheet_name='New Bullish Crossovers (MACD)', index=True)
            format_for_excel(near_support).to_excel(writer, sheet_name='Stocks Currently Near Support', index=True)

        status_text.info(f"تم حفظ تقرير الإكسل بنجاح: {excel_file_path}")
    except Exception as e:
        st.error(f"فشل حفظ ملف الإكسل: {e}")
        return False

    status_text.info("... (7/7) جارٍ حفظ تقرير PDF...")
    progress_bar.progress(0.99, text="Saving PDF report...")
    if REPORTLAB_AVAILABLE:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_pdf_path = os.path.splitext(excel_file_path)[0]
            pdf_file_path = f"{base_pdf_path}_{timestamp}.pdf"

            doc = SimpleDocTemplate(pdf_file_path, pagesize=landscape(letter))
            elements = []
            styles = getSampleStyleSheet()

            # --- ⭐️⭐️⭐️ PDF SUMMARY LOGIC STARTS HERE ⭐️⭐️⭐️ ---
            def create_pdf_table(title, df):
                if df.empty:
                    return [Paragraph(f"No data for: {title}", styles['h2']), Spacer(1, 0.1*inch)]

                df_formatted = format_for_excel(df.reset_index())
                
                cols_map = {
                    'Top 10 by Market Cap (from SPUS)': (['Ticker', 'Market Cap', 'Sector', 'Last Price', 'Final Quant Score', 'Relative P/E', 'Risk/Reward Ratio', 'Volatility (1Y)', 'Dividend Yield (%)'], ['Ticker', 'Mkt Cap', 'Sector', 'Price', 'Score', 'Rel. P/E', 'R/R', 'Volatility', 'Div %']),
                    'Top 20 by Final Quant Score': (['Ticker', 'Final Quant Score', 'Sector', 'Last Price', 'Relative P/E', 'Valuation (Graham)', 'Risk/Reward Ratio', 'Volatility (1Y)', '1-Year Momentum (12-1) (%)'], ['Ticker', 'Score', 'Sector', 'Price', 'Rel. P/E', 'Graham', 'R/R', 'Volatility', 'Momentum']),
                    'Top Quant & High R-R (Ratio > 1)': (['Ticker', 'Final Quant Score', 'Risk/Reward Ratio', 'Relative P/E', 'Last Price', 'Volatility (1Y)', 'Cut Loss Level (Support)'], ['Ticker', 'Score', 'R/R', 'Rel. P/E', 'Price', 'Volatility', 'Stop Loss']),
                    'Top 10 Undervalued (Rel & Graham)': (['Ticker', 'Final Quant Score', 'Relative P/E', 'Valuation (Graham)', 'Last Price', 'Fair Price (Graham)', 'Sector P/E', 'Forward P/E'], ['Ticker', 'Score', 'Rel. P/E', 'Graham', 'Price', 'Graham Price', 'Sector P/E', 'Stock P/E']),
                    'New Bullish Crossovers (MACD)': (['Ticker', 'Final Quant Score', 'MACD_Signal', 'Last Price', 'Trend (50/200 Day MA)', 'Risk/Reward Ratio', 'Cut Loss Level (Support)', 'Relative P/E'], ['Ticker', 'Score', 'MACD', 'Price', 'Trend', 'R/R', 'Stop Loss', 'Rel. P/E']),
                    'Stocks Currently Near Support': (['Ticker', 'Final Quant Score', 'Price vs. Levels', 'Last Price', 'Risk % (to Support)', 'Risk/Reward Ratio', 'Cut Loss Level (Support)', 'Volatility (1Y)'], ['Ticker', 'Score', 'vs. Levels', 'Price', 'Risk %', 'R/R', 'Stop Loss', 'Volatility'])
                }

                if title in cols_map:
                    cols, headers = cols_map[title]
                    existing_cols = [c for c in cols if c in df_formatted.columns]
                    df_pdf = df_formatted[existing_cols]
                    df_pdf.columns = [headers[cols.index(c)] for c in existing_cols]
                else:
                    df_pdf = df_formatted

                data = [df_pdf.columns.tolist()] + df_pdf.values.tolist()
                formatted_data = [data[0]]
                for row in data[1:]:
                    new_row = [str(item) for item in row]
                    formatted_data.append(new_row)

                table = Table(formatted_data, hAlign='LEFT')
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.green),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALTERNATINGBACKGROUND', (0, 1), (-1, -1), [colors.Color(0.9, 0.9, 0.9), colors.Color(0.98, 0.98, 0.98)])
                ])
                table.setStyle(table_style)

                # --- ⭐️ NEW: Add Summary Text ⭐️ ---
                SUMMARY_DESCRIPTIONS = {
                    'Top 10 by Market Cap (from SPUS)': "This table shows the 10 largest companies in the SPUS portfolio, sorted by their market capitalization.",
                    'Top 20 by Final Quant Score': "This table ranks the top 20 stocks based on the combined 6-factor quantitative score (Value, Momentum, Quality, Size, Volatility, Technicals).",
                    'Top Quant & High R-R (Ratio > 1)': "This table filters the top-ranked stocks to show only those with a favorable Risk/Reward Ratio (greater than 1).",
                    'Top 10 Undervalued (Rel & Graham)': "This table highlights the top 10 stocks considered 'Undervalued' by either the Graham Number or relative sector P/E.",
                    'New Bullish Crossovers (MACD)': "This table lists stocks that have just generated a 'Bullish Crossover' MACD signal, a positive momentum indicator.",
                    'Stocks Currently Near Support': "This table identifies stocks whose current price is very close to their 90-day technical support level, a potential entry point."
                }
                
                # Start elements list
                elements = [Paragraph(title, styles['h2']), Spacer(1, 0.1*inch), table, Spacer(1, 0.1*inch)]
                
                summary_text = SUMMARY_DESCRIPTIONS.get(title)
                
                if summary_text:
                    # Use a standard body text style
                    summary_paragraph = Paragraph(summary_text, styles['BodyText'])
                    elements.append(summary_paragraph)
                    
                elements.append(Spacer(1, 0.25*inch)) # Add the final spacer
                # --- ⭐️ END NEW ⭐️ ---

                return elements
            # --- ⭐️⭐️⭐️ PDF SUMMARY LOGIC ENDS HERE ⭐️⭐️⭐️ ---

            elements.append(Paragraph(f"SPUS Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['h1']))
            elements.extend(create_pdf_table("Top 10 by Market Cap (from SPUS)", top_10_market_cap))
            elements.extend(create_pdf_table("Top 20 by Final Quant Score", top_20_quant))
            elements.extend(create_pdf_table("Top Quant & High R-R (Ratio > 1)", top_quant_high_rr))
            elements.extend(create_pdf_table("Top 10 Undervalued (Rel & Graham)", top_10_undervalued))
            elements.extend(create_pdf_table("New Bullish Crossovers (MACD)", new_crossovers))
            elements.extend(create_pdf_table("Stocks Currently Near Support", near_support))

            doc.build(elements)
            status_text.info(f"تم حفظ تقرير PDF بنجاح: {pdf_file_path}")

        except Exception as e:
            st.error(f"فشل إنشاء تقرير PDF: {e}")
    else:
        st.warning("تم تخطي إنشاء PDF. (مكتبة reportlab غير مثبتة)")

    progress_bar.progress(1.0, text="اكتمل التحليل!")
    status_text.success("اكتمل التحليل بنجاح!")
    return True
# --- ⭐️ END UPDATED FUNCTION ---


# --- واجهة مستخدم Streamlit الرئيسية (Unchanged) ---
def main():
    st.set_page_config(page_title="SPUS Quantitative Analysis", layout="wide")
    st.title("SPUS Quantitative Analysis Dashboard")
    st.markdown("لوحة متابعة لتحليل محفظة SPUS بناءً على عوامل متعددة (قيمة، زخم، جودة، حجم).")

    CONFIG = load_config('config.json')

    if CONFIG is None:
        st.error("خطأ فادح: لم يتم العثور على ملف 'config.json'. لا يمكن تشغيل التطبيق.")
        st.error(f"المسار المتوقع: {os.path.join(BASE_DIR, 'config.json')}")
        st.stop()

    EXCEL_FILE = CONFIG.get('EXCEL_FILE_PATH', './spus_analysis_results.xlsx')
    ABS_EXCEL_PATH = os.path.join(BASE_DIR, EXCEL_FILE)


    with st.sidebar:
        st.image("https://www.sp-funds.com/wp-content/uploads/2022/02/SP-Funds-Logo-Primary-Wht-1.svg", width=200)
        st.header("أدوات التحكم")

        if st.button("Run Full Analysis (تشغيل التحليل الكامل)", type="primary"):
            with st.spinner("جارٍ تشغيل التحليل الكامل... قد يستغرق هذا عدة دقائق..."):
                analysis_success = run_full_analysis(CONFIG)
                if analysis_success:
                    st.cache_data.clear()
                    st.success("اكتمل التحليل! تم تحديث البيانات.")
                    st.rerun()
                else:
                    st.error("فشل التحليل. يرجى مراجعة اللوج (spus_analysis.log).")

        st.divider()

        st.header("Downloads")
        excel_path, pdf_path = get_latest_reports(ABS_EXCEL_PATH)
        
        if excel_path:
            with open(excel_path, "rb") as file:
                st.download_button(
                    label="📥 Download Excel Report",
                    data=file,
                    file_name=os.path.basename(excel_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.warning("لم يتم العثور على تقرير Excel. قم بتشغيل التحليل أولاً.")

        if pdf_path:
            with open(pdf_path, "rb") as file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=file,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                )
        else:
            st.info("لم يتم العثور على تقرير PDF. سيتم إنشاؤه بعد تشغيل التحليل.")
        
        st.divider()
        st.info("يعرض التطبيق آخر بيانات تم تحليلها. اضغط على الزر أعلاه لجلب أحدث البيانات.")

    data_sheets, mod_time = load_excel_data(EXCEL_FILE)

    if data_sheets is None:
        st.warning("لم يتم العثور على ملف نتائج (`spus_analysis_results.xlsx`).")
        st.info("👈 يرجى الضغط على زر 'Run Full Analysis' في الشريط الجانبي لبدء التحليل الأول.")
    else:
        st.success(f"يتم الآن عرض البيانات من آخر تحليل (بتاريخ: {datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')})")

        tab_titles = list(data_sheets.keys())
        
        if "Top 10 Undervalued (Graham)" in tab_titles:
            tab_titles[tab_titles.index("Top 10 Undervalued (Graham)")] = "Top 10 Undervalued (Rel & Graham)"
        elif "Top 10 Undervalued (Rel/Graham)" in tab_titles:
            tab_titles[tab_titles.index("Top 10 Undervalued (Rel/Graham)")] = "Top 10 Undervalued (Rel & Graham)"
            
        if "All Results" in tab_titles:
            tab_titles.remove("All Results")
            tab_titles.append("All Results")

        tabs = st.tabs(tab_titles)

        for i, sheet_name in enumerate(tab_titles):
            with tabs[i]:
                st.header(sheet_name)
                df_to_show = data_sheets[sheet_name]

                chart_df = df_to_show.copy().reset_index()

                if sheet_name == 'Top 20 Final Quant Score':
                    st.subheader("أعلى 20 شركة حسب التقييم (Quant Score)")
                    chart_df['Final Quant Score'] = pd.to_numeric(chart_df['Final Quant Score'], errors='coerce')
                    chart_df.dropna(subset=['Final Quant Score'], inplace=True)
                    st.bar_chart(chart_df.sort_values('Final Quant Score', ascending=False),
                                 x='Ticker', y='Final Quant Score', color="#00A600")

                elif sheet_name == 'Top Quant & High R-R':
                    st.subheader("أفضل الشركات (تقييم عالي ونسبة مخاطرة/عائد > 1)")
                    chart_df['Risk/Reward Ratio'] = pd.to_numeric(chart_df['Risk/Reward Ratio'], errors='coerce')
                    chart_df.dropna(subset=['Risk/Reward Ratio'], inplace=True)
                    st.bar_chart(chart_df.sort_values('Risk/Reward Ratio', ascending=False),
                                 x='Ticker', y='Risk/Reward Ratio', color="#004FB0")

                elif sheet_name == 'Top 10 by Market Cap (SPUS)':
                    st.subheader("أكبر 10 شركات (من محفظة SPUS)")
                    chart_df['Market Cap'] = pd.to_numeric(chart_df['Market Cap'], errors='coerce')
                    chart_df.dropna(subset=['Market Cap'], inplace=True)
                    st.bar_chart(chart_df.sort_values('Market Cap', ascending=False),
                                 x='Ticker', y='Market Cap')

                st.divider()

                styled_df = apply_comprehensive_styling(df_to_show)
                st.dataframe(styled_df, use_container_width=True)

                csv = df_to_show.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label=f"تنزيل {sheet_name} كـ CSV",
                    data=csv,
                    file_name=f"{sheet_name.replace(' ', '_')}.csv",
                    mime='text/csv',
                    key=f"csv_download_{sheet_name}"
                )

if __name__ == "__main__":
    main()
