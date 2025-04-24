# --- Base Imports ---
import streamlit as st
import norgatedata
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import coint, adfuller # Added ADF for potential future use/info
import statsmodels.api as sm
import traceback
import datetime
import time # For DTW/MI timing info
import io # Required for reading uploaded file buffer

# --- ML/Stats Imports ---
from sklearn.preprocessing import StandardScaler # Added import
from sklearn.cluster import KMeans # Added import
from sklearn.feature_selection import mutual_info_regression # Added import
from dtaidistance import dtw # Added import

# --- Configuration ---
DEFAULT_WATCHLIST = "Fut_Norgate" # Use your watchlist name
DEBUG_MODE = False # Set to True to print filter details, False for normal operation

# Ticker to Full Name Mapping (MUST BE COMPLETED BY USER)
# Only instruments listed here will be processed.
TICKER_MAP = {
    "&NQ_CCB": "E-mini Nasdaq-100",
    "&ES_CCB": "E-mini S&P 500",
    "&YM_CCB": "E-mini Dow Jones Industrial Average",
    "&RTY_CCB": "E-mini Russell 2000",
    "&ZT_CCB": "2-Year US Treasury Note",
    "&ZF_CCB": "5-Year US Treasury Note",
    "&ZN_CCB": "10-Year US Treasury Note",
    "&ZB_CCB": "30-Year US Treasury Bond",
    "&UB_CCB": "Ultra US Treasury Bond",
    "&DX_CCB": "US Dollar Index",
    "&6E_CCB": "Euro FX",
    "&6B_CCB": "British Pound Sterling",
    "&6J_CCB": "Japanese Yen",
    "&6C_CCB": "Canadian Dollar",
    "&6A_CCB": "Australian Dollar",
    "&6N_CCB": "New Zealand Dollar",
    "&6S_CCB": "Swiss Franc",
    "&6M_CCB": "Mexican Peso",
    "&BTC_CCB": "Bitcoin Futures",
    "&CL_CCB": "Crude Oil (WTI)",
    "&HO_CCB": "Heating Oil",
    "&RB_CCB": "RBOB Gasoline",
    "&NG_CCB": "Natural Gas",
    "&GC_CCB": "Gold",
    "&SI_CCB": "Silver",
    "&HG_CCB": "Copper",
    "&PA_CCB": "Palladium",
    "&PL_CCB": "Platinum",
    "&ZS_CCB": "Soybeans",
    "&ZM_CCB": "Soybean Meal",
    "&ZL_CCB": "Soybean Oil",
    "&ZC_CCB": "Corn",
    "&ZW_CCB": "Wheat",
    "&ZR_CCB": "Rough Rice",
    "&ZO_CCB": "Oats",
    "&KC_CCB": "Coffee",
    "&CC_CCB": "Cocoa",
    "&CT_CCB": "Cotton",
    "&SB_CCB": "Sugar",
    "&OJ_CCB": "Orange Juice",
    "&LBR_CCB": "Lumber",
    "&LE_CCB": "Live Cattle",
    "&GF_CCB": "Feeder Cattle",
    "&HE_CCB": "Lean Hogs"
    # --- ADD ANY OTHER TICKERS FROM YOUR WATCHLIST HERE ---
    # --- Instruments NOT listed here will be IGNORED ---
    # Example: "&BRN_CCB": "Brent Crude Oil Futures",
}

# --- Technique Definitions ---
TECHNIQUE_COINTEGRATION = "🪙 Cointegration (Engle-Granger)"
TECHNIQUE_CORRELATION = "🔗 Correlation (Returns)"
TECHNIQUE_DISTANCE = "📏 Distance (SSD)"
TECHNIQUE_CLUSTERING = "🧩 Clustering (K-Means + SSD Rank)"
TECHNIQUE_DTW = "〰️ Similarity (DTW)"
TECHNIQUE_MI = "ℹ️ Mutual Information (Prices)"

TECHNIQUES = [
    TECHNIQUE_COINTEGRATION,
    TECHNIQUE_CORRELATION,
    TECHNIQUE_DISTANCE,
    TECHNIQUE_CLUSTERING,
    TECHNIQUE_DTW,
    TECHNIQUE_MI
]

# --- Technique Information Dictionary (Updated Descriptions) ---
TECHNIQUE_INFO = {
    TECHNIQUE_COINTEGRATION: {
        "emoji": "🪙",
        "brief_description": "Tests if a *linear combination* (spread) of non-stationary prices is stationary, implying a long-run equilibrium.",
        "pros": "- Statistically rigorous test for mean-reversion.\n- Directly applicable to (non-stationary) prices.\n- Foundation for pairs trading strategies.",
        "cons": "- Assumes a *linear* relationship.\n- Sensitive to structural breaks.\n- Cointegration doesn't guarantee short-term correlation or profitability.",
        "use_case": "Finding pairs for long-term mean-reversion trading based on price equilibrium.",
        "differences": "Focuses on long-term equilibrium between prices (even if individually non-stationary). Unlike Correlation (short-term returns) or Distance/DTW (shape similarity).",
        "data_used": "Prices",
        "stationarity_note": "Specifically designed for non-stationary price series to find a stationary relationship."
    },
    TECHNIQUE_CORRELATION: {
        "emoji": "🔗",
        "brief_description": "Measures the *linear* relationship between *daily returns*.",
        "pros": "- Simple, fast, interpretable.\n- Captures short-term co-movement direction (+/-).\n- Returns are often more stationary than prices.",
        "cons": "- Only linear relationships.\n- High correlation of returns doesn't imply prices won't drift apart (cointegration).\n- Can be unstable over time.",
        "use_case": "Short-term relative value, spread betting on returns, identifying daily directional links.",
        "differences": "Uses returns (often near-stationary), not prices. Measures linear co-movement, unlike MI (non-linear) or Cointegration (long-run equilibrium).",
        "data_used": "Returns",
        "stationarity_note": "Applies to returns, which are typically less non-stationary than prices."
    },
    TECHNIQUE_DISTANCE: {
        "emoji": "📏",
        "brief_description": "Measures geometric similarity (Sum of Squared Differences) between *normalized price* chart shapes.",
        "pros": "- Intuitive (finds visually similar charts).\n- No linearity assumptions.\n- Doesn't require data stationarity.",
        "cons": "- Sensitive to outliers and scaling.\n- Ignores time lags (unlike DTW).\n- Shape similarity might not persist or be statistically significant.",
        "use_case": "Exploratory analysis, finding pairs that 'look' similar in overall price trajectory.",
        "differences": "Focuses on overall shape similarity on normalized prices. Less robust to time shifts than DTW. Doesn't test for equilibrium like Cointegration.",
        "data_used": "Normalized Prices",
        "stationarity_note": "Works directly on price patterns; does not assume stationarity."
    },
    TECHNIQUE_CLUSTERING: {
        "emoji": "🧩",
        "brief_description": "Groups instruments with similar *normalized price* patterns using K-Means, then ranks pairs within clusters by SSD.",
        "pros": "- Discovers groups automatically.\n- Can reveal non-obvious relationships based on overall shape.\n- Doesn't require data stationarity.",
        "cons": "- Requires choosing cluster count (k).\n- Sensitive to normalization.\n- Group membership doesn't guarantee pairwise tradability.",
        "use_case": "Identifying sectors/groups with similar price dynamics, then finding closest pairs within those groups.",
        "differences": "Groups assets first based on overall price patterns, then ranks pairs. Other methods compare all pairs directly.",
        "data_used": "Normalized Prices",
        "stationarity_note": "Works directly on price patterns; does not assume stationarity."
    },
    TECHNIQUE_DTW: {
        "emoji": "〰️",
        "brief_description": "Measures shape similarity between *normalized prices*, allowing for time shifts (lead/lag).",
        "pros": "- Robust to time shifts/warping.\n- Captures non-linear shape similarity.\n- Doesn't require data stationarity.",
        "cons": "- Computationally more intensive than SSD.\n- Can sometimes match unrelated series if patterns align by chance.\n- Distance value interpretation is relative.",
        "use_case": "Finding pairs with similar price patterns that aren't perfectly synchronized in time.",
        "differences": "Allows time-axis 'warping' for similarity, unlike Distance (SSD). More flexible but computationally heavier.",
        "data_used": "Normalized Prices",
        "stationarity_note": "Works directly on price patterns allowing time flexibility; does not assume stationarity."
    },
    TECHNIQUE_MI: {
        "emoji": "ℹ️",
        "brief_description": "Measures mutual dependence (linear & non-linear) between *normalized prices*.",
        "pros": "- Detects non-linear relationships missed by correlation.\n- Model-free measure of dependence.\n- Doesn't require data stationarity.",
        "cons": "- Computationally intensive.\n- Doesn't indicate direction (+/-) or type of relationship.\n- MI value interpretation is relative, not absolute.",
        "use_case": "Exploring complex, potentially non-linear dependencies between price series.",
        "differences": "Captures non-linear dependence, unlike Correlation (linear only). Measures information overlap, not just co-movement (Correlation) or equilibrium (Cointegration).",
        "data_used": "Normalized Prices",
        "stationarity_note": "Measures general dependence; does not assume stationarity."
    }
}


# --- Helper Functions ---
def get_instrument_name(norgate_ticker, ticker_map):
    """Gets the full instrument name from the Norgate ticker OR returns the ticker itself if not mapped (for uploaded files)."""
    return ticker_map.get(norgate_ticker, norgate_ticker) # Return ticker if not found

# --- Data Loading Functions ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_watchlist_symbols(watchlist_name=DEFAULT_WATCHLIST):
    """Loads symbols from a Norgate watchlist."""
    try:
        symbols = norgatedata.watchlist_symbols(watchlist_name)
        # Filter out potential None values if watchlist is empty or has issues
        return [s for s in symbols if s] if symbols else []
    except Exception as e:
        st.error(f"❌ Error loading watchlist '{watchlist_name}': {e}")
        return [] # Return empty list on error

# --- Filtering Function ---
def filter_symbols_by_map(symbols, ticker_map):
    """Filters the list of symbols to include only those present in the TICKER_MAP."""
    if not symbols: return [], []
    mapped_symbols = [s for s in symbols if s in ticker_map]
    skipped_symbols = [s for s in symbols if s not in ticker_map]
    return mapped_symbols, skipped_symbols

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_futures_data(_symbols, _start_date, _end_date):
    """Fetches historical price data for a list of known (mapped) futures symbols from Norgate."""
    if not _symbols:
        st.warning("⚠️ No mapped symbols provided to fetch data for.")
        return pd.DataFrame(), [] # Return empty DataFrame and list

    all_data = {}
    failed_symbols = []
    # Use st.empty() for smoother progress bar updates
    progress_placeholder = st.empty()
    total_symbols = len(_symbols)

    try:
        for i, symbol in enumerate(_symbols):
            # Update progress text within the placeholder
            full_name = get_instrument_name(symbol, TICKER_MAP)
            progress_text = f"Fetching data... ({i+1}/{total_symbols}) {full_name}"
            progress_placeholder.progress((i + 1) / total_symbols, text=progress_text)
            try:
                data = norgatedata.price_timeseries(
                    symbol,
                    start_date=_start_date,
                    end_date=_end_date,
                    stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.NONE,
                    padding_setting=norgatedata.PaddingType.NONE,
                    timeseriesformat='pandas-dataframe'
                )
                # Check data validity more thoroughly
                if data is not None and not data.empty and 'Close' in data.columns and data['Close'].notna().any():
                    all_data[symbol] = data['Close']
                else:
                    failed_symbols.append(symbol)
                    st.warning(f"⚠️ No valid 'Close' data found for {full_name} ({symbol}). Skipping.")
            except Exception as symbol_error:
                failed_symbols.append(symbol)
                st.warning(f"⚠️ Could not fetch data for {full_name} ({symbol}): {symbol_error}. Skipping.")

        progress_placeholder.empty() # Clear progress bar

        if not all_data:
            st.error("❌ No data could be fetched for any of the mapped symbols.")
            return pd.DataFrame(), []

        # Combine data and handle potential alignment issues
        combined_df = pd.DataFrame(all_data)
        combined_df.index = pd.to_datetime(combined_df.index)

        # Forward fill BEFORE dropping rows with any NaNs to handle missing values robustly
        combined_df = combined_df.ffill()
        # Drop rows where *any* instrument still has NaN (ensures all instruments have data for that date)
        combined_df = combined_df.dropna(axis=0, how='any')
        # Drop columns that are entirely NaN (if ffill didn't help)
        combined_df = combined_df.dropna(axis=1, how='all')

        fetched_symbols = list(combined_df.columns)
        if fetched_symbols:
             st.success(f"✅ Successfully fetched and processed data for {len(fetched_symbols)} mapped instruments.")
        else:
             st.error("❌ Data fetched, but no instruments remained after cleaning/alignment.")
             return pd.DataFrame(), []

        if failed_symbols:
            failed_names = [get_instrument_name(s, TICKER_MAP) for s in failed_symbols]
            st.warning(f"⚠️ Failed to fetch or process data for: {', '.join(failed_names)}")

        return combined_df, fetched_symbols

    except ImportError:
        st.error("❌ The 'norgatedata' library is not installed. Please install it (`pip install norgatedata`).")
        progress_placeholder.empty()
        return pd.DataFrame(), []
    except Exception as e:
        st.error(f"❌ Unexpected error during Norgate data fetching: {e}") # Clarified error source
        st.error(traceback.format_exc())
        progress_placeholder.empty()
        return pd.DataFrame(), []

@st.cache_data(ttl=3600) # Cache uploaded data as well
def load_uploaded_data(uploaded_file, sheet_name=None): # Added sheet_name parameter
    """Loads and processes data from an uploaded CSV or XLSX file."""
    if uploaded_file is None:
        return pd.DataFrame(), []

    try:
        # Read the file from the buffer
        file_name = uploaded_file.name
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            if sheet_name is None:
                # Should ideally not happen if UI logic is correct, but fallback
                st.warning("⚠️ No sheet selected for Excel file, reading the first one.")
                sheet_name = 0 # Default to first sheet if none provided
            # Read the specified sheet
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            st.error("❌ Unsupported file format. Please upload a CSV or XLSX file.")
            return pd.DataFrame(), []

        if df.empty:
            st.error(f"❌ Uploaded file ('{file_name}', Sheet: '{sheet_name}') is empty.")
            return pd.DataFrame(), []

        # --- Data Preprocessing ---
        # Assume the first column is the date/time index
        if df.columns.empty:
             st.error(f"❌ Uploaded file ('{file_name}', Sheet: '{sheet_name}') has no columns.")
             return pd.DataFrame(), []
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.set_index(date_col)
        df = df.sort_index() # Ensure chronological order

        # Drop rows where the index is NaT (failed date parsing) - Corrected method
        original_rows_before_nat_drop = len(df)
        df = df[pd.notna(df.index)]
        if DEBUG_MODE:
            st.caption(f"DEBUG: Rows before NaT index drop: {original_rows_before_nat_drop}, after: {len(df)}")

        if df.empty:
            st.error(f"❌ No valid dates found in the first column ('{date_col}') after parsing. Please ensure it contains dates.")
            return pd.DataFrame(), []

        # Convert remaining columns to numeric, coercing errors to NaN
        instrument_cols = df.columns
        for col in instrument_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Forward fill missing values
        df = df.ffill()

        # Drop columns that are entirely NaN after ffill
        df = df.dropna(axis=1, how='all')

        # Drop rows where *any* instrument still has NaN (ensures alignment)
        df = df.dropna(axis=0, how='any')

        if df.empty:
            st.error("❌ No valid data remained after cleaning and alignment. Check for missing values or non-numeric data.")
            return pd.DataFrame(), []

        processed_symbols = df.columns.tolist()
        st.success(f"✅ Successfully loaded and processed data for {len(processed_symbols)} instruments from '{file_name}' (Sheet: '{sheet_name}').")
        return df, processed_symbols

    except Exception as e:
        sheet_info = f"(Sheet: '{sheet_name}')" if sheet_name is not None else ""
        st.error(f"❌ Error processing uploaded file '{uploaded_file.name}' {sheet_info}: {e}")
        st.error(traceback.format_exc())
        return pd.DataFrame(), []


# --- Calculation Functions ---
@st.cache_data
def calculate_returns(price_df):
    """Calculates daily percentage returns."""
    if price_df is None or price_df.empty: return pd.DataFrame()
    # Ensure numeric types before pct_change
    price_df_numeric = price_df.apply(pd.to_numeric, errors='coerce')
    returns = price_df_numeric.pct_change()
    # Drop the first row (NaN) and any rows where all returns are NaN
    return returns.dropna(axis=0, how='all')

@st.cache_data
def normalize_prices(price_df):
    """Normalizes price data using Z-score (StandardScaler)."""
    if price_df is None or price_df.empty: return pd.DataFrame()
    # Ensure data is float before scaling
    try:
        price_df_float = price_df.astype(float)
    except ValueError as e:
        st.error(f"❌ Error converting price data to numeric for normalization: {e}")
        return pd.DataFrame()

    # --- Add Check for Zero Standard Deviation ---
    if isinstance(price_df_float, pd.DataFrame):
        zero_std_cols = price_df_float.columns[price_df_float.std() == 0].tolist()
        if zero_std_cols:
            st.warning(f"⚠️ Normalization Warning: The following columns have zero standard deviation within the selected date range and will likely result in NaN values after scaling: {', '.join(zero_std_cols)}")
            if DEBUG_MODE:
                 st.caption("This usually means all price values for these instruments are identical in the selected period.")
    elif isinstance(price_df_float, pd.Series):
         if price_df_float.std() == 0:
              st.warning(f"⚠️ Normalization Warning: The series '{price_df_float.name}' has zero standard deviation and will likely result in NaN values after scaling.")
              if DEBUG_MODE:
                   st.caption("This usually means all price values are identical in the selected period.")
    # --- End Check ---


    scaler = StandardScaler()
    # Handle both DataFrame and Series input
    if isinstance(price_df_float, pd.Series):
        # Reshape Series for scaler
        # Check std dev again before scaling to avoid division by zero warning from scaler itself if possible
        if price_df_float.std() == 0:
             # Create a series of NaNs manually if std is zero
             norm_data = np.full(price_df_float.shape, np.nan)
        else:
             norm_data = scaler.fit_transform(price_df_float.values.reshape(-1, 1))
        norm_df = pd.Series(norm_data.flatten(), index=price_df.index, name=price_df.name)
    elif isinstance(price_df_float, pd.DataFrame):
        # Check if DataFrame is empty after potential conversion errors
        if price_df_float.empty: return pd.DataFrame()
        # Scale column by column to handle zero std dev gracefully
        norm_data = {}
        for col in price_df_float.columns:
            series = price_df_float[col]
            if series.std() == 0:
                 # Create NaNs manually
                 norm_data[col] = np.full(series.shape, np.nan)
            else:
                 # Scale normally
                 norm_data[col] = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

        # norm_data = scaler.fit_transform(price_df_float) # Original line
        norm_df = pd.DataFrame(norm_data, index=price_df.index, columns=price_df.columns)
    else:
        st.error("❌ Invalid data type passed to normalize_prices.")
        return pd.DataFrame()

    return norm_df

@st.cache_data
def calculate_volatility(series, annualize=True, use_returns=True):
    """
    Calculates volatility.
    If use_returns=True, calculates std dev of percentage returns (annualized if specified).
    If use_returns=False, calculates std dev of the series values directly (annualization ignored).
    """
    if series is None or series.empty or len(series) < 2: return np.nan

    if use_returns:
        returns = series.pct_change().dropna()
        if returns.empty: return np.nan
        vol = returns.std()
        if annualize:
            # Simple approximation assuming daily data
            vol *= np.sqrt(252)
    else:
        # Calculate std dev of the series values directly
        vol = series.std()
        # Annualization doesn't apply in the same way to std dev of levels
        # if annualize: # Optional: could add logic if needed, but likely not for spread std dev
        #     pass

    return vol

@st.cache_data
def calculate_overlap_percentage(_series1, _series2, _full_range_length):
    """Calculates the percentage of overlapping data points."""
    if _series1 is None or _series2 is None or _full_range_length == 0: return 0.0
    common_index = _series1.dropna().index.intersection(_series2.dropna().index)
    return (len(common_index) / _full_range_length) * 100 if _full_range_length > 0 else 0.0


@st.cache_data
def calculate_half_life(series):
    """Calculates the half-life of mean reversion for a time series using OLS."""
    try:
        series = series.dropna()
        # Need at least a few points for regression
        if len(series) < 10: return np.inf
        # Calculate lagged series and delta
        lagged_series = series.shift(1).dropna()
        # Align series with lagged series
        aligned_series = series.loc[lagged_series.index]
        # Ensure enough points remain after alignment
        if len(aligned_series) < 10: return np.inf
        delta = aligned_series - lagged_series
        # Add constant for intercept term in OLS
        X = sm.add_constant(lagged_series)
        # Fit OLS model: delta = intercept + lambda * lagged_series + error
        model = sm.OLS(delta, X)
        results = model.fit()
        # Check if lambda (coefficient of lagged_series) exists and is valid
        if len(results.params) < 2 or not np.isfinite(results.params.iloc[1]): return np.inf
        lambda_ = results.params.iloc[1] # Coefficient for lagged series
        # Half-life requires lambda < 0 (mean reversion)
        if lambda_ >= 0: return np.inf
        # Calculate half-life: -ln(2) / lambda
        half_life = -np.log(2) / lambda_
        # Return half-life only if it's finite
        return half_life if np.isfinite(half_life) else np.inf
    except (ValueError, np.linalg.LinAlgError, IndexError):
        # Catch potential errors during regression or indexing
        return np.inf
    except Exception:
        # Catch any other unexpected errors
        # st.warning(f"⚠️ Unexpected error calculating half-life: {e}") # Optional: log if needed
        return np.inf

# --- Pair Finding Techniques ---
# Note: These functions now primarily return the core metric.
# Overlap and Volatility will be calculated later for filtering.

@st.cache_data
def find_cointegrated_pairs(_price_df, significance_level=0.05):
    """Finds cointegrated pairs using Engle-Granger test and calculates half-life."""
    if _price_df is None or _price_df.empty or _price_df.shape[1] < 2: return {}
    # Normalize prices for spread calculation later (more stable half-life)
    norm_df = normalize_prices(_price_df)
    if norm_df.empty:
        st.warning("⚠️ Price data became empty after normalization in cointegration check.")
        return {}

    n = _price_df.shape[1]
    pairs = {}
    tickers = _price_df.columns
    processed_pairs = set()
    progress_bar = st.progress(0, text="Finding cointegrated pairs & Half-Life...")
    total_checks = max(1, (n * (n - 1)) // 2)
    checks_done = 0
    min_obs = 50 # Minimum observations for reliable test

    for i in range(n):
        for j in range(i + 1, n):
            ticker1, ticker2 = tickers[i], tickers[j]
            pair_key = tuple(sorted((ticker1, ticker2)))
            if pair_key in processed_pairs: continue

            try:
                # Use original prices for cointegration test
                series1_orig, series2_orig = _price_df[ticker1].dropna(), _price_df[ticker2].dropna()
                common_index = series1_orig.index.intersection(series2_orig.index)

                if len(common_index) < min_obs: continue

                series1_aligned = series1_orig.loc[common_index]
                series2_aligned = series2_orig.loc[common_index]

                # Perform Engle-Granger cointegration test
                score, pvalue, _ = coint(series1_aligned, series2_aligned, trend='c') # 'c' includes constant term

                if pvalue < significance_level:
                    # Calculate spread using NORMALIZED prices for stable half-life calculation
                    half_life = np.inf # Default
                    if ticker1 in norm_df.columns and ticker2 in norm_df.columns:
                        norm_series1_aligned = norm_df.loc[common_index, ticker1]
                        norm_series2_aligned = norm_df.loc[common_index, ticker2]
                        # Simple spread (can be refined with hedge ratio if needed)
                        spread = norm_series1_aligned - norm_series2_aligned
                        if not spread.empty:
                            half_life = calculate_half_life(spread)

                    # Store only essential info initially
                    pair_info = {'pair_with': ticker2, 'p_value': pvalue, 'score': score, 'half_life': half_life}
                    pair_info_rev = {'pair_with': ticker1, 'p_value': pvalue, 'score': score, 'half_life': half_life}

                    if ticker1 not in pairs: pairs[ticker1] = []
                    if ticker2 not in pairs: pairs[ticker2] = []
                    pairs[ticker1].append(pair_info)
                    pairs[ticker2].append(pair_info_rev)

                processed_pairs.add(pair_key)
            except (ValueError, np.linalg.LinAlgError) as stat_err:
                st.warning(f"⚠️ Cointegration test failed for {get_instrument_name(ticker1,TICKER_MAP)} & {get_instrument_name(ticker2,TICKER_MAP)}: {stat_err}")
            except Exception as e:
                st.warning(f"⚠️ Unexpected error during cointegration for {get_instrument_name(ticker1,TICKER_MAP)} & {get_instrument_name(ticker2,TICKER_MAP)}: {e}")

            checks_done += 1
            # Update progress bar more frequently
            if checks_done % 20 == 0 or checks_done == total_checks:
                progress_bar.progress(min(1.0, checks_done / total_checks), text=f"Finding cointegrated pairs... ({checks_done}/{total_checks})")

    progress_bar.empty()
    # Initial sort by primary metric (p-value) - further sorting/filtering happens later
    for ticker in pairs:
        pairs[ticker] = sorted(pairs[ticker], key=lambda x: x['p_value']) # Only sort by p-value here
    return pairs

@st.cache_data
def find_correlated_pairs(_returns_df, threshold=0.7):
    """Finds pairs with absolute correlation in returns >= threshold."""
    if _returns_df is None or _returns_df.empty or _returns_df.shape[1] < 2: return {}

    try:
        corr_matrix = _returns_df.corr()
    except Exception as e:
        st.error(f"❌ Failed to calculate correlation matrix: {e}")
        return {}

    n = corr_matrix.shape[1]
    pairs = {}
    tickers = corr_matrix.columns

    for i in range(n):
        for j in range(i + 1, n):
            ticker1, ticker2 = tickers[i], tickers[j]
            # Check bounds explicitly before accessing iloc
            if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                correlation = corr_matrix.iloc[i, j]
                # Check if correlation is valid (not NaN)
                if pd.notna(correlation) and abs(correlation) >= threshold:
                    if ticker1 not in pairs: pairs[ticker1] = []
                    if ticker2 not in pairs: pairs[ticker2] = []
                    pairs[ticker1].append({'pair_with': ticker2, 'correlation': correlation})
                    pairs[ticker2].append({'pair_with': ticker1, 'correlation': correlation})
            else:
                # This should ideally not happen if loops are correct, but good failsafe
                st.warning(f"⚠️ Index out of bounds accessing correlation matrix ({i}, {j}). Skipping.")

    # Sort pairs by absolute correlation (descending)
    for ticker in pairs:
        pairs[ticker] = sorted(pairs[ticker], key=lambda x: abs(x['correlation']), reverse=True)
    return pairs

@st.cache_data
def find_distance_pairs(_price_df):
    """Finds pairs based on Sum of Squared Differences (SSD) of normalized prices."""
    if _price_df is None or _price_df.empty or _price_df.shape[1] < 2: return {}
    norm_df = normalize_prices(_price_df)
    if norm_df.empty:
        st.warning("⚠️ Price data became empty after normalization in distance check.")
        return {}

    n = norm_df.shape[1]
    pairs = {}
    tickers = norm_df.columns
    processed_pairs = set()
    progress_bar = st.progress(0, text="Finding distance pairs (SSD)...")
    total_checks = max(1, (n * (n - 1)) // 2)
    checks_done = 0
    min_obs = 10 # Need some overlap

    for i in range(n):
        for j in range(i + 1, n):
            ticker1, ticker2 = tickers[i], tickers[j]
            pair_key = tuple(sorted((ticker1, ticker2)))
            if pair_key in processed_pairs: continue

            try:
                series1, series2 = norm_df[ticker1].dropna(), norm_df[ticker2].dropna()
                common_index = series1.index.intersection(series2.index)

                if len(common_index) < min_obs: continue

                series1_aligned, series2_aligned = series1.loc[common_index], series2.loc[common_index]

                # Calculate SSD
                ssd = np.sum((series1_aligned - series2_aligned)**2)

                if np.isfinite(ssd): # Ensure SSD is a valid number
                    if ticker1 not in pairs: pairs[ticker1] = []
                    if ticker2 not in pairs: pairs[ticker2] = []
                    pairs[ticker1].append({'pair_with': ticker2, 'ssd': ssd})
                    pairs[ticker2].append({'pair_with': ticker1, 'ssd': ssd})

                processed_pairs.add(pair_key)
            except KeyError:
                 st.warning(f"⚠️ Key error during SSD for {get_instrument_name(ticker1,TICKER_MAP)} or {get_instrument_name(ticker2,TICKER_MAP)}. Skipping pair.")
            except Exception as e:
                 st.warning(f"⚠️ SSD calculation failed for {get_instrument_name(ticker1,TICKER_MAP)} & {get_instrument_name(ticker2,TICKER_MAP)}: {e}")

            checks_done += 1
            # Update progress bar more frequently
            if checks_done % 50 == 0 or checks_done == total_checks:
                progress_bar.progress(min(1.0, checks_done / total_checks), text=f"Finding distance pairs (SSD)... ({checks_done}/{total_checks})")

    progress_bar.empty()
    # Sort pairs by SSD (ascending)
    for ticker in pairs:
        pairs[ticker] = sorted(pairs[ticker], key=lambda x: x['ssd'])
    return pairs

@st.cache_data
def find_cluster_pairs(_price_df, num_clusters):
    """Finds pairs by clustering instruments (K-Means) based on normalized prices, then ranks pairs within clusters using SSD."""
    if _price_df is None or _price_df.empty or _price_df.shape[1] < 2 or num_clusters <= 1:
        return {}
    norm_df = normalize_prices(_price_df)
    if norm_df.empty:
        st.warning("⚠️ Price data became empty after normalization in clustering.")
        return {}

    # K-Means expects features (time points) as columns, samples (instruments) as rows
    data_for_clustering = norm_df.transpose().fillna(0) # Fill NaNs just in case, though dropna in fetch should handle most
    if data_for_clustering.empty or data_for_clustering.shape[0] <= num_clusters:
        st.warning("⚠️ Not enough instruments or data for clustering after processing.")
        return {}

    tickers = data_for_clustering.index
    try:
        # Use n_init='auto' for modern sklearn, random_state for reproducibility
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(data_for_clustering)
        labels = kmeans.labels_
    except ValueError as ve:
         st.error(f"❌ K-Means clustering failed (ValueError): {ve}. Check `num_clusters` ({num_clusters}) vs number of instruments ({data_for_clustering.shape[0]}).")
         return {}
    except Exception as e:
        st.error(f"❌ K-Means clustering failed: {e}")
        return {}

    # Map tickers to their cluster labels
    ticker_to_cluster = dict(zip(tickers, labels))
    # Group tickers by cluster
    clusters = {}
    for ticker, label in ticker_to_cluster.items():
        if label not in clusters: clusters[label] = []
        clusters[label].append(ticker)

    all_cluster_pairs = {}
    progress_bar = st.progress(0, text="Ranking pairs within clusters (SSD)...")
    total_instruments = len(tickers)
    min_obs_ssd = 10 # Min overlap for SSD calc

    for i, ticker1 in enumerate(tickers):
        cluster_label = ticker_to_cluster.get(ticker1)
        if cluster_label is None: continue # Should not happen if ticker was in clustering data

        cluster_members = clusters.get(cluster_label, [])
        # Only calculate SSD if there are other members in the cluster
        if len(cluster_members) <= 1: continue

        pairs_for_ticker1 = []
        for ticker2 in cluster_members:
            if ticker1 == ticker2: continue # Skip self-comparison

            try:
                # Calculate SSD between ticker1 and ticker2 using the normalized data
                series1, series2 = norm_df[ticker1].dropna(), norm_df[ticker2].dropna()
                common_index = series1.index.intersection(series2.index)

                if len(common_index) < min_obs_ssd: continue

                series1_aligned, series2_aligned = series1.loc[common_index], series2.loc[common_index]
                ssd = np.sum((series1_aligned - series2_aligned)**2)

                if np.isfinite(ssd):
                    pairs_for_ticker1.append({'pair_with': ticker2, 'ssd': ssd, 'cluster': cluster_label})

            except KeyError:
                 st.warning(f"⚠️ Key error during intra-cluster SSD for {get_instrument_name(ticker1,TICKER_MAP)} or {get_instrument_name(ticker2,TICKER_MAP)}. Skipping pair.")
            except Exception as e:
                 st.warning(f"⚠️ Intra-cluster SSD calculation failed for {get_instrument_name(ticker1,TICKER_MAP)} & {get_instrument_name(ticker2,TICKER_MAP)}: {e}")

        # Sort pairs for ticker1 by SSD (ascending)
        if pairs_for_ticker1:
            all_cluster_pairs[ticker1] = sorted(pairs_for_ticker1, key=lambda x: x['ssd'])

        progress_bar.progress(min(1.0, (i + 1) / total_instruments), text=f"Ranking pairs within clusters... ({i+1}/{total_instruments})")

    progress_bar.empty()
    return all_cluster_pairs


@st.cache_data
def find_dtw_pairs(_price_df):
    """Finds pairs based on Dynamic Time Warping distance of normalized prices (Optimized)."""
    if _price_df is None or _price_df.empty or _price_df.shape[1] < 2: return {}
    norm_df = normalize_prices(_price_df)
    if norm_df.empty:
        st.warning("⚠️ Price data became empty after normalization in DTW check.")
        return {}

    n = norm_df.shape[1]
    pairs = {}
    tickers = norm_df.columns
    processed_pairs = set()
    progress_bar = st.progress(0, text="Finding similar pairs (DTW - Optimized)...")
    total_checks = max(1, (n * (n - 1)) // 2)
    checks_done = 0
    start_time = time.time()
    use_c_dtw = True # Assume C is available, will fallback if error
    min_len = 10 # Minimum length for DTW calculation

    # Precompute numpy arrays for speed, ensuring they are C-contiguous double
    norm_arrays = {}
    for ticker in tickers:
        series = norm_df[ticker].dropna()
        if len(series) >= min_len:
             # Ensure correct dtype and C-contiguity for dtw_fast
             norm_arrays[ticker] = np.ascontiguousarray(series.values, dtype=np.double)
        else:
             norm_arrays[ticker] = None # Mark short series as unusable

    for i in range(n):
        for j in range(i + 1, n):
            ticker1, ticker2 = tickers[i], tickers[j]
            pair_key = tuple(sorted((ticker1, ticker2)))
            if pair_key in processed_pairs: continue

            try:
                series1 = norm_arrays.get(ticker1)
                series2 = norm_arrays.get(ticker2)

                # Skip if either series was too short or failed normalization
                if series1 is None or series2 is None: continue

                # --- Use Fast DTW with Pruning (dtaidistance) ---
                dtw_distance = np.inf # Default to infinity
                try:
                    # Try using the C implementation first with pruning for speed
                    # window: limits max shift (e.g., 10% of length) - adjust if needed
                    # max_dist: stop early if distance exceeds this (optional)
                    window_size = max(10, int(0.1 * max(len(series1), len(series2)))) # Example: 10% window
                    dtw_distance = dtw.distance_fast(series1, series2, window=window_size, use_pruning=True)
                except ImportError:
                     if use_c_dtw: # Show warning only once per run
                         st.warning(f"⚠️ `dtaidistance` C library optimization not available, using slower Python DTW. Install C library (`pip install dtaidistance[numpy]`) for speed.")
                         use_c_dtw = False
                     try:
                         # Pure Python version (slower)
                         dtw_distance = dtw.distance(series1, series2)
                     except Exception as py_err:
                          st.warning(f"⚠️ Python DTW calculation also failed for {get_instrument_name(ticker1,TICKER_MAP)} & {get_instrument_name(ticker2,TICKER_MAP)}: {py_err}")
                except Exception as c_err:
                    # Fallback to pure python if C fails for other reasons
                    if use_c_dtw: # Show warning only once per run
                         st.warning(f"⚠️ `dtaidistance` C library optimization failed ({c_err}), using slower Python DTW.")
                         use_c_dtw = False
                    try:
                        # Pure Python version (slower)
                        dtw_distance = dtw.distance(series1, series2)
                    except Exception as py_err:
                         st.warning(f"⚠️ Python DTW calculation also failed for {get_instrument_name(ticker1,TICKER_MAP)} & {get_instrument_name(ticker2,TICKER_MAP)}: {py_err}")


                # Add pair if DTW distance is finite
                if np.isfinite(dtw_distance):
                    if ticker1 not in pairs: pairs[ticker1] = []
                    if ticker2 not in pairs: pairs[ticker2] = []
                    pairs[ticker1].append({'pair_with': ticker2, 'dtw_distance': dtw_distance})
                    pairs[ticker2].append({'pair_with': ticker1, 'dtw_distance': dtw_distance})

                processed_pairs.add(pair_key)

            except KeyError:
                 st.warning(f"⚠️ Key error during DTW for {get_instrument_name(ticker1,TICKER_MAP)} or {get_instrument_name(ticker2,TICKER_MAP)}. Skipping pair.")
            except Exception as e:
                 st.warning(f"⚠️ Unexpected error during DTW calculation for {get_instrument_name(ticker1,TICKER_MAP)} & {get_instrument_name(ticker2,TICKER_MAP)}: {e}")

            checks_done += 1
            # --- Update progress bar more frequently ---
            if checks_done % 10 == 0 or checks_done == total_checks:
                 progress_bar.progress(min(1.0, checks_done / total_checks), text=f"Finding similar pairs (DTW)... ({checks_done}/{total_checks})")

    progress_bar.empty()
    end_time = time.time()
    st.info(f"ℹ️ DTW calculation took {end_time - start_time:.2f} seconds.")
    # Sort pairs by DTW distance (ascending)
    for ticker in pairs:
        pairs[ticker] = sorted(pairs[ticker], key=lambda x: x['dtw_distance'])
    return pairs

@st.cache_data
def find_mi_pairs(_price_df):
    """Finds pairs based on Mutual Information between normalized prices."""
    if _price_df is None or _price_df.empty or _price_df.shape[1] < 2: return {}
    norm_df = normalize_prices(_price_df)
    if norm_df.empty:
        st.warning("⚠️ Price data became empty after normalization in Mutual Information check.")
        return {}

    n = norm_df.shape[1]
    pairs = {}
    tickers = norm_df.columns
    processed_pairs = set()
    progress_bar = st.progress(0, text="Finding pairs (Mutual Information)...")
    total_checks = max(1, (n * (n - 1)) // 2)
    checks_done = 0
    min_obs = 50 # Minimum observations for reliable MI calculation
    start_time = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            ticker1, ticker2 = tickers[i], tickers[j]
            pair_key = tuple(sorted((ticker1, ticker2)))
            if pair_key in processed_pairs: continue

            try:
                # Align and drop NaNs for the specific pair
                temp_df = norm_df[[ticker1, ticker2]].dropna()
                if len(temp_df) < min_obs: continue

                # Prepare data for mutual_info_regression
                # X should be 2D, y should be 1D

                series1_aligned = temp_df[ticker1].values.reshape(-1, 1)
                series2_aligned = temp_df[ticker2].values

                # Calculate Mutual Information
                # discrete_features=False as prices are continuous
                # random_state for reproducibility of nearest neighbor calculations
                # n_neighbors can be tuned, default is 3
                mi_score = mutual_info_regression(series1_aligned, series2_aligned,
                                                  discrete_features=False, random_state=42)[0]

                if np.isfinite(mi_score): # Ensure MI score is valid
                    if ticker1 not in pairs: pairs[ticker1] = []
                    if ticker2 not in pairs: pairs[ticker2] = []
                    pairs[ticker1].append({'pair_with': ticker2, 'mutual_information': mi_score})
                    pairs[ticker2].append({'pair_with': ticker1, 'mutual_information': mi_score})

                processed_pairs.add(pair_key)
            except KeyError:
                 st.warning(f"⚠️ Key error during MI for {get_instrument_name(ticker1,TICKER_MAP)} or {get_instrument_name(ticker2,TICKER_MAP)}. Skipping pair.")
            except ValueError as ve:
                 st.warning(f"⚠️ MI calculation failed for {get_instrument_name(ticker1,TICKER_MAP)} & {get_instrument_name(ticker2,TICKER_MAP)} (ValueError): {ve}")
            except Exception as e:
                 st.warning(f"⚠️ Unexpected error during Mutual Information for {get_instrument_name(ticker1,TICKER_MAP)} & {get_instrument_name(ticker2,TICKER_MAP)}: {e}")

            checks_done += 1
            # Update progress bar more frequently
            if checks_done % 20 == 0 or checks_done == total_checks:
                progress_bar.progress(min(1.0, checks_done / total_checks), text=f"Finding pairs (Mutual Information)... ({checks_done}/{total_checks})")

    progress_bar.empty()
    end_time = time.time()
    st.info(f"ℹ️ Mutual Information calculation took {end_time - start_time:.2f} seconds.")
    # Sort pairs by Mutual Information (descending - higher MI means more dependence)
    for ticker in pairs:
        pairs[ticker] = sorted(pairs[ticker], key=lambda x: x['mutual_information'], reverse=True)
    return pairs


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="📈 Futures Pair Finder")

# --- Header ---
st.title("📈 Market Pair Detector") # Generalized title
st.markdown("""
Welcome! This application analyzes historical time series data to identify potential pairs based on various statistical and machine learning techniques.
Select your data source, desired instrument, analysis technique(s), date range, and filters from the sidebar to begin.
""")
# Display data source info later, after selection

st.write(f"🕒 App Run Time: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}")
st.divider() # Add a visual separator

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Analysis Configuration")
st.sidebar.markdown("Configure the data source, period, instrument, techniques, and filters.")

# --- Data Source Selection ---
st.sidebar.subheader("💾 Data Source")
data_source = st.sidebar.radio(
    "Select Data Source:",
    ("Norgate Data", "Uploaded File"),
    key="data_source_radio",
    horizontal=True,
)

uploaded_file = None
selected_sheet_name = None # Initialize sheet name variable

if data_source == "Uploaded File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or XLSX file",
        type=["csv", "xlsx", "xls"],
        key="file_uploader",
        help="Expected format: First column as Date/Time, subsequent columns as instrument prices."
    )
    if uploaded_file is not None:
        # --- Sheet Selection for Excel ---
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            try:
                # Use BytesIO to avoid issues with Streamlit's UploadedFile object
                excel_file_buffer = io.BytesIO(uploaded_file.getvalue())
                xls = pd.ExcelFile(excel_file_buffer)
                sheet_names = xls.sheet_names
                if not sheet_names:
                    st.sidebar.error("❌ Excel file contains no sheets.")
                    uploaded_file = None # Invalidate upload
                else:
                    selected_sheet_name = st.sidebar.selectbox(
                        "Select Sheet:",
                        options=sheet_names,
                        index=0,
                        key="sheet_selector"
                    )
            except Exception as e:
                st.sidebar.error(f"❌ Error reading Excel sheets: {e}")
                uploaded_file = None # Invalidate upload if sheets can't be read
        else:
            # For CSV files, sheet_name is not applicable
            selected_sheet_name = None

elif data_source == "Norgate Data":
    st.sidebar.caption(f"Using Norgate watchlist: **{DEFAULT_WATCHLIST}**")


# --- Data Options ---
st.sidebar.subheader("🗓️ Data Period")
# Date Range Selection - Keep defaults, but note they might be overridden by uploaded file range
today = datetime.date.today()
default_start = today - datetime.timedelta(days=5*365)
start_date_input = st.sidebar.date_input("Start Date", default_start, max_value=today - datetime.timedelta(days=1), key="start_date_input")
end_date_input = st.sidebar.date_input("End Date", today, max_value=today, key="end_date_input")

if start_date_input >= end_date_input:
    st.sidebar.error("❌ Error: Start date must be before end date.")
    st.stop() # Stop execution if dates are invalid

st.sidebar.divider() # Separator

# --- Initialize variables ---
price_data = pd.DataFrame()
available_symbols = []
instrument_names = {}
name_to_ticker = {}
skipped_symbols = [] # Specific to Norgate

# --- Load Data Based on Source ---
if data_source == "Norgate Data":
    # --- Load Symbols & Filter ---
    all_symbols = load_watchlist_symbols(DEFAULT_WATCHLIST)
    mapped_symbols, skipped_symbols = filter_symbols_by_map(all_symbols, TICKER_MAP)

    # Display skipped symbols warning (only for Norgate)
    if skipped_symbols:
        with st.sidebar.expander(f"⚠️ Skipped {len(skipped_symbols)} Norgate Symbols", expanded=False):
            skipped_names = [get_instrument_name(s, TICKER_MAP) for s in skipped_symbols]
            st.warning(f"Ignoring Norgate symbols not found in TICKER_MAP: {', '.join(skipped_names)}. Update TICKER_MAP to include them if desired.")

    if not mapped_symbols:
        st.error("❌ No symbols from the Norgate watchlist are defined in the TICKER_MAP. Cannot proceed with Norgate data.")
        # Don't st.stop() here, user might switch to upload
    else:
        # --- Fetch Data for MAPPED symbols ---
        price_data, available_symbols = fetch_futures_data(mapped_symbols, start_date_input, end_date_input)
        # --- Map Tickers to Names for UI (Norgate) ---
        if not price_data.empty:
            instrument_names = {ticker: get_instrument_name(ticker, TICKER_MAP) for ticker in available_symbols}

elif data_source == "Uploaded File":
    if uploaded_file is not None:
        # Pass the selected sheet name (will be None for CSV)
        price_data, available_symbols = load_uploaded_data(uploaded_file, selected_sheet_name)
        # Filter data based on sidebar date range AFTER loading
        if not price_data.empty:
            if DEBUG_MODE:
                st.write("--- DEBUG: Uploaded Data Info (Before Date Filter) ---")
                buffer = io.StringIO()
                price_data.info(buf=buffer)
                st.text(buffer.getvalue())
                st.dataframe(price_data.head())

            # Ensure index is datetime before filtering
            try: # Add try-except for robustness
                price_data.index = pd.to_datetime(price_data.index)
                start_datetime = pd.to_datetime(start_date_input)
                end_datetime = pd.to_datetime(end_date_input)

                # Filter based on date part of the index
                original_rows = len(price_data)
                price_data = price_data[
                    (price_data.index.normalize() >= start_datetime.normalize()) &
                    (price_data.index.normalize() <= end_datetime.normalize())
                ]
                filtered_rows = len(price_data)
                if DEBUG_MODE:
                    st.write(f"--- DEBUG: Date Filtering ---")
                    st.caption(f"Original rows: {original_rows}, Filtered rows: {filtered_rows}")
                    st.caption(f"Filtering range: {start_datetime.normalize().date()} to {end_datetime.normalize().date()}")
                    if not price_data.empty:
                        st.write("--- DEBUG: Uploaded Data Info (After Date Filter) ---")
                        buffer = io.StringIO()
                        price_data.info(buf=buffer)
                        st.text(buffer.getvalue())
                        st.dataframe(price_data.head())
                    else:
                        st.warning("DEBUG: price_data is empty after date filtering.")

            except Exception as e:
                 st.error(f"❌ Error during date conversion or filtering for uploaded data: {e}")
                 price_data = pd.DataFrame() # Invalidate data on error

            # price_data = price_data[(price_data.index >= pd.to_datetime(start_date_input)) & (price_data.index <= pd.to_datetime(end_date_input))] # Original line kept for reference
            if price_data.empty:
                 # Make warning more specific if filtering caused emptiness
                 if 'original_rows' in locals() and original_rows > 0: # Check if filtering happened
                     st.warning(f"⚠️ No data from the uploaded file falls within the selected Start/End date range ({start_date_input} to {end_date_input}).")
                 else:
                     # This case should be handled by load_uploaded_data, but as a fallback:
                     st.warning("⚠️ Uploaded data was empty even before date filtering.")
                 available_symbols = [] # Reset available symbols if date filter makes it empty
            else:
                 available_symbols = price_data.columns.tolist() # Update available symbols after date filtering
                 # --- Use column headers as names (Uploaded File) ---
                 instrument_names = {ticker: ticker for ticker in available_symbols} # Use ticker as name
    else:
        st.info("ℹ️ Please upload a CSV or XLSX file to proceed with the 'Uploaded File' option.")
        # Don't st.stop(), allow user to upload

# --- Check if data is available before proceeding ---
if price_data.empty or not available_symbols:
    if data_source == "Norgate Data" and not skipped_symbols and 'mapped_symbols' in locals() and not mapped_symbols:
        pass # Error already shown if no mapped symbols for Norgate
    elif data_source == "Uploaded File" and uploaded_file is None:
        pass # Info message already shown about needing to upload
    elif price_data.empty and available_symbols: # Should not happen, but safety check
         st.warning("⚠️ Inconsistent state: Symbols available but no price data.")
    elif price_data.empty and available_symbols: # General catch for empty dataframe after processing
         st.warning("⚠️ No price data available for the selected source, symbols, and date range after processing.")
    else:
         st.warning("⚠️ No data available to analyze. Please check your data source, symbol mapping (if Norgate), file content, or date range.")
    st.stop() # Stop if no data is ready for analysis


# --- Instrument Selection (Common Logic) ---
st.sidebar.subheader("🎯 Select Instrument")
name_to_ticker = {v: k for k, v in instrument_names.items()}

if not instrument_names:
    st.error("❌ No instruments available for selection after data loading and processing. Cannot proceed.")
    st.stop()

# Sort names alphabetically for dropdown
sorted_instrument_names = sorted(instrument_names.values())
selected_instrument_name = st.sidebar.selectbox(
    "Find pairs for:", sorted_instrument_names,
    index=0, # Default to the first instrument
    label_visibility="visible", # Make label visible
    key="instrument_select"
    )

if selected_instrument_name and selected_instrument_name in name_to_ticker:
    selected_ticker = name_to_ticker[selected_instrument_name]
else:
    st.warning("⚠️ Selected instrument name not found or invalid.")
    st.stop() # Stop if selection is somehow invalid

# --- Main Panel Setup (Common Logic) ---
# Ensure available_symbols reflects columns actually present after processing
available_symbols = price_data.columns.tolist() # Reconfirm based on final price_data

# --- Calculate Returns and Normalized Prices (needed for various calcs/plots) ---
returns_data = calculate_returns(price_data)
norm_price_data = normalize_prices(price_data) # Used for SSD, Clustering, DTW, MI, Spread Volatility, Plots
full_data_range_length = len(price_data.index) # For overlap calculation


st.sidebar.divider() # Separator

# --- Technique Selection (Context Dependent) ---
st.sidebar.subheader("🔬 Analysis Technique(s)")
# Selection widgets will be defined within tabs if needed, or use a general one here
# For simplicity, let's keep single selection for "Pair Analysis" and multi for "Comparison"
selected_technique_single = st.sidebar.selectbox(
    "Technique for Single Analysis:", options=TECHNIQUES, index=0,
    key="technique_select_single",
    help="Select one technique for the 'Pair Analysis' tab."
)
selected_techniques_multi = st.sidebar.multiselect(
    "Techniques for Comparison:", options=TECHNIQUES, default=[TECHNIQUES[0], TECHNIQUES[1]],
    key="technique_select_multi",
    help="Select multiple techniques for the 'Technique Comparison' tab."
)

# --- Display SINGLE Technique Info in Sidebar ---
if selected_technique_single and selected_technique_single in TECHNIQUE_INFO:
    with st.sidebar.expander(f"ℹ️ About: {selected_technique_single.split('(')[0].strip()}", expanded=False): # Start collapsed
        info = TECHNIQUE_INFO[selected_technique_single]
        st.markdown(f"**{info['brief_description']}**")
        st.markdown(f"**Data Used:** {info['data_used']}") # Show data used upfront
        st.markdown(f"**Stationarity:** {info['stationarity_note']}") # Show stationarity note
        st.markdown("**Pros:**\n" + info['pros'])
        st.markdown("**Cons:**\n" + info['cons'])
        st.markdown(f"**Ideal Use Case:** {info['use_case']}")
        st.markdown(f"**Key Differences:** {info['differences']}")

    st.sidebar.divider() # Separator

# --- Technique Specific Controls ---
num_clusters = 0
coint_p_value_threshold = 0.05
corr_threshold = 0.7

if selected_technique_single == TECHNIQUE_CLUSTERING:
    # Determine sensible range for k
    max_k = min(len(available_symbols) -1 , 25) # Use available_symbols count
    default_k = min(max(2, len(available_symbols) // 5), 8) # Use available_symbols count
    if max_k >= 2:
        num_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=max_k, value=default_k, key="k_slider")
    else:
        st.sidebar.warning("⚠️ Not enough instruments for clustering (need at least 2).")
        num_clusters = 0 # Disable clustering

elif selected_technique_single == TECHNIQUE_COINTEGRATION:
    coint_p_value_threshold = st.sidebar.slider("Cointegration P-value Threshold", 0.01, 0.10, 0.05, 0.01, key="p_value_slider", format="%.2f", help="Lower p-value means stronger statistical evidence for cointegration (long-run equilibrium).")

elif selected_technique_single == TECHNIQUE_CORRELATION:
    corr_threshold = st.sidebar.slider("Min. Absolute Correlation Threshold", 0.1, 0.99, 0.7, 0.05, key="corr_thresh_slider", format="%.2f", help="Minimum absolute correlation between daily returns to consider a pair.")
# Add placeholder for techniques without specific parameters
elif selected_technique_single in [TECHNIQUE_DISTANCE, TECHNIQUE_DTW, TECHNIQUE_MI]:
    st.sidebar.caption("No specific parameters for this technique.")
else:
     st.sidebar.caption("Select a technique to see its parameters.")

# --- Advanced Filtering Controls ---
st.sidebar.subheader("🔍 Advanced Filters")
min_overlap_pct = st.sidebar.slider(
    "Minimum Data Overlap (%)", 0, 100, 80, 5, key="overlap_filter",
    help="Minimum percentage of the selected date range where both instruments must have price data."
)
# Volatility of the SPREAD (normalized prices) - NOW Std Dev of Spread Values
volatility_range = st.sidebar.slider(
    "Spread Std Dev Range (Normalized)", 0.0, 3.0, (0.0, 2.5), 0.05, key="volatility_filter", # Adjusted range and label, Increased default max to 2.5
    help="Filter pairs based on the standard deviation of their normalized price spread (Z-score difference). Lower values suggest a more stable spread around its mean." # Adjusted help text
)
# Placeholder for Sector Filter (requires external data)
# selected_sector = st.sidebar.selectbox("Filter by Sector:", ["All", "Energy", "Metals", ...], key="sector_filter")


st.sidebar.divider() # Add final divider
st.sidebar.markdown("---")
# Conditional Tip based on data source
if data_source == "Norgate Data":
    st.sidebar.info("💡 Tip: Update `TICKER_MAP` in the script to include all desired instruments from your Norgate watchlist.")
else:
    st.sidebar.info("💡 Tip: Ensure uploaded file has dates in the first column and numeric price data.")
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ This app performs pair analysis on time series data.") # Generalized info


# --- Main Panel with Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Pair Analysis", "⚖️ Technique Comparison", "❓ Help / Documentation"])

with tab1:
    # Display actual data range used
    actual_start_date = price_data.index.min().strftime('%Y-%m-%d')
    actual_end_date = price_data.index.max().strftime('%Y-%m-%d')

    st.header(f"Single Technique Analysis: {selected_instrument_name}")
    st.subheader(f"Using Technique: {selected_technique_single}")

    # --- Display Analysis Summary ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Selected Instrument", value=selected_instrument_name)
    with col2:
        st.metric(label="Analysis Technique", value=selected_technique_single.split('(')[0].strip())
    with col3:
        # Show the actual date range of the data being analyzed
        st.metric(label="Data Range Analyzed", value=f"{actual_start_date} to {actual_end_date}")
    st.divider()

    # --- Find Pairs (Single Technique) ---
    found_pairs_data = {}
    metric_name, metric_format, sort_ascending = "", "", True
    extra_metric_name, extra_metric_format = None, None
    calculation_function = None

    start_calc_time = time.time()
    try:
        if selected_technique_single == TECHNIQUE_COINTEGRATION:
            calculation_function = find_cointegrated_pairs
            found_pairs_data = calculation_function(price_data, significance_level=coint_p_value_threshold)
            metric_name, metric_format, sort_ascending = "p_value", ".4f", True
            extra_metric_name, extra_metric_format = "half_life", ".2f"
        elif selected_technique_single == TECHNIQUE_CORRELATION:
             if returns_data is None or returns_data.empty: raise ValueError("Returns data missing.")
             calculation_function = find_correlated_pairs
             found_pairs_data = calculation_function(returns_data, threshold=corr_threshold)
             metric_name, metric_format, sort_ascending = "correlation", ".3f", False
        elif selected_technique_single == TECHNIQUE_DISTANCE:
             calculation_function = find_distance_pairs
             found_pairs_data = calculation_function(price_data)
             metric_name, metric_format, sort_ascending = "ssd", ".2f", True
        elif selected_technique_single == TECHNIQUE_CLUSTERING:
             if num_clusters < 2: raise ValueError("Clustering requires at least 2 clusters.")
             calculation_function = find_cluster_pairs
             found_pairs_data = calculation_function(price_data, num_clusters)
             metric_name, metric_format, sort_ascending = "ssd", ".2f", True
             extra_metric_name, extra_metric_format = "cluster", "d"
        elif selected_technique_single == TECHNIQUE_DTW:
             calculation_function = find_dtw_pairs
             found_pairs_data = calculation_function(price_data)
             metric_name, metric_format, sort_ascending = "dtw_distance", ".2f", True
        elif selected_technique_single == TECHNIQUE_MI:
             calculation_function = find_mi_pairs
             found_pairs_data = calculation_function(price_data)
             metric_name, metric_format, sort_ascending = "mutual_information", ".4f", False

    except Exception as calc_error:
         st.error(f"❌ An error occurred during the '{selected_technique_single}' calculation: {calc_error}")
         st.error(traceback.format_exc())
         found_pairs_data = {} # Reset pairs data on error

    end_calc_time = time.time()
    if calculation_function: # Only show timing if a calculation was attempted
         st.info(f"ℹ️ Pair calculation using {selected_technique_single.split('(')[0].strip()} took {end_calc_time - start_calc_time:.2f} seconds (before filtering).")

    # --- Filter and Display Results (Single Technique) ---
    if selected_ticker in found_pairs_data and found_pairs_data[selected_ticker]:
        raw_pairs_list = found_pairs_data[selected_ticker]
        filtered_display_data = []
        pair_names_for_selection = []

        # Get series for the selected instrument once
        series1_orig = price_data.get(selected_ticker)
        series1_norm = norm_price_data.get(selected_ticker)

        if series1_orig is None or series1_norm is None:
             st.warning(f"⚠️ Data missing for selected instrument {selected_instrument_name}. Cannot process pairs.")
        else:
            # --- Add debugging output header ---
            if DEBUG_MODE and raw_pairs_list:
                st.write("--- DEBUG: Pre-Filter Pair Details ---") # Add a header for debug info
                st.caption(f"Raw pairs found by {selected_technique_single.split('(')[0].strip()} before applying Overlap/Volatility filters:")

            progress_filter = st.progress(0, text="Applying filters...")
            total_raw_pairs = len(raw_pairs_list)

            for i, pair_info in enumerate(raw_pairs_list):
                pair_ticker = pair_info.get('pair_with')
                if not pair_ticker or pair_ticker not in instrument_names: continue

                series2_orig = price_data.get(pair_ticker)
                series2_norm = norm_price_data.get(pair_ticker)
                if series2_orig is None or series2_norm is None: continue # Skip if pair data missing

                # 1. Calculate Overlap
                overlap_pct = calculate_overlap_percentage(series1_orig, series2_orig, full_data_range_length)

                # 2. Calculate Spread Volatility (Standard Deviation of Normalized Spread)
                spread_vol = np.nan
                common_norm_index = series1_norm.index.intersection(series2_norm.index)
                if not common_norm_index.empty:
                    spread_norm = series1_norm.loc[common_norm_index] - series2_norm.loc[common_norm_index]
                    if not spread_norm.empty:
                        # Calculate Std Dev of the spread values directly, not returns
                        spread_vol = calculate_volatility(spread_norm, use_returns=False, annualize=False) # Changed parameters

                # --- Add debugging output for each pair ---
                if DEBUG_MODE:
                    debug_info = pair_info.copy() # Copy original pair info
                    debug_info['overlap_%'] = f"{overlap_pct:.1f}"
                    # Display the new spread_vol calculation
                    debug_info['spread_std_dev'] = f"{spread_vol:.3f}" if pd.notna(spread_vol) else "NaN" # Renamed for clarity
                    debug_info['passes_overlap'] = overlap_pct >= min_overlap_pct
                    # Check against the new spread_vol calculation
                    debug_info['passes_spread_std_dev'] = (volatility_range[0] <= spread_vol <= volatility_range[1]) if pd.notna(spread_vol) else False # Renamed for clarity
                    # Use st.json for better readability of the dictionary
                    st.json({instrument_names.get(pair_ticker, pair_ticker): debug_info})


                # Apply Filters
                if overlap_pct < min_overlap_pct: continue
                # Check against the new spread_vol calculation
                if not (volatility_range[0] <= spread_vol <= volatility_range[1]) and pd.notna(spread_vol): continue
                # Add sector filter here if implemented

                # Format and add to display list if filters passed
                pair_name = instrument_names[pair_ticker]
                metric_value = pair_info.get(metric_name)
                if metric_value is None or not np.isfinite(metric_value): continue

                try: metric_display = f"{metric_value:{metric_format}}"
                except (TypeError, ValueError): metric_display = str(metric_value)

                row_data = {
                    "Paired Instrument": pair_name,
                    metric_name.replace("_", " ").title(): metric_display,
                    "Overlap (%)": f"{overlap_pct:.1f}",
                    # Update column name in results table
                    "Spread Std Dev": f"{spread_vol:.3f}" if pd.notna(spread_vol) else "N/A",
                    "_ticker": pair_ticker # Internal lookup
                }

                # Add extra metric if exists
                if extra_metric_name and extra_metric_name in pair_info:
                     extra_value = pair_info[extra_metric_name]
                     extra_display_value = "N/A"
                     if extra_value is not None:
                         try:
                             if np.isinf(extra_value): extra_display_value = "inf"
                             else: extra_display_value = f"{extra_value:{extra_metric_format}}"
                         except (TypeError, ValueError): extra_display_value = str(extra_value)
                     row_data[extra_metric_name.replace("_", " ").title()] = extra_display_value

                filtered_display_data.append(row_data)
                pair_names_for_selection.append(pair_name) # Add only filtered pairs for visualization dropdown

                progress_filter.progress(min(1.0, (i + 1) / total_raw_pairs), text=f"Applying filters... ({i+1}/{total_raw_pairs})")
            progress_filter.empty()

            # --- Add debugging output footer ---
            if DEBUG_MODE and raw_pairs_list:
                st.write("--- END DEBUG ---")


        if filtered_display_data:
            # Sort final filtered list based on the technique's primary metric
            filtered_display_data = sorted(
                filtered_display_data,
                key=lambda x: float(x[metric_name.replace("_", " ").title()]) if x[metric_name.replace("_", " ").title()] not in ["N/A", "inf"] else (np.inf if sort_ascending else -np.inf),
                reverse=not sort_ascending
            )

            results_df = pd.DataFrame(filtered_display_data)

            # Define column order dynamically
            column_order = ["Paired Instrument"]
            if metric_name: column_order.append(metric_name.replace("_", " ").title())
            if extra_metric_name and extra_metric_name.replace("_", " ").title() in results_df.columns:
                column_order.append(extra_metric_name.replace("_", " ").title())
            # Update column name in display order
            column_order.extend(["Overlap (%)", "Spread Std Dev"]) # Add new filter columns

            existing_cols_in_order = [col for col in column_order if col in results_df.columns]
            results_df_display = results_df[existing_cols_in_order + ["_ticker"]]

            st.dataframe(
                results_df_display.drop(columns=['_ticker']),
                use_container_width=True,
                hide_index=True
            )

            # --- Visualization (using filtered pairs) ---
            st.divider()
            st.header("📈 Pair Visualization")
            # --- Add Debug Check 1 ---
            if DEBUG_MODE:
                st.write(f"--- DEBUG: Visualization Entry ---")
                st.caption(f"Number of filtered pairs available for selection: {len(pair_names_for_selection)}")
                if not pair_names_for_selection:
                    st.caption("No pairs passed filters, so no visualization dropdown will appear.")
            # --- End Debug Check 1 ---

            if pair_names_for_selection: # Use names from the filtered list
                selected_pair_name_viz = st.selectbox("Select a Filtered Pair to Visualize:", pair_names_for_selection, key="viz_select_tab1")

                selected_pair_ticker_viz = None
                if selected_pair_name_viz:
                    # Find the corresponding ticker using the results_df_display which includes the '_ticker' column
                    matching_rows = results_df_display[results_df_display['Paired Instrument'] == selected_pair_name_viz]
                    if not matching_rows.empty:
                        selected_pair_ticker_viz = matching_rows['_ticker'].iloc[0]
                    else:
                         # Fallback: try finding ticker from name_to_ticker (should not be needed if results_df_display is correct)
                         pair_name_to_ticker = {v: k for k, v in instrument_names.items()}
                         selected_pair_ticker_viz = pair_name_to_ticker.get(selected_pair_name_viz)

                    # --- Add Debug Check 2 ---
                    if DEBUG_MODE:
                        st.write(f"--- DEBUG: Ticker Lookup ---")
                        st.caption(f"Selected Pair Name for Viz: {selected_pair_name_viz}")
                        st.caption(f"Looked up Pair Ticker for Viz: {selected_pair_ticker_viz}")
                    # --- End Debug Check 2 ---


                # Ensure selected_ticker and selected_pair_ticker_viz are valid and exist in price_data and norm_price_data
                if selected_pair_ticker_viz and \
                   selected_ticker in price_data.columns and \
                   selected_pair_ticker_viz in price_data.columns and \
                   selected_ticker in norm_price_data.columns and \
                   selected_pair_ticker_viz in norm_price_data.columns:

                    # --- Plotting logic (Normalized Prices and Spread) ---
                    # Use pre-calculated normalized data
                    norm_series1_plot = norm_price_data[selected_ticker]
                    norm_series2_plot = norm_price_data[selected_pair_ticker_viz]

                    # --- Enhanced Debugging and Checks ---
                    # Check if the normalized series are valid for plotting (not empty and containat least one non-NaN value)
                    series1_valid = not norm_series1_plot.empty and norm_series1_plot.notna().any()
                    series2_valid = not norm_series2_plot.empty and norm_series2_plot.notna().any()

                    if DEBUG_MODE:
                        st.write(f"--- DEBUG: Pre-Plotting Checks ---")
                        st.caption(f"Plotting: {selected_instrument_name} ({selected_ticker}) vs {selected_pair_name_viz} ({selected_pair_ticker_viz})")
                        st.caption(f"Normalized Series 1 ({selected_ticker}) - Valid for Plotting: {series1_valid}")
                        if not norm_series1_plot.empty:
                            # Check for NaNs specifically
                            nan_count1 = norm_series1_plot.isna().sum()
                            st.text(norm_series1_plot.describe())
                            st.caption(f"NaN count: {nan_count1} (out of {len(norm_series1_plot)})") # Show total count
                        else:
                            st.caption("Series 1 is empty.")

                        st.caption(f"Normalized Series 2 ({selected_pair_ticker_viz}) - Valid for Plotting: {series2_valid}")
                        if not norm_series2_plot.empty:
                            # Check for NaNs specifically
                            nan_count2 = norm_series2_plot.isna().sum()
                            st.text(norm_series2_plot.describe())
                            st.caption(f"NaN count: {nan_count2} (out of {len(norm_series2_plot)})") # Show total count
                        else:
                            st.caption("Series 2 is empty.")
                        st.write("--- END DEBUG ---")

                    if series1_valid and series2_valid: # Use the pre-calculated validity flags
                        try: # --- Add Try-Except for Plotting ---
                            # Plot Normalized Prices
                            fig_norm = go.Figure()
                            fig_norm.add_trace(go.Scatter(x=norm_series1_plot.index, y=norm_series1_plot, mode='lines', name=f"{selected_instrument_name} (Norm)", line=dict(color='blue')))
                            fig_norm.add_trace(go.Scatter(x=norm_series2_plot.index, y=norm_series2_plot, mode='lines', name=f"{selected_pair_name_viz} (Norm)", line=dict(color='red')))
                            fig_norm.update_layout(title=f"Normalized Price Comparison: {selected_instrument_name} vs {selected_pair_name_viz}",
                                                xaxis_title="Date", yaxis_title="Normalized Price (Z-score)", legend_title="Instrument", hovermode="x unified",
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.02,
                                                    xanchor="right",
                                                    x=1
                                                ))
                            st.plotly_chart(fig_norm, use_container_width=True)

                            # Plot Spread (Normalized Difference)
                            spread_plot_df = pd.concat([norm_series1_plot, norm_series2_plot], axis=1, keys=['s1', 's2']).dropna()
                            if not spread_plot_df.empty:
                                spread_plot = spread_plot_df['s1'] - spread_plot_df['s2']
                                if not spread_plot.empty:
                                    fig_spread = go.Figure()
                                    fig_spread.add_trace(go.Scatter(x=spread_plot.index, y=spread_plot, mode='lines',name='Spread (Normalized)', line=dict(color='green')))
                                    spread_mean = spread_plot.mean()
                                    if np.isfinite(spread_mean):
                                         fig_spread.add_hline(y=spread_mean, line_dash="dash", line_color="grey", annotation_text=f"Mean: {spread_mean:.2f}")
                                    fig_spread.update_layout(title=f"Spread (Normalized Difference): {selected_instrument_name} - {selected_pair_name_viz}",
                                                            xaxis_title="Date", yaxis_title="Spread Value", hovermode="x unified")
                                    st.plotly_chart(fig_spread, use_container_width=True)
                                else: st.warning("⚠️ Could not plot spread (empty after calculation).")
                            else: st.warning("⚠️ Could not plot spread due to lack of overlapping data between normalized series.")
                        except Exception as plot_err: # --- Catch Plotting Errors ---
                             st.error(f"❌ An error occurred during plotting: {plot_err}")
                             st.error(traceback.format_exc())
                             if DEBUG_MODE:
                                 st.write("--- DEBUG: Data at Plotting Error ---")
                                 st.caption("Series 1 (Normalized):")
                                 st.dataframe(norm_series1_plot.head())
                                 st.caption("Series 2 (Normalized):")
                                 st.dataframe(norm_series2_plot.head())
                                 if 'spread_plot' in locals():
                                     st.caption("Spread Series:")
                                     st.dataframe(spread_plot.head())
                    else: # Provide more specific feedback
                        warning_msg = "⚠️ Cannot plot: "
                        if not series1_valid and not series2_valid:
                            warning_msg += f"Normalized data for *both* {selected_instrument_name} and {selected_pair_name_viz} is invalid (empty or all NaN)."
                        elif not series1_valid:
                            warning_msg += f"Normalized data for {selected_instrument_name} ({selected_ticker}) is invalid (empty or all NaN)."
                        elif not series2_valid:
                             warning_msg += f"Normalized data for {selected_pair_name_viz} ({selected_pair_ticker_viz}) is invalid (empty or all NaN)."
                        st.warning(warning_msg)
                        if not DEBUG_MODE:
                             st.caption("Enable DEBUG_MODE in the script for more details.")

                elif selected_pair_name_viz:
                    # Add more specific warnings if tickers were found but data is missing or invalid
                    if not selected_pair_ticker_viz:
                         st.warning(f"⚠️ Could not find the ticker for the selected pair name '{selected_pair_name_viz}' for plotting.")
                    elif selected_ticker not in price_data.columns or selected_ticker not in norm_price_data.columns:
                         st.warning(f"⚠️ Data missing for the primary instrument '{selected_instrument_name}' ({selected_ticker}) in price_data or norm_price_data.")
                    elif selected_pair_ticker_viz not in price_data.columns or selected_pair_ticker_viz not in norm_price_data.columns:
                         st.warning(f"⚠️ Data missing for the selected pair instrument '{selected_pair_name_viz}' ({selected_pair_ticker_viz}) in price_data or norm_price_data.")
                    else:
                         st.warning(f"⚠️ Could not find valid data for the selected pair '{selected_pair_name_viz}' for plotting (check data availability and normalization results).")

            else:
                st.info("✅ No pairs passed the filters for visualization.")
        else:
             # Modify the message slightly to acknowledge filtering
             st.info(f"✅ No pairs found for {selected_instrument_name} using {selected_technique_single} *that passed the selected filters*.")
             # Add a note if debugging is off but raw pairs might exist
             if not DEBUG_MODE and selected_ticker in found_pairs_data and found_pairs_data[selected_ticker]:
                 st.caption("Pairs might exist before filtering. Set DEBUG_MODE = True in the script to see pre-filter details.")

    with tab2:
        st.header(f"Technique Comparison for: {selected_instrument_name}")

        if not selected_techniques_multi:
            st.warning("⚠️ Please select at least one technique in the sidebar for comparison.")
            st.stop()

        # Display actual data range used
        actual_start_date_comp = price_data.index.min().strftime('%Y-%m-%d')
        actual_end_date_comp = price_data.index.max().strftime('%Y-%m-%d')

        st.write(f"Comparing techniques: {', '.join([t.split('(')[0].strip() for t in selected_techniques_multi])}")
        st.write(f"Data Range Analyzed: {actual_start_date_comp} to {actual_end_date_comp}") # Show actual range
        st.write(f"Filters Applied: Min Overlap {min_overlap_pct}%, Spread Std Dev Range {volatility_range[0]:.2f}-{volatility_range[1]:.2f}") # Renamed filter
        st.divider()

        # Get series for the selected instrument once
        series1_orig_comp = price_data.get(selected_ticker)
        series1_norm_comp = norm_price_data.get(selected_ticker)

        if series1_orig_comp is None or series1_norm_comp is None:
             st.error(f"❌ Data missing for selected instrument {selected_instrument_name}. Cannot perform comparison.")
             st.stop()

        # --- Run selected techniques and filter results ---

        all_results_filtered = {}

        for technique in selected_techniques_multi:
            st.subheader(f"Results: {technique}")
            tech_short_name = technique.split('(')[0].strip()
            found_pairs_data_comp = {}
            metric_name_comp, metric_format_comp, sort_ascending_comp = "", "", True
            extra_metric_name_comp, extra_metric_format_comp = None, None
            calculation_function_comp = None

            start_calc_time_comp = time.time()
            try:
                # --- (Similar try-except block as in tab1 to get raw pairs for 'technique') ---
                if technique ==                TECHNIQUE_COINTEGRATION:
                    calculation_function_comp = find_cointegrated_pairs
                    found_pairs_data_comp = calculation_function_comp(price_data, significance_level=coint_p_value_threshold)
                    metric_name_comp, metric_format_comp, sort_ascending_comp = "p_value", ".4f", True
                    extra_metric_name_comp, extra_metric_format_comp = "half_life", ".2f"
                elif technique == TECHNIQUE_CORRELATION:
                     if returns_data is None or returns_data.empty: raise ValueError("Returns data missing.")
                     calculation_function_comp = find_correlated_pairs
                     found_pairs_data_comp = calculation_function_comp(returns_data, threshold=corr_threshold)
                     metric_name_comp, metric_format_comp, sort_ascending_comp = "correlation", ".3f", False
                elif technique == TECHNIQUE_DISTANCE:
                     calculation_function_comp = find_distance_pairs
                     found_pairs_data_comp = calculation_function_comp(price_data)
                     metric_name_comp, metric_format_comp, sort_ascending_comp = "ssd", ".2f", True
                elif technique == TECHNIQUE_CLUSTERING:
                     if num_clusters < 2: raise ValueError("Clustering requires at least 2 clusters.")
                     calculation_function_comp = find_cluster_pairs
                     found_pairs_data_comp = calculation_function_comp(price_data, num_clusters)
                     metric_name_comp, metric_format_comp, sort_ascending_comp = "ssd", ".2f", True
                     extra_metric_name_comp, extra_metric_format_comp = "cluster", "d"
                elif technique == TECHNIQUE_DTW:
                     calculation_function_comp = find_dtw_pairs
                     found_pairs_data_comp = calculation_function_comp(price_data)
                     metric_name_comp, metric_format_comp, sort_ascending_comp = "dtw_distance", ".2f", True
                elif technique == TECHNIQUE_MI:
                     calculation_function_comp = find_mi_pairs
                     found_pairs_data_comp = calculation_function_comp(price_data)
                     metric_name_comp, metric_format_comp, sort_ascending_comp = "mutual_information", ".4f", False

            except Exception as calc_error:
                 st.error(f"❌ An error occurred during the '{technique}' calculation: {calc_error}")
                 # Continue to next technique
                 continue

            end_calc_time_comp = time.time()
            if calculation_function_comp:
                 # Corrected typo in variable name below
                 st.caption(f"Calculation took {end_calc_time_comp - start_calc_time_comp:.2f} seconds (before filtering).")


            # --- Filter results for this technique ---
            if selected_ticker in found_pairs_data_comp and found_pairs_data_comp[selected_ticker]:
                raw_pairs_list_comp = found_pairs_data_comp[selected_ticker]
                filtered_display_data_comp = []

                # (Similar filtering loop as in tab1, using _comp variables)
                for pair_info_comp in raw_pairs_list_comp:
                    pair_ticker_comp = pair_info_comp.get('pair_with')
                    if not pair_ticker_comp or pair_ticker_comp not in instrument_names: continue

                    series2_orig_comp = price_data.get(pair_ticker_comp)
                    series2_norm_comp = norm_price_data.get(pair_ticker_comp)
                    if series2_orig_comp is None or series2_norm_comp is None: continue

                    overlap_pct_comp = calculate_overlap_percentage(series1_orig_comp, series2_orig_comp, full_data_range_length)
                    spread_vol_comp = np.nan
                    common_norm_index_comp = series1_norm_comp.index.intersection(series2_norm_comp.index)
                    if not common_norm_index_comp.empty:
                        spread_norm_comp = series1_norm_comp.loc[common_norm_index_comp] - series2_norm_comp.loc[common_norm_index_comp]
                        if not spread_norm_comp.empty:
                            # Calculate Std Dev of the spread values directly, not returns
                            spread_vol_comp = calculate_volatility(spread_norm_comp, use_returns=False, annualize=False) # Changed parameters

                    if overlap_pct_comp < min_overlap_pct: continue
                    # Check against the new spread_vol calculation
                    if not (volatility_range[0] <= spread_vol_comp <= volatility_range[1]) and pd.notna(spread_vol_comp): continue

                    pair_name_comp = instrument_names[pair_ticker_comp]
                    metric_value_comp = pair_info_comp.get(metric_name_comp)
                    if metric_value_comp is None or not np.isfinite(metric_value_comp): continue

                    try: metric_display_comp = f"{metric_value_comp:{metric_format_comp}}"
                    except (TypeError, ValueError): metric_display_comp = str(metric_value_comp)

                    row_data_comp = {
                        "Paired Instrument": pair_name_comp,
                        metric_name_comp.replace("_", " ").title(): metric_display_comp,
                        "Overlap (%)": f"{overlap_pct_comp:.1f}",
                        # Update column name in results table
                        "Spread Std Dev": f"{spread_vol_comp:.3f}" if pd.notna(spread_vol_comp) else "N/A",
                        "_ticker": pair_ticker_comp
                    }

                    if extra_metric_name_comp and extra_metric_name_comp in pair_info_comp:
                         extra_value_comp = pair_info_comp[extra_metric_name_comp]
                         extra_display_value_comp = "N/A"
                         if extra_value_comp is not None:
                             try:
                                 if np.isinf(extra_value_comp): extra_display_value_comp = "inf"
                                 else: extra_display_value_comp = f"{extra_value_comp:{extra_metric_format_comp}}"
                             except (TypeError, ValueError): extra_display_value_comp = str(extra_value_comp)
                         row_data_comp[extra_metric_name_comp.replace("_", " ").title()] = extra_display_value_comp

                    filtered_display_data_comp.append(row_data_comp)

                # Sort and store filtered results
                if filtered_display_data_comp:
                    filtered_display_data_comp = sorted(
                        filtered_display_data_comp,
                        key=lambda x: float(x[metric_name_comp.replace("_", " ").title()]) if x[metric_name_comp.replace("_", " ").title()] not in ["N/A", "inf"] else (np.inf if sort_ascending_comp else -np.inf),
                        reverse=not sort_ascending_comp
                    )
                    results_df_comp = pd.DataFrame(filtered_display_data_comp)

                    # Define column order dynamically
                    column_order_comp = ["Paired Instrument"]
                    if metric_name_comp: column_order_comp.append(metric_name_comp.replace("_", " ").title())
                    if extra_metric_name_comp and extra_metric_name_comp.replace("_", " ").title() in results_df_comp.columns:
                        column_order_comp.append(extra_metric_name_comp.replace("_", " ").title())
                    # Update column name in display order
                    column_order_comp.extend(["Overlap (%)", "Spread Std Dev"])

                    existing_cols_in_order_comp = [col for col in column_order_comp if col in results_df_comp.columns]
                    results_df_display_comp = results_df_comp[existing_cols_in_order_comp] # No ticker needed here

                    st.dataframe(results_df_display_comp, use_container_width=True, hide_index=True)
                else:
                    st.info(f"✅ No pairs passed the filters for {tech_short_name}.")
            else:
                 st.info(f"✅ No pairs initially found for {tech_short_name} before filtering.")
            st.divider()


    with tab3:
        st.header("❓ Help / Documentation")

        st.subheader("🚀 Quickstart Guide")
        st.markdown("""
        1.  **Configure Sidebar:**
            *   Select the **Data Source** ('Norgate Data' or 'Uploaded File').
            *   If 'Uploaded File', use the **File Uploader** to select your CSV/XLSX file.
            *   Select the **Date Range** for the analysis (data outside this range will be excluded).
            *   Choose the primary **Instrument** you want to find pairs for (from Norgate or your file's columns).
            *   Select the **Analysis Technique(s)**:
                *   For the "Pair Analysis" tab, choose one technique.
                *   For the "Technique Comparison" tab, select multiple techniques.
            *   Adjust **Technique Parameters** (like p-value, correlation threshold, number of clusters) if applicable for the selected technique(s).
            *   Set **Advanced Filters** like Minimum Data Overlap and Spread Std Dev Range.
        2.  **View Results:**
            *   Go to the **"Pair Analysis"** tab to see detailed results and visualizations for the single selected technique.
            *   Go to the **"Technique Comparison"** tab to see filtered results from multiple techniques side-by-side.
        3.  **Explore:** Use the visualizations and tables to understand the relationships between instruments based on the chosen methods and filters.
        """)

        st.subheader("💾 Data Sources")
        st.markdown("""        
        *   **Norgate Data:** Uses the `norgatedata` library to fetch historical futures data based on symbols defined in the `TICKER_MAP` within the `app.py` script and present in the specified Norgate watchlist (`DEFAULT_WATCHLIST`). Requires `norgatedata` library and Norgate Data subscription.
        *   **Uploaded File:** Allows you to upload your own time series data in CSV or XLSX format.
            *   **Expected Format:**
                *   The **first column** must contain the dates or timestamps. These should be parsable by pandas (e.g., 'YYYY-MM-DD', 'MM/DD/YYYY HH:MM:SS').
                *   **Subsequent columns** should contain the price data for each instrument.
                *   The **header row** should contain the names of the instruments (these will be used for selection).
                *   Data should be numeric. Non-numeric values will be treated as missing.
            *   **Processing:** The app will attempt to parse dates, convert prices to numbers, forward-fill missing values, and align the data across all instruments.
        """)

        st.subheader("⚙️ Methodology Overview")
        st.markdown("""        
        This tool uses several methods to identify potential pairs.See the **Technique Comparison Guide** table below and the **sidebar expanders** for detailed descriptions, pros, cons, and use cases for each technique.

        *   **Data Used:** Note whether a technique uses raw Prices, Normalize Prices, or Returns.
        *   **Handles Non-Stationary Prices?:** Indicates if the technique can be applied directly to non-stationary price series.
        *   **Filters:** The "Overlap (%)" filter ensures pairs have sufficient common data within the date range. The "Spread Volatility" filter assesses the stability of the difference between the normalized prices of the pair.
        """)

        # --- Technique Comparison Table (Re-used from original code) ---
        st.subheader("📘 Technique Comparison Guide")
        table_data = []
        for tech_name in TECHNIQUES: # Use the defined order
            if tech_name in TECHNIQUE_INFO:
                info = TECHNIQUE_INFO[tech_name]
                table_data.append({
                    "Technique": tech_name.split('(')[0].strip(), # Cleaner name
                    "Data Used": info["data_used"],
                    "Handles Non-Stationary Prices?": "Yes" if info["stationarity_note"].startswith("Specifically") or info["stationarity_note"].startswith("Works directly") else ("Indirectly (via returns)" if info["stationarity_note"].startswith("Applies to returns")else "N/A"),
                    "Description": info["brief_description"],
                    "Ideal Use Case": info["use_case"],
                })

        comparison_df = pd.DataFrame(table_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)


        st.subheader("💡 Frequently Asked Questions (FAQ)")
        st.markdown("""        
        *   **Why are some Norgate symbols skipped?**
            Symbols listed in your Norgate watchlist but *not* defined in the `TICKER_MAP` dictionary at the top of the `app.py` script will be ignored when using the 'Norgate Data' source. Update the `TICKER_MAP` to include them. Also, symbols with insufficient or invalid data within the selected date range might be skipped during data fetching or processing.
        *   **Why did my file upload fail or result in no data?**
            Check the **Data Sources** section above for the expected file format. Common issues include:
                *   Incorrect file type (only CSV/XLSX supported).
                *   Date data not in the first column or in an unreadable format.
                *   Price data is non-numeric (e.g., contains text, currency symbols).
                *   The file is empty or contains too many missing values that cannot be filled.
                *   No data within the selected Start/End date range in the sidebar.
        *   **What does 'Half-Life' mean (Cointegration)?**
            It estimates the time (in periods, matching the input data frequency - likely days) it takes for the spread between two cointegrated assets to revert halfway back to its long-term mean after a deviation. A shorter half-life suggests faster mean reversion. Infinite half-life means the calculated relationship is not mean-reverting based on the test.
        *   **How to interpret P-value (Cointegration)?**
            The p-value indicates the probability of observing the data (or more extreme data) if the two series were *not* cointegrated. A low p-value (e.g., < 0.05) provides statistical evidence *against* the null hypothesis (no cointegration), suggesting the series *are* likely cointegrated.
        *   **What is Spread Std Dev (Normalized)?** # Updated FAQ item - Renamed from Spread Volatility
            It measures the standard deviation of the difference between the *normalized* prices (Z-scores) of the two instruments in a pair. A lower value (e.g., closer to 0) suggests the spread tends to stay closer to its average value, indicating more stability. A value of 1.0 would mean the spread typically fluctuates by about 1 standard deviation unit (in Z-score terms).
        *   **Which technique is best?**
            There's no single "best" technique. The ideal choice depends on your trading strategy and goals (e.g., long-term mean reversion vs. short-term correlation, tolerance for time lags). Use the **Technique Comparison Guide** and the **Flowchart** below to help decide. Experimenting with different techniques is recommended.
        """)

        # --- Flowchart for Technique Selection (Re-used from original code) ---
        st.subheader("🗺️ Technique Selection Flowchart")
        st.info("Use this guide to help choose the best technique for your objective:")
        try:
            from streamlit_agraph import agraph, Node, Edge, Config

            nodes = []
            edges = []

            # Define Nodes (id, label, shape, color)
            nodes.append(Node(id="A", label="Start: What's your primary goal?", shape="box"))
            nodes.append(Node(id="B", label="Long-term Mean Reversion?", shape="diamond")) # Decision
            nodes.append(Node(id="C", label="Cointegration", shape="box", color="#f9f")) # Result
            nodes.append(Node(id="D", label="Short-term Directional Co-movement?", shape="diamond")) # Decision
            nodes.append(Node(id="E", label="Correlation (Returns)", shape="box", color="#ccf")) # Result
            nodes.append(Node(id="F", label="Find Visually Similar Chart Shapes?", shape="diamond")) # Decision
            nodes.append(Node(id="G", label="Allow for Time Lags/Shifts?", shape="diamond")) # Decision
            nodes.append(Node(id="H", label="Similarity (DTW)", shape="box", color="#cfc")) # Result
            nodes.append(Node(id="I", label="Simple Geometric Distance OK?", shape="diamond")) # Decision
            nodes.append(Node(id="J", label="Distance (SSD)", shape="box", color="#ffc")) # Result
            nodes.append(Node(id="K", label="Explore Complex / Non-linear Dependence?", shape="diamond")) # Decision
            nodes.append(Node(id="L", label="Mutual Information (Prices)", shape="box", color="#fcc")) # Result
            nodes.append(Node(id="M", label="Discover Groups of Similar Assets First?", shape="diamond")) # Decision
            nodes.append(Node(id="N", label="Clustering (K-Means + SSD Rank)", shape="box", color="#cff")) # Result
            nodes.append(Node(id="O", label="End: Re-evaluate Goal / Explore", shape="box"))
            nodes.append(Node(id="P", label="Use Result", shape="ellipse")) # Terminator

            # Define Edges (source_id, target_id, label)
            edges.append(Edge(source="A", target="B", label=""))
            edges.append(Edge(source="B", target="C", label="Yes"))
            edges.append(Edge(source="B", target="D", label="No"))
            edges.append(Edge(source="D", target="E", label="Yes"))
            edges.append(Edge(source="D", target="F", label="No"))
            edges.append(Edge(source="F", target="G", label="Yes"))
            edges.append(Edge(source="F", target="K", label="No")) # Link F 'No' to K
            edges.append(Edge(source="G", target="H", label="Yes"))
            edges.append(Edge(source="G", target="I", label="No"))
            edges.append(Edge(source="I", target="J", label="Yes"))
            edges.append(Edge(source="I", target="K", label="No")) # Link I 'No' to K
            edges.append(Edge(source="K", target="L", label="Yes"))
            edges.append(Edge(source="K", target="M", label="No"))
            edges.append(Edge(source="M", target="N", label="Yes"))
            edges.append(Edge(source="M", target="O", label="No"))

            # Edges leading to the final result node
            edges.append(Edge(source="C", target="P", label=""))
            edges.append(Edge(source="E", target="P", label=""))
            edges.append(Edge(source="H", target="P", label=""))
            edges.append(Edge(source="J", target="P", label=""))
            edges.append(Edge(source="L", target="P", label=""))
            edges.append(Edge(source="N", target="P", label=""))


            # Configure graph layout and appearance
            # Ref: https://visjs.github.io/vis-network/docs/network/layout.html
            # Ref: https://visjs.github.io/vis-network/docs/network/nodes.html
            # Ref: https://visjs.github.io/vis-network/docs/network/edges.html
            config = Config(width='100%',  # Use full available width (string percentage)
                            height=750,   # Increased height for better vertical spacing
                            directed=True,
                            physics=False, # Keep physics disabled for static layout
                            hierarchical={
                                "enabled": True,
                                "levelSeparation": 200,  # Increase vertical distance between levels
                                "nodeSpacing": 150,      # Horizontal distance between nodes on same level
                                "treeSpacing": 250,      # Distance between different trees (if graph splits)
                                "direction": "UD",       # UD = Up-Down layout
                                "sortMethod": "directed" # Try to respect edge directions for layout
                            },
                            # Node defaults (can be overridden per node)
                            node={'shape': 'box',        # Default shape
                                  'margin': 10,          # Padding inside nodes
                                  'font': {'size': 12}},
                            # Edge defaults
                            edge={'font': {'align': 'top', 'size': 10},
                                  'arrows': 'to'},       # Ensure arrows are shown
                            # Interaction options
                            interaction={'navigationButtons': True, # Add zoom/pan buttons
                                         'tooltipDelay': 300,
                                         'hover': True}         # Enable hover effects
            )

            # Display the graph
            return_value = agraph(nodes=nodes, edges=edges, config=config) # Assign return value

        except ImportError:
            st.error("❌ The 'streamlit-agraph' library is not installed. Please install it: pip install streamlit-agraph")
        except Exception as e:
            st.error(f"❌ Failed to render flowchart using streamlit-agraph: {e}")
            st.error(traceback.format_exc())


# --- Fallback messages if data loading failed earlier ---
# (Modified slightly to account for data source selection)
if data_source == "Norgate Data" and 'mapped_symbols' in locals() and not mapped_symbols:
     # Error already shown if no mapped symbols for Norgate
     pass
elif data_source == "Uploaded File" and uploaded_file is None:
     # Info message already shown about needing to upload
     pass
elif price_data is None: # General catch for other None cases
    st.error("❌ Failed to load or process data. Check data source settings and file content.")
elif price_data.empty: # General catch for empty dataframe after processing
    st.warning("⚠️ No price data available for the selected source, symbols, and date range after processing.")

