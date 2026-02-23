"""
 Data Exploration Agent

Autonomously generates comprehensive data exploration code for MMM.

"""
import logging
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend, RAGBackend
from cerebro.spec.schema import MMMSpec

logger = logging.getLogger(__name__)


class DataExplorationAgent(BaseAgent):
    """
    Writes comprehensive data exploration code.
    
    Generates functions for:
    - Time series analysis
    - Autocorrelation checks
    - Channel spending patterns
    - Cross-correlations
    - Missing data analysis
    - Outlier detection
    """
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True):
        super().__init__(llm, "DataExplorationAgent")
        self.rag = RAGBackend() if use_rag else None
    
    def _get_reasoning_prompt(self, spec: MMMSpec, context: dict) -> str:
        """Generate COT reasoning prompt for data exploration"""
        
        # Extract actual data inspection results
        date_sample = context.get('date_sample_values')
        date_dtype = context.get('date_dtype')
        date_format_hint = context.get('date_format_hint')
        is_panel_data = context.get('is_panel_data', False)
        panel_id_column = context.get('panel_id_column')
        
        # Build data inspection section
        data_inspection = ""
        if date_sample is not None:
            data_structure = "PANEL DATA (multiple observations per time period)" if is_panel_data else "TIME SERIES (one observation per time period)"
            panel_info = f"\n- Panel ID column: {panel_id_column}" if panel_id_column else ""
            
            data_inspection = f"""
ACTUAL DATA INSPECTION (Use this - don't guess!):
- Sample values from '{spec.date_column}': {date_sample}
- Data type: {date_dtype}
- Inferred format: {date_format_hint}
- Data structure: {data_structure}{panel_info}
"""
        
        return f"""
================================================================================
                    MARKETING MIX MODELING (MMM) CONTEXT
================================================================================

You are building a MARKETING MIX MODEL (MMM) - a statistical model that quantifies 
the incremental impact of marketing channels on business outcomes (sales, visits, etc.).

 MMM GOAL: Decompose total outcome into contributions from:
   - Marketing channels (paid media: TV, digital ads, social, etc.)
   - Control variables (pricing, promotions, seasonality, holidays)
   - Baseline (organic demand)

 KEY MMM CONCEPTS YOU MUST UNDERSTAND:

1. ADSTOCK EFFECT: Marketing impact persists over time (ads today -> sales next week)
   -> Need lag analysis (cross-correlation) to find optimal lag windows

2. SATURATION: Diminishing returns at high spend (first $1000 >> last $1000)
   -> Need scatter plots (spend vs outcome) to visualize response curves

3. MULTICOLLINEARITY: Channels often launch together (e.g., TV + Digital)
   -> CRITICAL: Check correlation matrix & VIF. High correlation = can't separate effects!

4. ZERO-SPEND PERIODS: Channels may be off for weeks/months
   -> Need to identify: box plots, time series plots per channel

5. SEASONALITY: Weekly/monthly patterns (holidays, weekends, summer vs winter)
   -> Seasonal decomposition essential

6. DATA QUALITY ISSUES IN MMM:
   - Inconsistent spend (e.g., sudden spikes = campaign launches)
   - Missing data (channels not tracked consistently)
   - Outliers (Super Bowl ads, Black Friday)
   
================================================================================

YOUR TASK: Exploratory Data Analysis (EDA) for MMM

You will generate Python code to explore {len(spec.channels)} marketing channels 
and their relationship with the outcome '{spec.outcome}'.

SPEC INFO:
- Date column: {spec.date_column}
- Time unit: {spec.time_unit}
- Outcome: {spec.outcome}
- Channels ({len(spec.channels)}): {', '.join([ch.name for ch in spec.channels][:5])}{'...' if len(spec.channels) > 5 else ''}
- Controls: {len(spec.controls) if spec.controls else 0}

{data_inspection}

================================================================================

 REASONING QUESTIONS (Think step-by-step about MMM-specific needs):

1. DATA STRUCTURE: Is this panel data or time series?
   - DETECTED: {data_inspection if date_sample else "NOT AVAILABLE"}
   - If PANEL DATA (multiple DMAs/regions per time period):
     * DO NOT set date as index (not unique!)
     * DO NOT use df.asfreq()
     * Keep date as regular column
     * For time series plots: AGGREGATE first with df.groupby(DATE_COLUMN).sum()
   - If TIME SERIES (one observation per time period):
     * OK to set date as index
     * OK to use df.asfreq()

2. DATE PARSING: Based on ACTUAL sample values, how to parse '{spec.date_column}'?
   - Sample values: {date_sample if date_sample else "NOT AVAILABLE"}
   - Data type: {date_dtype if date_dtype else "UNKNOWN"}
   - FORMAT HINT: {date_format_hint if date_format_hint else "NOT AVAILABLE"}
   
   PARSING OPTIONS:
   - Sequential week numbers (0-108): pd.to_datetime('2023-01-01') + pd.to_timedelta(val, unit='W')
   - YYYYWW integer (202301): pd.to_datetime(str(val) + '-1', format='%Y%W-%w')
   - ISO week string ('2023-W01'): pd.to_datetime(val + '-1', format='%Y-W%W-%w')
   - Date string ('2023-01-15'): pd.to_datetime(val)
   
   -> USE THE FORMAT HINT ABOVE!

3. SEASONALITY: What seasonal period for decomposition?
   - time_unit='{spec.time_unit}' -> period = ?
   - week -> 52, month -> 12, day -> 365, quarter -> 4

4. LIBRARY-SPECIFIC API USAGE - CRITICAL:
   
   For seasonal_decompose (statsmodels):
   - decomposition = seasonal_decompose(ts_data, model='additive', period=52)
   - DO NOT use: decomposition.plot(ax=axes[0])  # ← WRONG! No ax parameter!
   - CORRECT way: decomposition.observed.plot(ax=axes[0])  # Plot each component
   - Components: decomposition.observed, .trend, .seasonal, .resid
   - Each component is a pandas Series with .plot(ax=...) method
   
   For matplotlib subplots:
   - Create axes: fig, axes = plt.subplots(4, 1, figsize=(16, 12))
   - Plot to specific axis: df[col].plot(ax=axes[0])
   - Always use plt.savefig() then plt.close()

5. MMM-SPECIFIC VISUALIZATIONS: What plots reveal MMM model assumptions?

   A. TIME SERIES PLOTS (Reveal trends, campaigns, seasonality):
      - Outcome over time (is there a trend? seasonal pattern?)
      - Each channel spend over time (when are campaigns active?)
      - Decomposition (trend + seasonal + residual)
      - Rolling statistics (4-week, 13-week, 52-week moving averages)
      
   B. LAG ANALYSIS (Find adstock/carry-over effects):
      - ACF/PACF of outcome (autocorrelation)
      - Cross-correlation (CCF) between channels and outcome (lag 0, 1, 2, 3, 4 weeks)
      -> CRITICAL FOR MMM: When does ad spend impact outcome?
      
   C. CORRELATION & MULTICOLLINEARITY (Can we separate channel effects?):
      - Correlation heatmap (all channels + controls + outcome)
      - VIF (Variance Inflation Factor) table
      -> HIGH CORRELATION = PROBLEM! Model can't attribute effects correctly.
      
   D. CHANNEL SPEND PATTERNS (Zero-spend, saturation, outliers):
      - Box plots per channel (outliers, zero-spend)
      - Scatter: each channel vs outcome (response curves, saturation)
      - Spend distribution histograms (are channels always-on or pulsing?)
      
   E. OUTLIER DETECTION (Data quality):
      - Outcome outliers (unusual sales days)
      - Channel spend outliers (Super Bowl, Black Friday)

5. EXECUTION ORDER: What sequence ensures correct analysis?
   Step 1: Load data, validate columns
   Step 2: Parse dates (handle panel vs time series correctly!)
   Step 3: Time series setup (index if not panel, frequency)
   Step 4: Descriptive stats (mean, median, skew, zeros per channel)
   Step 5: Time series plots (outcome, channels, decomposition, rolling stats)
   Step 6: ACF/PACF (autocorrelation)
   Step 7: Correlation matrix, VIF
   Step 8: Channel scatter plots (vs outcome)
   Step 9: Outlier detection
   Step 10: Save all plots with plt.savefig() + plt.close()

6. POTENTIAL MMM-SPECIFIC ISSUES:
   - Panel data mistakenly treated as time series (set_index fails!)
   - Date parsing errors (YYYYWW vs sequential weeks)
   - High multicollinearity (channels always launch together)
   - Zero-inflated channels (off 50% of the time)
   - Outliers dominating correlation (need robust correlation)

7. VALIDATION CHECKS:
   - Dates parsed correctly? (DatetimeIndex or regular column if panel)
   - All channels numeric? No NaNs in outcome?
   - Frequency set (if time series, not panel)?
   - No duplicate dates (if time series)?
   - VIF < 10 for all channels? (multicollinearity check)

OUTPUT FORMAT: Return ONLY valid JSON with this structure:

{{
    "data_format": "Panel data with weekid column containing sequential week numbers 0-108",
    "date_parsing": "df['{spec.date_column}'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df['{spec.date_column}'], unit='W')",
    "frequency": "W",
    "period_for_decompose": 52,
    "is_panel_data": {str(is_panel_data).lower()},
    "execution_order": [
        "Load CSV and validate columns exist",
        "Parse date column using inferred format",
        "Set up time series (index + freq if not panel, else keep as column)",
        "Calculate descriptive statistics for all channels",
        "Plot outcome and channel time series",
        "Seasonal decomposition of outcome",
        "ACF/PACF autocorrelation plots",
        "Correlation heatmap (channels + controls + outcome)",
        "VIF multicollinearity check",
        "Scatter plots (each channel vs outcome)",
        "Outlier detection (box plots, z-scores)",
        "Save all plots with savefig() and close()"
    ],
    "potential_issues": [
        "Panel data with duplicate dates per period",
        "High multicollinearity between channels (VIF > 10)",
        "Zero-inflated channels (50%+ zeros)",
        "Outliers from campaign events (Super Bowl, holidays)"
    ],
    "validation": [
        "Date column is datetime",
        "All channels are numeric",
        "No NaNs in outcome",
        "VIF < 10 for identifiability"
    ],
    "mmm_specific_plots": {{
        "time_series": ["outcome_over_time.png", "channels_over_time.png", "trend_decomposition.png", "rolling_stats.png"],
        "lag_analysis": ["acf.png", "pacf.png"],
        "multicollinearity": ["correlation_matrix.png"],
        "response_curves": ["channel_vs_outcome_scatter.png"],
        "data_quality": ["outlier_detection.png", "spend_distributions.png"]
    }}
}}

Think step-by-step about MMM modeling requirements. Output ONLY the JSON (no other text).
"""
    
    def generate_eda_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate data exploration code autonomously with COT reasoning"""
        logger.info(" DataExplorationAgent generating EDA code...")
        
        # ========== LOAD SAMPLE DATA FOR REASONING ==========
        date_sample = None
        date_dtype = None
        date_format_hint = None
        is_panel_data = False
        panel_id_column = None
        
        if data_path:
            try:
                import pandas as pd
                import os
                if os.path.exists(data_path):
                    # Load just first 100 rows to inspect date format AND data structure
                    df_sample = pd.read_csv(data_path, nrows=100)
                    if spec.date_column in df_sample.columns:
                        date_sample = df_sample[spec.date_column].head(5).tolist()
                        date_dtype = str(df_sample[spec.date_column].dtype)
                        
                        # Check if panel data (look for ID columns first, more robust)
                        for col in df_sample.columns:
                            if col.lower() in ['id', 'dmacode', 'region_id', 'store_id', 'customer_id', 'group_id']:
                                is_panel_data = True
                                panel_id_column = col
                                logger.info(f"[DATA INSPECTION] PANEL DATA DETECTED! Found ID column: {panel_id_column}")
                                break
                        
                        # If no ID column found, check for duplicate dates
                        if not is_panel_data:
                            n_rows = len(df_sample)
                            n_unique_dates = df_sample[spec.date_column].nunique()
                            if n_rows > n_unique_dates:
                                is_panel_data = True
                                logger.info(f"[DATA INSPECTION] PANEL DATA DETECTED! {n_rows} rows but only {n_unique_dates} unique dates")
                        
                        # Try to infer format
                        first_val = date_sample[0]
                        last_val = date_sample[-1]
                        if isinstance(first_val, int) or isinstance(first_val, float):
                            if 200000 < first_val < 210000:
                                date_format_hint = "YYYYWW integer (like 202301) - use pd.to_datetime(str(val) + '-1', format='%Y%W-%w')"
                            elif 20000000 < first_val < 21000000:
                                date_format_hint = "YYYYMMDD integer (like 20230115) - use pd.to_datetime(str(val), format='%Y%m%d')"
                            elif 0 <= first_val < 1000:
                                # Sequential week numbers
                                date_format_hint = f"Sequential week numbers ({first_val}-{last_val}) - use pd.to_datetime('2023-01-01') + pd.to_timedelta(val, unit='W')"
                        elif isinstance(first_val, str):
                            if '-W' in str(first_val):
                                date_format_hint = "ISO week string (like '2023-W01') - use pd.to_datetime(val + '-1', format='%Y-W%W-%w')"
                            elif '-' in str(first_val) and len(str(first_val)) == 10:
                                date_format_hint = "YYYY-MM-DD string - use pd.to_datetime(val)"
                        
                        logger.info(f"[DATA INSPECTION] Date column: {spec.date_column}")
                        logger.info(f"[DATA INSPECTION] Sample values: {date_sample}")
                        logger.info(f"[DATA INSPECTION] Data type: {date_dtype}")
                        logger.info(f"[DATA INSPECTION] Inferred format: {date_format_hint}")
            except Exception as e:
                logger.warning(f"Could not load sample data: {e}")
        
        # ========== PHASE 1: CHAIN OF THOUGHT REASONING ==========
        reasoning = self._reason_about_task(spec, {
            'data_path': data_path,
            'date_sample_values': date_sample,
            'date_dtype': date_dtype,
            'date_format_hint': date_format_hint,
            'is_panel_data': is_panel_data,
            'panel_id_column': panel_id_column
        })
        
        if reasoning:
            logger.info(f"[COT] Data format: {reasoning.get('data_format', 'N/A')}")
            logger.info(f"[COT] Operations needed: {len(reasoning.get('execution_order', []))} steps")
        
        # Get RAG examples
        rag_context = self._get_eda_examples() if self.rag else ""
        
        # Extract spec details
        channel_names = [ch.name for ch in spec.channels]
        control_names = [c.name if hasattr(c, 'name') else c for c in spec.controls] if spec.controls else []
        
        # Build prompt with reasoning guidance
        data_info = f"\nDATA PATH: {data_path}" if data_path else ""
        
        # Add reasoning context to prompt
        reasoning_context = ""
        if reasoning:
            reasoning_context = f"""
========== REASONING ANALYSIS (USE THIS!) ==========

DATA FORMAT UNDERSTANDING:
{reasoning.get('data_format', 'N/A')}

DATE PARSING STRATEGY:
{reasoning.get('date_parsing', 'N/A')}

REQUIRED FREQUENCY:
{reasoning.get('frequency', 'N/A')}

EXECUTION ORDER (FOLLOW THIS EXACTLY):
"""
            for i, step in enumerate(reasoning.get('execution_order', []), 1):
                reasoning_context += f"\nStep {i}: {step}"
            
            reasoning_context += f"""

POTENTIAL ISSUES TO HANDLE:
{', '.join(reasoning.get('potential_issues', []))}

VALIDATION CHECKS:
{', '.join(reasoning.get('validation', []))}

=======================================================

"""
        
        prompt = f"""OUTPUT ONLY PYTHON CODE. No markdown, no explanations, no prose. Just Python code and # comments.

{reasoning_context}

TARGET: 250-300 lines of detailed, production-grade Python code.

{rag_context}

========== MANDATORY: COPY-PASTE THESE EXACT VARIABLE ASSIGNMENTS ==========

At the top of your code, COPY these EXACT lines (DO NOT modify, DO NOT regenerate):

# Spec-provided values - DO NOT CHANGE
OUTCOME = '{spec.outcome}'
DATE_COLUMN = '{spec.date_column}'
CHANNEL_NAMES = {channel_names}
CONTROL_NAMES = {control_names}
TIME_UNIT = '{spec.time_unit}'

DO NOT REGENERATE THESE USING LOOPS OR FORMAT STRINGS.
DO NOT USE range() OR .format() TO CREATE COLUMN NAMES.
USE THE EXACT LISTS PROVIDED ABOVE.

{data_info}

{' PANEL DATA DETECTED! See below for CRITICAL handling instructions.' if is_panel_data else ''}

Write COMPREHENSIVE EDA functions for MMM:

# load_data(file_path) - DETAILED: Load CSV, validate columns using CHANNEL_NAMES and CONTROL_NAMES, check dtypes, memory usage, duplicates, missing value report, date parsing. Use DATE_COLUMN for date parsing.
  
  {'� CRITICAL - PANEL DATA HANDLING �' if is_panel_data else ''}
  {'This dataset has PANEL DATA (ID column: ' + str(panel_id_column) + ').' if is_panel_data else ''}
  {'RULES FOR PANEL DATA:' if is_panel_data else ''}
  {'1. After date parsing, DO NOT use df.set_index(DATE_COLUMN)' if is_panel_data else ''}
  {'2. DO NOT use df.asfreq()' if is_panel_data else ''}
  {'3. Keep date column as a REGULAR COLUMN, not an index' if is_panel_data else ''}
  {'4. For time series plots, aggregate first: df.groupby(DATE_COLUMN).sum()' if is_panel_data else ''}
  {'5. Return df with date as a column, NOT as index' if is_panel_data else ''}
# descriptive_statistics(df, channels) - DETAILED: For EACH channel - mean, median, std, min, max, percentiles (5,25,50,75,95,99), skewness, kurtosis, coefficient of variation, spend concentration
# time_series_analysis(df, outcome, channels) - DETAILED: Trend decomposition, seasonality (weekly/monthly), rolling stats (7/30 day), ACF/PACF for outcome AND all channels, stationarity tests (ADF, KPSS), changepoint detection. MUST CREATE matplotlib plot with plt.savefig('timeseries_analysis.png', dpi=300, bbox_inches='tight') followed by plt.close()
# correlation_analysis(df, channels, outcome) - DETAILED: Pearson/Spearman/Kendall, VIF calculation for multicollinearity, condition number, cross-correlation with lags. MUST CREATE matplotlib heatmap with plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight') followed by plt.close()
# outlier_detection(df, channels) - DETAILED: Z-score (threshold 3), IQR method, Isolation Forest, outlier counts per channel. MUST CREATE matplotlib plot with plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight') followed by plt.close()

Each function: full docstring, type hints, try/except error handling, logging statements. CRITICAL: Each analysis function MUST create matplotlib/seaborn visualizations and SAVE them using plt.savefig('filename.png', dpi=300, bbox_inches='tight') immediately followed by plt.close().

CRITICAL API USAGE EXAMPLES:

EXAMPLE 1 - GOOD (CORRECT seasonal_decompose usage):
```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_data, model='additive', period=52)
fig, axes = plt.subplots(4, 1, figsize=(16, 12))
decomposition.observed.plot(ax=axes[0])  # ← CORRECT: .observed.plot()
decomposition.trend.plot(ax=axes[1])      # ← CORRECT: .trend.plot()
decomposition.seasonal.plot(ax=axes[2])   # ← CORRECT: .seasonal.plot()
decomposition.resid.plot(ax=axes[3])      # ← CORRECT: .resid.plot()
plt.savefig('decomposition.png', dpi=300)
plt.close()
```
^ CORRECT: Plot each component separately

EXAMPLE 2 - BAD (WRONG seasonal_decompose usage):
```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_data, model='additive', period=52)
fig, axes = plt.subplots(4, 1, figsize=(16, 12))
decomposition.plot(ax=axes[0])  # ← WRONG! decomposition.plot() doesn't accept ax parameter!
```
^ REJECTED: decomposition.plot() creates its own figure, doesn't accept ax

CRITICAL - YOU MUST INCLUDE THIS ENTRY POINT FUNCTION (using exact spec values):

def run_exploration(data_path: str) -> dict:
    '''Main entry point for exploration module.
    
    Args:
        data_path (str): Path to input CSV file
        
    Returns:
        dict: Results with keys 'summary', 'plots_saved', 'warnings'
    '''
    df = load_data(data_path)
    
    # Use the EXACT variables defined at the top
    channels = CHANNEL_NAMES  # Do NOT regenerate
    outcome = OUTCOME
    date_col = DATE_COLUMN
    
    results = {{}}
    results['descriptive'] = descriptive_statistics(df, channels)
    results['timeseries'] = time_series_analysis(df, outcome, channels)
    results['correlation'] = correlation_analysis(df, channels, outcome)
    results['outliers'] = outlier_detection(df, channels)
    results['summary'] = 'Exploration completed successfully'
    results['plots_saved'] = ['eda_plots.png']
    
    return results

START WITH: import pandas as pd
END WITH: return results (inside run_exploration function)

Output 250-300 lines of valid Python code only. Every line must be Python or # comment.
MUST include the run_exploration(data_path: str) -> dict function as the main entry point."""

        # Agent writes code with STREAMING
        print("\n" + "="*80)
        print(" DATA EXPLORATION CODE (streaming):")
        print("="*80 + "\n")
        
        full_code = ""
        for token in self.llm.reason(prompt, stream=True):
            print(token, end="", flush=True)
            full_code += token
        
        print("\n\n" + "="*80)
        code = full_code
        # Aggressive cleanup - remove all non-Python lines
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip markdown fences
            if stripped.startswith('```'):
                continue
            # Skip prose (sentences ending with period, no Python keywords)
            if stripped and not any([
                line.startswith(' '),
                line.startswith('\t'),
                line.startswith('#'),
                line.startswith('def '),
                line.startswith('class '),
                line.startswith('import '),
                line.startswith('from '),
                line.startswith('@'),
                '=' in line,
                stripped.endswith(':'),
                any(kw in line for kw in ['if ', 'for ', 'while ', 'try:', 'except', 'return ', 'yield ', 'raise '])
            ]) and (
                stripped[0].isdigit() or
                '**' in stripped or
                (stripped.endswith('.') and len(stripped.split()) > 5)
            ):
                continue
            cleaned_lines.append(line)
        code = '\n'.join(cleaned_lines)

        
        # Clean up

        # AGGRESSIVE cleanup
        lines = code.split('\n')
        cleaned = []
        for line in lines:
            s = line.strip()
            # Skip markdown fences
            if s in ['```', '```python', '```py'] or s.startswith('```'):
                continue
            # Skip LLM tokens
            if '<|im_end|>' in line or '<|endoftext|>' in line:
                continue
            # Skip numbered prose
            if s and not line[0].isspace() and s[0].isdigit() and '. ' in s[:4]:
                continue
            # Skip "This function..." or "The function..."
            if s.startswith(('This function', 'The function', 'This code', 'Example usage:')):
                continue
            cleaned.append(line)
        code = '\n'.join(cleaned)
        code = self._clean_code(code)
        
        logger.info(f"[OK] Generated {len(code.splitlines())} lines of EDA code")
        return code
    
    def _get_eda_examples(self) -> str:
        """Get MMM-specific EDA few-shot examples"""
        return """
����������������������������������������������������������������������������������������������������������������������������������������������������������������
                         MMM-SPECIFIC FEW-SHOT EXAMPLES
����������������������������������������������������������������������������������������������������������������������������������������������������������������

EXAMPLE 1: VIF (Variance Inflation Factor) for Multicollinearity Detection

def calculate_vif(data, channel_cols):
    '''Calculate VIF for each channel to detect multicollinearity.
    
    VIF > 10 = high multicollinearity (channels too correlated)
    VIF > 5 = moderate concern
    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Select only channel columns
    X = data[channel_cols].fillna(0)
    
    # Calculate VIF for each channel
    vif_data = []
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({{'feature': col, 'VIF': vif}})
    
    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

# Usage in run_exploration:
vif_df = calculate_vif(df, CHANNEL_NAMES)
logger.info(f"VIF range: {{vif_df['VIF'].min():.2f}} to {{vif_df['VIF'].max():.2f}}")
results['correlation']['vif'] = vif_df.to_dict('records')


EXAMPLE 2: Channel-Specific Time Series Plots (Detect Campaign Patterns)

def plot_channel_timeseries(data, channel_cols, date_col, outcome):
    '''Plot each channel's spend and outcome over time to identify campaigns.'''
    import matplotlib.pyplot as plt
    
    # For panel data: aggregate by date first
    if date_col in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        df_agg = data.groupby(date_col).sum().reset_index()
    else:
        df_agg = data.copy()
    
    # Plot top 6 channels by total spend
    top_channels = data[channel_cols].sum().sort_values(ascending=False).head(6).index.tolist()
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()
    
    for idx, channel in enumerate(top_channels):
        ax = axes[idx]
        ax2 = ax.twinx()
        
        # Plot channel spend
        ax.plot(df_agg[date_col] if date_col in df_agg.columns else df_agg.index, 
                df_agg[channel], color='steelblue', linewidth=2, label='Spend')
        ax.set_ylabel('Channel Spend', color='steelblue')
        ax.tick_params(axis='y', labelcolor='steelblue')
        
        # Plot outcome on secondary axis
        ax2.plot(df_agg[date_col] if date_col in df_agg.columns else df_agg.index,
                 df_agg[outcome], color='coral', linewidth=1, alpha=0.7, label='Outcome')
        ax2.set_ylabel('Outcome', color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')
        
        ax.set_title(f'{{channel}} (Total: ${{df_agg[channel].sum():,.0f}})')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('channels_timeseries_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'channels_timeseries_grid.png'


EXAMPLE 3: Response Curves (Channel Spend vs Outcome Scatter)

def plot_response_curves(data, channel_cols, outcome):
    '''Scatter plots: each channel vs outcome to visualize saturation.'''
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    
    top_channels = data[channel_cols].sum().sort_values(ascending=False).head(8).index.tolist()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
    axes = axes.flatten()
    
    for idx, channel in enumerate(top_channels):
        ax = axes[idx]
        
        # Scatter plot
        x = data[channel].values
        y = data[outcome].values
        ax.scatter(x, y, alpha=0.3, s=10, color='steelblue')
        
        # Fit smoothing line (LOWESS)
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2, label='LOWESS fit')
        
        # Correlation
        corr, pval = pearsonr(x, y)
        
        ax.set_xlabel(f'{{channel}} Spend')
        ax.set_ylabel(outcome)
        ax.set_title(f'{{channel}}\\nCorr: {{corr:.3f}} (p={{pval:.3f}})')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('response_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'response_curves.png'


EXAMPLE 4: Zero-Spend Analysis (Critical for MMM)

def analyze_zero_spend(data, channel_cols):
    '''Identify channels with high zero-spend percentage (pulsing campaigns).'''
    zero_pct = {{}}
    for channel in channel_cols:
        n_zeros = (data[channel] == 0).sum()
        pct_zero = 100 * n_zeros / len(data)
        zero_pct[channel] = {{'n_zeros': n_zeros, 'pct_zero': pct_zero}}
    
    # Sort by % zeros
    zero_df = pd.DataFrame(zero_pct).T.sort_values('pct_zero', ascending=False)
    
    logger.info(f"Channels with >50% zero spend: {{(zero_df['pct_zero'] > 50).sum()}}")
    
    return zero_df.to_dict('index')


EXAMPLE 5: ACF/PACF with Interpretation (Autocorrelation for Adstock)

def plot_acf_pacf(data, outcome, date_col):
    '''Plot ACF/PACF to understand outcome autocorrelation (persistence).'''
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    # For panel data: aggregate by date first
    if date_col in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        y = data.groupby(date_col)[outcome].sum().values
    else:
        y = data[outcome].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # ACF (autocorrelation)
    plot_acf(y, lags=20, ax=ax1, alpha=0.05)
    ax1.set_title(f'ACF: {{outcome}} (Autocorrelation)\\nHigh values = outcome persists over time')
    ax1.set_xlabel('Lag (weeks)')
    
    # PACF (partial autocorrelation)
    plot_pacf(y, lags=20, ax=ax2, alpha=0.05, method='ywm')
    ax2.set_title(f'PACF: {{outcome}} (Partial Autocorrelation)\\nSignificant lags = AR order')
    ax2.set_xlabel('Lag (weeks)')
    
    plt.tight_layout()
    plt.savefig('acf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save separately for easier viewing
    fig2, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
    plot_pacf(y, lags=20, ax=ax, alpha=0.05, method='ywm')
    ax.set_title(f'PACF: {{outcome}}')
    plt.tight_layout()
    plt.savefig('pacf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return ['acf.png', 'pacf.png']


EXAMPLE 6: Seasonal Decomposition (Trend + Seasonality + Residual)

def plot_seasonal_decomposition(data, outcome, date_col, period=52):
    '''Decompose outcome into trend, seasonal, and residual components.'''
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # For panel data: aggregate by date first
    if date_col in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        ts_data = data.groupby(date_col)[outcome].sum()
        ts_data.index = pd.to_datetime(ts_data.index)
    else:
        ts_data = data[outcome]
    
    # Decompose
    decomposition = seasonal_decompose(ts_data, model='additive', period=period, extrapolate_trend='freq')
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), dpi=300)
    
    decomposition.observed.plot(ax=axes[0], color='black')
    axes[0].set_ylabel('Observed')
    axes[0].set_title(f'Seasonal Decomposition: {{outcome}} (period={{period}})')
    
    decomposition.trend.plot(ax=axes[1], color='steelblue')
    axes[1].set_ylabel('Trend')
    
    decomposition.seasonal.plot(ax=axes[2], color='green')
    axes[2].set_ylabel('Seasonal')
    
    decomposition.resid.plot(ax=axes[3], color='red')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    
    for ax in axes:
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trend_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'trend_decomposition.png'


EXAMPLE 7: Correlation Heatmap with Annotations (MMM Multicollinearity Check)

def plot_correlation_heatmap(data, channel_cols, control_cols, outcome):
    '''Correlation heatmap: channels + controls + outcome.'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Select relevant columns
    cols = channel_cols + control_cols + [outcome]
    corr_matrix = data[cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 14), dpi=300)
    
    # Use mask to show only lower triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, 
                square=True, linewidths=0.5, cbar_kws={{"shrink": 0.8}}, ax=ax)
    
    ax.set_title('MMM Correlation Matrix\\n High channel-channel correlation = multicollinearity risk', 
                 fontsize=14)
    
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Identify high correlations (>0.7) between channels
    high_corr_pairs = []
    for i in range(len(channel_cols)):
        for j in range(i+1, len(channel_cols)):
            corr_val = corr_matrix.loc[channel_cols[i], channel_cols[j]]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({{
                    'channel1': channel_cols[i],
                    'channel2': channel_cols[j],
                    'correlation': corr_val
                }})
    
    if high_corr_pairs:
        logger.warning(f" {{len(high_corr_pairs)}} channel pairs with |corr| > 0.7 (multicollinearity risk)")
    
    return 'correlation_matrix.png', high_corr_pairs


EXAMPLE 8: Handling Panel Data (Multiple DMAs/Regions per Week)

# CORRECT approach for panel data:

def load_data(file_path):
    '''Load and validate MMM panel data.'''
    df = pd.read_csv(file_path)
    
    # Parse date (sequential week numbers)
    df[DATE_COLUMN] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df[DATE_COLUMN], unit='W')
    
    # DO NOT set index for panel data (multiple rows per date)
    # Keep date as a regular column
    
    # Check if panel data
    n_rows = len(df)
    n_unique_dates = df[DATE_COLUMN].nunique()
    if n_rows > n_unique_dates:
        logger.info(f"PANEL DATA DETECTED: {{n_rows}} rows, {{n_unique_dates}} unique dates")
    
    return df

# For time series plots with panel data, AGGREGATE FIRST:
def plot_outcome_timeseries_panel(data, outcome, date_col):
    '''Plot outcome over time (panel data requires aggregation).'''
    import matplotlib.pyplot as plt
    
    # Aggregate by date
    df_agg = data.groupby(date_col)[outcome].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.plot(df_agg[date_col], df_agg[outcome], linewidth=2, color='steelblue')
    ax.set_xlabel('Date')
    ax.set_ylabel(outcome)
    ax.set_title(f'{{outcome}} Over Time (Aggregated Across All DMAs)')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outcome_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'outcome_timeseries.png'


EXAMPLE 9: Rolling Statistics (4-week, 13-week, 52-week Moving Averages)

def plot_rolling_stats(data, outcome, date_col):
    '''Plot rolling averages to smooth out noise.'''
    import matplotlib.pyplot as plt
    
    # For panel data: aggregate by date first
    if date_col in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        ts_data = data.groupby(date_col)[outcome].sum().reset_index()
        ts_data = ts_data.set_index(date_col)
    else:
        ts_data = data[[outcome]]
    
    # Calculate rolling stats
    ts_data['MA_4W'] = ts_data[outcome].rolling(window=4, min_periods=1).mean()
    ts_data['MA_13W'] = ts_data[outcome].rolling(window=13, min_periods=1).mean()
    ts_data['MA_52W'] = ts_data[outcome].rolling(window=52, min_periods=1).mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    
    ax.plot(ts_data.index, ts_data[outcome], label='Actual', alpha=0.4, linewidth=1)
    ax.plot(ts_data.index, ts_data['MA_4W'], label='4-week MA', linewidth=2)
    ax.plot(ts_data.index, ts_data['MA_13W'], label='13-week MA (Quarter)', linewidth=2)
    ax.plot(ts_data.index, ts_data['MA_52W'], label='52-week MA (Year)', linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(outcome)
    ax.set_title(f'{{outcome}}: Rolling Averages (Smoothing)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rolling_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'rolling_stats.png'


EXAMPLE 10: Complete run_exploration Structure for MMM

def run_exploration(data_path: str) -> dict:
    '''Main entry point for MMM data exploration.'''
    logger.info(f"Starting MMM exploration: {{data_path}}")
    
    # 1. Load and validate
    df = load_data(data_path)
    
    # 2. Descriptive stats
    descriptive = {{}}
    for channel in CHANNEL_NAMES:
        descriptive[channel] = {{
            'mean': float(df[channel].mean()),
            'median': float(df[channel].median()),
            'std': float(df[channel].std()),
            'min': float(df[channel].min()),
            'max': float(df[channel].max()),
            'pct_zero': float(100 * (df[channel] == 0).sum() / len(df))
        }}
    
    # 3. Time series analysis
    timeseries_plots = []
    timeseries_plots.append(plot_outcome_timeseries_panel(df, OUTCOME, DATE_COLUMN))
    timeseries_plots.append(plot_channel_timeseries(df, CHANNEL_NAMES, DATE_COLUMN, OUTCOME))
    timeseries_plots.append(plot_seasonal_decomposition(df, OUTCOME, DATE_COLUMN, period=52))
    timeseries_plots.append(plot_rolling_stats(df, OUTCOME, DATE_COLUMN))
    
    # 4. Lag analysis
    acf_pacf_plots = plot_acf_pacf(df, OUTCOME, DATE_COLUMN)
    
    # 5. Correlation and multicollinearity
    corr_plot, high_corr = plot_correlation_heatmap(df, CHANNEL_NAMES, CONTROL_NAMES, OUTCOME)
    vif_df = calculate_vif(df, CHANNEL_NAMES)
    
    # 6. Response curves
    response_plot = plot_response_curves(df, CHANNEL_NAMES, OUTCOME)
    
    # 7. Zero-spend analysis
    zero_spend = analyze_zero_spend(df, CHANNEL_NAMES)
    
    # 8. Return results
    results = {{
        'descriptive': descriptive,
        'timeseries_plots': timeseries_plots,
        'acf_pacf_plots': acf_pacf_plots,
        'correlation': {{
            'plot': corr_plot,
            'high_correlations': high_corr,
            'vif': vif_df.to_dict('records')
        }},
        'response_curves': response_plot,
        'zero_spend': zero_spend
    }}
    
    logger.info("MMM exploration complete!")
    return results


KEY PATTERNS FROM EXAMPLES:
 Always check for panel vs time series data structure
 For panel data: aggregate with groupby() before time series plots
 Use plt.savefig() + plt.close() for EVERY plot
 VIF and correlation checks are CRITICAL for MMM
 Zero-spend % per channel reveals campaign patterns
 Response curves show saturation effects
 ACF/PACF show persistence (autocorrelation)

����������������������������������������������������������������������������������������������������������������������������������������������������������������
"""
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown and artifacts"""
        code = code.strip()
        if code.startswith('```python'):
            code = code[len('```python'):].strip()
        if code.startswith('```'):
            code = code[3:].strip()
        if code.endswith('```'):
            code = code[:-3].strip()
        for token in ['<|im_end|>', '<|endoftext|>']:
            code = code.replace(token, '')
        return code.strip()
