"""
 Preprocessing Agent

Autonomously generates data preprocessing code for MMM.

"""
import logging
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend, RAGBackend
from cerebro.spec.schema import MMMSpec

logger = logging.getLogger(__name__)


class PreprocessingAgent(BaseAgent):
    """
    Writes data preprocessing code.
    
    Generates functions for:
    - Missing value imputation
    - Outlier detection and capping
    - Feature scaling
    - Time feature engineering
    - Data quality checks
    """
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True):
        super().__init__(llm, "PreprocessingAgent")
        self.rag = RAGBackend() if use_rag else None
    
    def _get_reasoning_prompt(self, spec: MMMSpec, context: dict) -> str:
        """Generate COT reasoning prompt for preprocessing"""
        return f"""
================================================================================
                    MARKETING MIX MODELING (MMM) CONTEXT
================================================================================

You are preprocessing data for a MARKETING MIX MODEL (MMM).

 MMM GOAL: Build a regression model that attributes business outcome to marketing channels.

 MMM-SPECIFIC PREPROCESSING CHALLENGES:

1. **MARKETING SPEND IS RIGHT-SKEWED**: 
   - Channels have occasional huge spikes (Black Friday, Super Bowl)
   - Most weeks: low-moderate spend
   - Problem: Standard scaling treats spikes as outliers, but they're REAL marketing events!
   
2. **ZERO-INFLATED CHANNELS**:
   - Many channels OFF for weeks/months (50%+ zeros)
   - Problem: Imputation/scaling must preserve zeros (they're not missing!)
   
3. **MULTICOLLINEARITY AFTER FEATURE ENGINEERING**:
   - If you create lags/rolling features, channels become more correlated
   - Problem: Worsens multicollinearity -> model can't attribute correctly
   
4. **PRESERVE INTERPRETABILITY**:
   - MMM stakeholders need to understand: "$1 in Channel A -> X visits"
   - Problem: Complex transformations (StandardScaler -> negative values -> log -> NaN) break interpretability
   
5. **NO DATA LEAKAGE**:
   - Don't use future data (e.g., rolling mean that looks ahead)
   - Panel data: Don't leak across DMAs/regions

================================================================================

YOUR TASK: Preprocess MMM data for regression modeling

SPEC INFO:
- Outcome: {spec.outcome}
- Date column: {spec.date_column}
- Channels: {len(spec.channels)}
- Controls: {len(spec.controls) if spec.controls else 0}
- Time unit: {spec.time_unit}

================================================================================

 MMM-SPECIFIC REASONING QUESTIONS:

1. MISSING VALUES: How to handle gaps in {spec.time_unit} time series?
   
   MMM-APPROPRIATE STRATEGIES:
   - Channels: fillna(0) - missing spend = no spend
   - Controls: forward fill (last known value)
   - Outcome: interpolate (short gaps) or forward fill (long gaps)
   
    AVOID: Mean imputation (hides zero-spend patterns)

2. OUTLIERS: How to handle WITHOUT removing real marketing events?
   
   MMM-APPROPRIATE STRATEGIES:
   - Use PERCENTILE CAPPING (1st/99th) instead of z-score
   - DON'T remove outliers (Super Bowl ads are real!)
   - Consider log transformation ONLY if needed, but be careful:
     * log(0) = -inf �� Need log(x + 1)
     * But log breaks linear interpretability!
   
    CRITICAL: For this iteration, AVOID log transforms to prevent NaN issues!
   
3. SCALING: What type preserves MMM interpretability?
   
   MMM-APPROPRIATE STRATEGIES:
   - MinMaxScaler [0, 1]: BEST - preserves zeros, all positive
   - NO StandardScaler: Creates negative values �� breaks log transforms �� NaN cascade
   - NO multiple scaling steps: StandardScaler + MinMaxScaler = overkill
   
    USE: MinMaxScaler ONLY

4. FEATURE ENGINEERING: What features help MMM modeling?
   
   MMM-USEFUL FEATURES (optional, but be careful):
   - Lag features: channel_lag1, channel_lag2 (adstock proxies)
   - Rolling averages: 4-week rolling mean (smoothing)
   - Trend: time index (captures base trend)
   
    WARNING: 
   - Lags/rolling create NaNs �� MUST fillna(0) or use min_periods
   - Lags increase multicollinearity
   - For now, prefer SIMPLE preprocessing (no complex features)

5. DATA LEAKAGE PREVENTION:
   - Ensure rolling windows don't look ahead (use min_periods)
   - Panel data: Don't leak across DMAs (groupby ID for rolling features)
   - No future data in lags

6. VALIDATION: What checks prevent model failure?
   
   CRITICAL MMM VALIDATIONS:
   - No NaNs remain (models fail!)
   - No infinite values (from log(0) or division by zero)
   - All channels are non-negative after scaling
   - Date column is datetime
   - Output is NUMERIC (no strings)
   
    IMPORTANT: Make validation FORGIVING:
   - If NaN found: fillna(0) instead of raising error
   - If inf found: clip to finite range instead of failing
   - Goal: ROBUST preprocessing that always succeeds

7. OUTPUT FORMAT:
   - MUST return: string path to saved CSV file
   - NOT: DataFrame object (causes type errors in pipeline)
   - File naming: {{input_path}}_preprocessed.csv

8. VISUALIZATIONS: What plots validate preprocessing quality?
   
   MMM-RELEVANT PLOTS:
   - Before/after distributions (box plots for top 6 channels)
   - Outlier detection (scatter showing capped values)
   - Zero-spend preservation (histogram showing % zeros before/after)
   - File names: 'preprocessing_outliers.png', 'preprocessing_distributions.png'

��������������������������������������������������������������������������������������������������������������������������������������������������������������

OUTPUT FORMAT: Return ONLY valid JSON:

{{
    "missing_value_strategy": "Channels: fillna(0), Controls: forward fill, Outcome: interpolate",
    "outlier_strategy": "Percentile capping (1st/99th), NO log transform to avoid NaN",
    "scaling_strategy": "MinMaxScaler ONLY (no StandardScaler, no log transform)",
    "feature_engineering": "Minimal (avoid lags/rolling to prevent NaN and multicollinearity)",
    "data_leakage_checks": [
        "Rolling windows use min_periods",
        "No future data in features",
        "Panel data grouped by ID for rolling features"
    ],
    "output_format": "Save to CSV: {{input_path}}_preprocessed.csv, return path string",
    "validation_checks": [
        "fillna(0) for any remaining NaN",
        "Clip inf to finite range",
        "Verify all numeric columns",
        "Check date is datetime"
    ],
    "execution_order": [
        "Load data",
        "Handle missing values (fillna)",
        "Outlier treatment (percentile capping)",
        "Scaling (MinMaxScaler only)",
        "Feature engineering (if any, with min_periods)",
        "Robust validation (fillna/clip instead of error)",
        "Save to CSV",
        "Return file path string"
    ],
    "visualizations": {{
        "before_after": ["Box plots showing distribution changes"],
        "outliers": ["Scatter showing capped values"],
        "zero_preservation": ["Histogram of % zeros"],
        "filenames": ["preprocessing_outliers.png", "preprocessing_distributions.png"]
    }}
}}

Think step-by-step about MMM preprocessing requirements. Output ONLY the JSON.
"""
    
    def generate_preprocessing_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate preprocessing code autonomously with COT reasoning"""
        logger.info(" PreprocessingAgent generating preprocessing code...")
        
        # ========== LOAD SAMPLE DATA FOR REASONING ==========
        date_sample = None
        date_dtype = None
        
        if data_path:
            try:
                import pandas as pd
                import os
                if os.path.exists(data_path):
                    df_sample = pd.read_csv(data_path, nrows=10)
                    if spec.date_column in df_sample.columns:
                        date_sample = df_sample[spec.date_column].head(5).tolist()
                        date_dtype = str(df_sample[spec.date_column].dtype)
                        logger.info(f"[DATA INSPECTION] Date sample: {date_sample}, dtype: {date_dtype}")
            except Exception as e:
                logger.warning(f"Could not load sample data: {e}")
        
        # ========== PHASE 1: COT REASONING ==========
        reasoning = self._reason_about_task(spec, {
            'data_path': data_path,
            'date_sample_values': date_sample,
            'date_dtype': date_dtype
        })
        
        if reasoning:
            logger.info(f"[COT] Missing value strategy: {reasoning.get('missing_value_strategy', 'N/A')}")
            logger.info(f"[COT] Output format: {reasoning.get('output_format', 'N/A')}")
        
        rag_context = self._get_preprocessing_examples() if self.rag else ""
        
        # Extract spec details
        channel_names = [ch.name for ch in spec.channels]
        control_names = [c.name if hasattr(c, 'name') else c for c in spec.controls] if spec.controls else []
        
        # Add reasoning context
        reasoning_context = ""
        if reasoning:
            reasoning_context = f"""
========== REASONING ANALYSIS (FOLLOW THIS!) ==========

MISSING VALUE STRATEGY:
{reasoning.get('missing_value_strategy', 'N/A')}

OUTLIER HANDLING:
{reasoning.get('outlier_strategy', 'N/A')}

DATA LEAKAGE PREVENTION:
{', '.join(reasoning.get('data_leakage_checks', []))}

OUTPUT FORMAT (CRITICAL):
{reasoning.get('output_format', 'Return DataFrame')}

VALIDATION CHECKS:
{', '.join(reasoning.get('validation_checks', []))}

EXECUTION ORDER:
"""
            for i, step in enumerate(reasoning.get('execution_order', []), 1):
                reasoning_context += f"\n{i}. {step}"
            
            #  ADD VISUALIZATION REASONING (THIS WAS MISSING!)
            viz = reasoning.get('visualizations', {})
            if viz:
                reasoning_context += f"""

VISUALIZATIONS REQUIRED (YOU REASONED THESE ARE NEEDED!):
- Before/After: {viz.get('before_after', [])}
- Outliers: {viz.get('outliers', [])}
- Zero Preservation: {viz.get('zero_preservation', [])}
- Required filenames: {viz.get('filenames', [])}

YOU MUST IMPLEMENT ALL VISUALIZATION FUNCTIONS THAT YOU REASONED ABOUT!
Each plot MUST use plt.savefig(filename, dpi=300, bbox_inches='tight') + plt.close()
"""
            
            reasoning_context += "\n\n=======================================================\n\n"
        
        prompt = f"""{reasoning_context}You are an expert ML engineer. Write COMPREHENSIVE, DETAILED, PRODUCTION-GRADE preprocessing code for Marketing Mix Modeling.

{rag_context}

CRITICAL - USE THESE EXACT VALUES FROM SPEC:
- OUTCOME: '{spec.outcome}'
- DATE_COLUMN: '{spec.date_column}'
- CHANNEL_NAMES: {channel_names}
- CONTROL_NAMES: {control_names}
- TIME_UNIT: '{spec.time_unit}'

CRITICAL REQUIREMENTS:
- The code MUST NOT create NaN or infinite values
- Use conservative transformations (cap, not remove)
- Fill NaNs immediately after any operation that creates them
- Test: NO StandardScaler + log transform combo (creates NaNs)
- Test: NO multiple scalers (pick ONE: MinMaxScaler)
- Test: Use min_periods=1 in rolling operations
- Code should be ~200-250 lines (COMPREHENSIVE preprocessing for MMM)
- MANDATORY: Generate ALL functions shown in few-shot examples above
- MANDATORY: Create ALL 3 visualization plots (see below)

Write COMPREHENSIVE preprocessing functions following the examples above:

1. COMPREHENSIVE MISSING VALUE HANDLING (40 lines):
   - Detect missing patterns
   - Forward fill for time series continuity
   - Backward fill for edge cases
   - Interpolation (linear, polynomial) for gaps
   - Mean/median imputation as fallback
   - Report missing value statistics before/after
   - Handle zero vs NaN distinction
   - FOLLOW EXAMPLE 1 from few-shot examples

2. COMPREHENSIVE OUTLIER TREATMENT (50 lines):
   - Winsorization (cap at 1st/99th percentile) to preserve marketing events
   - DO NOT use Z-score or IQR removal (they remove too many observations)
   - DO NOT apply log transformations here (causes NaNs on negative values)
   - MANDATORY PLOT 1: Box plots showing capped distributions
     * plt.savefig('preprocessing_outliers.png', dpi=300, bbox_inches='tight')
     * plt.close()
   - FOLLOW EXAMPLE 2 from few-shot examples

3. COMPREHENSIVE FEATURE ENGINEERING (60 lines):
   - Create lag[1, 2] features for ALL channels (not just lag_1!)
   - Create rolling[4, 13] week averages (not just rolling_7!)
   - Use min_periods=1 to avoid NaNs
   - IMMEDIATELY fill any NaNs after each operation: df = df.fillna(0)
   - Pattern: for lag in [1, 2]: df[f'{{channel}}_lag{{lag}}'] = df[channel].shift(lag).fillna(0)
   - Pattern: for window in [4, 13]: df[f'{{channel}}_ma{{window}}'] = df[channel].rolling(window, min_periods=1).mean().fillna(0)
   - FOLLOW EXAMPLE 4 from few-shot examples EXACTLY

4. MMM-APPROPRIATE SCALING (15 lines):
   - Use ONLY MinMaxScaler (keeps values in [0,1], no negatives)
   - DO NOT use StandardScaler (creates negative values)
   - DO NOT apply log transformations (causes NaNs/infinities)
   - Scale: CHANNEL_NAMES + lag features + rolling features + CONTROL_NAMES
   - FOLLOW EXAMPLE 3 from few-shot examples

5. BEFORE/AFTER VISUALIZATIONS (40 lines):
   - MANDATORY PLOT 2: Before/after distribution histograms for top 6 channels
     * plt.savefig('preprocessing_distributions.png', dpi=300, bbox_inches='tight')
     * plt.close()
   - MANDATORY PLOT 3: Zero-preservation histogram (% zeros before/after)
     * plt.savefig('preprocessing_zero_preservation.png', dpi=300, bbox_inches='tight')
     * plt.close()
   - FOLLOW EXAMPLE 7 from few-shot examples

6. ROBUST DATA VALIDATION (30 lines):
   - Check for NaNs and fill with 0 if found (don't raise error, fix it)
   - Check for infinities and clip to reasonable bounds (don't raise error, fix it)
   - Log warnings, not errors
   - Final safety check: if df.isnull().sum().sum() > 0 or np.isinf(df.select_dtypes(include=[np.number])).sum().sum() > 0, RAISE ValueError
   - FOLLOW EXAMPLE 5 from few-shot examples

Include:
- Full docstrings with parameter descriptions
- Type hints (from typing import)
- Print statements with progress
- Comprehensive try/except error handling
- Logging for debugging
- Helper functions for reusability
- Comments explaining each transformation

CRITICAL: Output ONLY valid Python code with # comments.
- NO markdown code fences (``` or ```python)
- NO explanatory paragraphs or prose
- NO numbered lists or bullet points
- Every line must be executable Python or a # comment
- Do NOT explain what the code does - just write the code

========== MANDATORY ENTRY POINT REQUIREMENTS ==========

YOU MUST END YOUR CODE WITH A run_preprocessing() FUNCTION following EXAMPLE 6 above.

CRITICAL REQUIREMENTS:
1. Function signature: def run_preprocessing(data_path: str) -> str
2. MUST save df_before = df.copy() at the start (for visualizations)
3. MUST call all preprocessing functions in order
4. MUST call plot_preprocessing_effects(df_before, df, CHANNEL_NAMES) before saving
5. MUST return output_path string (NOT DataFrame!)

Follow EXAMPLE 6 structure exactly - it shows the complete pattern including visualization integration.
Every line must be valid Python or a # comment."""

        # Stream the code generation
        print("\n" + "="*80)
        print(" PREPROCESSING CODE (streaming):")
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
        
        logger.info(f"[OK] Generated {len(code.splitlines())} lines of preprocessing code")
        return code
    
    def _get_preprocessing_examples(self) -> str:
        """Get MMM-specific preprocessing few-shot examples"""
        return """
����������������������������������������������������������������������������������������������������������������������������������������������������������������
                    MMM-SPECIFIC PREPROCESSING FEW-SHOT EXAMPLES
����������������������������������������������������������������������������������������������������������������������������������������������������������������

EXAMPLE 1: Handle Missing Values (MMM-Appropriate Strategies)

def handle_missing_values(df, channel_cols, control_cols, outcome):
    '''Handle missing values for MMM data.
    
    Strategy:
    - Channels: fillna(0) - missing = no spend
    - Controls: forward fill - carry forward last known value
    - Outcome: interpolate for short gaps
    '''
    # Channels: Missing spend = zero spend
    for channel in channel_cols:
        if df[channel].isna().any():
            n_missing = df[channel].isna().sum()
            logger.info(f"{{channel}}: {{n_missing}} missing values �� filling with 0")
            df[channel] = df[channel].fillna(0)
    
    # Controls: Forward fill (last observation carried forward)
    for control in control_cols:
        if df[control].isna().any():
            n_missing = df[control].isna().sum()
            logger.info(f"{{control}}: {{n_missing}} missing values �� forward fill")
            df[control] = df[control].fillna(method='ffill').fillna(0)  # ffill then 0 for leading NaNs
    
    # Outcome: Interpolate (for short gaps only)
    if df[outcome].isna().any():
        n_missing = df[outcome].isna().sum()
        logger.info(f"{{outcome}}: {{n_missing}} missing values �� interpolate")
        df[outcome] = df[outcome].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    return df


EXAMPLE 2: Outlier Treatment (Percentile Capping, NOT Removal)

def treat_outliers_percentile(df, channel_cols, lower_pct=1, upper_pct=99):
    '''Cap outliers at percentiles (preserves marketing events like Super Bowl).
    
    DON'T remove outliers - they're real marketing spend!
    '''
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()
    
    # Get top 6 channels by total spend
    top_channels = df[channel_cols].sum().sort_values(ascending=False).head(6).index.tolist()
    
    for idx, channel in enumerate(top_channels):
        # Calculate percentile thresholds
        lower_bound = df[channel].quantile(lower_pct / 100)
        upper_bound = df[channel].quantile(upper_pct / 100)
        
        # Count outliers
        n_lower = (df[channel] < lower_bound).sum()
        n_upper = (df[channel] > upper_bound).sum()
        
        if n_lower > 0 or n_upper > 0:
            logger.info(f"{{channel}}: Capping {{n_lower}} low outliers, {{n_upper}} high outliers")
        
        # Cap (NOT remove!)
        df[channel] = df[channel].clip(lower=lower_bound, upper=upper_bound)
        
        # Plot before/after
        ax = axes[idx]
        ax.boxplot([df[channel]], vert=True)
        ax.set_title(f'{{channel}}\\nCapped at {{lower_pct}}th-{{upper_pct}}th percentile')
        ax.set_ylabel('Spend')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('preprocessing_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


EXAMPLE 3: Scaling for MMM (MinMaxScaler ONLY, NO StandardScaler)

def scale_features_mmm(df, channel_cols, control_cols):
    '''Scale features using MinMaxScaler [0, 1].
    
    WHY MinMaxScaler?
    - Preserves zeros (important for zero-spend periods)
    - All values stay positive (no negative values that break log transforms)
    - Interpretable: 0 = min spend, 1 = max spend
    
    WHY NOT StandardScaler?
    - Creates negative values
    - If followed by log transform �� NaN!
    - Harder to interpret
    '''
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale channels
    df[channel_cols] = scaler.fit_transform(df[channel_cols])
    logger.info(f"Scaled {{len(channel_cols)}} channels to [0, 1] with MinMaxScaler")
    
    # Scale controls (if numeric)
    numeric_controls = df[control_cols].select_dtypes(include=[np.number]).columns.tolist()
    if numeric_controls:
        df[numeric_controls] = scaler.fit_transform(df[numeric_controls])
        logger.info(f"Scaled {{len(numeric_controls)}} controls to [0, 1]")
    
    return df


EXAMPLE 4: Feature Engineering (Conservative, with NaN Handling)

def create_lag_features(df, channel_cols, lags=[1, 2], date_col=None, id_col=None):
    '''Create lag features for adstock modeling (optional).
    
     WARNINGS:
    - Creates NaNs for first `lag` rows
    - Increases multicollinearity
    - For panel data: must group by ID!
    
    Use ONLY if needed for adstock.
    '''
    if id_col and id_col in df.columns:
        # Panel data: group by ID
        for channel in channel_cols:
            for lag in lags:
                df[f'{{channel}}_lag{{lag}}'] = df.groupby(id_col)[channel].shift(lag).fillna(0)
        logger.info(f"Created {{len(channel_cols) * len(lags)}} lag features (panel data, grouped by {{id_col}})")
    else:
        # Time series: simple shift
        for channel in channel_cols:
            for lag in lags:
                df[f'{{channel}}_lag{{lag}}'] = df[channel].shift(lag).fillna(0)
        logger.info(f"Created {{len(channel_cols) * len(lags)}} lag features (time series)")
    
    return df


def create_rolling_features(df, channel_cols, windows=[4, 13], date_col=None, id_col=None):
    '''Create rolling mean features (smoothing, optional).
    
     WARNINGS:
    - Creates NaNs for first `window` rows (use min_periods!)
    - For panel data: must group by ID!
    '''
    if id_col and id_col in df.columns:
        # Panel data: group by ID
        for channel in channel_cols:
            for window in windows:
                df[f'{{channel}}_ma{{window}}'] = df.groupby(id_col)[channel].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(level=0, drop=True).fillna(0)
        logger.info(f"Created {{len(channel_cols) * len(windows)}} rolling features (panel data)")
    else:
        # Time series: simple rolling
        for channel in channel_cols:
            for window in windows:
                df[f'{{channel}}_ma{{window}}'] = df[channel].rolling(
                    window=window, min_periods=1
                ).mean().fillna(0)
        logger.info(f"Created {{len(channel_cols) * len(windows)}} rolling features (time series)")
    
    return df


EXAMPLE 5: Robust Validation (Forgiving, NOT Strict)

def validate_preprocessed_data(df, channel_cols, control_cols, outcome):
    '''Validate preprocessed data with ROBUST error handling.
    
    Philosophy: FIX issues instead of raising errors.
    '''
    issues_found = []
    
    # Check for NaNs
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logger.warning(f" NaNs found in {{len(nan_cols)}} columns: {{nan_cols[:5]}}")
        # FIX: Fill with 0
        df[nan_cols] = df[nan_cols].fillna(0)
        logger.info(" Fixed: Filled NaNs with 0")
        issues_found.append(f"NaNs in {{len(nan_cols)}} columns (filled with 0)")
    
    # Check for infinite values
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        logger.warning(f" Infinite values found in {{len(inf_cols)}} columns: {{inf_cols[:5]}}")
        # FIX: Clip to finite range
        for col in inf_cols:
            finite_max = df[col][np.isfinite(df[col])].max()
            finite_min = df[col][np.isfinite(df[col])].min()
            df[col] = df[col].replace([np.inf, -np.inf], [finite_max, finite_min])
        logger.info(" Fixed: Clipped infinite values to finite range")
        issues_found.append(f"Infinite values in {{len(inf_cols)}} columns (clipped)")
    
    # Check outcome is numeric
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        logger.error(f" Outcome '{{outcome}}' is not numeric!")
        issues_found.append(f"Non-numeric outcome")
    
    # Check all channels are numeric
    non_numeric_channels = [ch for ch in channel_cols if not pd.api.types.is_numeric_dtype(df[ch])]
    if non_numeric_channels:
        logger.warning(f" Non-numeric channels: {{non_numeric_channels}}")
        # FIX: Convert to numeric, coerce errors to NaN, then fill
        for ch in non_numeric_channels:
            df[ch] = pd.to_numeric(df[ch], errors='coerce').fillna(0)
        logger.info(" Fixed: Converted to numeric and filled NaNs")
        issues_found.append(f"Non-numeric channels (converted)")
    
    # Summary
    if issues_found:
        logger.warning(f"Validation found and FIXED {{len(issues_found)}} issues")
        return df, issues_found
    else:
        logger.info(" All validation checks passed!")
        return df, []


EXAMPLE 6: Complete run_preprocessing for MMM (WITH VISUALIZATIONS!)

def run_preprocessing(data_path: str) -> str:
    '''Main entry point for MMM preprocessing.
    
    CRITICAL: Must return STRING path, NOT DataFrame!
    '''
    logger.info(f"Starting MMM preprocessing: {{data_path}}")
    
    # 1. Load data
    df = pd.read_csv(data_path)
    df_before = df.copy()  # �� CRITICAL: Save original for before/after visualization!
    logger.info(f"Loaded {{len(df)}} rows, {{len(df.columns)}} columns")
    
    # 2. Handle missing values
    df = handle_missing_values(df, CHANNEL_NAMES, CONTROL_NAMES, OUTCOME)
    
    # 3. Outlier treatment
    df = treat_outliers_percentile(df, CHANNEL_NAMES, lower_pct=1, upper_pct=99)
    
    # 4. Feature engineering (create lag[1,2] + rolling[4,13])
    df = create_lag_features(df, CHANNEL_NAMES, lags=[1, 2])
    df = create_rolling_features(df, CHANNEL_NAMES, windows=[4, 13])
    
    # 5. Scaling (MinMaxScaler ONLY)
    df = scale_features_mmm(df, CHANNEL_NAMES, CONTROL_NAMES)
    
    # 6. Robust validation
    df, issues = validate_preprocessed_data(df, CHANNEL_NAMES, CONTROL_NAMES, OUTCOME)
    
    # 7. CRITICAL: Generate before/after visualizations (from EXAMPLE 7)
    plot_preprocessing_effects(df_before, df, CHANNEL_NAMES)  # �� Creates 2 plots!
    logger.info("Generated before/after visualization plots")
    
    # 8. Save to CSV
    output_path = data_path.replace('.csv', '_preprocessed.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved preprocessed data: {{output_path}}")
    
    # 9. CRITICAL: Return STRING path, NOT DataFrame!
    return output_path


EXAMPLE 7: Visualizations (Before/After Comparison)

def plot_preprocessing_effects(df_before, df_after, channel_cols):
    '''Plot before/after distributions to validate preprocessing.'''
    import matplotlib.pyplot as plt
    
    top_channels = df_before[channel_cols].sum().sort_values(ascending=False).head(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    axes = axes.flatten()
    
    for idx, channel in enumerate(top_channels):
        ax = axes[idx]
        
        # Before
        ax.hist(df_before[channel], bins=50, alpha=0.5, label='Before', color='red', edgecolor='black')
        
        # After
        ax.hist(df_after[channel], bins=50, alpha=0.5, label='After', color='green', edgecolor='black')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{{channel}}\\nBefore: μ={{df_before[channel].mean():.2f}}, After: μ={{df_after[channel].mean():.2f}}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('preprocessing_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'preprocessing_distributions.png'


KEY PATTERNS FROM EXAMPLES:
 Channels: fillna(0) - missing = no spend
 Outliers: Cap with percentiles, DON'T remove (Super Bowl is real!)
 Scaling: MinMaxScaler ONLY (no StandardScaler �� no negative values �� no NaN)
 Feature engineering: Use min_periods, fillna(0) immediately
 Validation: FORGIVING - fix issues instead of raising errors
 Return type: STRING path to CSV, NOT DataFrame!
 Plots: Always use plt.savefig() + plt.close()

����������������������������������������������������������������������������������������������������������������������������������������������������������������
"""
    
    def _clean_code(self, code: str) -> str:
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

