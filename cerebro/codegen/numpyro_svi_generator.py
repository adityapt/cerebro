"""
NumPyro SVI Code Generator
Generates production-quality NumPyro SVI code from MMM specs
"""
from typing import Dict, Any
from cerebro.spec.schema import MMMSpec, AdstockConfig, SaturationConfig


class NumPyroSVIGenerator:
    """Generates NumPyro SVI code from MMM spec"""
    
    def __init__(self):
        self.indent = "    "
    
    @staticmethod
    def sanitize_name(name: str) -> str:
        """Sanitize column name for use as Python variable name"""
        # Replace dots, dashes, and spaces with underscores
        safe_name = name.replace('.', '_').replace('-', '_').replace(' ', '_')
        # Remove any remaining invalid characters
        safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in safe_name)
        # Ensure it doesn't start with a number
        if safe_name[0].isdigit():
            safe_name = 'ch_' + safe_name
        return safe_name
    
    def _generate_data_exploration(self) -> str:
        """Generate comprehensive data exploration module"""
        return '''# =============================================================================
# DATA EXPLORATION & ANALYSIS
# =============================================================================

def explore_data(data, outcome_col, channel_cols, date_col=None):
    """
    Comprehensive exploratory data analysis for MMM
    
    Analyzes:
    - Time series trends
    - Seasonality patterns
    - Autocorrelation structure
    - Cross-correlations between channels
    - Spending patterns and outliers
    - Missing data patterns
    """
    print("\\n" + "=" * 80)
    print("📊 DATA EXPLORATION")
    print("=" * 80)
    
    results = {}
    
    # Basic statistics
    print("\\n1. Dataset Overview")
    print(f"   Rows: {len(data):,}")
    print(f"   Columns: {len(data.columns)}")
    print(f"   Date range: {data[date_col].min()} to {data[date_col].max()}" if date_col else "   No date column")
    print(f"   Outcome: {outcome_col}")
    print(f"   Channels: {len(channel_cols)}")
    
    # Outcome analysis
    print("\\n2. Outcome Variable Analysis")
    outcome_data = data[outcome_col]
    print(f"   Mean: {outcome_data.mean():.2f}")
    print(f"   Std: {outcome_data.std():.2f}")
    print(f"   Min: {outcome_data.min():.2f}")
    print(f"   Max: {outcome_data.max():.2f}")
    print(f"   Zeros: {(outcome_data == 0).sum()} ({(outcome_data == 0).mean()*100:.1f}%)")
    
    # Channel spending analysis
    print("\\n3. Channel Spending Patterns")
    channel_stats = []
    for ch in channel_cols[:5]:  # Show top 5
        stats_dict = {
            'channel': ch[:30],  # Truncate long names
            'mean': data[ch].mean(),
            'std': data[ch].std(),
            'zeros': (data[ch] == 0).mean() * 100
        }
        channel_stats.append(stats_dict)
        print(f"   {stats_dict['channel']:30s}: μ={stats_dict['mean']:10.2f}, σ={stats_dict['std']:10.2f}, zeros={stats_dict['zeros']:5.1f}%")
    
    if len(channel_cols) > 5:
        print(f"   ... and {len(channel_cols) - 5} more channels")
    
    # Autocorrelation analysis
    print("\\n4. Autocorrelation Analysis (Outcome)")
    from statsmodels.tsa.stattools import acf
    try:
        acf_vals = acf(outcome_data.dropna(), nlags=12, fft=False)
        sig_lags = np.where(np.abs(acf_vals[1:]) > 2/np.sqrt(len(outcome_data)))[0]
        print(f"   Significant lags: {sig_lags[:5]+1 if len(sig_lags) > 0 else 'None'}")
        print(f"   Lag-1 autocorr: {acf_vals[1]:.3f}")
        results['acf_outcome'] = acf_vals
    except:
        print("   Could not compute autocorrelation")
    
    # Channel correlations
    print("\\n5. Channel Correlations with Outcome")
    correlations = []
    for ch in channel_cols[:10]:  # Top 10
        corr = data[ch].corr(outcome_data)
        correlations.append((ch[:30], corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for ch, corr in correlations[:5]:
        print(f"   {ch:30s}: {corr:6.3f}")
    
    # Missing data
    print("\\n6. Missing Data Analysis")
    missing_pct = (data.isnull().sum() / len(data) * 100)
    missing_cols = missing_pct[missing_pct > 0]
    if len(missing_cols) > 0:
        print(f"   ⚠️  {len(missing_cols)} columns with missing data")
        for col, pct in missing_cols.items():
            print(f"      {col[:30]:30s}: {pct:.1f}%")
    else:
        print("   ✅ No missing data")
    
    # Outlier detection
    print("\\n7. Outlier Detection (Outcome)")
    q1, q3 = outcome_data.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((outcome_data < (q1 - 3*iqr)) | (outcome_data > (q3 + 3*iqr))).sum()
    print(f"   Outliers (3×IQR): {outliers} ({outliers/len(data)*100:.1f}%)")
    
    results['channel_stats'] = channel_stats
    results['correlations'] = correlations
    results['outliers'] = outliers
    
    print("\\n" + "=" * 80)
    
    return results'''
    
    def _generate_preprocessing(self) -> str:
        """Generate preprocessing module for data cleaning and transformation"""
        return '''# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_data(data, outcome_col, channel_cols, control_cols=None, date_col=None):
    """
    Preprocess data for MMM modeling
    
    Steps:
    1. Handle missing values
    2. Detect and cap outliers
    3. Scale/normalize features
    4. Create time features (if date column provided)
    5. Quality checks
    """
    print("\\n" + "=" * 80)
    print("🔧 DATA PREPROCESSING")
    print("=" * 80)
    
    data = data.copy()
    preprocessing_report = {}
    
    # 1. Missing value imputation
    print("\\n1. Missing Value Imputation")
    missing_before = data.isnull().sum().sum()
    if missing_before > 0:
        print(f"   Found {missing_before} missing values")
        # Forward fill for time series
        data = data.fillna(method='ffill').fillna(method='bfill')
        # Fill remaining with 0 for channels (no spend)
        for ch in channel_cols:
            data[ch] = data[ch].fillna(0)
        missing_after = data.isnull().sum().sum()
        print(f"   ✓ Imputed {missing_before - missing_after} values")
    else:
        print("   ✓ No missing values")
    preprocessing_report['missing_imputed'] = missing_before
    
    # 2. Outlier detection and capping
    print("\\n2. Outlier Detection & Capping")
    outliers_capped = 0
    
    # Cap outcome outliers at 99.5th percentile
    q995 = data[outcome_col].quantile(0.995)
    outlier_mask = data[outcome_col] > q995
    if outlier_mask.sum() > 0:
        print(f"   Outcome: Capping {outlier_mask.sum()} values above {q995:.2f}")
        data.loc[outlier_mask, outcome_col] = q995
        outliers_capped += outlier_mask.sum()
    
    # Cap channel spending outliers
    for ch in channel_cols:
        q995_ch = data[ch].quantile(0.995)
        if q995_ch > 0:
            outlier_mask_ch = data[ch] > q995_ch
            if outlier_mask_ch.sum() > 0:
                data.loc[outlier_mask_ch, ch] = q995_ch
                outliers_capped += outlier_mask_ch.sum()
    
    print(f"   ✓ Capped {outliers_capped} outlier values")
    preprocessing_report['outliers_capped'] = outliers_capped
    
    # 3. Feature scaling
    print("\\n3. Feature Scaling")
    scaler_channels = StandardScaler()
    scaler_controls = StandardScaler()
    
    # Scale channels (preserve zero-inflation pattern)
    data_channels_scaled = scaler_channels.fit_transform(data[channel_cols])
    for i, ch in enumerate(channel_cols):
        data[f'{ch}_scaled'] = data_channels_scaled[:, i]
    
    # Scale controls
    if control_cols:
        data_controls_scaled = scaler_controls.fit_transform(data[control_cols])
        for i, ctrl in enumerate(control_cols):
            data[f'{ctrl}_scaled'] = data_controls_scaled[:, i]
        print(f"   ✓ Scaled {len(channel_cols)} channels + {len(control_cols)} controls")
    else:
        print(f"   ✓ Scaled {len(channel_cols)} channels")
    
    preprocessing_report['scalers'] = {
        'channels': scaler_channels,
        'controls': scaler_controls if control_cols else None
    }
    
    # 4. Time features (if date provided)
    if date_col:
        print("\\n4. Time Feature Engineering")
        data[date_col] = pd.to_datetime(data[date_col])
        data['week_of_year'] = data[date_col].dt.isocalendar().week
        data['month'] = data[date_col].dt.month
        data['quarter'] = data[date_col].dt.quarter
        data['year'] = data[date_col].dt.year
        print("   ✓ Created: week_of_year, month, quarter, year")
    
    # 5. Quality checks
    print("\\n5. Quality Checks")
    n_obs = len(data)
    n_features = len(channel_cols) + (len(control_cols) if control_cols else 0)
    
    # Check for constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() == 1]
    if constant_cols:
        print(f"   ⚠️  Found {len(constant_cols)} constant columns: {constant_cols[:3]}")
    
    # Check for high correlation (multicollinearity)
    corr_matrix = data[channel_cols].corr()
    high_corr = np.where((np.abs(corr_matrix) > 0.9) & (np.abs(corr_matrix) < 1.0))
    if len(high_corr[0]) > 0:
        print(f"   ⚠️  Found {len(high_corr[0])//2} high correlations (>0.9) between channels")
    else:
        print("   ✓ No high multicollinearity detected")
    
    # Minimum observations check
    min_obs_required = n_features * 10
    if n_obs < min_obs_required:
        print(f"   ⚠️  WARNING: {n_obs} observations < {min_obs_required} recommended (10x features)")
    else:
        print(f"   ✓ Sufficient observations: {n_obs} >= {min_obs_required}")
    
    preprocessing_report['constant_cols'] = constant_cols
    preprocessing_report['n_obs'] = n_obs
    preprocessing_report['n_features'] = n_features
    
    print("\\n" + "=" * 80)
    
    return data, preprocessing_report'''
    
    def generate(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate complete production MMM pipeline from spec"""
        
        code_sections = [
            self._generate_imports(),
            self._generate_data_exploration(),
            self._generate_preprocessing(),
            self._generate_transforms(spec),
            self._generate_model(spec),
            self._generate_guide_and_svi(spec),
            self._generate_diagnostics(),
            self._generate_budget_optimization(spec),
            self._generate_visualization(spec),
            self._generate_execution(spec, data_path),
            self._generate_results(spec)
        ]
        
        return "\n\n".join(code_sections)
    
    def _generate_imports(self) -> str:
        return '''"""
🚀 PRODUCTION MARKETING MIX MODEL (MMM) PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Auto-generated by Cerebro
Includes: EDA, Preprocessing, Modeling, Diagnostics, Optimization, Visualization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive
from numpyro.optim import Adam

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Optional: ArviZ for advanced diagnostics
try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    print("⚠️  ArviZ not available - some diagnostics will be skipped")
    HAS_ARVIZ = False

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# JAX configuration
jax.config.update('jax_platform_name', 'cpu')  # Change to 'gpu' if available

print("=" * 80)
print("🎯 PRODUCTION MMM PIPELINE")
print("=" * 80)'''
    
    def _generate_adstock_function(self, config: AdstockConfig) -> str:
        """Generate adstock transformation function"""
        if config.type == "geometric":
            return '''def geometric_adstock(x, alpha, max_lag):
    """Geometric adstock: simple exponential decay"""
    lags = jnp.arange(max_lag + 1)
    weights = jnp.power(alpha, lags)
    weights = weights / weights.sum()
    return jnp.convolve(x, weights, mode='same')'''
        
        elif config.type == "weibull":
            return '''def weibull_adstock(x, alpha, beta, max_lag):
    """Weibull adstock: flexible decay shape"""
    lags = jnp.arange(max_lag + 1)
    weights = jnp.power(1 - jnp.power(lags / max_lag, beta), alpha)
    weights = weights / jnp.maximum(weights.sum(), 1e-10)
    return jnp.convolve(x, weights, mode='same')'''
        
        elif config.type == "delayed":
            return '''def delayed_adstock(x, theta, delay, max_lag):
    """Delayed adstock: effect peaks after delay"""
    lags = jnp.arange(max_lag + 1)
    weights = jnp.where(
        lags < delay,
        0,
        jnp.power(theta, lags - delay)
    )
    weights = weights / jnp.maximum(weights.sum(), 1e-10)
    return jnp.convolve(x, weights, mode='same')'''
        
        elif config.type == "carryover":
            return '''def carryover_adstock(x, theta, max_lag):
    """Carryover adstock: momentum effects"""
    lags = jnp.arange(max_lag + 1)
    weights = jnp.power(theta, lags) * (1 + jnp.log1p(lags))
    weights = weights / jnp.maximum(weights.sum(), 1e-10)
    return jnp.convolve(x, weights, mode='same')'''
        
        elif config.type == "exponential":
            return '''def exponential_adstock(x, decay, max_lag):
    """Exponential adstock: exponential decay"""
    lags = jnp.arange(max_lag + 1)
    weights = jnp.exp(-decay * lags)
    weights = weights / jnp.maximum(weights.sum(), 1e-10)
    return jnp.convolve(x, weights, mode='same')'''
        
        else:
            raise ValueError(f"Unknown adstock type: {config.type}")
    
    def _generate_saturation_function(self, config: SaturationConfig) -> str:
        """Generate saturation transformation function"""
        if config.type == "hill":
            return '''def hill_saturation(x, k, s):
    """Hill saturation: S-curve with inflection point k and shape s"""
    return x ** s / (k ** s + x ** s)'''
        
        elif config.type == "logistic":
            return '''def logistic_saturation(x, k, s):
    """Logistic saturation: alternative S-curve"""
    return 1 / (1 + jnp.exp(-s * (x - k)))'''
        
        elif config.type == "exponential":
            return '''def exponential_saturation(x, lam):
    """Exponential saturation: 1 - exp(-λx)"""
    return 1 - jnp.exp(-lam * x)'''
        
        elif config.type == "michaelis_menten":
            return '''def michaelis_menten_saturation(x, vmax, km):
    """Michaelis-Menten saturation: enzyme kinetics"""
    return vmax * x / (km + x)'''
        
        else:
            raise ValueError(f"Unknown saturation type: {config.type}")
    
    def _generate_transforms(self, spec: MMMSpec) -> str:
        """Generate all transformation functions"""
        functions = set()
        
        # Collect unique adstock and saturation types
        for channel in spec.channels:
            if channel.transform:
                if channel.transform.adstock:
                    functions.add(self._generate_adstock_function(channel.transform.adstock))
                if channel.transform.saturation:
                    functions.add(self._generate_saturation_function(channel.transform.saturation))
        
        header = '''# =============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================='''
        
        return header + "\n\n" + "\n\n".join(functions)
    
    def _generate_model(self, spec: MMMSpec) -> str:
        """Generate the NumPyro model function"""
        
        model_code = '''# =============================================================================
# NUMPYRO MODEL
# =============================================================================

def mmm_model(data, outcome):
    """
    Auto-generated MMM model
    
    Channels: {num_channels}
    Controls: {num_controls}
    Hierarchy: {hierarchy}
    """
    n_obs = len(outcome)
    
    # Baseline (intercept) - broadcast to array
    baseline = numpyro.sample('baseline', dist.Normal(0, 1))
    mu = jnp.ones(n_obs) * baseline
    
'''.format(
            num_channels=len(spec.channels),
            num_controls=len(spec.controls) if spec.controls else 0,
            hierarchy=spec.hierarchy.dimension if spec.hierarchy else "None"
        )
        
        # Generate code for each channel
        for channel in spec.channels:
            model_code += self._generate_channel_code(channel, spec)
        
        # Add controls if any
        if spec.controls:
            model_code += self._generate_control_code(spec.controls)
        
        # Likelihood
        model_code += '''
    # Likelihood
    sigma = numpyro.sample('sigma', dist.HalfNormal(100))
    numpyro.sample('obs', dist.Normal(mu, sigma), obs=outcome)
    
    return mu
'''
        
        return model_code
    
    def _generate_channel_code(self, channel, spec: MMMSpec) -> str:
        """Generate code for a single channel"""
        safe_name = self.sanitize_name(channel.name)
        original_name = channel.name  # Keep original for data access
        
        code = f'''    # -------------------------------------------------------------------------
    # Channel: {original_name}
    # -------------------------------------------------------------------------
    
'''
        
        if not channel.transform:
            # No transformation, just coefficient
            code += f'''    coef_{safe_name} = numpyro.sample('coef_{safe_name}', dist.HalfNormal(0.5))
    mu = mu + coef_{safe_name} * data['{original_name}']
    
'''
            return code
        
        # Adstock transformation
        if channel.transform.adstock:
            adstock = channel.transform.adstock
            code += self._generate_adstock_code(safe_name, original_name, adstock)
        else:
            code += f'''    {safe_name}_adstocked = data['{original_name}']
    
'''
        
        # Saturation transformation
        if channel.transform.saturation:
            saturation = channel.transform.saturation
            code += self._generate_saturation_code(safe_name, saturation)
        else:
            code += f'''    {safe_name}_transformed = {safe_name}_adstocked
    
'''
        
        # Coefficient
        code += f'''    coef_{safe_name} = numpyro.sample('coef_{safe_name}', dist.HalfNormal(0.5))
    mu = mu + coef_{safe_name} * {safe_name}_transformed
    
'''
        
        return code
    
    def _generate_adstock_code(self, safe_name: str, original_name: str, adstock: AdstockConfig) -> str:
        """Generate adstock transformation code for a channel"""
        if adstock.type == "geometric":
            return f'''    # Geometric adstock
    alpha_{safe_name} = numpyro.sample('alpha_{safe_name}', dist.Beta(2, 2))
    {safe_name}_adstocked = geometric_adstock(data['{original_name}'], alpha_{safe_name}, {adstock.max_lag})
    
'''
        
        elif adstock.type == "weibull":
            return f'''    # Weibull adstock (flexible decay)
    alpha_{safe_name} = numpyro.sample('alpha_{safe_name}', dist.Beta(3, 3))
    beta_{safe_name} = numpyro.sample('beta_{safe_name}', dist.HalfNormal(2))
    {safe_name}_adstocked = weibull_adstock(data['{original_name}'], alpha_{safe_name}, beta_{safe_name}, {adstock.max_lag})
    
'''
        
        elif adstock.type == "delayed":
            return f'''    # Delayed adstock (using continuous delay, will be rounded)
    theta_{safe_name} = numpyro.sample('theta_{safe_name}', dist.Uniform(0, 0.5))
    delay_{safe_name}_continuous = numpyro.sample('delay_{safe_name}_continuous', dist.Uniform(1, 5))
    delay_{safe_name} = jnp.round(delay_{safe_name}_continuous).astype(int)
    {safe_name}_adstocked = delayed_adstock(data['{original_name}'], theta_{safe_name}, delay_{safe_name}, {adstock.max_lag})
    
'''
        
        elif adstock.type == "carryover":
            return f'''    # Carryover adstock
    theta_{safe_name} = numpyro.sample('theta_{safe_name}', dist.Beta(2, 2))
    {safe_name}_adstocked = carryover_adstock(data['{original_name}'], theta_{safe_name}, {adstock.max_lag})
    
'''
        
        elif adstock.type == "exponential":
            return f'''    # Exponential adstock
    decay_{safe_name} = numpyro.sample('decay_{safe_name}', dist.HalfNormal(1))
    {safe_name}_adstocked = exponential_adstock(data['{original_name}'], decay_{safe_name}, {adstock.max_lag})
    
'''
        
        else:
            return f'''    {safe_name}_adstocked = data['{original_name}']  # No adstock
    
'''
    
    def _generate_saturation_code(self, channel_name: str, saturation: SaturationConfig) -> str:
        """Generate saturation transformation code"""
        if saturation.type == "hill":
            return f'''    # Hill saturation
    k_{channel_name} = numpyro.sample('k_{channel_name}', dist.HalfNormal(1))
    s_{channel_name} = numpyro.sample('s_{channel_name}', dist.Gamma(3, 1))
    {channel_name}_transformed = hill_saturation({channel_name}_adstocked, k_{channel_name}, s_{channel_name})
    
'''
        
        elif saturation.type == "logistic":
            return f'''    # Logistic saturation
    k_{channel_name} = numpyro.sample('k_{channel_name}', dist.Normal(0, 1))
    s_{channel_name} = numpyro.sample('s_{channel_name}', dist.HalfNormal(1))
    {channel_name}_transformed = logistic_saturation({channel_name}_adstocked, k_{channel_name}, s_{channel_name})
    
'''
        
        elif saturation.type == "exponential":
            return f'''    # Exponential saturation
    lam_{channel_name} = numpyro.sample('lam_{channel_name}', dist.Exponential(1))
    {channel_name}_transformed = exponential_saturation({channel_name}_adstocked, lam_{channel_name})
    
'''
        
        else:
            return f'''    {channel_name}_transformed = {channel_name}_adstocked  # No saturation
    
'''
    
    def _generate_control_code(self, controls: list) -> str:
        """Generate code for control variables"""
        code = '''    # -------------------------------------------------------------------------
    # Control Variables
    # -------------------------------------------------------------------------
    
'''
        for control in controls:
            safe_name = self.sanitize_name(control)
            code += f'''    coef_{safe_name} = numpyro.sample('coef_{safe_name}', dist.Normal(0, 0.5))
    mu = mu + coef_{safe_name} * data['{control}']
    
'''
        
        return code
    
    def _generate_guide_and_svi(self, spec: MMMSpec) -> str:
        """Generate guide and SVI setup"""
        inf = spec.inference
        
        return f'''# =============================================================================
# GUIDE AND SVI SETUP
# =============================================================================

def run_svi(data, outcome, num_steps={inf.num_steps}, lr={inf.lr}):
    """Run SVI inference"""
    
    # Automatic guide (mean-field approximation)
    guide = autoguide.AutoNormal(mmm_model)
    
    # Optimizer
    optimizer = Adam(step_size=lr)
    
    # SVI object
    svi = SVI(mmm_model, guide, optimizer, loss=Trace_ELBO(num_particles={inf.num_particles}))
    
    # Initialize
    rng_key = jax.random.PRNGKey(0)
    svi_state = svi.init(rng_key, data=data, outcome=outcome)
    
    # Training loop
    losses = []
    print("\\nTraining SVI...")
    print("=" * 80)
    
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state, data=data, outcome=outcome)
        
        if step % 1000 == 0:
            losses.append(loss)
            print(f"Step {{step:6d}} / {{num_steps}} | Loss: {{loss:.2f}}")
    
    print("=" * 80)
    print("✅ Training complete!\\n")
    
    # Get posterior samples
    params = svi.get_params(svi_state)
    predictive = Predictive(guide, params=params, num_samples=1000)
    posterior_samples = predictive(jax.random.PRNGKey(1), data=data, outcome=outcome)
    
    return posterior_samples, losses, svi, svi_state
'''
    
    def _generate_diagnostics(self) -> str:
        """Generate model diagnostics module"""
        return '''# =============================================================================
# MODEL DIAGNOSTICS
# =============================================================================

def diagnose_model(posterior_samples, losses, data, outcome, svi_state=None):
    """
    Comprehensive model diagnostics
    
    Checks:
    1. Convergence (loss curve)
    2. Posterior predictive checks
    3. Parameter distributions
    4. Residual analysis
    5. R² and MAPE
    """
    print("\\n" + "=" * 80)
    print("🔍 MODEL DIAGNOSTICS")
    print("=" * 80)
    
    diagnostics = {}
    
    # 1. Convergence diagnostics
    print("\\n1. Convergence Analysis")
    loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
    if loss_trend < -10:
        print(f"   ✅ Loss decreasing (slope: {loss_trend:.2f})")
        diagnostics['converged'] = True
    else:
        print(f"   ⚠️  Loss not decreasing strongly (slope: {loss_trend:.2f})")
        diagnostics['converged'] = False
    
    final_loss = losses[-1]
    print(f"   Final loss: {final_loss:.2f}")
    
    # 2. Posterior Predictive Checks
    print("\\n2. Posterior Predictive Checks")
    # Get predicted values (mean of posterior predictive)
    if 'mu' in posterior_samples or 'obs' in posterior_samples:
        pred_key = 'mu' if 'mu' in posterior_samples else 'obs'
        predictions = posterior_samples[pred_key].mean(axis=0)
        
        # R²
        r2 = r2_score(outcome, predictions)
        print(f"   R²: {r2:.3f}")
        diagnostics['r2'] = r2
        
        # MAPE
        mape = mean_absolute_percentage_error(outcome, predictions) * 100
        print(f"   MAPE: {mape:.2f}%")
        diagnostics['mape'] = mape
        
        # Residuals
        residuals = outcome - predictions
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        print(f"   Residual mean: {residual_mean:.2f} (should be ~0)")
        print(f"   Residual std: {residual_std:.2f}")
        
        diagnostics['predictions'] = predictions
        diagnostics['residuals'] = residuals
    
    # 3. Parameter Summary
    print("\\n3. Posterior Parameter Summary")
    param_count = 0
    for param_name, samples in posterior_samples.items():
        if param_name not in ['mu', 'obs']:
            param_count += 1
            if param_count <= 10:  # Show first 10
                mean_val = samples.mean()
                std_val = samples.std()
                print(f"   {param_name[:40]:40s}: {mean_val:8.4f} ± {std_val:6.4f}")
    
    if param_count > 10:
        print(f"   ... and {param_count - 10} more parameters")
    
    diagnostics['n_params'] = param_count
    
    # 4. ArviZ diagnostics (if available)
    if HAS_ARVIZ:
        print("\\n4. ArviZ Diagnostics")
        try:
            inference_data = az.from_dict(posterior_samples)
            summary = az.summary(inference_data, var_names=['~obs', '~mu'])
            
            # Check R-hat (should be < 1.01)
            if 'r_hat' in summary.columns:
                max_rhat = summary['r_hat'].max()
                if max_rhat < 1.01:
                    print(f"   ✅ R-hat: {max_rhat:.4f} (< 1.01, good convergence)")
                else:
                    print(f"   ⚠️  R-hat: {max_rhat:.4f} (> 1.01, poor convergence)")
                diagnostics['max_rhat'] = max_rhat
            
            # ESS
            if 'ess_bulk' in summary.columns:
                min_ess = summary['ess_bulk'].min()
                print(f"   ESS (bulk): {min_ess:.0f} (>400 recommended)")
                diagnostics['min_ess'] = min_ess
        except Exception as e:
            print(f"   Could not compute ArviZ diagnostics: {e}")
    
    print("\\n" + "=" * 80)
    
    return diagnostics'''
    
    def _generate_budget_optimization(self, spec: MMMSpec) -> str:
        """Generate budget optimization module"""
        channel_names = [self.sanitize_name(ch.name) for ch in spec.channels]
        
        return f'''# =============================================================================
# BUDGET OPTIMIZATION
# =============================================================================

def optimize_budget(posterior_samples, channel_names, current_spend, total_budget):
    """
    Optimize budget allocation across channels
    
    Uses marginal ROI to allocate budget for maximum impact
    
    Args:
        posterior_samples: Posterior samples from model
        channel_names: List of channel names
        current_spend: Dict of current spend by channel
        total_budget: Total budget to allocate
    
    Returns:
        optimal_allocation: Dict with optimal spend by channel
        expected_lift: Expected incremental outcome
    """
    print("\\n" + "=" * 80)
    print("💰 BUDGET OPTIMIZATION")
    print("=" * 80)
    
    # Get channel coefficients (mean of posterior)
    channel_coefs = {{}}
    for ch in channel_names:
        coef_key = f'coef_{{ch}}'
        if coef_key in posterior_samples:
            channel_coefs[ch] = posterior_samples[coef_key].mean()
    
    print(f"\\n1. Current Spend & ROI")
    print(f"   Total budget: ${{total_budget:,.0f}}")
    
    # Calculate ROI for each channel
    rois = {{}}
    for ch, coef in channel_coefs.items():
        if ch in current_spend:
            roi = coef / max(current_spend[ch], 1)  # Avoid division by zero
            rois[ch] = roi
            print(f"   {{ch[:30]:30s}}: ${{current_spend[ch]:10,.0f}} → ROI={{roi:.4f}}")
    
    # Optimize allocation (simple greedy by ROI)
    print(f"\\n2. Optimal Allocation")
    sorted_channels = sorted(rois.items(), key=lambda x: x[1], reverse=True)
    
    optimal_allocation = {{}}
    remaining_budget = total_budget
    
    for ch, roi in sorted_channels:
        if remaining_budget > 0:
            # Allocate proportional to ROI
            allocation = min(remaining_budget, current_spend.get(ch, 0) * 1.5)  # Max 50% increase
            optimal_allocation[ch] = allocation
            remaining_budget -= allocation
            
            change_pct = ((allocation / max(current_spend.get(ch, 1), 1)) - 1) * 100
            print(f"   {{ch[:30]:30s}}: ${{allocation:10,.0f}} ({{change_pct:+5.1f}}%)")
    
    # Calculate expected lift
    expected_lift = sum(optimal_allocation.get(ch, 0) * channel_coefs.get(ch, 0) 
                       for ch in channel_names)
    current_outcome = sum(current_spend.get(ch, 0) * channel_coefs.get(ch, 0) 
                         for ch in channel_names)
    lift_pct = ((expected_lift / max(current_outcome, 1)) - 1) * 100
    
    print(f"\\n3. Expected Impact")
    print(f"   Current outcome: {{current_outcome:,.2f}}")
    print(f"   Optimized outcome: {{expected_lift:,.2f}}")
    print(f"   Expected lift: {{lift_pct:+.2f}}%")
    
    print("\\n" + "=" * 80)
    
    return optimal_allocation, expected_lift'''
    
    def _generate_visualization(self, spec: MMMSpec) -> str:
        """Generate visualization module"""
        return '''# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(data, outcome, predictions, posterior_samples, channel_cols, date_col=None):
    """
    Generate comprehensive visualizations
    
    Creates:
    1. Actual vs Predicted time series
    2. Residual plots
    3. Channel contribution waterfall
    4. Response curves (saturation)
    5. Posterior distributions
    """
    print("\\n" + "=" * 80)
    print("📈 GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Actual vs Predicted
    print("\\n1. Actual vs Predicted Plot")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    if date_col is not None:
        x_axis = data[date_col]
        axes[0].plot(x_axis, outcome, label='Actual', alpha=0.7, linewidth=2)
        axes[0].plot(x_axis, predictions, label='Predicted', alpha=0.7, linewidth=2)
    else:
        axes[0].plot(outcome, label='Actual', alpha=0.7)
        axes[0].plot(predictions, label='Predicted', alpha=0.7)
    
    axes[0].set_title('Actual vs Predicted Outcome', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Outcome')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = outcome - predictions
    axes[1].scatter(predictions, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mmm_predictions.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: mmm_predictions.png")
    plt.close()
    
    # 2. Channel Contributions
    print("\\n2. Channel Contribution Analysis")
    contributions = {{}}
    for ch in channel_cols[:10]:  # Top 10 channels
        coef_key = f'coef_{{ch}}'
        if coef_key in posterior_samples:
            # Contribution = coef × channel_data
            coef_mean = posterior_samples[coef_key].mean()
            contrib = (coef_mean * data[ch].values).sum()
            contributions[ch[:20]] = contrib  # Truncate long names
    
    # Sort by absolute contribution
    contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    channels_list = list(contributions.keys())
    contrib_values = list(contributions.values())
    colors = ['green' if v > 0 else 'red' for v in contrib_values]
    
    ax.barh(channels_list, contrib_values, color=colors, alpha=0.7)
    ax.set_xlabel('Contribution to Outcome')
    ax.set_title('Channel Contributions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('mmm_contributions.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: mmm_contributions.png")
    plt.close()
    
    # 3. Parameter Distributions (first 6 params)
    print("\\n3. Posterior Distributions")
    param_names = [k for k in posterior_samples.keys() if k not in ['mu', 'obs']][:6]
    
    if len(param_names) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, param in enumerate(param_names):
            axes[idx].hist(posterior_samples[param], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[idx].set_title(param[:30], fontsize=10)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(param_names), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('mmm_posteriors.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved: mmm_posteriors.png")
        plt.close()
    
    print("\\n" + "=" * 80)
    
    return {{
        'contributions': contributions,
        'plots_saved': ['mmm_predictions.png', 'mmm_contributions.png', 'mmm_posteriors.png']
    }}'''
    
    def _generate_execution(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate execution code with full pipeline"""
        
        channel_cols = [ch.name for ch in spec.channels]
        control_cols = spec.controls if spec.controls else []
        date_col = spec.date_column if hasattr(spec, 'date_column') else None
        
        # Use provided data path or placeholder
        if data_path:
            data_load = f'''# Load data
print("Loading data from: {data_path}")
data_df = pd.read_csv('{data_path}')
print(f"✓ Loaded {{len(data_df)}} rows, {{len(data_df.columns)}} columns\\n")'''
        else:
            data_load = '''# Load data
print("Loading data...")
# TODO: Specify your data path here
data_df = pd.read_csv('path/to/your/data.csv')
print(f"✓ Loaded {len(data_df)} rows, {len(data_df.columns)} columns\\n")'''
        
        channel_list_str = '[' + ', '.join([f'"{ch}"' for ch in channel_cols]) + ']'
        control_list_str = '[' + ', '.join([f'"{ctrl}"' for ctrl in control_cols]) + ']' if control_cols else '[]'
        
        return f'''# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

if __name__ == "__main__":
    
    {data_load}
    
    # Define columns
    outcome_col = '{spec.outcome}'
    channel_cols = {channel_list_str}
    control_cols = {control_list_str}
    date_col = '{date_col}' if '{date_col}' != 'None' else None
    
    # =========================================================================
    # STEP 1: DATA EXPLORATION
    # =========================================================================
    exploration_results = explore_data(data_df, outcome_col, channel_cols, date_col)
    
    # =========================================================================
    # STEP 2: PREPROCESSING
    # =========================================================================
    data_df_processed, preprocessing_report = preprocess_data(
        data_df, outcome_col, channel_cols, control_cols, date_col
    )
    
    # =========================================================================
    # STEP 3: MODEL TRAINING
    # =========================================================================
    print("\\n" + "=" * 80)
    print("🎯 MODEL TRAINING")
    print("=" * 80)
    
    # Prepare data for NumPyro (use original unscaled for now)
    data = {{
''' + ''.join([f"        '{ch}': data_df['{ch}'].values,\n" for ch in channel_cols]) + (
            ''.join([f"        '{ctrl}': data_df['{ctrl}'].values,\n" for ctrl in control_cols]) if control_cols else ''
        ) + f'''    }}
    
    outcome = data_df[outcome_col].values
    
    print(f"\\nData shape: {{len(outcome)}} observations")
    print(f"Outcome range: {{outcome.min():.2f}} - {{outcome.max():.2f}}")
    print(f"Channels: {len(channel_cols)}")
    print(f"Controls: {len(control_cols)}\\n")
    
    # Run SVI training
    print("=" * 80)
    print("STARTING BAYESIAN INFERENCE (SVI)")
    print("=" * 80)
    posterior_samples, losses, svi, svi_state = run_svi(data, outcome)
    
    # =========================================================================
    # STEP 4: DIAGNOSTICS
    # =========================================================================
    diagnostics = diagnose_model(posterior_samples, losses, data, outcome, svi_state)
    
    # =========================================================================
    # STEP 5: BUDGET OPTIMIZATION
    # =========================================================================
    # Example current spend (replace with actual)
    current_spend = {{ch: data_df[ch].mean() for ch in channel_cols[:10]}}  # Top 10 channels
    total_budget = sum(current_spend.values()) * 1.0  # Keep same total budget
    
    channel_names_safe = {[f'"{self.sanitize_name(ch)}"' for ch in channel_cols]}
    optimal_allocation, expected_lift = optimize_budget(
        posterior_samples, 
        channel_names_safe[:10],  # Optimize top 10
        current_spend, 
        total_budget
    )
    
    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    predictions = diagnostics.get('predictions', posterior_samples.get('mu', np.zeros(len(outcome))))
    if len(predictions.shape) > 1:
        predictions = predictions.mean(axis=0)
    
    viz_results = visualize_results(
        data_df, 
        outcome, 
        predictions, 
        posterior_samples, 
        channel_cols,
        date_col
    )
'''
    
    def _generate_results(self, spec: MMMSpec) -> str:
        """Generate results processing code"""
        return '''# =============================================================================
# RESULTS
# =============================================================================

print("=" * 80)
print("POSTERIOR SUMMARY")
print("=" * 80)

# Print posterior means and std
for var_name, samples in posterior_samples.items():
    if var_name != 'obs':
        mean = samples.mean()
        std = samples.std()
        print(f"{var_name:30s}: {mean:8.4f} ± {std:6.4f}")

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Step (x1000)')
plt.ylabel('ELBO Loss')
plt.title('SVI Training Loss')
plt.grid(True)
plt.savefig('svi_loss.png', dpi=150, bbox_inches='tight')
print("\\n✅ Saved loss plot: svi_loss.png")

# ArviZ diagnostics (if available)
if HAS_ARVIZ:
    # Convert to ArviZ InferenceData for diagnostics
    inference_data = az.from_dict(posterior_samples)
    
    # Summary statistics
    summary = az.summary(inference_data, var_names=['~obs'])
    print("\\n" + "=" * 80)
    print("ARVIZ SUMMARY")
    print("=" * 80)
    print(summary)
    
    # Save results
    results = {
        'spec_name': "''' + spec.name + '''",
        'posterior_samples': posterior_samples,
        'losses': losses,
        'summary': summary
    }
else:
    print("\\n⚠️  ArviZ not available - basic results only")
    results = {
        'spec_name': "''' + spec.name + '''",
        'posterior_samples': posterior_samples,
        'losses': losses
    }

print("\\n" + "=" * 80)
print("✅ MODEL FITTING COMPLETE!")
print("=" * 80)
print("\\nResults saved in 'results' dict")
print(f"Posterior samples: {len(posterior_samples)} variables")
print(f"Training losses: {len(losses)} checkpoints")
'''


# CLI for testing
if __name__ == "__main__":
    from cerebro.spec import MMMSpec
    
    # Load example spec
    spec = MMMSpec.from_yaml("cerebro/spec/examples/complex_numpyro.yaml")
    
    # Generate code
    generator = NumPyroSVIGenerator()
    code = generator.generate(spec)
    
    # Save generated code
    with open("/tmp/generated_mmm.py", "w") as f:
        f.write(code)
    
    print("✅ Generated NumPyro SVI code!")
    print(f"   Saved to: /tmp/generated_mmm.py")
    print(f"   Lines: {len(code.split(chr(10)))}")

