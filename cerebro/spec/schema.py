"""
MMM Spec Schema - Pydantic models for type-safe spec parsing
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
import yaml


class AdstockConfig(BaseModel):
    """Adstock transformation configuration"""
    type: Literal["geometric", "delayed", "carryover", "exponential"]
    max_lag: int = Field(ge=1, le=52, description="Maximum lag periods")
    
    # Prior configurations (optional, will use defaults if not specified)
    alpha_prior: Optional[str] = None  # e.g., "beta(3, 3)"
    beta_prior: Optional[str] = None   # e.g., "half_normal(2)"
    theta_prior: Optional[str] = None  # e.g., "uniform(0, 0.5)"
    delay_prior: Optional[str] = None  # e.g., "discrete_uniform(1, 4)"
    
    reasoning: Optional[str] = None  # Agent's explanation


class SaturationConfig(BaseModel):
    """Saturation transformation configuration"""
    type: Literal["hill", "logistic", "exponential", "michaelis_menten"]
    
    # Prior configurations
    k_prior: Optional[str] = None      # e.g., "half_normal(1)"
    s_prior: Optional[str] = None      # e.g., "gamma(3, 1)"
    lambda_prior: Optional[str] = None # e.g., "exponential(1)"
    
    reasoning: Optional[str] = None


class ChannelTransform(BaseModel):
    """Full transformation pipeline for a channel"""
    adstock: Optional[AdstockConfig] = None
    saturation: Optional[SaturationConfig] = None


class DataAnalysis(BaseModel):
    """Agent's analysis of the channel (for documentation)"""
    autocorrelation: Optional[str] = None
    spending_pattern: Optional[str] = None
    variance: Optional[float] = None
    seasonality: Optional[str] = None


class Channel(BaseModel):
    """Single media channel specification"""
    name: str
    data_analysis: Optional[DataAnalysis] = None
    transform: Optional[ChannelTransform] = None
    
    @validator('name')
    def validate_name(cls, v):
        if not v or v.isspace():
            raise ValueError("Channel name cannot be empty")
        return v


class HierarchyConfig(BaseModel):
    """Hierarchical model configuration"""
    dimension: str  # e.g., "region", "product"
    levels: List[str]  # e.g., ["US", "CA", "EU"]
    pooling: Literal["pooled", "unpooled", "hierarchical"] = "hierarchical"


class PriorsConfig(BaseModel):
    """Prior distributions for model parameters"""
    baseline: str = "normal(0, 1)"
    channel_effects: str = "half_normal(0.5)"
    control_effects: str = "normal(0, 1)"
    regional_variance: Optional[str] = None  # For hierarchical models
    sigma: str = "half_normal(100)"


class AdvancedModelConfig(BaseModel):
    """Advanced model structure (TVPs, correlated priors). Default True for agentic modeling."""
    time_varying_parameters: bool = Field(
        True,
        description="If True, channel coefficients evolve over time (e.g. random walk); requires priors for TVP process."
    )
    correlated_channel_priors: bool = Field(
        True,
        description="If True, use MultivariateNormal with Cholesky/LKJ for channel coefficients instead of independent priors."
    )
    tvp_prior_scale: Optional[str] = Field(
        "half_normal(0.05)",
        description="Prior for TVP innovation scale, e.g. 'half_normal(0.05)' for random walk sigma."
    )


class InferenceConfig(BaseModel):
    """Inference configuration"""
    backend: Literal["auto", "jax_optim", "numpyro_svi", "numpyro_nuts", "pymc", "stan"] = "auto"
    
    # SVI specific
    optimizer: Optional[str] = "adam"
    lr: Optional[float] = 0.01
    num_steps: Optional[int] = 50000
    num_particles: Optional[int] = 10
    
    # NUTS specific
    num_warmup: Optional[int] = 1000
    num_samples: Optional[int] = 1000
    num_chains: Optional[int] = 4
    
    reasoning: Optional[str] = None


class EvaluationConfig(BaseModel):
    """Model evaluation configuration"""
    metrics: List[Literal["loo", "waic", "r2", "mape", "experiment_calibration"]] = ["loo", "r2"]
    holdout_periods: Optional[int] = None  # For time-series CV


class MMMSpec(BaseModel):
    """Complete MMM specification"""
    # Metadata
    name: Optional[str] = "MMM Model"
    description: Optional[str] = None
    version: Optional[str] = "1.0"
    
    # Data configuration
    outcome: str
    time_unit: Literal["day", "week", "month"] = "week"
    date_column: Optional[str] = None  # Auto-detected if not specified
    
    # Model components
    channels: List[Channel]
    controls: Optional[List[str]] = []
    hierarchy: Optional[HierarchyConfig] = None
    
    # Advanced structure (TVPs, multivariate priors). Default enabled for agentic modeling.
    advanced: Optional[AdvancedModelConfig] = Field(
        default_factory=lambda: AdvancedModelConfig(),
        description="When omitted, defaults to time_varying_parameters=True, correlated_channel_priors=True."
    )
    
    # Priors and inference
    priors: PriorsConfig = PriorsConfig()
    inference: InferenceConfig = InferenceConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    # Agent's reasoning (for autonomous mode)
    data_insights: Optional[Dict[str, Any]] = None
    
    @validator('channels')
    def validate_channels(cls, v):
        if not v:
            raise ValueError("At least one channel must be specified")
        return v
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MMMSpec':
        """Load spec from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, output_path: str):
        """Save spec to YAML file"""
        import numpy as np
        
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        spec_dict = self.dict(exclude_none=True)
        spec_dict = convert_numpy_types(spec_dict)
        
        with open(output_path, 'w') as f:
            yaml.dump(
                spec_dict,
                f,
                default_flow_style=False,
                sort_keys=False
            )
    
    def summary(self) -> str:
        """Human-readable summary"""
        return f"""
MMM Spec: {self.name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Outcome: {self.outcome}
Channels: {len(self.channels)} ({', '.join([c.name for c in self.channels[:3]])}{'...' if len(self.channels) > 3 else ''})
Controls: {len(self.controls)} variables
Hierarchy: {self.hierarchy.dimension if self.hierarchy else 'None'}
Backend: {self.inference.backend}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Example usage
if __name__ == "__main__":
    # Example spec
    spec = MMMSpec(
        name="Search + TV MMM",
        outcome="bookings",
        time_unit="week",
        channels=[
            Channel(
                name="search",
                data_analysis=DataAnalysis(
                    autocorrelation="decays slowly over 8 weeks",
                    spending_pattern="exponential diminishing returns"
                ),
                transform=ChannelTransform(
                    adstock=AdstockConfig(type="geometric", max_lag=8),
                    saturation=SaturationConfig(type="hill")
                )
            ),
            Channel(
                name="tv",
                transform=ChannelTransform(
                    adstock=AdstockConfig(type="delayed", max_lag=4),
                    saturation=SaturationConfig(type="logistic")
                )
            )
        ],
        controls=["price_index", "macro_index"],
        inference=InferenceConfig(backend="numpyro_svi", num_steps=50000)
    )
    
    print(spec.summary())
    
    # Save to YAML
    spec.to_yaml("/tmp/test_spec.yaml")
    
    # Load from YAML
    loaded = MMMSpec.from_yaml("/tmp/test_spec.yaml")
    print("Spec validated and loaded successfully!")

