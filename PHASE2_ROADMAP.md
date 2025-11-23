# CEREBRO - PHASE 2 ROADMAP: CAUSAL INFERENCE EXTENSIONS

## Vision: Universal Causal & Experimental Analysis System

**Expand from**: Marketing Mix Modeling only  
**Expand to**: Complete causal inference & experimentation toolkit

---

## Phase 2 Scope

### 1. A/B Testing & Experimentation
- Frequentist tests (t-test, Mann-Whitney, chi-square)
- Bayesian A/B testing (PyMC, NumPyro)
- Multi-armed bandits
- Sequential testing
- Power analysis & sample size calculation
- Stratified sampling design

### 2. Causal Inference
- Propensity score matching
- Inverse probability weighting (IPW)
- Difference-in-differences (DiD)
- Regression discontinuity design (RDD)
- Synthetic control methods
- Instrumental variables (IV)

### 3. Counterfactual Analysis
- Backdoor adjustment (DoWhy)
- Front-door adjustment
- Mediation analysis
- Sensitivity analysis
- Bounds estimation

### 4. Geo Testing
- Geo-level experiments
- Matched market testing
- Geographic lift studies
- Time-based geolift (Google's GeoLift)

### 5. Uplift Modeling
- Treatment effect heterogeneity
- CATE estimation (EconML)
- Meta-learners (S, T, X, R learners)
- Causal forests

---

## Technical Implementation

### New Spec Schemas

```python
# cerebro/spec/ab_test_schema.py
class ABTestSpec(BaseModel):
    test_name: str
    treatment_column: str
    outcome_column: str
    stratification_vars: Optional[List[str]]
    test_type: Literal["ttest", "welch", "mann_whitney", "bayesian", "sequential"]
    alpha: float = 0.05
    power: float = 0.8
    mde: Optional[float]  # Minimum detectable effect

# cerebro/spec/causal_schema.py
class CausalInferenceSpec(BaseModel):
    treatment: str
    outcome: str
    confounders: List[str]
    method: Literal["psm", "ipw", "did", "rdd", "iv", "backdoor"]
    estimand: Literal["ate", "att", "atc"]
    
# cerebro/spec/geo_test_schema.py
class GeoTestSpec(BaseModel):
    geo_column: str
    treatment_geos: List[str]
    control_geos: List[str]
    outcome: str
    covariates: List[str]
    method: Literal["matched_markets", "synthetic_control", "geolift"]
    
# cerebro/spec/uplift_schema.py
class UpliftSpec(BaseModel):
    treatment: str
    outcome: str
    features: List[str]
    model_type: Literal["s_learner", "t_learner", "x_learner", "causal_forest"]
```

### New Agents

```python
# cerebro/agents/ab_testing_agent.py
class ABTestingAgent(BaseAgent):
    """Generates A/B test analysis code"""
    - Statistical tests
    - Bayesian inference
    - Power analysis
    - Sequential monitoring
    
# cerebro/agents/causal_agent.py
class CausalInferenceAgent(BaseAgent):
    """Generates causal inference code using DoWhy/EconML"""
    - Identify causal graph
    - Estimate causal effects
    - Refute estimates (sensitivity)
    
# cerebro/agents/geo_testing_agent.py
class GeoTestingAgent(BaseAgent):
    """Generates geo experiment code"""
    - Matched market selection
    - Synthetic control
    - Geolift analysis
    
# cerebro/agents/uplift_agent.py
class UpliftAgent(BaseAgent):
    """Generates uplift modeling code"""
    - Meta-learner selection
    - CATE estimation
    - Treatment effect heterogeneity
```

### Enhanced RAG Database

Expand from 4,049 to ~10,000 examples:

**Current Sources (keep):**
- PyMC-Marketing, LightweightMMM, Meridian, Robyn
- NumPyro, PyMC, JAX examples
- CausalML, DoWhy, EconML (already have 789 examples)

**New Sources to Add:**

#### A/B Testing
- [Bayesian A/B Testing (PyMC)](https://github.com/pymc-devs/pymc-examples)
- [Bandits library](https://github.com/bgalbraith/bandits)
- [AB Testing Calculator](https://github.com/bookingcom/powercalculator)

#### Causal Inference
- [CausalML](https://github.com/uber/causalml) - MORE examples
- [DoWhy](https://github.com/py-why/dowhy) - MORE examples  
- [EconML](https://github.com/microsoft/EconML) - MORE examples
- [CausalPy](https://github.com/pymc-labs/CausalPy)
- [CausalImpact (Python)](https://github.com/jamalsenouci/causalimpact)

#### Geo Testing
- [GeoLift (R → Python)](https://github.com/facebookincubator/GeoLift)
- Synthetic Control Method examples

#### Uplift Modeling
- CausalML uplift models
- [PyLift](https://github.com/wayfair/pylift)
- EconML heterogeneous treatment effects

**Total Expected**: ~10,000 examples after Phase 2

---

## Implementation Plan

### Step 1: RAG Enhancement (Week 1)
```bash
# Add to fine_tuning/rebuild_rag_with_api_examples.py
new_repos = {
    'causalml': 'https://github.com/uber/causalml.git',
    'dowhy': 'https://github.com/py-why/dowhy.git',
    'econml': 'https://github.com/microsoft/EconML.git',
    'causalpy': 'https://github.com/pymc-labs/CausalPy.git',
    'pylift': 'https://github.com/wayfair/pylift.git',
    'pymc_examples': 'https://github.com/pymc-devs/pymc-examples.git',
}

# Extract A/B testing, causal inference, uplift examples
python fine_tuning/rebuild_rag_with_causal_examples.py
```

### Step 2: Schema Definition (Week 1)
- Create `cerebro/spec/ab_test_schema.py`
- Create `cerebro/spec/causal_schema.py`
- Create `cerebro/spec/geo_test_schema.py`
- Create `cerebro/spec/uplift_schema.py`
- Add Pydantic validation

### Step 3: Agent Implementation (Week 2-3)
- Implement `ABTestingAgent`
- Implement `CausalInferenceAgent`
- Implement `GeoTestingAgent`
- Implement `UpliftAgent`
- Test with sample data

### Step 4: CLI Extension (Week 3)
```bash
# New commands
cerebro abtest data.csv --treatment variant --outcome conversion
cerebro causal data.csv --treatment campaign --outcome sales --confounders age,income
cerebro geotest data.csv --treatment-geos "NY,CA" --control-geos "TX,FL"
cerebro uplift data.csv --treatment offer --outcome purchase --features age,history
```

### Step 5: Examples & Documentation (Week 4)
- `examples/abtest_bayesian.py`
- `examples/causal_did_analysis.py`
- `examples/geo_matched_markets.py`
- `examples/uplift_metalearner.py`
- Update README with new capabilities

### Step 6: Testing & Validation (Week 4)
- Unit tests for each agent
- Integration tests with real data
- Validate generated code correctness
- Benchmark against hand-written code

---

## Success Metrics

### Quantitative
- RAG examples: 4,049 → 10,000+
- Supported methods: 1 (MMM) → 5 (MMM, A/B, Causal, Geo, Uplift)
- Agent count: 6 → 10
- Example coverage: 100+ real-world scenarios

### Qualitative
- Generate correct A/B test code (Bayesian & frequentist)
- Generate valid DoWhy/EconML causal inference pipelines
- Generate geo experiment analysis code
- Generate uplift modeling pipelines
- All outputs: 1000+ lines, production-quality

---

## Dependencies to Add

```python
# setup.py additions
install_requires=[
    # Existing
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pymc>=5.0.0",
    "numpyro>=0.13.0",
    
    # Phase 2: Causal & Experimental
    "dowhy>=0.11.0",           # Causal inference
    "econml>=0.14.0",          # Causal ML
    "causalml>=0.15.0",        # Uber's causal lib
    "pylift>=0.1.0",           # Uplift modeling
    "causalpy>=0.2.0",         # PyMC causal
    "scipy>=1.10.0",           # Statistical tests
]
```

---

## File Structure After Phase 2

```
cerebro/
├── cerebro/
│   ├── agents/
│   │   ├── ab_testing_agent.py          [NEW]
│   │   ├── causal_agent.py              [NEW]
│   │   ├── geo_testing_agent.py         [NEW]
│   │   ├── uplift_agent.py              [NEW]
│   │   └── ... (existing MMM agents)
│   │
│   ├── spec/
│   │   ├── ab_test_schema.py            [NEW]
│   │   ├── causal_schema.py             [NEW]
│   │   ├── geo_test_schema.py           [NEW]
│   │   ├── uplift_schema.py             [NEW]
│   │   └── mmm_schema.py                (existing)
│   │
│   └── cli.py                            (extended)
│
├── examples/
│   ├── mmm_autonomous.py                 (existing)
│   ├── abtest_bayesian.py               [NEW]
│   ├── causal_did.py                    [NEW]
│   ├── geo_synthetic_control.py         [NEW]
│   └── uplift_metalearner.py            [NEW]
│
└── fine_tuning/
    ├── rebuild_rag_with_causal_examples.py  [NEW]
    └── rag_complete_causal.jsonl            [NEW - 10K examples]
```

---

## Timeline

**Phase 2 Duration**: 4 weeks (1 month)

- Week 1: RAG + Schemas
- Week 2-3: Agents
- Week 3: CLI
- Week 4: Examples + Tests

**Total Development**: ~80-100 hours

---

## Risks & Mitigations

### Risk 1: Library Compatibility
**Issue**: DoWhy/EconML/CausalML may have conflicting dependencies  
**Mitigation**: Use optional dependencies, test in isolated environments

### Risk 2: Code Quality
**Issue**: Causal inference code is more complex than MMM  
**Mitigation**: More RAG examples, better prompts, validation layer

### Risk 3: RAG Size
**Issue**: 10K examples = large index  
**Mitigation**: Chunk more aggressively, use better retrieval (hybrid search)

---

## Post-Phase 2 Vision

**Cerebro becomes**: Universal causal & experimental analysis system

**Use Cases**:
1. Marketing: MMM, geo tests, A/B tests
2. Product: Feature experiments, uplift modeling
3. Economics: Policy evaluation, causal inference
4. Healthcare: Treatment effect estimation
5. Social Science: Observational causal studies

**Differentiation**: 
- Only tool that generates production causal inference code autonomously
- Multi-method support (Bayesian, frequentist, causal ML)
- RAG-powered with 10K+ real examples

---

## Ready to Start Phase 2?

After pushing Phase 1 to GitHub, begin with:

```bash
# Create Phase 2 branch
git checkout -b phase2-causal-extensions

# Start with RAG enhancement
python fine_tuning/rebuild_rag_with_causal_examples.py
```

Then iteratively add agents, schemas, and examples.

---

**Phase 2 will transform Cerebro from MMM tool → Universal Causal Inference System**

