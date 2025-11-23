# Cerebro Capabilities: Current & Planned

## Phase 1: Marketing Mix Modeling (CURRENT - Ready to Push)

### What It Does
Autonomously generates production-grade Marketing Mix Model code from your data.

### Supported Methods
- **Bayesian MMM**: NumPyro SVI, PyMC NUTS, Stan sampling
- **Transformations**: Adstock (geometric, Weibull, delayed), Saturation (Hill, logistic)
- **Features**: Data exploration, preprocessing, modeling, diagnostics, optimization, visualization

### CLI Commands
```bash
# Generate complete MMM pipeline
cerebro auto data.csv --output mmm_pipeline.py

# Generate from spec
cerebro generate mmm_spec.yaml --output pipeline.py
```

### Example Use Cases
- TV/Digital/Social media attribution
- Budget optimization across channels
- ROI analysis and forecasting
- Channel contribution analysis

### RAG Database
- 4,049 examples from production repos
- Sources: PyMC-Marketing, LightweightMMM, Meridian, Robyn, NumPyro, PyMC, JAX

---

## Phase 2: Causal & Experimental Analysis (PLANNED - 4 weeks)

### New Capabilities

#### 1. A/B Testing & Experimentation
**Methods:**
- Frequentist tests (t-test, Mann-Whitney, chi-square)
- Bayesian A/B testing
- Multi-armed bandits
- Sequential testing
- Power analysis
- Stratified sampling design

**CLI:**
```bash
cerebro abtest data.csv \
  --treatment variant \
  --outcome conversion \
  --type bayesian

cerebro power-analysis \
  --baseline 0.05 \
  --mde 0.01 \
  --alpha 0.05 \
  --power 0.8
```

**Use Cases:**
- Website A/B testing
- Email campaign experiments
- Feature rollout experiments
- Price testing

---

#### 2. Causal Inference
**Methods:**
- Propensity Score Matching (PSM)
- Inverse Probability Weighting (IPW)
- Difference-in-Differences (DiD)
- Regression Discontinuity (RDD)
- Instrumental Variables (IV)
- Backdoor/Front-door adjustment

**CLI:**
```bash
cerebro causal data.csv \
  --treatment campaign \
  --outcome sales \
  --confounders age,income,history \
  --method did

cerebro causal data.csv \
  --treatment policy \
  --outcome outcome \
  --method psm \
  --estimand ate
```

**Use Cases:**
- Policy evaluation
- Treatment effect estimation
- Observational study analysis
- Marketing campaign causality

---

#### 3. Geo Testing
**Methods:**
- Matched market testing
- Synthetic control
- Geographic lift studies
- Time-based geo experiments

**CLI:**
```bash
cerebro geotest data.csv \
  --geo-column dma \
  --treatment-geos "NY,CA" \
  --control-geos "TX,FL" \
  --outcome sales \
  --method synthetic_control

cerebro geotest data.csv \
  --method matched_markets \
  --matching-vars population,income
```

**Use Cases:**
- Store expansion testing
- Regional campaign testing
- Geographic experiments
- Market-level interventions

---

#### 4. Uplift Modeling
**Methods:**
- S-learner, T-learner, X-learner, R-learner
- Causal forests
- CATE estimation
- Treatment effect heterogeneity

**CLI:**
```bash
cerebro uplift data.csv \
  --treatment offer \
  --outcome purchase \
  --features age,history,value \
  --model x_learner

cerebro uplift data.csv \
  --treatment campaign \
  --outcome conversion \
  --model causal_forest
```

**Use Cases:**
- Targeted marketing
- Personalized treatment
- Customer segmentation by treatment response
- Heterogeneous treatment effects

---

#### 5. Counterfactual Analysis
**Methods:**
- Counterfactual prediction
- Mediation analysis
- Sensitivity analysis
- Bounds estimation

**CLI:**
```bash
cerebro counterfactual data.csv \
  --treatment intervention \
  --outcome result \
  --mediator engagement \
  --confounders age,gender

cerebro sensitivity data.csv \
  --treatment campaign \
  --outcome sales \
  --unobserved-confounder-strength 0.3
```

**Use Cases:**
- "What if" scenario analysis
- Mediation pathway analysis
- Sensitivity to hidden confounders
- Robustness checks

---

## Comparison Matrix

| Capability | Phase 1 | Phase 2 |
|------------|---------|---------|
| **Marketing Mix Modeling** | ✅ Full | ✅ Full |
| **A/B Testing** | ❌ None | ✅ Full |
| **Causal Inference** | ❌ None | ✅ Full |
| **Geo Testing** | ❌ None | ✅ Full |
| **Uplift Modeling** | ❌ None | ✅ Full |
| **Counterfactual** | ❌ None | ✅ Full |
| | | |
| **RAG Examples** | 4,049 | 10,000+ |
| **Agents** | 6 | 10 |
| **Methods** | 1 domain | 6 domains |
| **CLI Commands** | 2 | 12+ |

---

## Technical Architecture (Phase 2)

### New Components

**Schemas:**
- `ABTestSpec` - A/B test configuration
- `CausalInferenceSpec` - Causal analysis setup
- `GeoTestSpec` - Geo experiment design
- `UpliftSpec` - Uplift modeling config

**Agents:**
- `ABTestingAgent` - Generates A/B test code
- `CausalInferenceAgent` - Generates DoWhy/EconML code
- `GeoTestingAgent` - Generates geo test code
- `UpliftAgent` - Generates uplift modeling code

**RAG Sources (Added):**
- CausalML (Uber) - More examples
- DoWhy (Microsoft) - More examples
- EconML (Microsoft) - More examples
- CausalPy (PyMC Labs)
- PyLift (Wayfair)
- PyMC A/B testing examples

---

## Dependencies (Phase 2 Additions)

```python
# Causal & Experimental
"dowhy>=0.11.0",           # Causal inference
"econml>=0.14.0",          # Causal ML
"causalml>=0.15.0",        # Uber causal lib
"pylift>=0.1.0",           # Uplift modeling
"causalpy>=0.2.0",         # PyMC causal
```

---

## Timeline

**Phase 1 (Current):**
- Status: Complete
- Capabilities: MMM only
- Ready: Yes

**Phase 2 (Planned):**
- Duration: 4 weeks
- Capabilities: MMM + A/B + Causal + Geo + Uplift
- Start: After Phase 1 push

---

## Vision

**Phase 1**: Best-in-class autonomous MMM system  
**Phase 2**: Universal causal & experimental analysis system

**End Goal**: One tool for all causal inference needs
- Marketing: MMM, geo tests, A/B tests
- Product: Experiments, uplift, CATE
- Economics: Policy evaluation, causal inference
- Healthcare: Treatment effects
- Social Science: Observational studies

---

**Current Status**: Phase 1 ready to push  
**Next**: Push to GitHub, then start Phase 2

