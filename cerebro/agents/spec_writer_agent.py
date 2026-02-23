"""
Autonomous Spec Writer Agent
Analyzes data and generates MMM specs (not code!)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import yaml
import logging
from pathlib import Path

from cerebro.spec import MMMSpec, Channel, ChannelTransform, AdstockConfig, SaturationConfig, DataAnalysis

logger = logging.getLogger(__name__)


class AutonomousSpecWriter:
    """
    Agent that analyzes MMM data and writes specs autonomously
    
    Key insight: Agent generates STRUCTURED YAML, not raw Python code
    This makes it validatable and deterministic code generation possible
    """
    
    def __init__(self, llm, use_rag: bool = True):
        """
        Args:
            llm: LLM backend (MLXBackend, GeminiBackend, etc.)
            use_rag: Whether to use RAG to enhance spec generation
        """
        self.llm = llm
        self.use_rag = use_rag
        
        if use_rag:
            try:
                from cerebro.llm.rag_backend import RAGBackend
                self.rag = RAGBackend()
                logger.info("[OK] RAG system loaded (production MMM examples available)")
            except Exception as e:
                logger.warning(f"Could not load RAG: {e}")
                self.rag = None
                self.use_rag = False
        else:
            self.rag = None
    
    def analyze_and_generate_spec(
        self,
        data_path: str,
        target_column: Optional[str] = None
    ) -> MMMSpec:
        """
        Analyze data and generate MMM spec autonomously
        
        Args:
            data_path: Path to CSV file
            target_column: Target column name (auto-detected if None)
        
        Returns:
            MMMSpec object
        """
        logger.info("AUTONOMOUS SPEC GENERATION")
        
        # Step 1: Load and profile data
        logger.info("Step 1: Loading and profiling data...")
        data = pd.read_csv(data_path)
        profile = self._profile_data(data)
        
        logger.info(f"  [OK] Loaded {len(data)} rows, {len(data.columns)} columns")
        logger.info(f"  [OK] Detected {len(profile['media_channels'])} media channels")
        logger.info(f"  [OK] Target: {profile['target_column']}")
        
        # Step 2: Ask agent to generate spec
        logger.info("\nStep 2: Generating spec with LLM...")
        spec_yaml = self._generate_spec_with_llm(data, profile)
        
        # Step 3: Parse and validate
        logger.info("\nStep 3: Parsing and validating spec...")
        spec = self._parse_and_validate_spec(spec_yaml)
        
        # Step 4: Add data path to spec metadata
        if not spec.data_insights:
            spec.data_insights = {}
        spec.data_insights['data_path'] = data_path
        spec.data_insights['n_observations'] = len(data)
        
        logger.info("[OK] SPEC GENERATION COMPLETE!")
        logger.info(spec.summary())
        
        return spec
    
    def _classify_columns_semantic(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Use LLM to semantically classify columns instead of keyword matching"""
        
        # Build column statistics for LLM
        col_stats = []
        for col in data.columns:
            stats = {
                'name': col,
                'dtype': str(data[col].dtype),
                'nunique': int(data[col].nunique()),
                'pct_unique': float(data[col].nunique() / len(data) * 100),
                'sample_values': data[col].head(5).tolist(),
                'has_nulls': bool(data[col].isnull().any())
            }
            if data[col].dtype in [np.float64, np.int64]:
                stats['min'] = float(data[col].min())
                stats['max'] = float(data[col].max())
                stats['mean'] = float(data[col].mean())
                stats['std'] = float(data[col].std())
            col_stats.append(stats)
        
        # Prompt for LLM
        prompt = f"""You are an expert data scientist analyzing MMM (Marketing Mix Model) data.

TASK: Classify each column as one of: DATE, TARGET, MEDIA_CHANNEL, CONTROL, or ID.

RULES:
1. DATE: Time identifier (week, date, day, month, year)
2. TARGET: The outcome variable (sales, visits, revenue, conversions)
3. MEDIA_CHANNEL: Marketing spend/impressions (channel, ad, media, impression, spend, cost)
4. CONTROL: External factors (seasonality, weather, price, macro indicators)
5. ID: Identifiers that should NOT be used as features (dmacode, store_id, user_id, etc.)
   - High cardinality (>10% unique values)
   - Sequential patterns (500, 501, 502...)
   - Names with "id", "code", "key"

DATASET INFO:
- Total rows: {len(data)}
- Total columns: {len(data.columns)}

COLUMNS:
"""
        for i, stats in enumerate(col_stats, 1):
            prompt += f"\n{i}. {stats['name']}\n"
            prompt += f"   dtype: {stats['dtype']}, unique: {stats['nunique']} ({stats['pct_unique']:.1f}%)\n"
            prompt += f"   sample: {stats['sample_values']}\n"
            if 'min' in stats:
                prompt += f"   range: [{stats['min']:.2f}, {stats['max']:.2f}], mean: {stats['mean']:.2f}\n"
        
        prompt += """
OUTPUT FORMAT (JSON):
{
  "date": "column_name or null",
  "target": "column_name",
  "media_channels": ["channel1", "channel2", ...],
  "controls": ["control1", "control2", ...],
  "ids_excluded": ["id1", "id2", ...]
}

IMPORTANT:
- dmacode, store_id, user_id, etc. should go in "ids_excluded", NOT "controls"
- Only put true exogenous variables in "controls"
- Use exact column names as they appear

JSON:"""
        
        # Get LLM response
        response = ""
        for chunk in self.llm.reason(prompt, stream=True):
            response += chunk
        
        # Parse JSON
        import json
        # Extract JSON from response (might have markdown)
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        try:
            classification = json.loads(response)
            logger.info(f"[OK] Semantic column classification:")
            logger.info(f"  Date: {classification['date']}")
            logger.info(f"  Target: {classification['target']}")
            logger.info(f"  Media channels: {len(classification['media_channels'])}")
            logger.info(f"  Controls: {len(classification['controls'])}")
            logger.info(f"  IDs excluded: {classification.get('ids_excluded', [])}")
            return classification
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response: {response}")
            # Fallback to simple heuristics
            return self._fallback_classification(data)
    
    def _fallback_classification(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback to simple heuristics if LLM fails"""
        logger.warning("Using fallback heuristic classification")
        
        # Date
        date_col = None
        for col in data.columns:
            if any(kw in col.lower() for kw in ['date', 'week', 'day', 'month', 'year', 'time']):
                date_col = col
                break
        
        # Target
        target_col = None
        for col in data.columns:
            if any(kw in col.lower() for kw in ['target', 'sales', 'revenue', 'visits', 'bookings', 'conversions']):
                target_col = col
                break
        if not target_col:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            target_col = numeric_cols[-1] if len(numeric_cols) > 0 else data.columns[-1]
        
        # Media channels
        media_keywords = ['spend', 'impression', 'cost', 'channel', 'ad', 'media']
        media_channels = []
        for col in data.columns:
            if col in [target_col, date_col]:
                continue
            if any(kw in col.lower() for kw in media_keywords):
                media_channels.append(str(col))
        
        # Controls - EXCLUDE high cardinality IDs
        id_keywords = ['id', 'code', 'key', 'index']
        controls = []
        for col in data.columns:
            if col in [target_col, date_col] + media_channels:
                continue
            # Skip if looks like ID
            if any(kw in col.lower() for kw in id_keywords):
                continue
            # Skip if high cardinality (>10% unique)
            if data[col].nunique() / len(data) > 0.10:
                continue
            if data[col].dtype in [np.float64, np.int64]:
                controls.append(col)
        
        return {
            'date': date_col,
            'target': target_col,
            'media_channels': media_channels,
            'controls': controls,
            'ids_excluded': []
        }
    
    def _profile_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Profile the dataset to understand structure using semantic LLM-based classification"""
        profile = {}
        
        # Use LLM to classify columns semantically
        column_classification = self._classify_columns_semantic(data)
        
        date_col = column_classification['date']
        target_col = column_classification['target']
        media_channels = column_classification['media_channels']
        controls = column_classification['controls']
        
        profile['date_column'] = date_col
        profile['target_column'] = target_col
        profile['media_channels'] = media_channels
        profile['controls'] = controls
        
        # Basic statistics
        profile['n_obs'] = len(data)
        profile['date_range'] = f"{data[date_col].min()} to {data[date_col].max()}" if date_col else "unknown"
        
        # Channel statistics
        profile['channel_stats'] = {}
        for ch in media_channels:
            profile['channel_stats'][ch] = {
                'mean': float(data[ch].mean()),
                'std': float(data[ch].std()),
                'min': float(data[ch].min()),
                'max': float(data[ch].max()),
                'zero_pct': float((data[ch] == 0).mean() * 100)
            }
        
        return profile
    
    def _generate_spec_with_llm(self, data: pd.DataFrame, profile: Dict[str, Any]) -> str:
        """Use LLM to generate spec YAML"""
        
        # Get RAG examples if available
        rag_context = ""
        if self.use_rag and self.rag:
            logger.info("Retrieving production MMM examples from RAG...")
            rag_context = self._get_rag_examples()
        
        # Create data summary for LLM
        data_summary = f"""
Dataset Overview:
- Observations: {profile['n_obs']}
- Date range: {profile['date_range']}
- Target: {profile['target_column']}
- Media channels: {len(profile['media_channels'])}
- Controls: {len(profile['controls'])}

Media Channels:
{self._format_channel_stats(profile['channel_stats'])}

Sample data (first 5 rows):
{data.head().to_string()}
"""
        
        # Format exact channel names
        exact_channels = '\n'.join([f'  - "{ch}"' for ch in profile['media_channels']])
        
        prompt = f"""You are an expert Marketing Mix Modeler. Analyze this dataset and generate a YAML spec for the MMM model.

{data_summary}

{rag_context}

**CRITICAL: Use EXACT column names from the data - DO NOT SHORTEN OR SIMPLIFY THEM!**

Exact channel names to use:
{exact_channels}

Your task: Generate a complete YAML spec following this structure. Use the production examples above as inspiration for sophisticated transformations.

```yaml
name: "Descriptive Name"
description: "Brief description"
outcome: "{profile['target_column']}"
time_unit: "week"
date_column: "{profile['date_column']}"

channels:
  - name: "USE EXACT NAME FROM LIST ABOVE - DO NOT MODIFY!"
    data_analysis:
      autocorrelation: "your analysis of lag structure"
      spending_pattern: "your analysis of diminishing returns"
    transform:
      adstock:
        type: "geometric"  # CHOOSE: geometric, delayed, carryover, exponential
        max_lag: 6  # Based on your analysis
        reasoning: "Why you chose this"
      saturation:
        type: "hill"  # CHOOSE: hill, logistic, exponential
        reasoning: "Why you chose this"
  
  # ... repeat for each channel

controls: {profile['controls']}

priors:
  baseline: "normal(0, 1)"
  channel_effects: "half_normal(0.5)"
  control_effects: "normal(0, 0.5)"
  sigma: "half_normal(100)"

inference:
  backend: "numpyro_nuts"  # ALWAYS use numpyro_nuts (MCMC) for production - most robust and accurate
  num_warmup: 500
  num_samples: 500
  num_chains: 1  # Use 1 chain for faster execution (sequential)
  reasoning: "MCMC provides full posterior uncertainty quantification"

evaluation:
  metrics: ["loo", "r2", "mape"]
```

IMPORTANT GUIDELINES:
1. For each channel, ANALYZE the data characteristics
2. Choose adstock type based on:
   - geometric: simple decay (good default)
   - delayed: if effects peak after delay (e.g., TV)
   - carryover: momentum effects
   - exponential: exponential decay
3. Choose saturation based on spending pattern
4. Explain your reasoning
5. Output ONLY valid YAML, no markdown fences, no extra text

Generate the complete spec now:"""
        
        logger.info("Calling LLM to generate spec...")
        response = self.llm.reason(prompt)
        
        # Clean up response (remove markdown fences and model artifacts)
        yaml_text = response.strip()
        
        # Remove markdown code fences
        if yaml_text.startswith('```yaml'):
            yaml_text = yaml_text[7:]
        if yaml_text.startswith('```'):
            yaml_text = yaml_text[3:]
        if yaml_text.endswith('```'):
            yaml_text = yaml_text[:-3]
        
        # Remove model special tokens
        for token in ['<|im_end|>', '<|endoftext|>', '<|end|>', '<|im_start|>']:
            yaml_text = yaml_text.replace(token, '')
        
        # Remove any trailing backticks
        yaml_text = yaml_text.rstrip('`').strip()
        
        return yaml_text.strip()
    
    def _get_rag_examples(self) -> str:
        """Retrieve production MMM examples from RAG"""
        if not self.rag:
            return ""
        
        queries = [
            "geometric adstock transformation implementation",
            "delayed adstock with peak effect after lag",
            "hill saturation curve for diminishing returns",
            "hierarchical bayesian MMM with regional effects",
            "numpyro SVI for large scale MMM"
        ]
        
        examples = []
        for query in queries:
            try:
                results = self.rag.search(query, n_results=1)
                if results:
                    examples.append(results[0])
            except:
                continue
        
        if not examples:
            return ""
        
        rag_section = """

PRODUCTION MMM CODE EXAMPLES (from RAG)


These are real production MMM patterns from frameworks like PyMC-Marketing,
LightweightMMM, Meridian, etc. Use these as inspiration for sophisticated
transformations:

"""
        for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
            rag_section += f"\nExample {i}:\n```python\n{example[:500]}...\n```\n"
        
        rag_section += """


"""
        return rag_section
    
    def _format_channel_stats(self, stats: Dict[str, Dict]) -> str:
        """Format channel statistics for LLM"""
        lines = []
        for ch, st in stats.items():
            lines.append(f"  {ch}:")
            lines.append(f"    mean={st['mean']:.2f}, std={st['std']:.2f}, zero%={st['zero_pct']:.1f}%")
        return "\n".join(lines)
    
    def _parse_and_validate_spec(self, yaml_text: str) -> MMMSpec:
        """Parse YAML and validate with Pydantic"""
        try:
            spec_dict = yaml.safe_load(yaml_text)
            spec = MMMSpec(**spec_dict)
            logger.info("[OK] Spec validated successfully")
            return spec
        except Exception as e:
            logger.error(f"Failed to parse/validate spec: {e}")
            logger.error("Generated YAML:")
            logger.error(yaml_text)
            raise
    
    def save_spec(self, spec: MMMSpec, output_path: str):
        """Save spec to YAML file"""
        spec.to_yaml(output_path)
        logger.info(f"[OK] Spec saved to: {output_path}")


# CLI for testing
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Test with real data
    data_path = "/Users/adityapu/Documents/GitHub/deepcausalmmm/examples/data/MMM Data.csv"
    
    if not Path(data_path).exists():
        print(f"[ERROR] Data not found: {data_path}")
        sys.exit(1)
    
    # Use Gemini (free) or MLX (local)
    from cerebro.llm import AutoBackend
    
    llm = AutoBackend(model="qwen2.5-coder:7b")  # Use 7B for speed
    
    # Create agent
    agent = AutonomousSpecWriter(llm=llm)
    
    # Generate spec
    spec = agent.analyze_and_generate_spec(data_path)
    
    # Save
    agent.save_spec(spec, "/tmp/autonomous_mmm_spec.yaml")
    
    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print(f"Spec saved to: /tmp/autonomous_mmm_spec.yaml")
    print(f"\nTo generate code:")
    print(f"  cerebro generate /tmp/autonomous_mmm_spec.yaml")

