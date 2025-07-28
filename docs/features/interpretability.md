# Interpretability Guide

Comprehensive guide to understanding and using the 15+ interpretability methods available in the ML Pipeline Framework for model explanations, regulatory compliance, and business insights.

## 📋 Table of Contents

- [Overview](#overview)
- [Method Categories](#method-categories)
- [Global Interpretability](#global-interpretability)
- [Local Interpretability](#local-interpretability)
- [Advanced Methods](#advanced-methods)
- [Fraud-Specific Explanations](#fraud-specific-explanations)
- [Choosing the Right Method](#choosing-the-right-method)
- [Configuration Guide](#configuration-guide)
- [Usage Examples](#usage-examples)
- [Regulatory Compliance](#regulatory-compliance)
- [Best Practices](#best-practices)

## 🔍 Overview

Model interpretability is crucial for fraud detection and regulated industries. Our framework provides 15+ explanation methods that help you understand:

- **Why** the model made a specific prediction
- **Which** features are most important
- **How** features interact with each other
- **What** would change the prediction
- **When** to trust the model's predictions

### Interpretability Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Model       │───▶│ Interpretability│───▶│   Explanations  │
│                 │    │    Pipeline     │    │                 │
│ • Any ML Model  │    │                 │    │ • Global        │
│ • Ensemble      │    │ • 15+ Methods   │    │ • Local         │
│ • Black Box     │    │ • Unified API   │    │ • Advanced      │
└─────────────────┘    │ • Compliance    │    │ • Reports       │
                       └─────────────────┘    └─────────────────┘
```

## 📊 Method Categories

### Global Interpretability
Methods that explain overall model behavior across all predictions:

| Method | Purpose | Best For | Output |
|--------|---------|----------|--------|
| **SHAP** | Feature importance with interactions | All models | Feature contributions |
| **Functional ANOVA** | Feature interactions | Complex models | Interaction effects |
| **ALE Plots** | Unbiased feature effects | Correlated features | Feature response curves |
| **Permutation Importance** | Model-agnostic importance | Any model | Feature rankings |
| **Surrogate Models** | Interpretable approximations | Black box models | Simple model rules |

### Local Interpretability
Methods that explain individual predictions:

| Method | Purpose | Best For | Output |
|--------|---------|----------|--------|
| **LIME** | Local linear approximations | Instance explanations | Feature contributions |
| **Anchors** | Rule-based explanations | Categorical data | If-then rules |
| **Counterfactuals** | Minimal changes to flip prediction | Actionable insights | Alternative scenarios |
| **ICE Plots** | Individual feature responses | Understanding variance | Response curves |

### Advanced Methods
Specialized methods for specific use cases:

| Method | Purpose | Best For | Output |
|--------|---------|----------|--------|
| **Trust Scores** | Prediction reliability | Uncertainty quantification | Confidence scores |
| **Prototypes** | Representative examples | Case-based reasoning | Similar instances |
| **Concept Activation** | High-level concept detection | Neural networks | Concept importance |
| **Causal Analysis** | True causal effects | Decision making | Causal relationships |

## 🌍 Global Interpretability

### SHAP (SHapley Additive exPlanations)

SHAP provides unified framework for feature importance with game-theoretic foundations.

#### Configuration
```yaml
explainability:
  global_interpretability:
    shap:
      enabled: true
      explainer_type: "auto"  # auto, tree, linear, kernel
      sample_size: 1000
      interaction_analysis: true
      plots:
        - "summary_plot"
        - "waterfall_plot"
        - "dependence_plot"
        - "interaction_plot"
```

#### Usage Example
```python
from explainability.interpretability_pipeline import InterpretabilityPipeline

# Initialize pipeline
pipeline = InterpretabilityPipeline(model)

# Generate SHAP explanations
shap_results = pipeline.explain_global(
    X_test, 
    method='shap',
    sample_size=1000
)

# Access results
feature_importance = shap_results['feature_importance']
interactions = shap_results['interactions']
```

#### Interpreting SHAP Results

**Feature Importance**: Shows average contribution of each feature
```python
# Top 10 most important features
top_features = feature_importance.head(10)
print("Most Important Features:")
for feature, importance in top_features.items():
    print(f"{feature}: {importance:.3f}")
```

**Feature Interactions**: Identifies feature pairs that interact
```python
# Top feature interactions
interactions_df = shap_results['interaction_matrix']
top_interactions = interactions_df.unstack().sort_values(ascending=False).head(10)
```

### Functional ANOVA

Decomposes model predictions into main effects and interactions using functional analysis of variance.

#### Configuration
```yaml
functional_anova:
  enabled: true
  max_order: 2  # Include up to 2-way interactions
  n_permutations: 100
  interaction_strength_threshold: 0.1
```

#### Key Insights
- **Main Effects**: Individual feature contributions
- **Interaction Effects**: How features work together
- **Higher-Order Effects**: Complex multi-feature interactions

### ALE (Accumulated Local Effects) Plots

Shows unbiased feature effects, especially useful when features are correlated.

#### Configuration
```yaml
ale_plots:
  enabled: true
  n_bins: 20
  center: true
  plot_pdp_comparison: true
  features: "auto"  # or specify: ["amount", "merchant_risk"]
```

#### Advantages over PDP
- Unbiased when features are correlated
- Shows local effects rather than marginal effects
- More reliable for real-world data

### Permutation Importance

Model-agnostic method that measures feature importance by observing score decrease when feature values are randomly shuffled.

#### Configuration
```yaml
permutation_importance:
  enabled: true
  n_repeats: 10
  scoring_metric: "precision_at_k"
  plot_top_n: 20
```

### Surrogate Models

Creates interpretable models that approximate the behavior of complex models.

#### Configuration
```yaml
surrogate_models:
  enabled: true
  models: ["decision_tree", "linear_model"]
  max_depth: 5
  fidelity_threshold: 0.8
```

## 🎯 Local Interpretability

### LIME (Local Interpretable Model-agnostic Explanations)

Explains individual predictions by fitting local linear models around the instance of interest.

#### Configuration
```yaml
local_interpretability:
  lime:
    enabled: true
    mode: "tabular"
    n_samples: 5000
    n_features: 10
    discretize_continuous: true
```

#### Usage Example
```python
# Explain a single prediction
instance_idx = 42
lime_explanation = pipeline.explain_local(
    X_test.iloc[instance_idx:instance_idx+1],
    method='lime'
)

# Get feature contributions
contributions = lime_explanation['feature_contributions']
prediction = lime_explanation['prediction']

print(f"Prediction: {prediction}")
print("Feature Contributions:")
for feature, contribution in contributions.items():
    print(f"  {feature}: {contribution:+.3f}")
```

#### Interpreting LIME Results
- **Positive values**: Features supporting the prediction
- **Negative values**: Features opposing the prediction
- **Magnitude**: Strength of feature influence

### Anchors

Provides rule-based explanations that define regions where the model behaves consistently.

#### Configuration
```yaml
anchors:
  enabled: true
  threshold: 0.95  # Precision threshold
  max_anchor_size: 5  # Maximum rule size
  coverage_samples: 10000
```

#### Example Output
```
IF transaction_amount <= 100 AND merchant_risk <= 0.3 
THEN prediction = NOT_FRAUD (precision: 0.97, coverage: 0.23)
```

### Counterfactual Explanations

Shows minimal changes needed to flip the model's prediction, providing actionable insights.

#### Configuration
```yaml
counterfactuals:
  enabled: true
  method: "dice"  # dice, wachter, prototype
  n_counterfactuals: 5
  max_features_changed: 3
  diversity_weight: 0.5
  actionability_constraints: true
```

#### Usage Example
```python
# Generate counterfactuals for a fraud case
fraud_instance = X_test[y_test == 1].iloc[0:1]
counterfactuals = pipeline.explain_counterfactual(
    fraud_instance,
    desired_class=0  # Want to see how to make it NOT fraud
)

print("To avoid fraud detection, change:")
for cf in counterfactuals['counterfactuals']:
    for feature, change in cf['changes'].items():
        print(f"  {feature}: {change['from']} → {change['to']}")
```

### ICE (Individual Conditional Expectation) Plots

Shows how predictions change as a single feature varies, for individual instances.

#### Configuration
```yaml
ice_plots:
  enabled: true
  n_samples: 100
  feature_subset: "top_10"
  center: true
```

## 🔬 Advanced Methods

### Trust Scores

Quantifies prediction reliability by measuring distance to nearest training examples.

#### Configuration
```yaml
advanced_interpretability:
  trust_scores:
    enabled: true
    k_neighbors: 10
    confidence_threshold: 0.8
    uncertainty_quantification: true
```

#### Usage Example
```python
# Calculate trust scores
trust_results = pipeline.calculate_trust_scores(X_test)

# Identify low-confidence predictions
low_confidence = trust_results['trust_scores'] < 0.5
risky_predictions = X_test[low_confidence]

print(f"Low confidence predictions: {len(risky_predictions)}")
```

### Prototypes and Criticisms

Finds representative examples (prototypes) and atypical examples (criticisms) from training data.

#### Configuration
```yaml
prototypes:
  enabled: true
  n_prototypes_per_class: 10
  selection_method: "mmcriticism"  # Maximum Mean Discrepancy
  include_criticisms: true
```

### Concept Activation Vectors

Identifies high-level concepts that the model has learned (primarily for neural networks).

#### Configuration
```yaml
concept_activation:
  enabled: true
  n_concepts: 20
  significance_level: 0.05
  concept_sensitivity: true
```

### Causal Analysis

Attempts to identify causal relationships rather than just correlations.

#### Configuration
```yaml
causal_analysis:
  enabled: true
  method: "do_calculus"
  confounding_adjustment: true
  backdoor_criterion: true
```

## 🚨 Fraud-Specific Explanations

### Reason Codes

Regulatory-compliant reason codes for fraud decisions.

#### Configuration
```yaml
fraud_specific:
  reason_codes:
    enabled: true
    code_mapping:
      "high_amount": "Transaction amount significantly above normal"
      "unusual_time": "Transaction at unusual time of day"
      "merchant_risk": "High-risk merchant category"
      "velocity": "High transaction velocity detected"
    max_codes: 4
```

#### Example Output
```
Fraud Decision Reasons:
1. High transaction amount (Score: 0.85)
2. Unusual transaction time (Score: 0.72)
3. High-risk merchant (Score: 0.64)
4. High velocity pattern (Score: 0.58)
```

### Narrative Explanations

Human-readable explanations in natural language.

#### Configuration
```yaml
narrative_explanations:
  enabled: true
  template_based: true
  include_confidence: true
  include_alternatives: true
  language: "en"
```

#### Example Output
```
This transaction was flagged as fraudulent with 89% confidence. 
The primary concern is the transaction amount of $2,450, which is 
significantly higher than the customer's typical spending of $85. 
Additionally, this transaction occurred at 3:47 AM, which is unusual 
for this customer who typically transacts between 9 AM and 8 PM.

To avoid this flag, the transaction amount should be below $500, 
or the transaction should occur during normal hours (9 AM - 8 PM).
```

### Risk Factor Analysis

Detailed analysis of contributing risk factors.

#### Configuration
```yaml
risk_factors:
  enabled: true
  factor_weights: true
  interactive_effects: true
  temporal_patterns: true
```

## 🧭 Choosing the Right Method

### Decision Tree for Method Selection

```
Need to explain...
├── Overall model behavior?
│   ├── Feature importance → SHAP, Permutation Importance
│   ├── Feature interactions → Functional ANOVA
│   └── Feature effects → ALE Plots
├── Individual predictions?
│   ├── Why this prediction? → LIME, SHAP
│   ├── What rules apply? → Anchors
│   └── How to change outcome? → Counterfactuals
├── Model reliability?
│   ├── Prediction confidence → Trust Scores
│   ├── Similar examples → Prototypes
│   └── Outlier detection → Trust Scores + Prototypes
└── Regulatory compliance?
    ├── Reason codes → Fraud-specific reasons
    ├── Human explanations → Narrative explanations
    └── Audit trails → SHAP + LIME + Anchors
```

### Method Comparison Matrix

| Method | Speed | Accuracy | Interpretability | Regulatory | Use Case |
|--------|-------|----------|------------------|------------|----------|
| **SHAP** | Medium | High | High | ✅ | General purpose |
| **LIME** | Fast | Medium | High | ✅ | Quick local explanations |
| **Anchors** | Slow | High | Very High | ✅ | Rule-based decisions |
| **Counterfactuals** | Medium | High | Medium | ⚠️ | Actionable insights |
| **Trust Scores** | Fast | High | Low | ⚠️ | Uncertainty estimation |
| **ALE** | Medium | Very High | Medium | ⚠️ | Unbiased feature effects |

### Recommendations by Use Case

#### Fraud Detection
```yaml
recommended_methods:
  - "shap"              # Feature importance
  - "lime"              # Instance explanations
  - "anchors"           # Decision rules
  - "reason_codes"      # Regulatory compliance
  - "trust_scores"      # Confidence assessment
```

#### Credit Risk
```yaml
recommended_methods:
  - "shap"              # Feature contributions
  - "counterfactuals"   # Improvement recommendations
  - "surrogate_models"  # Simple model approximation
  - "narrative_explanations"  # Customer communication
```

#### Healthcare
```yaml
recommended_methods:
  - "lime"              # Case-by-case explanations
  - "prototypes"        # Similar patient cases
  - "trust_scores"      # Clinical confidence
  - "causal_analysis"   # Treatment effects
```

## ⚙️ Configuration Guide

### Complete Interpretability Configuration

```yaml
# configs/explainability_config.yaml
explainability_settings:
  # Global methods
  global:
    shap:
      enabled: true
      max_samples: 1000
      algorithm: "auto"
      interaction_analysis: true
    
    ale_plots:
      enabled: true
      n_bins: 20
      center: true
      features: ["amount", "merchant_risk", "velocity"]
    
    permutation_importance:
      enabled: true
      n_repeats: 10
      scoring_metric: "precision_at_k"
  
  # Local methods
  local:
    lime:
      enabled: true
      n_samples: 5000
      n_features: 10
    
    anchors:
      enabled: true
      threshold: 0.95
      max_anchor_size: 5
    
    counterfactuals:
      enabled: true
      n_counterfactuals: 5
      max_features_changed: 3
  
  # Advanced methods
  advanced:
    trust_scores:
      enabled: true
      k_neighbors: 10
      confidence_threshold: 0.8
    
    prototypes:
      enabled: true
      n_prototypes_per_class: 10
  
  # Fraud-specific
  fraud_specific:
    reason_codes: true
    narrative_explanations: true
    risk_factors: true
  
  # Reporting
  reporting:
    auto_generate: true
    formats: ["html", "pdf", "json"]
    business_summary: true
    regulatory_format: true
```

### Performance Optimization

```yaml
# Performance settings
performance:
  parallel_processing: true
  n_jobs: -1
  cache_explanations: true
  cache_location: "./cache/explanations"
  
  # Approximation for speed
  approximation:
    enabled: true
    sample_size_reduction: 0.8
    feature_selection: true
    early_stopping: true
```

## 🚀 Usage Examples

### Basic Explanation Pipeline

```python
from explainability.interpretability_pipeline import InterpretabilityPipeline
import pandas as pd

# Load your trained model and test data
model = load_model('artifacts/best_model.pkl')
X_test = pd.read_csv('data/test.csv')

# Initialize interpretability pipeline
pipeline = InterpretabilityPipeline(model)

# Generate comprehensive explanations
explanations = pipeline.explain_all(
    X_test.head(100),  # Explain first 100 instances
    methods=['shap', 'lime', 'anchors'],
    generate_report=True,
    output_dir='explanations/'
)

# Access results
global_importance = explanations['shap']['feature_importance']
local_explanations = explanations['lime']['local_explanations']
decision_rules = explanations['anchors']['rules']
```

### Fraud Detection Example

```python
# Fraud-specific interpretability
fraud_pipeline = InterpretabilityPipeline(
    model,
    config_path='configs/explainability_config.yaml'
)

# Explain a fraud case
fraud_instance = X_test[y_test == 1].iloc[0:1]
fraud_explanation = fraud_pipeline.explain_fraud_case(
    fraud_instance,
    include_reason_codes=True,
    include_counterfactuals=True,
    include_narrative=True
)

# Print explanation
print("🚨 Fraud Detection Explanation")
print("=" * 40)
print(f"Fraud Probability: {fraud_explanation['probability']:.2%}")
print(f"Confidence: {fraud_explanation['confidence']:.2%}")

print("\n📋 Reason Codes:")
for i, reason in enumerate(fraud_explanation['reason_codes'], 1):
    print(f"{i}. {reason['description']} (Score: {reason['score']:.2f})")

print(f"\n📖 Explanation:")
print(fraud_explanation['narrative'])

print("\n🔄 To avoid fraud detection:")
for cf in fraud_explanation['counterfactuals']:
    print(f"  Change {cf['feature']}: {cf['from']} → {cf['to']}")
```

### Batch Explanation Processing

```python
def explain_batch(model, data, batch_size=100):
    """Process explanations in batches for large datasets."""
    
    pipeline = InterpretabilityPipeline(model)
    all_explanations = []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        
        # Generate explanations for batch
        batch_explanations = pipeline.explain_local(
            batch, 
            method='lime',
            parallel=True
        )
        
        all_explanations.extend(batch_explanations)
        print(f"Processed {i+len(batch)}/{len(data)} instances")
    
    return all_explanations

# Process large dataset
explanations = explain_batch(model, X_test, batch_size=500)
```

### Model Comparison with Explanations

```python
# Compare explanations across different models
models = {
    'xgboost': load_model('models/xgboost.pkl'),
    'lightgbm': load_model('models/lightgbm.pkl'),
    'random_forest': load_model('models/rf.pkl')
}

comparison_results = {}
test_instance = X_test.iloc[0:1]

for model_name, model in models.items():
    pipeline = InterpretabilityPipeline(model)
    explanation = pipeline.explain_local(test_instance, method='shap')
    
    comparison_results[model_name] = {
        'prediction': explanation['prediction'],
        'top_features': explanation['feature_contributions'].head(5)
    }

# Display comparison
print("Model Explanation Comparison:")
for model_name, results in comparison_results.items():
    print(f"\n{model_name.upper()}:")
    print(f"  Prediction: {results['prediction']:.3f}")
    print("  Top Features:")
    for feature, contrib in results['top_features'].items():
        print(f"    {feature}: {contrib:+.3f}")
```

## 📋 Regulatory Compliance

### GDPR Article 22 Compliance

Automated decision-making explanations for GDPR:

```yaml
regulatory_compliance:
  gdpr:
    enabled: true
    article_22_compliance: true
    right_to_explanation: true
    automated_decision_explanations: true
    human_reviewable: true
    
    # Required explanation elements
    required_elements:
      - "decision_reasoning"
      - "feature_contributions"
      - "alternative_outcomes"
      - "human_contact"
```

### SR 11-7 Banking Compliance

Model risk management for banking:

```yaml
regulatory_compliance:
  sr11_7:
    enabled: true
    model_documentation: true
    ongoing_monitoring: true
    validation_framework: true
    
    # Required documentation
    documentation:
      - "model_development"
      - "model_validation"
      - "ongoing_monitoring"
      - "model_limitations"
```

### Fair Lending Compliance

Fairness analysis for lending decisions:

```yaml
regulatory_compliance:
  fair_lending:
    enabled: true
    protected_attributes: ["age", "gender", "race", "ethnicity"]
    fairness_metrics: ["demographic_parity", "equalized_odds"]
    bias_testing: true
    disparate_impact_analysis: true
```

## 🎯 Best Practices

### 1. Method Selection Strategy

```python
def select_explanation_methods(use_case, data_size, model_type, time_budget):
    """Recommend explanation methods based on context."""
    
    methods = []
    
    # Always include SHAP for feature importance
    methods.append('shap')
    
    # Add based on use case
    if use_case == 'fraud_detection':
        methods.extend(['lime', 'reason_codes', 'trust_scores'])
    elif use_case == 'credit_risk':
        methods.extend(['counterfactuals', 'narrative_explanations'])
    elif use_case == 'healthcare':
        methods.extend(['prototypes', 'trust_scores'])
    
    # Add based on data size
    if data_size < 10000:
        methods.append('anchors')  # More thorough for small data
    
    # Add based on model type
    if model_type in ['xgboost', 'lightgbm', 'random_forest']:
        methods.append('ale_plots')  # Great for tree models
    
    # Add based on time budget
    if time_budget > 3600:  # > 1 hour
        methods.extend(['functional_anova', 'causal_analysis'])
    
    return list(set(methods))  # Remove duplicates
```

### 2. Explanation Quality Assurance

```python
def validate_explanations(explanations, model, X_test):
    """Validate explanation quality and consistency."""
    
    quality_metrics = {}
    
    # 1. Consistency check
    if 'shap' in explanations and 'lime' in explanations:
        shap_importance = explanations['shap']['feature_importance']
        lime_importance = explanations['lime']['average_importance']
        
        # Calculate correlation between methods
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(shap_importance, lime_importance)
        quality_metrics['method_consistency'] = correlation
    
    # 2. Fidelity check for surrogate models
    if 'surrogate_models' in explanations:
        surrogate = explanations['surrogate_models']['model']
        original_preds = model.predict(X_test)
        surrogate_preds = surrogate.predict(X_test)
        
        from sklearn.metrics import accuracy_score
        fidelity = accuracy_score(original_preds, surrogate_preds)
        quality_metrics['surrogate_fidelity'] = fidelity
    
    # 3. Stability check
    # Test explanation stability with small perturbations
    stability_scores = []
    for i in range(10):
        # Add small noise
        X_perturbed = X_test + np.random.normal(0, 0.01, X_test.shape)
        perturbed_explanation = explain_instance(model, X_perturbed.iloc[0])
        
        # Compare with original
        stability = calculate_explanation_similarity(
            explanations['lime']['local_explanations'][0],
            perturbed_explanation
        )
        stability_scores.append(stability)
    
    quality_metrics['explanation_stability'] = np.mean(stability_scores)
    
    return quality_metrics
```

### 3. Explanation Caching Strategy

```python
class ExplanationCache:
    """Cache explanations to avoid recomputation."""
    
    def __init__(self, cache_dir='./cache/explanations'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(self, model, instance, method):
        """Generate unique cache key."""
        import hashlib
        
        # Hash model parameters
        model_hash = hashlib.md5(
            str(model.get_params()).encode()
        ).hexdigest()[:8]
        
        # Hash instance
        instance_hash = hashlib.md5(
            instance.to_string().encode()
        ).hexdigest()[:8]
        
        return f"{model_hash}_{instance_hash}_{method}"
    
    def get(self, model, instance, method):
        """Get cached explanation if available."""
        cache_key = self.get_cache_key(model, instance, method)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, model, instance, method, explanation):
        """Cache explanation."""
        cache_key = self.get_cache_key(model, instance, method)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(explanation, f)
```

### 4. Business-Friendly Reporting

```python
def generate_business_report(explanations, model_performance):
    """Generate business-friendly explanation report."""
    
    report = {
        'executive_summary': {
            'model_performance': {
                'accuracy': f"{model_performance['accuracy']:.1%}",
                'fraud_detection_rate': f"{model_performance['recall']:.1%}",
                'false_alarm_rate': f"{model_performance['fpr']:.1%}"
            },
            'key_insights': [
                f"Transaction amount is the most important factor ({explanations['shap']['feature_importance'].iloc[0]:.1%} of decisions)",
                f"Time of day affects {explanations['ale']['time_effect']:.1%} of fraud cases",
                f"Model is {explanations['trust_scores']['avg_confidence']:.1%} confident on average"
            ]
        },
        'risk_factors': {
            'top_fraud_indicators': explanations['fraud_specific']['top_risk_factors'],
            'protective_factors': explanations['fraud_specific']['protective_factors'],
            'interaction_effects': explanations['functional_anova']['interactions']
        },
        'model_reliability': {
            'high_confidence_rate': f"{explanations['trust_scores']['high_confidence_rate']:.1%}",
            'avg_explanation_stability': f"{explanations['quality_metrics']['stability']:.1%}",
            'explanation_consistency': f"{explanations['quality_metrics']['consistency']:.1%}"
        },
        'recommendations': [
            "Focus fraud prevention on high-amount transactions during off-hours",
            "Implement additional verification for transactions flagged with low confidence",
            "Review merchant risk scoring based on explanation insights"
        ]
    }
    
    return report
```

---

**Ready to Start Explaining?** 🚀

1. **Quick Start**: Enable SHAP and LIME for basic explanations
2. **Customize**: Add fraud-specific reason codes and narratives
3. **Validate**: Check explanation quality and consistency
4. **Report**: Generate business-friendly explanation reports
5. **Comply**: Enable regulatory compliance features

For more examples and advanced usage, check out our [explanation notebooks](../examples/interpretability_examples.ipynb) and [regulatory compliance guide](../compliance/interpretability_requirements.md).