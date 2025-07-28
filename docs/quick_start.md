# Quick Start Guide

Get up and running with the ML Pipeline Framework in just 10 minutes! This guide will walk you through creating your first machine learning pipeline.

## 🚀 Prerequisites

- ML Pipeline Framework installed ([Installation Guide](installation.md))
- Python 3.8+ environment activated
- Basic familiarity with machine learning concepts

## 📊 Step 1: Prepare Your Data

Let's start with a sample dataset. We'll create a synthetic fraud detection dataset:

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic credit card transaction data
n_samples = 10000
n_features = 15

# Create feature data
data = {
    'transaction_amount': np.random.lognormal(3, 1.5, n_samples),
    'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_samples),
    'transaction_hour': np.random.randint(0, 24, n_samples),
    'days_since_last_transaction': np.random.exponential(2, n_samples),
    'customer_age': np.random.normal(45, 15, n_samples).clip(18, 80),
    'account_balance': np.random.lognormal(8, 1, n_samples),
    'credit_limit': np.random.lognormal(9, 0.5, n_samples),
    'previous_transactions_today': np.random.poisson(2, n_samples),
    'is_weekend': np.random.choice([0, 1], n_samples, p=[5/7, 2/7]),
    'merchant_risk_score': np.random.beta(2, 5, n_samples),
    'location_risk_score': np.random.beta(1, 10, n_samples),
    'velocity_score': np.random.gamma(2, 2, n_samples),
}

# Add some derived features
data['amount_to_limit_ratio'] = data['transaction_amount'] / data['credit_limit']
data['unusual_time'] = ((data['transaction_hour'] <= 5) | (data['transaction_hour'] >= 23)).astype(int)
data['high_velocity'] = (data['velocity_score'] > np.percentile(data['velocity_score'], 95)).astype(int)

# Create target variable (fraud indicator)
# Fraud is more likely with high amounts, unusual times, high risk scores
fraud_probability = (
    0.1 * (data['transaction_amount'] > np.percentile(data['transaction_amount'], 95)) +
    0.2 * data['unusual_time'] +
    0.3 * (data['merchant_risk_score'] > 0.8) +
    0.2 * (data['location_risk_score'] > 0.8) +
    0.1 * (data['amount_to_limit_ratio'] > 0.9) +
    0.05  # base fraud rate
)

data['is_fraud'] = np.random.binomial(1, fraud_probability.clip(0, 1), n_samples)

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values to make it realistic
missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[missing_indices, 'merchant_risk_score'] = np.nan

# Save to CSV
df.to_csv('fraud_detection_data.csv', index=False)
print(f"✅ Created dataset with {len(df)} samples, {df['is_fraud'].mean():.1%} fraud rate")
print(f"📁 Saved to: fraud_detection_data.csv")
```

## ⚙️ Step 2: Initialize Configuration

Generate a configuration file tailored for fraud detection:

```bash
# Create configuration directory
mkdir -p configs

# Generate fraud detection configuration
ml-pipeline init --config-type fraud-detection --output configs/
```

This creates a comprehensive configuration file at `configs/fraud_detection_config.yaml`.

## 🔧 Step 3: Customize Configuration

Edit the generated configuration to point to your data:

```bash
# Edit the configuration file
nano configs/fraud_detection_config.yaml
```

Update the data source section:

```yaml
data_source:
  type: "csv"
  csv_options:
    file_paths:
      - "fraud_detection_data.csv"  # Update this path
    separator: ","
    encoding: "utf-8"
    header_row: 0
    validate_headers: true
    optimize_dtypes: true
```

Update the target variable:

```yaml
model_training:
  target:
    column: "is_fraud"  # Update target column name
    type: "classification"
    classes: [0, 1]
```

## 🎯 Step 4: Train Your First Model

Now let's train a machine learning model:

```bash
# Train with AutoML enabled
ml-pipeline train --config configs/fraud_detection_config.yaml --mode automl --verbose

# Or train with specific algorithms
ml-pipeline train --config configs/fraud_detection_config.yaml --mode train
```

Expected output:
```
🚀 Starting training in automl mode...
📊 Data loaded: 10000 rows, 16 columns
🔍 Target variable: is_fraud (1.7% positive class)
🤖 Starting AutoML with 6 algorithms...
⏰ Time budget: 3600 seconds

🔄 Training logistic_regression... ✅ Completed (AUC: 0.834)
🔄 Training random_forest... ✅ Completed (AUC: 0.892)
🔄 Training xgboost... ✅ Completed (AUC: 0.913)
🔄 Training lightgbm... ✅ Completed (AUC: 0.908)
🔄 Training catboost... ✅ Completed (AUC: 0.905)

🏆 Best model: xgboost (AUC: 0.913)
💾 Model saved to: ./artifacts/models/best_model.pkl
📊 Training report: ./artifacts/reports/training_report.html
```

## 🔮 Step 5: Make Predictions

Use your trained model to make predictions on new data:

```bash
# Create test data (first 1000 rows)
head -n 1001 fraud_detection_data.csv > test_data.csv

# Make predictions
ml-pipeline predict \
  --model ./artifacts/models/best_model.pkl \
  --data test_data.csv \
  --output predictions.csv \
  --include-probabilities
```

View your predictions:
```python
import pandas as pd

# Load predictions
predictions = pd.read_csv('predictions.csv')

# Display summary
print("🔮 Prediction Summary:")
print(f"Total predictions: {len(predictions)}")
print(f"Predicted fraud cases: {predictions['prediction'].sum()}")
print(f"Average fraud probability: {predictions['probability'].mean():.3f}")

# Show high-risk transactions
high_risk = predictions[predictions['probability'] > 0.8]
print(f"\n⚠️ High-risk transactions (>80% fraud probability): {len(high_risk)}")
if len(high_risk) > 0:
    print(high_risk[['transaction_amount', 'merchant_category', 'probability']].head())
```

## 🔍 Step 6: Explain Model Decisions

Generate explanations for your model's predictions:

```bash
# Generate SHAP explanations
ml-pipeline explain \
  --model ./artifacts/models/best_model.pkl \
  --data test_data.csv \
  --method shap \
  --generate-plots \
  --output explanations/

# Generate multiple explanation types
ml-pipeline explain \
  --model ./artifacts/models/best_model.pkl \
  --data test_data.csv \
  --method all \
  --generate-plots \
  --output explanations/
```

This creates:
- SHAP summary plots showing feature importance
- Individual prediction explanations
- Partial dependence plots
- Feature interaction analysis

## 📊 Step 7: View Results

Open the generated reports in your browser:

```bash
# View training report
open ./artifacts/reports/training_report.html

# View explanation dashboard
open ./explanations/shap_summary.html
```

The reports include:
- **Model Performance**: Accuracy, precision, recall, AUC-ROC
- **Feature Importance**: Which features matter most
- **Model Comparison**: How different algorithms performed
- **Business Impact**: Expected cost savings and ROI
- **Compliance**: Regulatory compliance metrics

## 🚀 Step 8: Deploy Your Model (Optional)

Deploy your trained model to a production environment:

```bash
# Generate Kubernetes deployment
ml-pipeline deploy \
  --model ./artifacts/models/best_model.pkl \
  --environment production \
  --platform kubernetes \
  --dry-run

# View generated deployment files
ls deploy_production_*/
```

## 📈 Step 9: Monitor Your Model (Optional)

Set up monitoring for your deployed model:

```bash
# Start monitoring dashboard
ml-pipeline monitor \
  --deployment-name fraud-model \
  --environment production \
  --dashboard \
  --metrics performance drift fairness
```

## 🎉 Congratulations!

You've successfully:
- ✅ Created a synthetic fraud detection dataset
- ✅ Configured an ML pipeline
- ✅ Trained multiple models with AutoML
- ✅ Made predictions on new data
- ✅ Generated model explanations
- ✅ Viewed comprehensive reports

## 🔄 Next Steps

### Explore Advanced Features

1. **Custom Feature Engineering**:
   ```yaml
   feature_engineering:
     enabled: true
     derived_features:
       - name: "risk_score"
         expression: "merchant_risk_score * location_risk_score"
         type: "numeric"
   ```

2. **Hyperparameter Tuning**:
   ```bash
   ml-pipeline train --config configs/fraud_detection_config.yaml --mode tune
   ```

3. **A/B Testing**:
   ```yaml
   experimentation:
     ab_testing:
       enabled: true
       test_traffic_percentage: 5
   ```

### Production Readiness

1. **Data Validation**:
   ```yaml
   preprocessing:
     data_quality:
       validation_level: "strict"
       checks:
         - type: "schema_validation"
         - type: "bias_detection"
   ```

2. **Model Monitoring**:
   ```yaml
   monitoring:
     drift_detection: true
     performance_monitoring: true
     fairness_monitoring: true
   ```

3. **Security & Compliance**:
   ```yaml
   security:
     data_encryption: true
     audit_logging: true
   compliance:
     frameworks: ["gdpr", "pci_dss"]
   ```

### Learning Resources

- **[Feature Engineering Guide](features/feature_engineering.md)** - Advanced feature creation
- **[AutoML Deep Dive](features/automl.md)** - Comprehensive AutoML usage
- **[Explainability Guide](features/explainability.md)** - Model interpretability
- **[Production Deployment](deployment/README.md)** - Enterprise deployment
- **[Example Notebooks](../notebooks/README.md)** - More complex examples

## 🆘 Need Help?

### Common Issues

1. **Configuration Errors**:
   ```bash
   # Validate your configuration
   ml-pipeline validate configs/fraud_detection_config.yaml
   ```

2. **Memory Issues**:
   ```yaml
   # Reduce batch size
   data_source:
     csv_options:
       chunk_size: 10000
   ```

3. **Performance Issues**:
   ```yaml
   # Enable parallel processing
   resources:
     compute:
       n_jobs: -1
   ```

### Getting Support

- 📖 **Documentation**: [Full Documentation](README.md)
- 💬 **Community**: [GitHub Discussions](https://github.com/ml-pipeline-framework/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ml-pipeline-framework/issues)
- 📧 **Email**: support@ml-pipeline-framework.io

---

**Ready for more?** Explore our [tutorials](tutorials/README.md) and [examples](examples/README.md) for advanced use cases!