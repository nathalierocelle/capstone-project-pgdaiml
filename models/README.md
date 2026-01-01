# Models Directory

This directory contains trained machine learning models, preprocessing artifacts, and model configuration files.

## Files

### Trained Models

1. **`best_binary_model.pkl`** 
   - **Algorithm**: Gradient Boosting Classifier
   - **Purpose**: Binary classification (Healthy vs Sick)
   - **Performance**:
     - Test Accuracy: 94%
     - F1-Score: 89.3%
     - Precision: 89% (Sick class)
     - Recall: 89% (Sick class)

2. **`best_multiclass_model.pkl`**
   - **Algorithm**: Random Forest Classifier
   - **Purpose**: Multi-class classification (31 thyroid conditions)
   - **Performance**:
     - Test Accuracy: 91%
     - Weighted F1-Score: 91%
     - Handles 31 diagnosis categories

### Preprocessing Artifacts

3. **`scaler_binary.pkl`** 
   - StandardScaler fitted on binary classification training data
   - Ensures consistent feature scaling for binary predictions

4. **`scaler_multiclass.pkl`**
   - StandardScaler fitted on multiclass classification training data
   - Ensures consistent feature scaling for multiclass predictions

5. **`label_encoder_multiclass.pkl`** 
   - LabelEncoder for multiclass diagnosis codes
   - Maps diagnosis codes to numerical labels and vice versa
   - Handles 31 unique thyroid condition categories

### Configuration

6. **`config.json`** 
   - Model configuration and metadata
   - Contains:
     - Feature column names
     - Model hyperparameters
     - Training parameters (random_state, test_size, cv_folds)
     - Performance metrics
     - Class distributions
     - Execution metadata

## 📊 Model Architecture

### Binary Classification Pipeline
```
Input Features (21) 
    ↓
StandardScaler (scaler_binary.pkl)
    ↓
Gradient Boosting Classifier (best_binary_model.pkl)
    ↓
Binary Prediction (0: Not Sick, 1: Sick)
```

### Multiclass Classification Pipeline
```
Input Features (21)
    ↓
StandardScaler (scaler_multiclass.pkl)
    ↓
Random Forest Classifier (best_multiclass_model.pkl)
    ↓
Label Prediction (31 diagnosis codes)
    ↓
LabelEncoder.inverse_transform (label_encoder_multiclass.pkl)
    ↓
Diagnosis Code (A, B, C, ..., GKJ)
```

## 🎯 Model Selection Process

### Binary Classification
Models evaluated (9 algorithms):
- ✅ **Gradient Boosting** (Selected - Best F1: 89.3%)
- Random Forest (F1: 88.1%)
- XGBoost (F1: 87.5%)
- SVM RBF (F1: 86.2%)
- Logistic Regression (F1: 84.7%)
- Decision Tree (F1: 82.3%)
- SVM Linear (F1: 81.9%)
- K-Nearest Neighbors (F1: 79.4%)
- Naive Bayes (F1: 76.8%)

### Multiclass Classification
Models evaluated (8 algorithms):
- **Random Forest** (Selected - Best Weighted F1: 91%)
- XGBoost (Weighted F1: 89.7%)
- Gradient Boosting (Weighted F1: 89.2%)
- SVM RBF (Weighted F1: 86.5%)
- Logistic Regression (Weighted F1: 84.1%)
- Decision Tree (Weighted F1: 82.7%)
- K-Nearest Neighbors (Weighted F1: 78.9%)
- Naive Bayes (Weighted F1: 75.3%)

## 🔧 Training Configuration

From `config.json`:

```json
{
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.2,
    "cv_folds": 5,
    "dataset_shape": [9172, 21],
    "binary_class_distribution": {
        "0": 6771,
        "1": 2401
    }
}
```

### Training Parameters
- **Data Split**: 80% training, 20% testing (stratified)
- **Cross-Validation**: 5-fold stratified CV
- **Random State**: 42 (for reproducibility)
- **Class Handling**: Stratified sampling to maintain class distribution

## Loading Models

### Python Example

```python
import joblib
import json

# Load binary classification pipeline
binary_model = joblib.load('models/best_binary_model.pkl')
binary_scaler = joblib.load('models/scaler_binary.pkl')

# Load multiclass classification pipeline
multiclass_model = joblib.load('models/best_multiclass_model.pkl')
multiclass_scaler = joblib.load('models/scaler_multiclass.pkl')
label_encoder = joblib.load('models/label_encoder_multiclass.pkl')

# Load configuration
with open('models/config.json', 'r') as f:
    config = json.load(f)

# Make prediction
import pandas as pd
import numpy as np

# Prepare features
features = config['feature_columns']
X = pd.DataFrame(...)  # Your input data

# Binary prediction
X_scaled = binary_scaler.transform(X[features])
binary_pred = binary_model.predict(X_scaled)
binary_proba = binary_model.predict_proba(X_scaled)

# Multiclass prediction
X_scaled = multiclass_scaler.transform(X[features])
multiclass_pred = multiclass_model.predict(X_scaled)
diagnosis_code = label_encoder.inverse_transform(multiclass_pred)
```

## Model Performance Details

### Binary Classification (Gradient Boosting)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Sick | 96% | 96% | 96% | 1,355 |
| Sick | 89% | 89% | 89% | 480 |
| **Accuracy** | | | **94%** | 1,835 |

### Multiclass Classification (Random Forest)

- **Overall Accuracy**: 91%
- **Weighted F1-Score**: 91%
- **Macro F1-Score**: 85%
- **Handles**: 31 unique diagnosis categories

### Top Performing Classes
- Class "-" (Healthy): 96% F1-score
- Class "K" (Concurrent Non-Thyroidal Illness): 88% F1-score
- Class "G" (Compensated Hypothyroid): 85% F1-score

## Model Versioning

| Version | Date | Binary Model | Multiclass Model | Notes |
|---------|------|--------------|------------------|-------|
| v1.0.0 | 2024-12-18 | Gradient Boosting (94% acc) | Random Forest (91% F1) | Initial release |

## Feature Importance

Top 10 most important features (from SHAP analysis):

1. TSH (Thyroid Stimulating Hormone)
2. FTI (Free Thyroxine Index)
3. TT4 (Total Thyroxine)
4. T3 (Triiodothyronine)
5. T4U (Thyroid hormone uptake)
6. age
7. T3_T4_ratio
8. TSH_T4_product
9. any_treatment
10. medical_complexity_score

## Usage Notes

### Important Considerations

1. **Feature Scaling Required**: Always use the corresponding scaler before prediction
2. **Feature Order**: Maintain exact feature order as in `config.json`
3. **Missing Values**: Handle missing values before scaling (median imputation recommended)
4. **Version Compatibility**: 
   - Python: 3.9.7+
   - scikit-learn: 1.3.0+
   - XGBoost: 2.0.0+
   - joblib: 1.3.0+

### Model Limitations

- Trained on specific dataset distribution (may not generalize to all populations)
- Performance may vary with significantly different patient demographics
- Should be used as decision support, not replacement for medical diagnosis
- Regular retraining recommended with new data

## Retraining

To retrain models with new data:

```python
# See notebook/4_Model_Implementation.ipynb for complete pipeline
```

Steps:
1. Prepare new data with same feature schema
2. Run preprocessing pipeline
3. Train new models with same hyperparameters
4. Evaluate on holdout test set
5. Compare with current model performance
6. Update model files if performance improves

## Related Files

- **Training Notebook**: `notebook/4_Model_Implementation.ipynb`
- **Evaluation Results**: `results/binary_classification_report.txt`
- **Explainability Analysis**: `notebook/5_Ethical_AI_Bias_Auditing.ipynb`
- **Web Application**: `src/app.py` (uses these models)

## Support

For issues with model loading or predictions, please refer to:
- Model implementation notebook
- Source code in `src/utils/model_loader.py`
- Open an issue on GitHub

---

**Note**: Models are serialized using joblib for efficient storage and loading. Ensure compatible versions of all dependencies when loading models in production environments.

**Disclaimer**: These models are for educational and research purposes. Not intended for clinical use without proper validation and regulatory approval.