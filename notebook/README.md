# Notebooks Directory

This directory contains Jupyter notebooks documenting the complete data science workflow from data collection to ethical AI auditing.

## Notebooks Overview

### Sequential Workflow

The notebooks are numbered to follow a logical progression through the machine learning pipeline:

## Notebook 2: Data Collection & Understanding

**File**: `2_Data_Collection_&_Understanding.ipynb`

### Purpose
Initial exploration and understanding of the thyroid disease dataset.

### Contents
- **Data Source**: Kaggle thyroid disease dataset download and loading
- **Initial Exploration**: 
  - Dataset shape and structure
  - Feature data types
  - Missing value analysis
  - Basic statistical summaries
- **Target Variable Analysis**:
  - Binary classification target distribution
  - Multiclass diagnosis code distribution
  - Class imbalance assessment
- **Preliminary Insights**:
  - Data quality assessment
  - Identified preprocessing needs
  - Feature relationship hypotheses

### Key Findings
- Dataset contains 9,172 patient records
- 26.2% sick vs 73.8% healthy (binary)
- 31 unique diagnosis categories (multiclass)
- Significant class imbalance requiring handling strategies

### Outputs
- Understanding of data structure
- List of preprocessing requirements
- Initial feature correlation insights

---

## Notebook 3: Data Preprocessing, EDA & Feature Engineering

**File**: `3_Data_Preprocessing,_EDA_&_Feature_Engineering.ipynb`

### Purpose
Comprehensive data cleaning, exploratory analysis, and creation of engineered features.

### Contents

#### Data Preprocessing
- **Missing Value Handling**:
  - Median imputation for numerical features
  - Mode imputation for categorical features
  - Systematic approach to preserve data integrity
- **Outlier Detection & Treatment**:
  - IQR method for outlier identification
  - Clinical domain-informed outlier handling
- **Encoding**:
  - One-hot encoding for categorical variables
  - Label encoding for ordinal features
- **Data Type Conversions**:
  - Ensuring appropriate data types
  - Boolean to integer conversions

#### Exploratory Data Analysis (EDA)
- **Univariate Analysis**:
  - Distribution of individual features
  - Histogram and density plots
  - Statistical summaries
- **Bivariate Analysis**:
  - Feature correlations with target variable
  - Scatter plots and relationship visualization
  - Clinical measurement interactions
- **Multivariate Analysis**:
  - Correlation heatmaps
  - Feature interaction patterns
  - Multicollinearity detection
- **Class Distribution Visualization**:
  - Binary class balance
  - Multiclass diagnosis distribution
  - Stratification requirements

#### Feature Engineering
- **Derived Features**:
  - `T3_T4_ratio`: Ratio of T3 to TT4 levels
  - `TSH_T4_product`: Product of TSH and TT4
  - `any_treatment`: Consolidated treatment indicator
  - `medical_complexity_score`: Composite medical history score
  - `symptom_query_score`: Aggregated symptom indicators
- **Feature Selection**:
  - Correlation-based selection
  - Variance threshold filtering
  - Domain knowledge integration

#### Dimensionality Reduction
- **PCA (Principal Component Analysis)**:
  - Variance explained analysis
  - Component interpretation
  - Comparison with original features

### Visualizations
- 50+ plots including:
  - Distribution plots
  - Correlation heatmaps
  - Box plots for outliers
  - Feature importance preliminary analysis
  - PCA variance plots

### Outputs
- `df_cleaned_engineered.csv`: Fully preprocessed dataset
- `df_selected_features.csv`: Feature-selected dataset
- `df_pca_transformed.csv`: PCA-transformed features

---

## Notebook 4: Model Implementation

**File**: `4_Model_Implementation.ipynb`

### Purpose
Comprehensive model training, evaluation, and selection for both binary and multiclass classification.

### Contents

#### Model Training Pipeline
- **Train-Test Split**:
  - 80-20 stratified split
  - Maintains class distribution
  - Random state 42 for reproducibility
- **Feature Scaling**:
  - StandardScaler normalization
  - Fitted on training data only
  - Applied to test data

#### Binary Classification Models (9 Algorithms)
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier
5. **Gradient Boosting Classifier** (Best)
6. Support Vector Machine (RBF kernel)
7. Support Vector Machine (Linear kernel)
8. Gaussian Naive Bayes
9. K-Nearest Neighbors

#### Multiclass Classification Models (8 Algorithms)
1. Logistic Regression (OvR)
2. Decision Tree Classifier
3. **Random Forest Classifier** (Best)
4. XGBoost Classifier
5. Gradient Boosting Classifier
6. Support Vector Machine (RBF kernel)
7. Gaussian Naive Bayes
8. K-Nearest Neighbors

#### Model Evaluation

**Metrics Calculated**:
- Accuracy
- Precision (macro, weighted)
- Recall (macro, weighted)
- F1-Score (macro, weighted)
- ROC-AUC (binary only)
- Confusion Matrix
- Classification Report

**Validation Strategy**:
- 5-Fold Stratified Cross-Validation
- Mean and standard deviation of CV scores
- Test set evaluation for final model

#### Hyperparameter Tuning
- Grid search for optimal parameters
- Cross-validated hyperparameter selection
- Performance comparison across configurations

#### Model Comparison
- **Visualizations**:
  - Model performance comparison bar charts
  - ROC curves (binary classification)
  - Confusion matrices (heatmaps)
  - Cross-validation score distributions
- **Statistical Comparison**:
  - Pairwise model performance tests
  - Confidence intervals
  - Significance testing

#### Model Explainability (Preliminary)
- Feature importance plots
- Top feature identification
- Model-specific interpretability

### Results Summary

**Binary Classification**:
- Winner: Gradient Boosting
- Test Accuracy: 94%
- F1-Score: 89.3%

**Multiclass Classification**:
- Winner: Random Forest
- Test Accuracy: 91%
- Weighted F1-Score: 91%

### Outputs
- `best_binary_model.pkl`: Trained Gradient Boosting model
- `best_multiclass_model.pkl`: Trained Random Forest model
- `scaler_binary.pkl`: Binary feature scaler
- `scaler_multiclass.pkl`: Multiclass feature scaler
- `label_encoder_multiclass.pkl`: Diagnosis code encoder
- `config.json`: Model configuration and metadata
- `binary_classification_results.csv`: Detailed results
- `multiclass_classification_results.csv`: Detailed results

---

## Notebook 5: Ethical AI & Bias Auditing

**File**: `5_Ethical_AI_Bias_Auditing.ipynb`

### Purpose
Comprehensive analysis of model explainability, fairness, and potential biases.

### Contents

#### Explainability Analysis

**SHAP (SHapley Additive exPlanations)**:
- Global feature importance
- Feature contribution analysis
- Summary plots showing feature impact
- Dependence plots for feature interactions
- Waterfall plots for individual predictions
- Force plots for prediction explanation

**LIME (Local Interpretable Model-agnostic Explanations)**:
- Instance-level explanations
- Feature contribution for specific predictions
- Model-agnostic interpretability
- Healthy vs sick patient comparisons

**Partial Dependence Plots (PDP)**:
- Individual feature effects on predictions
- ICE (Individual Conditional Expectation) curves
- 2D interaction plots for feature pairs
- Top 6 feature PDP analysis

**Permutation Importance**:
- Feature importance via performance degradation
- Comparison with SHAP importance
- Validation of feature relevance

#### Fairness & Bias Analysis

**Demographic Parity**:
- Equal prediction rates across protected groups
- Sex-based fairness assessment
- Age group fairness evaluation
- Statistical parity calculations

**Equalized Odds**:
- Equal True Positive Rate (TPR) across groups
- Equal False Positive Rate (FPR) across groups
- Fairness in both positive and negative predictions

**Disparate Impact Analysis**:
- Ratio of positive prediction rates
- Protected attribute impact assessment
- 80% rule compliance checking

**Class Imbalance Analysis**:
- Training vs test class distributions
- Impact on model performance
- Mitigation strategies evaluation

**Overfitting Assessment**:
- Train vs test performance comparison
- Cross-validation score variance
- Generalization capability analysis

**Threshold Optimization**:
- Fairness-accuracy trade-off analysis
- Optimal decision threshold selection
- ROC curve analysis for fairness

#### Protected Attributes Examined
- Sex (Male vs Female)
- Age groups (<30, 30-50, 50-70, 70+)
- Pregnancy status (where applicable)

### Visualizations (25+ Plots)

**Explainability**:
- SHAP summary plots
- SHAP feature importance
- SHAP dependence plots
- SHAP waterfall plots
- LIME explanations
- PDP/ICE plots
- Permutation importance charts

**Fairness**:
- Demographic parity charts
- Equalized odds comparisons
- Disparate impact ratios
- Class distribution plots
- Threshold optimization curves
- Overfitting assessment plots

### Key Findings

**Model Fairness**:
- Demographic parity within acceptable range (<10% difference)
- Equalized odds maintained across sex groups
- No significant disparate impact detected
- Age-based predictions show minimal bias

**Model Explainability**:
- TSH is most important feature (SHAP value: 2.3)
- FTI and TT4 are critical for classification
- Clinical measurements align with medical knowledge
- Feature interactions are interpretable

**Ethical Considerations**:
- Model suitable for decision support
- Requires human oversight for final diagnosis
- Bias mitigation successful
- Transparent and interpretable predictions

### Outputs
- `figures/explainability/`: All SHAP, LIME, PDP visualizations
- `figures/fairness/`: Bias auditing charts

---

## Running the Notebooks

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Recommended Execution Order

1. Start with **Notebook 2** for data understanding
2. Proceed to **Notebook 3** for preprocessing and EDA
3. Run **Notebook 4** for model training
4. Finish with **Notebook 5** for explainability and fairness

## Troubleshooting

**Common Issues**:

1. **Import errors**: Ensure all packages installed
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Memory errors**: Close other applications, restart kernel

3. **Kaggle dataset download**: May require Kaggle API setup
   ```bash
   pip install kagglehub
   ```
