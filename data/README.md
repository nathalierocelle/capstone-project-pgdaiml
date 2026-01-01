# Data Directory

This directory contains processed datasets used in the thyroid disease classification project.

## Files

### Main Datasets

1. **`df_cleaned_engineered.csv`**
   - Cleaned dataset with engineered features
   - Records: 9,172 patient records
   - Features: 21 columns including clinical measurements and derived features
   - Used for: Final model training and evaluation

2. **`df_pca_transformed.csv`**
   - PCA-transformed feature set for dimensionality reduction analysis
   - Used for: Comparing performance with reduced feature space
   - Components: Principal components capturing variance

3. **`df_selected_features.csv`**
   - Feature-selected dataset based on importance and correlation analysis
   - Used for: Optimized model training with most relevant features
   - Selection method: Based on SHAP values and statistical analysis

## Data Schema

### Clinical Measurements
- `TSH`: Thyroid Stimulating Hormone level
- `T3`: Triiodothyronine level
- `TT4`: Total Thyroxine level
- `T4U`: Thyroid hormone uptake
- `FTI`: Free Thyroxine Index

### Engineered Features
- `T3_T4_ratio`: Ratio of T3 to TT4
- `TSH_T4_product`: Product of TSH and TT4
- `any_treatment`: Binary indicator of treatment history
- `medical_complexity_score`: Composite medical history score
- `symptom_query_score`: Aggregated symptom indicator

### Patient Demographics
- `sex`: Patient gender (0: Female, 1: Male)
- `age`: Patient age
- `pregnant`: Pregnancy status

### Clinical Indicators
- `goitre`: Presence of goitre
- `query_hyperthyroid`: Query for hyperthyroid symptoms
- `psych`: Psychological symptoms present

### Measurement Indicators
- `TSH_measured`: TSH measurement taken
- `T3_measured`: T3 measurement taken
- `T4U_measured`: T4U measurement taken

## Data Source

Original data sourced from **Kaggle Thyroid Disease Dataset** (UCI Machine Learning Repository)

## Privacy & Ethics

- All data is anonymized
- No personally identifiable information (PII) included
- Used strictly for educational and research purposes
- Complies with academic data usage policies

## Data Statistics

- **Total Records**: 9,172
- **Binary Classes**: 
  - Not Sick: 6,771 (73.8%)
  - Sick: 2,401 (26.2%)
- **Multiclass Classes**: 31 unique diagnosis codes
- **Missing Values**: Handled through imputation (see preprocessing notebook)
- **Class Balance**: Addressed using stratified sampling and SMOTE techniques

## Data Processing Pipeline

1. **Raw Data Collection** → Kaggle dataset download
2. **Data Cleaning** → Missing value handling, outlier detection
3. **Feature Engineering** → Creating derived features
4. **Feature Selection** → Importance-based selection
5. **Dimensionality Reduction** → PCA transformation
6. **Final Dataset** → Ready for model training

## Usage

```python
import pandas as pd

# Load cleaned and engineered data
df = pd.read_csv('data/df_cleaned_engineered.csv')

# Load feature-selected data
df_selected = pd.read_csv('data/df_selected_features.csv')

# Load PCA-transformed data
df_pca = pd.read_csv('data/df_pca_transformed.csv')
```

## Related Files

- **Preprocessing Notebook**: `notebook/3_Data_Preprocessing,_EDA_&_Feature_Engineering.ipynb`
- **Data Understanding**: `notebook/2_Data_Collection_&_Understanding.ipynb`
