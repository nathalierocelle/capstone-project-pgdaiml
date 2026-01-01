# Thyroid Disease Likelihood 

A hierarchical machine learning system for thyroid disease detection and multi-class classification, developed as part of the Post Graduate Diploma in AI and Machine Learning at Asian Institute of Management.

## Project Overview

This project implements a comprehensive thyroid disease classification system using hierarchical machine learning models. The system first performs binary classification to detect thyroid abnormalities, then uses multi-class classification to identify specific thyroid conditions across 31 different diagnosis categories.

**Key Features:**
- **Binary Classification**: Detects whether a patient has thyroid disease (94% accuracy, 89% F1-score)
- **Multi-Class Classification**: Identifies specific thyroid conditions among 31 diagnosis types (91% weighted F1-score)
- **Explainable AI**: Implements SHAP, LIME, and PDP for model interpretability
- **Ethical AI**: Comprehensive bias auditing and fairness analysis
- **Interactive Web App**: Streamlit-based interface with AI-powered insights using OpenAI GPT

## Model Performance

### Binary Classification (Healthy vs Sick)
- **Best Model**: Gradient Boosting
- **Test Accuracy**: 94%
- **F1-Score**: 89.3%
- **ROC-AUC**: High discriminative capability

### Multi-Class Classification (31 Diagnosis Types)
- **Best Model**: Random Forest
- **Test Accuracy**: 91%
- **Weighted F1-Score**: 91%
- **Handles**: 31 different thyroid condition categories

## Dataset Information

- **Source**: Kaggle Thyroid Disease Dataset
- **Total Records**: 9,172 patient records
- **Features**: 21 engineered features including clinical measurements and symptom indicators
- **Classes**: 
  - Binary: 2 classes (Sick: 2,401 | Not Sick: 6,771)
  - Multi-class: 31 diagnosis categories

### Key Features
- Clinical measurements: TSH, T3, T4, TT4, T4U, FTI
- Derived features: T3_T4_ratio, TSH_T4_product
- Patient information: sex, age, pregnancy status
- Symptoms: goitre, hyperthyroid query, psychological symptoms
- Treatment history: any_treatment, medical_complexity_score

## Project Structure

```
capstone-project-pgdaiml/
├── data/                               # Processed datasets
│   ├── df_cleaned_engineered.csv       # Feature-engineered dataset
│   ├── df_pca_transformed.csv          # PCA-transformed features
│   └── df_selected_features.csv        # Selected features for modeling
├── models/                             # Trained models and artifacts
│   ├── best_binary_model.pkl           # Gradient Boosting (binary)
│   ├── best_multiclass_model.pkl       # Random Forest (multiclass)
│   ├── scaler_binary.pkl               # Feature scaler for binary
│   ├── scaler_multiclass.pkl           # Feature scaler for multiclass
│   ├── label_encoder_multiclass.pkl    # Label encoder
│   └── config.json                     # Model configuration
├── notebook/                           # Jupyter notebooks
│   ├── 2_Data_Collection_&_Understanding.ipynb
│   ├── 3_Data_Preprocessing,_EDA_&_Feature_Engineering.ipynb
│   ├── 4_Model_Implementation.ipynb
│   └── 5_Ethical_AI_Bias_Auditing.ipynb
├── figures/                            # Visualizations
│   ├── explainability/                 # SHAP, LIME, PDP plots
│   └── fairness/                       # Bias auditing visualizations
├── results/                            # Model evaluation results
│   ├── binary_classification_results.csv
│   ├── binary_classification_report.txt
│   ├── multiclass_classification_results.csv
│   └── multiclass_classification_report.txt
├── src/                                # Source code
│   ├── app.py                          # Streamlit web application
│   └── utils/                          # Utility modules
│       ├── data_processor.py           # Data processing utilities
│       ├── model_loader.py             # Model loading utilities
│       ├── predictor.py                # Prediction engine
│       └── ai_insights.py              # OpenAI GPT integration
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Getting Started

### Prerequisites
- Python 3.9.7 or higher
- pip package manager

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd capstone-project-pgdaiml-main
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables (for AI insights feature)
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

Launch the Streamlit web application:
```bash
streamlit run src/app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Jupyter Notebooks

Navigate through the notebooks in order:
1. `2_Data_Collection_&_Understanding.ipynb` - Data exploration and initial analysis
2. `3_Data_Preprocessing,_EDA_&_Feature_Engineering.ipynb` - Data cleaning and feature creation
3. `4_Model_Implementation.ipynb` - Model training and evaluation
4. `5_Ethical_AI_Bias_Auditing.ipynb` - Fairness analysis and bias detection

```bash
jupyter notebook notebook/
```

## Machine Learning Pipeline

### 1. Data Preprocessing
- Missing value imputation
- Feature encoding (one-hot, label encoding)
- Feature scaling (StandardScaler)
- Handling class imbalance

### 2. Feature Engineering
- T3_T4_ratio: Ratio of T3 to TT4
- TSH_T4_product: Product of TSH and TT4
- any_treatment: Binary indicator of any treatment history
- medical_complexity_score: Composite score of medical history
- symptom_query_score: Aggregated symptom indicators

### 3. Model Training
**Binary Classification Models Evaluated:**
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Gradient Boosting ⭐ (Best: 89.3% F1)
- SVM (RBF & Linear)
- Naive Bayes
- K-Nearest Neighbors

**Multi-Class Classification Models Evaluated:**
- Logistic Regression
- Decision Tree
- Random Forest ⭐ (Best: 91% Weighted F1)
- XGBoost
- Gradient Boosting
- SVM (RBF)
- Naive Bayes
- K-Nearest Neighbors

### 4. Model Evaluation
- 5-fold Cross-validation
- Test set evaluation with stratified split (80/20)
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices for error analysis

## 🔍 Explainability & Interpretability

### SHAP (SHapley Additive exPlanations)
- Global feature importance
- Feature contribution analysis
- Dependence plots for feature interactions
- Individual prediction explanations

### LIME (Local Interpretable Model-agnostic Explanations)
- Instance-level explanations
- Feature contribution for individual predictions
- Model-agnostic approach

### PDP (Partial Dependence Plots)
- Feature effect visualization
- ICE (Individual Conditional Expectation) curves
- 2D interaction plots

### Permutation Importance
- Feature importance based on model performance degradation

## ⚖️ Ethical AI & Bias Auditing

Comprehensive fairness analysis including:

### Fairness Metrics
- **Demographic Parity**: Equal prediction rates across groups
- **Equalized Odds**: Equal TPR and FPR across protected attributes
- **Disparate Impact**: Ratio of positive prediction rates

### Bias Detection
- Sex-based bias analysis
- Age group fairness evaluation
- Protected attribute impact assessment

### Model Robustness
- Overfitting assessment (train vs test performance)
- Class imbalance analysis
- Threshold optimization for fairness-accuracy trade-offs

## Key Insights

### Most Important Features (SHAP Analysis)
1. **TSH** (Thyroid Stimulating Hormone) - Primary indicator
2. **FTI** (Free Thyroxine Index) - Critical for diagnosis
3. **TT4** (Total Thyroxine) - Core thyroid hormone
4. **T3** - Triiodothyronine levels
5. **T4U** - Thyroid hormone uptake
6. **Age** - Patient age correlation

### Clinical Implications
- Model enables early detection of thyroid abnormalities
- Multi-class classification assists in specific diagnosis
- Explainability features support clinical decision-making
- Bias auditing ensures equitable healthcare delivery

## Web Application Features

### Interactive Prediction Interface
- Interactive data-entry with validation
- Real-time prediction with probability

### AI-Powered Insights
- GPT-4 integration for natural language explanations
- Personalized health recommendations
- Clinical context and interpretation

## Technical Stack

- **ML/AI**: scikit-learn, XGBoost, SHAP, LIME
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **AI Integration**: OpenAI GPT API
- **Model Persistence**: joblib
- **Environment**: Python 3.9.7

## Contributing

This is an academic capstone project. For questions or suggestions, please reach out through the appropriate academic channels.

## License

This project is developed for educational purposes as part of the AIM PGDAIML program.

---

**Note**: This system is designed for educational and research purposes. It should not be used as a substitute for professional medical diagnosis and treatment. Always consult qualified healthcare professionals for medical decisions.

**Dataset Citation**: Thyroid Disease Dataset from Kaggle (UCI Machine Learning Repository)

**Last Updated**: January 2026