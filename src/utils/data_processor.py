"""
Data preprocessing utilities for thyroid disease classification.
Replicates the exact feature engineering from the training notebooks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ThyroidDataProcessor:
    """Handles all data preprocessing and feature engineering for thyroid classification."""
    
    def __init__(self):
        """Initialize the data processor with feature definitions."""
        self.numeric_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        self.binary_features = [
            'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
            'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
            'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre',
            'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured',
            'TT4_measured', 'T4U_measured', 'FTI_measured'
        ]
        self.sex_feature = 'sex'
        
        # Clinical thresholds
        self.age_bins = [0, 18, 35, 50, 65, 120]
        self.age_labels = [0, 1, 2, 3, 4]  # child, young_adult, middle_aged, senior, elderly
        
        self.tsh_bins = [0, 0.4, 4.0, 10.0, float('inf')]
        self.tsh_labels = [0, 1, 2, 3]  # low, normal, elevated, very_high
        
    def process_single_input(self, data: Dict) -> pd.DataFrame:
        """
        Process a single patient input from manual form.
        
        Args:
            data: Dictionary containing patient information
            
        Returns:
            Processed DataFrame ready for prediction
        """
        df = pd.DataFrame([data])
        return self.process_dataframe(df)
    
    def process_file_upload(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process uploaded CSV file.
        
        Args:
            df: Raw DataFrame from uploaded file
            
        Returns:
            Processed DataFrame ready for prediction
        """
        return self.process_dataframe(df)
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to a DataFrame.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Fully processed DataFrame with all engineered features
        """
        df = df.copy()
        
        # 1. Impute missing values
        df = self._impute_missing_values(df)
        
        # 2. Encode binary features
        df = self._encode_binary_features(df)
        
        # 3. Encode sex
        df = self._encode_sex(df)
        
        # 4. Engineer features
        df = self._engineer_features(df)
        
        # 5. Encode referral source (if present)
        df = self._encode_referral_source(df)
        
        return df
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values for numeric features with median."""
        # Pre-defined median values from training (you should load these from config)
        # For now, using reasonable defaults
        default_medians = {
            'TSH': 1.3,
            'T3': 1.5,
            'TT4': 102.0,
            'T4U': 0.97,
            'FTI': 105.0,
            'age': 50.0
        }
        
        for col in self.numeric_features:
            if col in df.columns and df[col].isnull().any():
                median_value = default_medians.get(col, df[col].median())
                # FIXED: No more inplace=True
                df[col] = df[col].fillna(median_value)
        
        return df
    
    def _encode_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode binary t/f features as 1/0."""
        for col in self.binary_features:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Handle string values
                    df[col] = df[col].map({'t': 1, 'f': 0, 'True': 1, 'False': 0, 
                                          'yes': 1, 'no': 0, 'Y': 1, 'N': 0})
                elif df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)
                # FIXED: No more inplace=True
                df[col] = df[col].fillna(0)
                df[col] = df[col].astype(int)
        
        return df
    
    def _encode_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode sex as 0 (M) or 1 (F)."""
        if self.sex_feature in df.columns:
            if df[self.sex_feature].dtype == 'object':
                df[self.sex_feature] = df[self.sex_feature].map({
                    'M': 0, 'F': 1, 'm': 0, 'f': 1,
                    'Male': 0, 'Female': 1, 'male': 0, 'female': 1
                })
            # FIXED: No more inplace=True
            df[self.sex_feature] = df[self.sex_feature].fillna(0)
            df[self.sex_feature] = df[self.sex_feature].astype(int)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        
        # Log transformation for TSH
        if 'TSH' in df.columns:
            df['TSH_log'] = np.log1p(df['TSH'])
        
        # Age grouping
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'], 
                bins=self.age_bins, 
                labels=self.age_labels,
                include_lowest=True
            )
            df['age_group'] = df['age_group'].astype(float)
        
        # TSH categorization
        if 'TSH' in df.columns:
            df['TSH_category'] = pd.cut(
                df['TSH'],
                bins=self.tsh_bins,
                labels=self.tsh_labels,
                include_lowest=True
            )
            df['TSH_category'] = df['TSH_category'].astype(float)
        
        # Medical complexity score
        medical_history_features = [
            'thyroid_surgery', 'I131_treatment', 'goitre', 'tumor',
            'hypopituitary', 'psych', 'sick', 'lithium'
        ]
        available_medical = [f for f in medical_history_features if f in df.columns]
        if available_medical:
            df['medical_complexity_score'] = df[available_medical].sum(axis=1)
        
        # Symptom query score
        symptom_queries = ['query_hypothyroid', 'query_hyperthyroid', 'query_on_thyroxine']
        available_symptoms = [f for f in symptom_queries if f in df.columns]
        if available_symptoms:
            df['symptom_query_score'] = df[available_symptoms].sum(axis=1)
        
        return df
    
    def _encode_referral_source(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode referral source if present."""
        if 'referral_source' in df.columns:
            # Get dummies
            referral_dummies = pd.get_dummies(df['referral_source'], prefix='referral')
            df = pd.concat([df, referral_dummies], axis=1)
            # FIXED: No more inplace=True
            df = df.drop('referral_source', axis=1)
        
        return df
    
    def get_feature_names(self, include_referral: bool = True) -> List[str]:
        """
        Get list of all expected feature names after processing.
        
        Args:
            include_referral: Whether to include referral source features
            
        Returns:
            List of feature names
        """
        features = (
            self.numeric_features +
            self.binary_features +
            [self.sex_feature] +
            ['TSH_log', 'age_group', 'TSH_category', 
             'medical_complexity_score', 'symptom_query_score']
        )
        
        if include_referral:
            # Common referral sources (would be better to load from config)
            referral_sources = ['SVHC', 'STMW', 'SVI', 'SVHD', 'other']
            features.extend([f'referral_{src}' for src in referral_sources])
        
        return features
    
    def validate_input(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that input has required features.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_missing_features)
        """
        required_features = set(self.numeric_features + ['sex'])
        present_features = set(df.columns)
        missing_features = required_features - present_features
        
        return len(missing_features) == 0, list(missing_features)
    
    def create_feature_template(self) -> Dict:
        """
        Create a template dictionary for manual input.
        
        Returns:
            Dictionary with all required features and default values
        """
        template = {}
        
        # Numeric features
        for feat in self.numeric_features:
            template[feat] = 0.0
        
        # Binary features
        for feat in self.binary_features:
            template[feat] = 0
        
        # Sex
        template[self.sex_feature] = 0
        
        # Optional
        template['referral_source'] = 'other'
        
        return template
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions for all features."""
        return {
            'age': 'Patient age in years',
            'sex': 'Sex (Male/Female)',
            'TSH': 'Thyroid Stimulating Hormone level (normal: 0.4-4.0 mIU/L)',
            'T3': 'Triiodothyronine level (normal: 0.8-2.0 nmol/L)',
            'TT4': 'Total Thyroxine level (normal: 60-150 nmol/L)',
            'T4U': 'T4 Uptake (normal: 0.75-1.15)',
            'FTI': 'Free Thyroxine Index (normal: 60-170)',
            'on_thyroxine': 'Currently taking thyroxine medication',
            'query_on_thyroxine': 'Query regarding thyroxine treatment',
            'on_antithyroid_meds': 'Currently taking anti-thyroid medication',
            'sick': 'Currently ill',
            'pregnant': 'Pregnancy status',
            'thyroid_surgery': 'History of thyroid surgery',
            'I131_treatment': 'Previous I-131 treatment',
            'query_hypothyroid': 'Suspected hypothyroidism',
            'query_hyperthyroid': 'Suspected hyperthyroidism',
            'lithium': 'Taking lithium medication',
            'goitre': 'Presence of goitre',
            'tumor': 'Presence of tumor',
            'hypopituitary': 'Hypopituitary condition',
            'psych': 'Psychiatric condition',
            'TSH_measured': 'TSH was measured',
            'T3_measured': 'T3 was measured',
            'TT4_measured': 'TT4 was measured',
            'T4U_measured': 'T4U was measured',
            'FTI_measured': 'FTI was measured',
            'referral_source': 'Source of referral'
        }
    
    def get_normal_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get normal ranges for clinical measurements."""
        return {
            'age': (0, 120),
            'TSH': (0.4, 4.0),
            'T3': (0.8, 2.0),
            'TT4': (60, 150),
            'T4U': (0.75, 1.15),
            'FTI': (60, 170)
        }