"""
Prediction pipeline for hierarchical thyroid disease classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import streamlit as st
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')


class ThyroidPredictor:
    """Handles the two-stage hierarchical prediction pipeline."""
    
    def __init__(self, models: Dict[str, Any], processor):
        """
        Initialize predictor with loaded models.
        
        Args:
            models: Dictionary containing all model artifacts
            processor: ThyroidDataProcessor instance
        """
        self.binary_model = models.get('binary_model')
        self.binary_scaler = models.get('binary_scaler')
        self.multiclass_model = models.get('multiclass_model')
        self.multiclass_scaler = models.get('multiclass_scaler')
        self.label_encoder = models.get('label_encoder')
        self.config = models.get('config', {})
        self.processor = processor
        
        # Get feature columns from config
        self.feature_columns = self.config.get('feature_columns', [])
        
    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Perform hierarchical prediction on input data.
        
        Args:
            df: Processed DataFrame (after data_processor)
            
        Returns:
            List of prediction dictionaries, one per row
        """
        predictions = []
        
        for idx, row in df.iterrows():
            result = self._predict_single(row)
            predictions.append(result)
        
        return predictions
    
    def _predict_single(self, row: pd.Series) -> Dict[str, Any]:
        """
        Predict for a single patient.
        
        Args:
            row: Single row of processed data
            
        Returns:
            Dictionary with prediction results
        """
        result = {
            'stage1_prediction': None,
            'stage1_probability': None,
            'stage1_confidence': None,
            'stage2_prediction': None,
            'stage2_probability': None,
            'stage2_confidence': None,
            'final_diagnosis': None,
            'risk_level': None,
            'feature_importance': {}
        }
        
        try:
            # Prepare features
            X = self._prepare_features(row)
            
            # Safety check for scaler
            if self.binary_scaler is None:
                st.error("❌ Binary scaler is not loaded! Cannot make predictions.")
                result['error'] = "Binary scaler not loaded"
                return result
            
            # Stage 1: Binary classification (Healthy vs Sick)
            X_scaled_binary = self.binary_scaler.transform(X)
            binary_pred = self.binary_model.predict(X_scaled_binary)[0]
            
            # Try to get probabilities
            if hasattr(self.binary_model, 'predict_proba'):
                try:
                    binary_proba = self.binary_model.predict_proba(X_scaled_binary)[0]
                    result['stage1_probability'] = {
                        'not_sick': float(binary_proba[0]),
                        'sick': float(binary_proba[1])
                    }
                    result['stage1_confidence'] = float(max(binary_proba))
                except Exception as e:
                    st.warning(f"Could not calculate probabilities: {str(e)}")
                    result['stage1_confidence'] = None
            else:
                st.info("Model does not support probability predictions")
                result['stage1_confidence'] = None
            
            result['stage1_prediction'] = 'Sick' if binary_pred == 1 else 'Not Sick'
            
            # Stage 2: Multi-class classification (only if sick)
            if binary_pred == 1:
                
                try:
                    X_scaled_multi = self.multiclass_scaler.transform(X)
                    multi_pred_raw = self.multiclass_model.predict(X_scaled_multi)[0]
                    
                    # Use the prediction directly - no encoding/decoding needed
                    # If it's a string, use it as is. If it's a number, convert to string
                    multi_pred = str(multi_pred_raw)
                    result['stage2_prediction'] = multi_pred
                    
                    # Try to get probabilities
                    if hasattr(self.multiclass_model, 'predict_proba'):
                        try:
                            multi_proba = self.multiclass_model.predict_proba(X_scaled_multi)[0]
                            
                            # Get class names from model if available
                            if hasattr(self.multiclass_model, 'classes_'):
                                class_names = self.multiclass_model.classes_
                            elif self.label_encoder is not None:
                                class_names = self.label_encoder.classes_
                            else:
                                class_names = [f"Class_{i}" for i in range(len(multi_proba))]
                            
                            result['stage2_probability'] = {
                                str(class_names[i]): float(multi_proba[i]) 
                                for i in range(len(multi_proba))
                            }
                            result['stage2_confidence'] = float(max(multi_proba))
                        except Exception as e:
                            st.warning(f"Could not calculate stage 2 probabilities: {str(e)}")
                            st.error(f"Probability error details: {e}")
                            result['stage2_confidence'] = None
                    else:
                        result['stage2_confidence'] = None
                        
                except Exception as e:
                    st.error(f"Stage 2 prediction error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    result['stage2_prediction'] = f"Error: {str(e)}"
                
                result['final_diagnosis'] = multi_pred
                result['risk_level'] = self._assess_risk_level(
                    result['stage1_confidence'],
                    result['stage2_confidence']
                )
            else:
                result['final_diagnosis'] = 'No thyroid condition detected'
                result['risk_level'] = 'Low'
            
            # Calculate feature importance
            result['feature_importance'] = self._calculate_feature_importance(row)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _prepare_features(self, row: pd.Series) -> pd.DataFrame:
        """
        Prepare features in the exact order expected by models.
        
        Args:
            row: Single row of data
            
        Returns:
            DataFrame with features (not numpy array)
        """
        # If we have feature columns from config, use them
        if self.feature_columns:
            # Ensure all features are present
            features = {}
            for col in self.feature_columns:
                if col in row.index:
                    features[col] = row[col]
                else:
                    # Default to 0 for missing features
                    features[col] = 0
            # Return as DataFrame to preserve column names
            return pd.DataFrame([features])
        else:
            # Fallback: use all numeric columns except targets
            exclude_cols = ['target', 'target_binary', 'patient_id']
            feature_cols = [col for col in row.index if col not in exclude_cols]
            # Return as DataFrame
            return pd.DataFrame([row[feature_cols].to_dict()])
    
    def _assess_risk_level(self, stage1_conf: float, stage2_conf: float = None) -> str:
        """
        Assess overall risk level based on prediction confidence.
        
        Args:
            stage1_conf: Stage 1 confidence
            stage2_conf: Stage 2 confidence (if applicable)
            
        Returns:
            Risk level string
        """
        if stage2_conf is not None:
            if stage1_conf >= 0.9:
                return 'High Confidence - Strong Recommendation: Seek Medical Attention'
            elif stage1_conf >= 0.7:
                return 'Medium Confidence - Recommendation: Consult Healthcare Provider'
            else:
                return 'Low Confidence - Suggestion: Monitor Symptoms and Follow-up if Needed'
    
    def _calculate_feature_importance(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate feature contribution to prediction.
        Uses model's feature_importances_ if available.
        
        Args:
            row: Patient data
            
        Returns:
            Dictionary of feature names to importance scores
        """
        importance_dict = {}
        
        try:
            if hasattr(self.binary_model, 'feature_importances_'):
                importances = self.binary_model.feature_importances_
                
                # Get feature names
                if self.feature_columns:
                    feature_names = self.feature_columns
                else:
                    exclude_cols = ['target', 'target_binary', 'patient_id']
                    feature_names = [col for col in row.index if col not in exclude_cols]
                
                # Create importance dictionary
                for i, name in enumerate(feature_names[:len(importances)]):
                    importance_dict[name] = float(importances[i])
                
                # Sort by importance
                importance_dict = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
        
        except Exception as e:
            st.warning(f"Could not calculate feature importance: {str(e)}")
        
        return importance_dict