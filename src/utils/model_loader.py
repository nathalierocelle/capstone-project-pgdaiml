"""
Model loading and management utilities.
"""

import joblib
import json
import os
from typing import Dict, Any, Tuple
import streamlit as st


class ModelLoader:
    """Handles loading and caching of trained models and artifacts."""
    
    def __init__(self, models_dir: str = "../models"):
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory containing model artifacts
        """
        self.models_dir = models_dir
        self.config = None
        
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def load_models(_self) -> Dict[str, Any]:
        """
        Load all required models and artifacts.
        Uses Streamlit caching to avoid reloading on every interaction.
        
        Returns:
            Dictionary containing all loaded artifacts
        """
        artifacts = {}
        
        try:
            # Load configuration
            config_path = os.path.join(_self.models_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    artifacts['config'] = json.load(f)
                    _self.config = artifacts['config']
                st.success("✓ Configuration loaded")
            else:
                st.warning(f"Config file not found at {config_path}")
                artifacts['config'] = {}
            
            # Load binary classification model
            binary_model_path = os.path.join(_self.models_dir, 'best_binary_model.pkl')
            if os.path.exists(binary_model_path):
                artifacts['binary_model'] = joblib.load(binary_model_path)
                st.success("✓ Binary classification model loaded")
            else:
                st.error(f"❌ Binary model not found at {binary_model_path}")
                
            # Load binary scaler
            binary_scaler_path = os.path.join(_self.models_dir, 'scaler_binary.pkl')
            if os.path.exists(binary_scaler_path):
                artifacts['binary_scaler'] = joblib.load(binary_scaler_path)
                # Verify it has transform method
                if hasattr(artifacts['binary_scaler'], 'transform'):
                    st.success("✓ Binary scaler loaded")
                else:
                    st.error("❌ Binary scaler is invalid - missing transform method!")
            else:
                st.error(f"❌ Binary scaler not found at {binary_scaler_path}")
                
            # Load multi-class model
            multi_model_path = os.path.join(_self.models_dir, 'best_multiclass_model.pkl')
            if os.path.exists(multi_model_path):
                artifacts['multiclass_model'] = joblib.load(multi_model_path)
                st.success("✓ Multi-class classification model loaded")
            else:
                st.error(f"❌ Multi-class model not found at {multi_model_path}")
                
            # Load multi-class scaler
            multi_scaler_path = os.path.join(_self.models_dir, 'scaler_multiclass.pkl')
            if os.path.exists(multi_scaler_path):
                artifacts['multiclass_scaler'] = joblib.load(multi_scaler_path)
                # Verify it has transform method
                if hasattr(artifacts['multiclass_scaler'], 'transform'):
                    st.success("✓ Multi-class scaler loaded")
                else:
                    st.error("❌ Multi-class scaler is invalid - missing transform method!")
            else:
                st.error(f"❌ Multi-class scaler not found at {multi_scaler_path}")
                
            # Load label encoder
            encoder_path = os.path.join(_self.models_dir, 'label_encoder_multiclass.pkl')
            if os.path.exists(encoder_path):
                artifacts['label_encoder'] = joblib.load(encoder_path)
                st.success("✓ Label encoder loaded")
            else:
                st.warning("⚠️ Label encoder not found - class names may not display correctly")
            
            # Summary
            loaded_count = len([k for k in ['binary_model', 'binary_scaler', 'multiclass_model', 'multiclass_scaler', 'label_encoder'] if k in artifacts])
            st.info(f"✅ Successfully loaded {loaded_count}/5 required model files")
            
            return artifacts
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return artifacts
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model metadata
        """
        if self.config is None:
            return {}
            
        info = {
            'binary_model_name': self.config.get('best_binary_model', 'Unknown'),
            'multiclass_model_name': self.config.get('best_multiclass_model', 'Unknown'),
            'binary_test_f1': self.config.get('binary_test_f1', 'N/A'),
            'multiclass_test_f1': self.config.get('multiclass_test_f1_weighted', 'N/A'),
            'feature_columns': self.config.get('feature_columns', []),
            'execution_date': self.config.get('execution_date', 'Unknown')
        }
        
        return info
    
    def get_feature_columns(self) -> list:
        """Get the exact feature columns used during training."""
        if self.config is None:
            return []
        return self.config.get('feature_columns', [])