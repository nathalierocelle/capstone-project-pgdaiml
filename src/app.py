"""
Thyroid Disease Classification - Streamlit Application
A hierarchical ML system for thyroid disease detection and classification.
Compatible with Streamlit 1.12 and Python 3.9.7
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import re
import warnings
from utils.data_processor import ThyroidDataProcessor
from utils.model_loader import ModelLoader
from utils.predictor import ThyroidPredictor
from utils.ai_insights import AIInsightsGenerator
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))

load_dotenv()
warnings.filterwarnings("ignore")

# Diagnosis mapping dictionary
DIAGNOSIS_MAPPING = {
    # Hyperthyroid conditions
    'A': 'Hyperthyroid',
    'B': 'T3 Toxic',
    'C': 'Toxic Goitre',
    'D': 'Secondary Toxic',
    
    # Hypothyroid conditions
    'E': 'Hypothyroid',
    'F': 'Primary Hypothyroid',
    'G': 'Compensated Hypothyroid',
    'H': 'Secondary Hypothyroid',
    
    # Binding protein
    'I': 'Increased Binding Protein',
    'J': 'Decreased Binding Protein',
    
    # General health
    'K': 'Concurrent Non-Thyroidal Illness',
    
    # Replacement therapy
    'L': 'Consistent with Replacement Therapy',
    'M': 'Underreplaced',
    'N': 'Overreplaced',
    
    # Antithyroid treatment
    'O': 'Antithyroid Drugs',
    'P': 'I131 Treatment',
    'Q': 'Surgery',
    
    # Miscellaneous
    'R': 'Discordant Assay Results',
    'S': 'Elevated TBG',
    'T': 'Elevated Thyroid Hormones',
    
    # Combination diagnoses
    'AK': 'Consistent with Hyperthyroid, but more likely Concurrent Non-Thyroidal Illness',
    'GK': 'Consistent with Compensated Hypothyroid, but more likely Concurrent Non-Thyroidal Illness',
    'FK': 'Consistent with Primary Hypothyroid, but more likely Concurrent Non-Thyroidal Illness',
    'MK': 'Consistent with Underreplaced, but more likely Concurrent Non-Thyroidal Illness',
    'KJ': 'Consistent with Concurrent Non-Thyroidal Illness, but more likely Decreased Binding Protein',
    'GI': 'Consistent with Compensated Hypothyroid, but more likely Increased Binding Protein',
    'C|I': 'Consistent with Toxic Goitre, but more likely Increased Binding Protein',
    'H|K': 'Consistent with Secondary Hypothyroid, but more likely Concurrent Non-Thyroidal Illness',
    'MI': 'Consistent with Underreplaced, but more likely Increased Binding Protein',
    'LJ': 'Consistent with Replacement Therapy, but more likely Decreased Binding Protein',
    'GKJ': 'Consistent with Compensated Hypothyroid and Concurrent Non-Thyroidal Illness, but more likely Decreased Binding Protein',
    'OI': 'Consistent with Antithyroid Drugs, but more likely Increased Binding Protein',
    'D|R': 'Consistent with Secondary Toxic, but more likely Discordant Assay Results'
}


def get_diagnosis_name(code):
    """Convert diagnosis code to full name."""
    if code in DIAGNOSIS_MAPPING:
        return f"{code} - {DIAGNOSIS_MAPPING[code]}"
    return code


def get_diagnosis_category(code):
    """Get the category of the diagnosis."""
    categories = {
        'A': 'Hyperthyroid Condition',
        'B': 'Hyperthyroid Condition',
        'C': 'Hyperthyroid Condition',
        'D': 'Hyperthyroid Condition',
        'E': 'Hypothyroid Condition',
        'F': 'Hypothyroid Condition',
        'G': 'Hypothyroid Condition',
        'H': 'Hypothyroid Condition',
        'I': 'Binding Protein Issue',
        'J': 'Binding Protein Issue',
        'K': 'General Health',
        'L': 'Replacement Therapy',
        'M': 'Replacement Therapy',
        'N': 'Replacement Therapy',
        'O': 'Antithyroid Treatment',
        'P': 'Antithyroid Treatment',
        'Q': 'Antithyroid Treatment',
        'R': 'Miscellaneous',
        'S': 'Miscellaneous',
        'T': 'Miscellaneous'
    }
    return categories.get(code, 'Unknown')


# Page configuration
st.set_page_config(
    page_title="Thyroid Disease Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Prediction Box Styles */
    .prediction-box {
        padding: 2rem;
        border-radius: 12px;
        border: none;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .prediction-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .healthy {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    
    .sick {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
    
    .prediction-box h2 {
        margin: 0 0 0.5rem 0;
        color: #2c3e50;
        font-size: 1.8rem;
    }
    
    .prediction-box p {
        margin: 0.5rem 0;
        color: #495057;
    }
    
    .diagnosis-category {
        font-size: 0.95rem;
        color: #6c757d;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Info Box Styles */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .input-group-title {
        color: #2c3e50;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        padding-top: 0.5rem;
        border-bottom: 2px solid #3498db;
        background-color: #f8f9fa;
        padding-left: 0.5rem;
        border-radius: 4px 4px 0 0;
    }
    
    /* Button Enhancements */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background-color: #ffffff;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #f8f9fa;
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Alert/Message Boxes */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Range Reference */
    .range-ref {
        font-size: 0.85rem;
        color: #6c757d;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'ai_generator' not in st.session_state:
    st.session_state.ai_generator = None


@st.cache(allow_output_mutation=True)
def initialize_app():
    """Initialize and cache all resources."""
    processor = ThyroidDataProcessor()
    model_loader = ModelLoader(models_dir="../models")
    models = model_loader.load_models()
    predictor = ThyroidPredictor(models, processor)
    ai_generator = AIInsightsGenerator()
    
    return processor, model_loader, models, predictor, ai_generator


def reset_session():
    """Reset the session to start fresh."""
    st.session_state.predictions = None
    st.session_state.patient_data = None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<p class="main-header">🏥 Thyroid Disease Classifier</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Hierarchical Screening System</p>', 
                unsafe_allow_html=True)
    st.warning("""
        **This is a screening tool only.**
        Not intended for diagnosis. Always consult qualified healthcare professionals for medical advice and treatment.
        """)
    
    # Initialize components
    try:
        processor, model_loader, models, predictor, ai_generator = initialize_app()
        st.session_state.ai_generator = ai_generator
    except Exception as e:
        st.error("⚠️ **Application Initialization Failed**")
        st.error(f"Error: {str(e)}")
        st.info("💡 **Tip:** Ensure model files are in the `models` directory")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ️ System Overview")
        
        st.markdown("""
        <div class="info-box">
        <strong>Two-Stage Classification:</strong><br>
        <strong>Stage 1:</strong> Initial screening (Healthy/Sick)<br>
        <strong>Stage 2:</strong> Specific disorder classification
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Model Performance")
        
        model_info = model_loader.get_model_info()
        if model_info:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Binary F1", f"{model_info.get('binary_test_f1', 0)*100:.1f}%")
            with col2:
                st.metric("Multi-class F1", f"{model_info.get('multiclass_test_f1', 0)*100:.1f}%")
            
            with st.expander("📋 Model Details", expanded=True):
                st.text(f"Binary: {model_info.get('binary_model_name', 'N/A')}")
                st.text(f"Multi-class: {model_info.get('multiclass_model_name', 'N/A')}")
        
        st.markdown("---")
        
        # New Session Button
        if st.button("🔄 New Session", help="Clear current results and start fresh"):
            reset_session()
            st.experimental_rerun()
            st.success("✅ Session cleared! Ready for new patient.")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🔬 Patient Assessment", "📚 User Guide"])
    
    with tab1:
        prediction_tab(processor, predictor, st.session_state.ai_generator)
    
    with tab2:
        information_tab()


def prediction_tab(processor, predictor, ai_generator):
    """Single patient prediction interface."""
    
    st.markdown('<p class="section-header">Patient Assessment</p>', unsafe_allow_html=True)
    
    manual_input_form(processor, predictor, ai_generator)


def manual_input_form(processor, predictor, ai_generator):
    """Manual input form for patient data."""
    
    st.markdown("### Enter Patient Information")
    st.markdown("Fill in the patient details below. Required fields are marked with clinical reference ranges.")
    
    patient_data = {}
    
    # Demographics in full width row
    st.markdown('<p class="input-group-title">👤 Demographics</p>', unsafe_allow_html=True)
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        patient_data['age'] = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=50,
            help="Patient's age in years"
        )
    with col_demo2:
        patient_data['sex'] = st.selectbox(
            "Biological Sex",
            options=[("Male", 0), ("Female", 1)],
            format_func=lambda x: x[0]
        )[1]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create columns for organized input - all sections now start at same level
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<p class="input-group-title">🧪 Lab Results</p>', unsafe_allow_html=True)
        
        patient_data['TSH'] = st.number_input(
            "TSH (mIU/L)",
            min_value=0.0,
            max_value=50.0,
            value=1.3,
            step=0.1,
            help="Thyroid Stimulating Hormone"
        )
        st.markdown('<p class="range-ref">Normal: 0.4-4.0 mIU/L</p>', unsafe_allow_html=True)
        
        patient_data['T3'] = st.number_input(
            "T3 (nmol/L)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1,
            help="Triiodothyronine"
        )
        st.markdown('<p class="range-ref">Normal: 0.8-2.0 nmol/L</p>', unsafe_allow_html=True)
        
        patient_data['TT4'] = st.number_input(
            "TT4 (nmol/L)",
            min_value=0.0,
            max_value=300.0,
            value=102.0,
            step=1.0,
            help="Total Thyroxine"
        )
        st.markdown('<p class="range-ref">Normal: 60-150 nmol/L</p>', unsafe_allow_html=True)
        
        patient_data['T4U'] = st.number_input(
            "T4U",
            min_value=0.0,
            max_value=2.0,
            value=0.97,
            step=0.01,
            help="T4 Uptake"
        )
        st.markdown('<p class="range-ref">Normal: 0.75-1.15</p>', unsafe_allow_html=True)
        
        patient_data['FTI'] = st.number_input(
            "FTI",
            min_value=0.0,
            max_value=300.0,
            value=105.0,
            step=1.0,
            help="Free Thyroxine Index"
        )
        st.markdown('<p class="range-ref">Normal: 60-170</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<p class="input-group-title">💊 Current Medications</p>', unsafe_allow_html=True)
        patient_data['on_thyroxine'] = st.checkbox("Taking Thyroxine")
        patient_data['query_on_thyroxine'] = st.checkbox("Thyroxine Query")
        patient_data['on_antithyroid_meds'] = st.checkbox("Taking Anti-thyroid Medication")
        patient_data['lithium'] = st.checkbox("Taking Lithium")
        
        st.markdown('<p class="input-group-title">🏥 Treatment History</p>', unsafe_allow_html=True)
        patient_data['I131_treatment'] = st.checkbox("Received I-131 Treatment")
        patient_data['thyroid_surgery'] = st.checkbox("Previous Thyroid Surgery")
        
        st.markdown('<p class="input-group-title">📋 Medical History</p>', unsafe_allow_html=True)
        patient_data['goitre'] = st.checkbox("Goitre")
        patient_data['tumor'] = st.checkbox("Tumor")
        patient_data['hypopituitary'] = st.checkbox("Hypopituitary Condition")
        patient_data['psych'] = st.checkbox("Psychiatric Condition")
    
    with col3:
        st.markdown('<p class="input-group-title">🩺 Current Status</p>', unsafe_allow_html=True)
        patient_data['sick'] = st.checkbox("Currently Unwell")
        patient_data['pregnant'] = st.checkbox("Pregnant")
        patient_data['query_hypothyroid'] = st.checkbox("Suspected Hypothyroidism")
        patient_data['query_hyperthyroid'] = st.checkbox("Suspected Hyperthyroidism")
        
        st.markdown('<p class="input-group-title">✅ Tests Performed</p>', unsafe_allow_html=True)
        patient_data['TSH_measured'] = st.checkbox("TSH Measured", value=True)
        patient_data['T3_measured'] = st.checkbox("T3 Measured", value=True)
        patient_data['TT4_measured'] = st.checkbox("TT4 Measured", value=True)
        patient_data['T4U_measured'] = st.checkbox("T4U Measured", value=True)
        patient_data['FTI_measured'] = st.checkbox("FTI Measured", value=True)
        
        st.markdown('<p class="input-group-title">🏢 Referral Information</p>', unsafe_allow_html=True)
        patient_data['referral_source'] = st.selectbox(
            "Referral Source",
            ["SVHC", "STMW", "SVI", "SVHD", "other"],
            help="Select the referring healthcare facility"
        )
    
    # Convert boolean to int
    for key, value in patient_data.items():
        if isinstance(value, bool):
            patient_data[key] = 1 if value else 0
    
    # Predict button - full width and prominent
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔬 Generate Prediction", key="predict_btn")
    if predict_button:
        make_prediction(patient_data, processor, predictor)
    
    # Display results
    if st.session_state.predictions:
        display_prediction_results(st.session_state.predictions[0], patient_data, processor, ai_generator)


def make_prediction(patient_data, processor, predictor):
    """Make prediction for single patient."""
    try:
        with st.spinner("🔄 Analyzing patient data..."):
            # Process data
            processed_df = processor.process_single_input(patient_data)
            
            # Make prediction
            predictions = predictor.predict(processed_df)
            
            # Store in session state
            st.session_state.predictions = predictions
            st.session_state.patient_data = patient_data
            
            st.success("✅ Analysis complete!")
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.exception(e)


def display_prediction_results(prediction, patient_data, processor, ai_generator):
    """Display prediction results with visualizations."""
    
    st.markdown("---")
    st.markdown('<p class="section-header">📋 Analysis Results</p>', unsafe_allow_html=True)
    
    # Main results
    stage1 = prediction.get('stage1_prediction')
    final_diagnosis_code = prediction.get('final_diagnosis')
    final_diagnosis = get_diagnosis_name(final_diagnosis_code)
    diagnosis_category = get_diagnosis_category(final_diagnosis_code) if final_diagnosis_code in DIAGNOSIS_MAPPING else None
    risk_level = prediction.get('risk_level')
    
    is_negative = stage1 == "Not Sick"
    if is_negative:
        final_diagnosis = "Negative Thyroid Disease"
        diagnosis_category = None
    else:
        final_diagnosis = get_diagnosis_name(final_diagnosis_code)
        diagnosis_category = get_diagnosis_category(final_diagnosis_code) if final_diagnosis_code in DIAGNOSIS_MAPPING else None

    box_class = "healthy" if is_negative else "sick"
    icon = "✅" if is_negative else "⚠️"
    title = "Negative Thyroid Disease" if is_negative else "Positive Thyroid Disease"

    category_html = f'<p class="diagnosis-category">Category: {diagnosis_category}</p>' if diagnosis_category else ''

    risk_html = ''
    if risk_level:
        risk_colors = {
            "Low": "#28a745",
            "Medium": "#ffc107",
            "High": "#dc3545"
        }
        risk_color = risk_colors.get(risk_level, "#6c757d")
        risk_html = f'''<p style="font-size: 1.2rem;">
            <strong>Risk Assessment:</strong> 
            <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span>
        </p>'''

    if not is_negative:
        st.markdown(f"""<div class="prediction-box {box_class}">
            <h2>{icon} {title}</h2>
            <p style="font-size: 1.3rem; margin-top: 1rem;"><strong>Diagnosis:</strong> {final_diagnosis}</p>
            {category_html}
            {risk_html}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="prediction-box {box_class}">
            <h2>{icon} {title}</h2>
            <p style="font-size: 1.3rem; margin-top: 1rem;"><strong>Diagnosis:</strong> {final_diagnosis}</p>""", unsafe_allow_html=True)
    
    # Detailed metrics
    st.markdown("### 📊 Confidence Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        confidence = prediction.get('stage1_confidence')
        st.metric(
            "Initial Screening",
            stage1,
            f"{confidence:.1%} confidence" if confidence is not None else "N/A"
        )
    
    with col2:
        if prediction.get('stage2_prediction'):
            stage2_pred = get_diagnosis_name(prediction.get('stage2_prediction'))
            st.metric(
                "Specific Classification",
                stage2_pred,
                f"{prediction.get('stage2_confidence', 0):.1%} confidence"
            )
        else:
            st.metric("Specific Classification", "Not Applicable", "No condition detected")
    
    # Clinical context
    with st.expander("📊 Clinical Values Comparison", expanded=True):
        st.markdown("#### Laboratory Values vs. Normal Ranges")
        
        normal_ranges = processor.get_normal_ranges()
        
        clinical_metrics = []
        for metric, (low, high) in normal_ranges.items():
            if metric in patient_data:
                value = patient_data[metric]
                if metric == 'age':
                    status = "—"
                else:
                    if low <= value <= high:
                        status = "✅ Within Range"
                    elif value < low:
                        status = "⬇️ Below Normal"
                    else:
                        status = "⬆️ Above Normal"
                
                clinical_metrics.append({
                    'Test': metric.upper(),
                    'Patient Value': f"{float(value):.2f}" if isinstance(value, (int, float)) else str(value),
                    'Normal Range': f"{low} - {high}",
                    'Status': status
                })
        
        if clinical_metrics:
            df_display = pd.DataFrame(clinical_metrics)
            # Ensure all columns are strings to avoid type conversion issues
            for col in df_display.columns:
                df_display[col] = df_display[col].astype(str)
            st.dataframe(df_display)
    
    # Feature importance
    if prediction.get('feature_importance'):
        st.markdown("### 🔍 Key Contributing Factors")
        
        # Ensure importance is float and sort
        top_features = [
            (str(k).upper(), float(v)) 
            for k, v in sorted(
                prediction['feature_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        ]
        
        if top_features:
            feat_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
            # Sort so highest importance is at the top of the chart
            feat_df = feat_df.sort_values('Importance', ascending=True)
            
            fig = px.bar(
                feat_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Influential Features',
                color='Importance',
                color_continuous_scale='Viridis',
                text=feat_df['Importance'].apply(lambda x: f'{x:.3f}')
            )
            fig.update_layout(
                height=400,
                xaxis_title="Importance Score",
                yaxis_title="",
                xaxis=dict(type='linear'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # AI-Powered Insights Section
        st.markdown("### 🤖 AI-Powered Health Insights")
        
        if ai_generator and ai_generator.client:
            with st.spinner("🔄 Generating personalized insights..."):
                insights = ai_generator.generate_insights(
                    patient_data,
                    prediction,
                    top_features
                )
            
            if 'error' not in insights or not insights.get('error'):
                def clean_insight(text):
                    """Remove trailing newlines and separators from insight text."""
                    if text:
                        if text.endswith('---'):
                            text = text[:-3].rstrip()
                            text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
                        return text
                    return text
                
                # Display insights with better spacing - each section in its own expander
                
                with st.expander("📋 Result Interpretation", expanded=True):
                    st.markdown(clean_insight(insights["interpretation"]))
                
                with st.expander("⚠️ Key Risk Factors", expanded=True):
                    st.markdown(clean_insight(insights["risk_factors"]))
                
                with st.expander("💡 Recommendations", expanded=True):
                    st.markdown(clean_insight(insights["recommendations"]))
                
                with st.expander("📚 Educational Information", expanded=True):
                    st.markdown(clean_insight(insights["education"]))
            else:
                st.info("💡 AI insights unavailable. " + insights.get('interpretation', 'Please configure OpenAI API key in .env file to enable this feature.'))
        else:
            st.info("💡 AI insights are currently disabled. Add your OPENAI_API_KEY to the .env file to unlock personalized health insights powered by GPT-4.")
    
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    if stage1 == "Not Sick":
        st.markdown("""
        <div class="success-box">
        <strong>✅ No immediate concerns detected</strong><br><br>
        • Continue regular health monitoring<br>
        • Maintain thyroid medication if prescribed<br>
        • Schedule routine follow-up as recommended by your doctor
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-box">
        <strong>⚠️ Further evaluation recommended</strong><br><br>
        • Consult with an endocrinologist<br>
        • Discuss the {final_diagnosis} classification<br>
        • Consider additional diagnostic tests<br>
        • Follow up on treatment options
        </div>
        """, unsafe_allow_html=True)


def information_tab():
    """Information and help tab."""
    st.markdown('<p class="section-header">📚 User Guide & Information</p>', unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("## 🚀 Quick Start Guide")
    
    st.markdown("""
    ### How to Use This System
    
    1. **Enter Patient Demographics**
       - Fill in age and biological sex at the top of the form
    
    2. **Input Laboratory Results**
       - Enter all thyroid hormone test values (TSH, T3, TT4, T4U, FTI)
       - Reference ranges are shown below each field
    
    3. **Provide Medical History**
       - Check boxes for current medications
       - Indicate any previous treatments
       - Note relevant medical conditions
       - Specify current symptoms
    
    4. **Confirm Tests Performed**
       - Indicate which tests were actually measured
       - Select the referral source
    
    5. **Generate Prediction**
       - Click the "Generate Prediction" button
       - Wait for analysis to complete
    
    6. **Review Results**
       - Examine the diagnosis and risk assessment
       - Review confidence metrics and probability distributions
       - Check clinical value comparisons
       - Read AI-powered insights (if enabled)
    
    7. **Start New Session**
       - Click "New Session" button in sidebar to assess another patient
    """)
    
    st.markdown("---")
    
    # System Overview
    st.markdown("## 🎯 How the System Works")
    
    st.markdown("""
    This AI-powered system uses a sophisticated **two-stage hierarchical** approach for thyroid disease screening:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>🔍 Stage 1: Initial Screening</strong><br><br>
        <strong>Purpose:</strong> Determine if thyroid abnormality exists<br>
        <strong>Output:</strong> Healthy or Sick classification<br>
        <strong>Features:</strong> Uses all clinical and demographic data<br>
        <strong>Goal:</strong> High sensitivity to catch potential cases
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <strong>🎯 Stage 2: Disease Classification</strong><br><br>
        <strong>Purpose:</strong> Identify specific thyroid disorder<br>
        <strong>Output:</strong> Specific condition classification<br>
        <strong>Trigger:</strong> Only runs if Stage 1 detects condition<br>
        <strong>Goal:</strong> Accurate disorder type identification
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Diagnosis Classification Guide
    st.markdown("## 🏥 Diagnosis Classifications")
    
    st.markdown("### Understanding the Diagnosis Codes")
    st.markdown("The system classifies thyroid conditions into the following categories:")
    
    # Hyperthyroid Conditions
    with st.expander("🔴 Hyperthyroid Conditions", expanded=False):
        st.markdown("""
        **A - Hyperthyroid:** General overactive thyroid condition with elevated thyroid hormone levels.
        
        **B - T3 Toxic:** Elevated T3 hormone levels causing thyrotoxicosis.
        
        **C - Toxic Goitre:** Enlarged thyroid gland producing excess hormones.
        
        **D - Secondary Toxic:** Hyperthyroidism caused by factors outside the thyroid gland.
        """)
    
    # Hypothyroid Conditions
    with st.expander("🔵 Hypothyroid Conditions", expanded=False):
        st.markdown("""
        **E - Hypothyroid:** General underactive thyroid condition with insufficient hormone production.
        
        **F - Primary Hypothyroid:** Hypothyroidism originating from thyroid gland dysfunction.
        
        **G - Compensated Hypothyroid:** Mild thyroid dysfunction with normal thyroid hormone levels but elevated TSH.
        
        **H - Secondary Hypothyroid:** Hypothyroidism caused by pituitary or hypothalamus problems.
        """)
    
    # Binding Protein Issues
    with st.expander("🟡 Binding Protein Conditions", expanded=False):
        st.markdown("""
        **I - Increased Binding Protein:** Elevated levels of proteins that bind thyroid hormones, affecting hormone availability.
        
        **J - Decreased Binding Protein:** Reduced binding protein levels affecting thyroid hormone transport.
        """)
    
    # General Health
    with st.expander("🟢 General Health Issues", expanded=False):
        st.markdown("""
        **K - Concurrent Non-Thyroidal Illness:** Thyroid function changes due to other systemic illnesses.
        """)
    
    # Replacement Therapy
    with st.expander("💊 Replacement Therapy Status", expanded=False):
        st.markdown("""
        **L - Consistent with Replacement Therapy:** Thyroid levels appropriate for current medication.
        
        **M - Underreplaced:** Insufficient thyroid hormone replacement medication.
        
        **N - Overreplaced:** Excessive thyroid hormone replacement causing hyperthyroid symptoms.
        """)
    
    # Antithyroid Treatment
    with st.expander("🏥 Antithyroid Treatment", expanded=False):
        st.markdown("""
        **O - Antithyroid Drugs:** Patient on medication to reduce thyroid hormone production.
        
        **P - I131 Treatment:** Patient has undergone radioactive iodine therapy.
        
        **Q - Surgery:** Patient has had thyroid surgery.
        """)
    
    # Miscellaneous
    with st.expander("⚪ Miscellaneous Findings", expanded=False):
        st.markdown("""
        **R - Discordant Assay Results:** Laboratory test results that don't match expected patterns.
        
        **S - Elevated TBG:** Elevated Thyroxine-Binding Globulin levels.
        
        **T - Elevated Thyroid Hormones:** Increased thyroid hormone levels without clear hyperthyroid symptoms.
        """)
    
    st.markdown("---")
    
    # Clinical Tests Explained
    st.markdown("## 🧪 Understanding Lab Tests")
    
    tests_info = {
        "TSH (Thyroid Stimulating Hormone)": {
            "range": "0.4 - 4.0 mIU/L",
            "description": "Produced by pituitary gland to regulate thyroid function. Elevated TSH suggests hypothyroidism; low TSH suggests hyperthyroidism."
        },
        "T3 (Triiodothyronine)": {
            "range": "0.8 - 2.0 nmol/L",
            "description": "Active thyroid hormone. Affects metabolism, heart rate, and body temperature."
        },
        "TT4 (Total Thyroxine)": {
            "range": "60 - 150 nmol/L",
            "description": "Primary hormone produced by thyroid gland. Converted to T3 in body tissues."
        },
        "T4U (T4 Uptake)": {
            "range": "0.75 - 1.15",
            "description": "Measures binding proteins in blood. Used to calculate Free Thyroxine Index."
        },
        "FTI (Free Thyroxine Index)": {
            "range": "60 - 170",
            "description": "Calculated measure of free (unbound) thyroxine. More accurate than TT4 alone."
        }
    }
    
    for test, info in tests_info.items():
        with st.expander(f"{test}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Normal Range:**  \n{info['range']}")
            with col2:
                st.markdown(f"{info['description']}")
    
    st.markdown("---")
    
    # Interpreting Results
    st.markdown("## 📊 Interpreting Your Results")
    
    st.markdown("### Confidence Levels")
    confidence_data = pd.DataFrame({
        'Confidence Range': ['> 90%', '70% - 90%', '< 70%'],
        'Interpretation': ['High Confidence', 'Moderate Confidence', 'Low Confidence'],
        'Recommendation': [
            'Results are highly reliable',
            'Results are generally reliable',
            'Consult healthcare provider for confirmation'
        ]
    })
    st.table(confidence_data)
    
    st.markdown("### Risk Levels")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <strong>🟢 Low Risk</strong><br>
        • No significant findings<br>
        • Continue routine monitoring<br>
        • Maintain current treatment
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>🟡 Medium Risk</strong><br>
        • Some indicators present<br>
        • Schedule follow-up<br>
        • Monitor symptoms
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 1rem; border-radius: 8px;">
        <strong>🔴 High Risk</strong><br>
        • Multiple indicators<br>
        • Seek medical attention<br>
        • Further testing needed
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI Insights Section
    st.markdown("## 🤖 AI-Powered Insights")
    
    st.markdown("""
    ### What are AI Insights?
    
    When enabled, this feature uses GPT-4 to provide:
    
    - **Result Interpretation**: Plain-language explanation of your classification results
    - **Risk Factors Analysis**: Identification of key factors contributing to the diagnosis
    - **Personalized Recommendations**: Tailored next steps based on your specific results
    - **Educational Information**: Context about thyroid conditions and their management
    
    ### How to Enable
    
    1. Create a `.env` file in your project root directory
    2. Add your OpenAI API key: `OPENAI_API_KEY=your-api-key-here`
    3. Restart the application
    4. AI insights will automatically be available when generating predictions
    
    ### How to Get an OpenAI API Key
    
    1. Visit [platform.openai.com](https://platform.openai.com/)
    2. Sign up or log in to your account
    3. Navigate to API Keys section
    4. Create a new API key
    5. Copy and add it to your .env file
    
    ### Privacy Note
    
    When AI insights are enabled, patient data is sent to OpenAI's API for analysis. Patient data is processed securely and is not stored by OpenAI beyond the duration of the API call.
    """)
    
    st.markdown("---")
    
    # Important Disclaimers
    st.markdown("## ⚠️ Important Information")
    
    st.error("""
    ### Medical Disclaimer
    
    **This system is a screening tool, NOT a diagnostic instrument.**
    
    - ❌ **Do not use** for self-diagnosis
    - ❌ **Do not use** to make treatment decisions
    - ❌ **Do not use** as substitute for professional medical advice
    
    **Always consult qualified healthcare professionals:**
    - ✅ For diagnosis confirmation
    - ✅ For treatment planning
    - ✅ For medical advice and care
    
    Machine learning models can make errors. Clinical judgment and professional expertise are essential.
    """)
    
    st.markdown("---")
    
    # FAQ Section
    st.markdown("## ❓ Frequently Asked Questions")
    
    faqs = {
        "How accurate is this system?": "The system achieves high F1 scores on test data, but accuracy varies by case. Always verify with medical professionals.",
        "Can I use this for diagnosis?": "No. This is a screening tool only. Diagnosis requires comprehensive medical evaluation by qualified professionals.",
        "What do the diagnosis codes mean?": "Each code (A-T) represents a specific thyroid condition or status. See the 'Diagnosis Classifications' section for detailed explanations.",
        "What if I get conflicting results?": "Consult with your healthcare provider. Lab values can fluctuate, and clinical context is crucial.",
        "How often should I get tested?": "Follow your doctor's recommendations. Typically, thyroid testing is done annually or as symptoms arise.",
        "Is my data secure?": "Data processed in this session is not stored permanently. When AI insights are enabled, data is sent to OpenAI's API but is not stored beyond the API call duration.",
        "What should I do with high-risk results?": "Schedule an appointment with your healthcare provider as soon as possible for proper evaluation.",
        "What are AI-powered insights?": "AI insights use GPT-4 to provide personalized explanations of your results, risk factor analysis, and educational information about thyroid health. Enable by adding OPENAI_API_KEY to your .env file.",
        "How do I start a new assessment?": "Click the 'New Session' button in the sidebar to clear current results and start fresh with a new patient.",
        "Why aren't AI insights working?": "Ensure you have added OPENAI_API_KEY to your .env file and restarted the application. Check the sidebar to see if AI insights are available."
    }
    
    for question, answer in faqs.items():
        with st.expander(f"{question}"):
            st.markdown(answer)
    
    st.markdown("---")
    
    # Features Overview
    st.markdown("## ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Core Functionality
        
        ✅ **Two-stage hierarchical classification**
        - Initial health screening
        - Specific condition identification
        
        ✅ **Comprehensive input form**
        - Demographics
        - Laboratory results
        - Medical history
        - Current status
        
        ✅ **Detailed results**
        - Confidence metrics
        - Probability distributions
        - Feature importance analysis
        - Clinical value comparison
        """)
    
    with col2:
        st.markdown("""
        ### Advanced Features
        
        🤖 **AI-Powered Insights** (Optional)
        - GPT-4 result interpretation
        - Risk factor analysis
        - Personalized recommendations
        - Automatic via .env configuration
        
        🔄 **Session Management**
        - New session button
        - Clear results easily
        - Assess multiple patients
        
        📊 **Visualization**
        - Interactive charts
        - Probability graphs
        - Feature importance
        """)
    
    st.markdown("---")
    
    # Contact and Support
    st.markdown("## 📞 Support & Feedback")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Need Help?
        
        If you encounter technical issues:
        - Check your input data format
        - Ensure all required fields are filled
        - Review error messages carefully
        - Click "New Session" to restart
        - Verify .env configuration for AI insights
        
        **📧 Contact Support:**  
        Email: [nathalierocelle19@gmail.com](mailto:nathalierocelle19@gmail.com)
        """)
    
    with col2:
        st.markdown("""
        ### Provide Feedback
        
        Help us improve this system:
        - Report bugs or errors
        - Suggest new features
        - Share user experience
        - Request documentation updates
        - Ask questions about setup
        
        **📧 Send Feedback:**  
        Email: [nathalierocelle19@gmail.com](mailto:nathalierocelle19@gmail.com)
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>📬 We'd love to hear from you!</strong><br>
    For any questions, concerns, or suggestions, please contact us at 
    <a href="mailto:nathalierocelle19@gmail.com">nathalierocelle19@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()