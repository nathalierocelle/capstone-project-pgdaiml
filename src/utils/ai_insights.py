"""
AI Insights Generator using OpenAI GPT-4
Provides personalized health insights based on thyroid classification results.
"""

import os
from openai import OpenAI


class AIInsightsGenerator:
    """Generate AI-powered health insights using OpenAI's GPT-4."""
    
    def __init__(self, api_key=None):
        """
        Initialize the AI insights generator.
        
        Args:
            api_key: OpenAI API key (optional, will check environment variable if not provided)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                self.client = None
    
    def generate_insights(self, patient_data, prediction, top_features):
        """
        Generate personalized health insights based on prediction results.
        
        Args:
            patient_data: Dictionary containing patient information
            prediction: Dictionary containing prediction results
            top_features: List of tuples with (feature_name, importance_score)
        
        Returns:
            Dictionary containing different types of insights
        """
        if not self.client:
            return {
                'error': True,
                'interpretation': 'OpenAI API key not configured.'
            }
        
        try:
            # Safely extract values with defaults and convert to strings
            stage1 = str(prediction.get('stage1_prediction') or 'Unknown')
            stage2 = str(prediction.get('stage2_prediction') or 'Not classified')
            
            # Handle confidence values carefully
            confidence1_raw = prediction.get('stage1_confidence')
            confidence1 = f"{float(confidence1_raw):.1%}" if confidence1_raw is not None else "Not available"
            
            confidence2_raw = prediction.get('stage2_confidence')
            confidence2 = f"{float(confidence2_raw):.1%}" if confidence2_raw is not None else "Not available"
            
            risk_level = str(prediction.get('risk_level') or 'Unknown')
            final_diagnosis = str(prediction.get('final_diagnosis') or 'Unknown')
            
            # Safely extract patient data
            age = str(patient_data.get('age') or 'Unknown')
            sex_val = patient_data.get('sex')
            sex = 'Female' if sex_val == 1 else 'Male' if sex_val == 0 else 'Unknown'
            
            # Handle lab values
            tsh = patient_data.get('TSH')
            tsh_str = f"{float(tsh):.2f}" if tsh is not None else 'Not measured'
            
            t3 = patient_data.get('T3')
            t3_str = f"{float(t3):.2f}" if t3 is not None else 'Not measured'
            
            tt4 = patient_data.get('TT4')
            tt4_str = f"{float(tt4):.2f}" if tt4 is not None else 'Not measured'
            
            # Build feature importance string safely
            feature_list = []
            if top_features and len(top_features) > 0:
                for item in top_features[:5]:
                    try:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            feat, importance = item[0], item[1]
                            if feat is not None and importance is not None:
                                feat_str = str(feat).upper()
                                imp_str = f"{float(importance):.3f}"
                                feature_list.append(f"- {feat_str}: {imp_str}")
                    except (ValueError, TypeError, IndexError):
                        continue
            
            features_str = "\n".join(feature_list) if feature_list else "No feature importance data available"
            
            # Create comprehensive prompt with all strings
            prompt = f"""You are a medical AI assistant helping to explain thyroid disease screening results. 
Provide clear, compassionate, and accurate insights based on this patient's data.

Patient Information:
- Age: {age} years
- Sex: {sex}
- TSH: {tsh_str} mIU/L (Normal: 0.4-4.0)
- T3: {t3_str} nmol/L (Normal: 0.8-2.0)
- TT4: {tt4_str} nmol/L (Normal: 60-150)

Classification Results:
- Stage 1 (Initial Screening): {stage1}
- Stage 1 Confidence: {confidence1}
- Stage 2 Classification: {stage2}
- Stage 2 Confidence: {confidence2}
- Final Diagnosis: {final_diagnosis}
- Risk Level: {risk_level}

Top Contributing Features:
{features_str}

Please provide insights in all of these areas:

RESULT INTERPRETATION: Explain what these results mean or the overview of the results to a patient. If the patient is positive with thyroid disease, explain the diagnosis classification.
RISK FACTORS: Identify the key factors that influenced this classification.
RECOMMENDATIONS: Suggest practical next steps (emphasize consulting healthcare professionals).
EDUCATION: Provide brief educational information about thyroid health relevant to these results.

Format your response with clear section headers and use markdown formatting."""

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical AI assistant specializing in thyroid health. Provide clear, accurate, and compassionate explanations. Always emphasize that these are screening results and not a diagnosis, and that patients should consult healthcare professionals."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            # Extract the response
            full_response = response.choices[0].message.content
            
            # Parse the response into sections
            insights = self._parse_response(full_response)
            
            return insights
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in generate_insights: {error_details}")
            return {
                'error': True,
                'interpretation': f'Unable to generate insights at this time.'
            }
    
    def _parse_response(self, response_text):
        """
        Parse the GPT-4 response into structured sections.
        
        Args:
            response_text: The full text response from GPT-4
        
        Returns:
            Dictionary with parsed sections
        """
        sections = {
            'interpretation': '',
            'risk_factors': '',
            'recommendations': '',
            'education': ''
        }
        
        if not response_text:
            return sections
        
        # Try to split by numbered sections or headers
        lines = response_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            line_original = line.strip()
            
            # Detect section headers
            if any(keyword in line_lower for keyword in ['result interpretation', 'interpretation:']):
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'interpretation'
                current_content = []
                continue
            
            elif any(keyword in line_lower for keyword in ['risk factors', 'risk factor']):
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'risk_factors'
                current_content = []
                continue
            
            elif any(keyword in line_lower for keyword in ['recommendations', 'next steps']):
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'recommendations'
                current_content = []
                continue
            
            elif any(keyword in line_lower for keyword in ['education', 'educational information', 'about thyroid']):
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'education'
                current_content = []
                continue
            
            else:
                # Add content to current section (skip empty lines at the start)
                if current_section:
                    if line_original or current_content:  # Add line if not empty or if we already have content
                        current_content.append(line_original)
        
        # Don't forget the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If parsing failed, put everything in interpretation
        if not any(sections.values()):
            sections['interpretation'] = response_text
        
        # Remove empty sections
        sections = {k: v for k, v in sections.items() if v}
        
        return sections