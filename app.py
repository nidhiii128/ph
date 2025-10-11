import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.feature_engineering import URLFeatureExtractor
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="PhishGuard 🛡️",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .safe-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
    }
    .phishing-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
    }
    .feature-plot {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and feature extractor"""
    try:
        model_data = joblib.load('models/trained_model.joblib')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_url(url, model_data):
    """Predict if a URL is phishing or benign"""
    extractor = URLFeatureExtractor()
    
    # Extract features
    features = extractor.extract_features(url)
    feature_df = pd.DataFrame([features])
    
    # Ensure all feature columns are present
    for col in model_data['feature_columns']:
        if col not in feature_df.columns:
            feature_df[col] = 0
    
    # Reorder columns to match training
    feature_df = feature_df[model_data['feature_columns']]
    
    # Scale features if needed
    if model_data['needs_scaling']:
        features_scaled = model_data['scaler'].transform(feature_df)
        prediction = model_data['model'].predict(features_scaled)[0]
        probability = model_data['model'].predict_proba(features_scaled)[0]
    else:
        prediction = model_data['model'].predict(feature_df)[0]
        probability = model_data['model'].predict_proba(feature_df)[0]
    
    return prediction, probability, features

def main():
    # Header Section
    st.markdown('<h1 class="main-header">PhishGuard 🛡️</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A real-time phishing URL scanner powered by Machine Learning</p>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.error("⚠️ Model could not be loaded. Please check if the model file exists.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("About PhishGuard")
        st.markdown("""
        This tool analyzes URLs for phishing patterns using a **{}** model 
        trained on 500,000+ examples.
        
        **How it works:**
        - Analyzes URL structure and patterns
        - Doesn't visit the website
        - Uses lexical features for safety
        - Provides instant results
        
        **Model Performance:**
        - Accuracy: {:.1f}%
        - Precision: {:.1f}%
        - Recall: {:.1f}%
        """.format(
            model_data['model_name'],
            model_data['performance']['accuracy'] * 100,
            model_data['performance']['precision'] * 100,
            model_data['performance']['recall'] * 100
        ))
        
        st.markdown("---")
        st.markdown("**⚠️ Disclaimer:** This tool is for educational purposes. Always use multiple security measures.")
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scanner Section
        st.header("🔍 URL Scanner")
        url_input = st.text_input(
            "Enter a URL to analyze:",
            placeholder="https://www.example.com",
            help="Paste the complete URL including http:// or https://"
        )
        
        analyze_btn = st.button("🚀 Analyze URL", type="primary", use_container_width=True)
        
        # Results Section
        if analyze_btn and url_input:
            if not url_input.startswith(('http://', 'https://')):
                st.warning("⚠️ Please include http:// or https:// in the URL for accurate analysis.")
                url_input = 'https://' + url_input
            
            with st.spinner("Analyzing URL features..."):
                prediction, probability, features = predict_url(url_input, model_data)
            
            # Display Results
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Prediction with visual feedback
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                # Confidence gauge
                confidence = probability[1] if prediction == 1 else probability[0]
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    title = {'text': "Confidence"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_res2:
                if prediction == 0:  # Benign
                    st.markdown('<div class="safe-box">', unsafe_allow_html=True)
                    st.success("✅ **SAFE URL**")
                    st.markdown(f"**Result:** BENIGN - This URL appears to be safe.")
                    st.markdown(f"**Confidence:** {probability[0]*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:  # Phishing
                    st.markdown('<div class="phishing-box">', unsafe_allow_html=True)
                    st.error("🚨 **PHISHING URL**")
                    st.markdown(f"**Result:** PHISHING - This URL is suspicious!")
                    st.markdown(f"**Confidence:** {probability[1]*100:.1f}%")
                    st.markdown("**Advice:** Do not enter any personal information!")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature Analysis
            st.subheader("🔍 Feature Analysis")
            
            # Display key features
            key_features = {
                'URL Length': features['url_length'],
                'Uses HTTPS': 'Yes' if features['uses_https'] else 'No',
                'Contains IP': 'Yes' if features['ip_in_url'] else 'No',
                'Shortened URL': 'Yes' if features['is_shortened'] else 'No',
                'Suspicious Keywords': features['suspicious_words_count'],
                'Special Characters': features['count_special_chars'] if 'count_special_chars' in features else 
                                    sum(features.get(f'count_{char}', 0) for char in 
                                       ['dot', 'hyphen', 'underscore', 'slash', 'questionmark', 
                                        'equal', 'at', 'and', 'exclamation', 'space', 'tilde', 
                                        'comma', 'plus', 'asterisk', 'hash', 'dollar', 'percent'])
            }
            
            feat_cols = st.columns(3)
            for i, (feature, value) in enumerate(key_features.items()):
                with feat_cols[i % 3]:
                    st.metric(feature, value)
    
    with col2:
        # Quick Info Section
        st.header("📈 Model Info")
        st.info(f"**Algorithm:** {model_data['model_name']}")
        
        # Performance metrics
        perf_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [
                model_data['performance']['accuracy'],
                model_data['performance']['precision'],
                model_data['performance']['recall'],
                model_data['performance']['f1']
            ]
        }
        perf_df = pd.DataFrame(perf_data)
        perf_df['Score'] = (perf_df['Score'] * 100).round(1)
        
        fig = px.bar(perf_df, x='Score', y='Metric', orientation='h',
                     title="Model Performance (%)",
                     color='Score', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance preview
        st.subheader("🔑 Key Features")
        important_features = [
            "URL Length", "Special Characters", "Suspicious Keywords",
            "Domain Characteristics", "HTTPS Usage", "IP Address Presence"
        ]
        
        for feature in important_features:
            st.markdown(f"• {feature}")
        
        # Test URLs section
        st.markdown("---")
        st.subheader("🧪 Test Examples")
        
        test_urls = {
            "Safe": "https://www.google.com",
            "Safe": "https://github.com",
            "Suspicious": "http://secure-login-verify-account.com",
            "Suspicious": "https://bit.ly/suspicious-link-123"
        }
        
        for desc, url in test_urls.items():
            if st.button(f"Test: {desc}", key=url):
                st.session_state.test_url = url

if __name__ == "__main__":
    main()