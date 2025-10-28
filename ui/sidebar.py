import streamlit as st
import plotly.express as px
import pandas as pd

def show_sidebar(model_data):
    st.markdown("## ⚡ Model Insights")
    st.info(f"**Algorithm:** {model_data['model_name']}")
    
    perf_df = pd.DataFrame({
        'Metric': ['Accuracy','Precision','Recall','F1-Score'],
        'Score': [
            model_data['performance']['accuracy'],
            model_data['performance']['precision'],
            model_data['performance']['recall'],
            model_data['performance']['f1']
        ]
    })
    perf_df['Score'] = (perf_df['Score']*100).round(1)
    
    fig = px.bar(perf_df, x='Score', y='Metric', orientation='h', color='Score',
                 color_continuous_scale='plasma', title='Model Performance (%)')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("**🔐 Disclaimer:** For educational use only. Always verify suspicious links.")
