import streamlit as st
import plotly.graph_objects as go

def result_card(url, prediction, probability, features=None):
    if prediction == 0:
        st.markdown(f"""
        <div style='background:#d4edda; border-radius:15px; padding:20px; margin:10px 0;
                    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);'>
            ✅ <b>SAFE URL</b><br>
            {url}<br>
            Confidence: {probability[0]*100:.1f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#f8d7da; border-radius:15px; padding:20px; margin:10px 0;
                    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);'>
            🚨 <b>PHISHING URL</b><br>
            {url}<br>
            Confidence: {probability[1]*100:.1f}%<br>
            <b style='color:red;'>Do NOT enter personal info!</b>
        </div>
        """, unsafe_allow_html=True)
    
    # Animated gauge for confidence
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability[1]*100 if prediction==1 else probability[0]*100,
        domain={'x':[0,1],'y':[0,1]},
        title={'text': "Confidence %"},
        gauge={'axis': {'range':[0,100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range':[0,50],'color':'#FF4136'},
                   {'range':[50,80],'color':'#FFDC00'},
                   {'range':[80,100],'color':'#2ECC40'}
               ],
               'threshold': {'line': {'color':'red','width':4}, 'value':90}}
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
