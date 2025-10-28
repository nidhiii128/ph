import streamlit as st

def show_header():
    st.markdown("""
    <div style='text-align:center; padding:20px;'>
        <h1 style='font-size:4rem; background: linear-gradient(90deg,#ff0080,#7928ca);
                   -webkit-background-clip: text; color: transparent;'>
            🛡️ PhishGuard
        </h1>
        <p style='font-size:1.5rem; color:#999;'>Real-time phishing URL scanner ⚡</p>
    </div>
    """, unsafe_allow_html=True)
