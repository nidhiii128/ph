import streamlit as st

def theme():
    st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(90deg,#ff0080,#7928ca);
        color:white; font-size:1.2rem; border-radius:15px; padding:10px 20px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.1);
        background: linear-gradient(90deg,#7928ca,#ff0080);
    }
    .stTextInput>div>input {
        border-radius:15px; border:2px solid #7928ca; padding:12px;
        transition: all 0.3s ease-in-out;
    }
    .stTextInput>div>input:focus {
        border-color:#ff0080;
        box-shadow:0 0 10px rgba(255,0,128,0.5);
    }
    </style>
    """, unsafe_allow_html=True)
