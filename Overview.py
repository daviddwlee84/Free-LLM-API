import streamlit as st
from shared_components import api_settings_ui

with st.sidebar:
    api_settings_ui()

st.title("Overview")
