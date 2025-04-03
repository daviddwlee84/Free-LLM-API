import streamlit as st
from shared_components import api_settings_ui

with st.sidebar:
    st.link_button(
        "OpenRouter Models",
        "https://openrouter.ai/models?fmt=table&input_modalities=text&output_modalities=text&max_price=0",
    )

    api_settings_ui()

st.title("Text -> Text")
