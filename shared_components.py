from dotenv import load_dotenv
import streamlit as st
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(curr_dir, ".env"))


def api_settings_ui():
    st.title("API Settings")
    st.session_state["HF_TOKEN"] = st.text_input(
        "HF Token",
        value=os.getenv("HF_TOKEN"),
        type="password",
        help="Get your token at: https://huggingface.co/settings/tokens",
    )
    st.session_state["OPENROUTER_API_KEY"] = st.text_input(
        "OpenRouter API Key",
        value=os.getenv("OPENROUTER_API_KEY"),
        type="password",
        help="Get your key at: https://openrouter.ai/settings/keys",
    )
    st.session_state["TOGETHER_API_KEY"] = st.text_input(
        "Together API Key",
        value=os.getenv("TOGETHER_API_KEY"),
        type="password",
    )
