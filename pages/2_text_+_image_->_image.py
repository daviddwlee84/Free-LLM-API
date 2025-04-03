import streamlit as st
from shared_components import api_settings_ui
import os
from huggingface_hub import InferenceClient
from PIL import Image
import tempfile
import io
import requests
import base64
from io import BytesIO

# Page config
st.set_page_config(page_title="Text + Image to Image", layout="wide")
st.title("Text + Image -> Image")
st.caption("Text-Guided Image Generation")


# Sidebar for API settings
with st.sidebar:
    api_settings_ui()

# TODO: able to fetch image from clipboard

client = InferenceClient()

curr_dir = os.path.dirname(os.path.abspath(__file__))

# BUG: huggingface_hub.errors.HfHubHTTPError: 402 Client Error: Payment Required for url: https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-2-inpainting (Request ID: Root=1-67ee3d50-44017165644014b531529bab;db652428-0066-47f8-8f27-9845b7e6bfbe) You have exceeded your monthly included credits for Inference Providers. Subscribe to PRO to get 20x more monthly included credits.

image = client.image_to_image(
    os.path.join(curr_dir, "../sample/pig_draft.png"),
    prompt="turn the pig into a horse",
)

st.image(image)
