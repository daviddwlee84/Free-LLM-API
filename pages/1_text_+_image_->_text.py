import streamlit as st
from shared_components import api_settings_ui

with st.sidebar:
    st.link_button(
        "OpenRouter Models",
        "https://openrouter.ai/models?fmt=table&input_modalities=image%2Ctext&output_modalities=text&max_price=0",
    )

    api_settings_ui()

"""
[What is Image-Text-to-Text? - Hugging Face](https://huggingface.co/tasks/image-text-to-text)

curl https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.2-11B-Vision-Instruct \
    -X POST \
    -d '{"messages": [{"role": "user","content": [{"type": "image"}, {"type": "text", "text": "Can you describe the image?"}]}]}' \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer hf_***"
"""

st.title("Text + Image -> Text")
