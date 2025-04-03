import streamlit as st
from shared_components import api_settings_ui
import os
from huggingface_hub import InferenceClient
from PIL import Image
import tempfile

# Page config
st.set_page_config(page_title="Text to Image", layout="wide")
st.title("Text -> Image")

# Sidebar for API settings
with st.sidebar:
    api_settings_ui()

    # Provider selection
    st.subheader("Provider Settings")
    provider = st.selectbox(
        "Select Provider",
        options=["hf-inference", "replicate", "openai", "fireworks-ai", "stability"],
        index=1,
        help="Select the provider for image generation",
    )

    # Model selection based on provider
    model_options = {
        "hf-inference": [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/sdxl-turbo",
        ],
        "replicate": [
            "black-forest-labs/FLUX.1-schnell",
            "stability-ai/sdxl",
            "stability-ai/stable-diffusion",
            "midjourney/midjourney",
        ],
        "openai": ["dall-e-3"],
        "fireworks-ai": ["fireworks/stable-diffusion-xl"],
        "stability": ["stable-diffusion-xl-1024-v1-0"],
    }

    selected_model = st.selectbox(
        "Select Model",
        options=model_options.get(provider, ["default-model"]),
        index=0,
        help="Select the model for image generation",
    )

    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        num_inference_steps = st.slider(
            "Number of Steps", min_value=1, max_value=100, value=30
        )
        guidance_scale = st.slider(
            "Guidance Scale", min_value=1.0, max_value=20.0, value=7.5
        )
        width = st.select_slider("Width", options=[256, 512, 768, 1024], value=512)
        height = st.select_slider("Height", options=[256, 512, 768, 1024], value=512)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    # Input area
    st.subheader("Text Prompt")

    # Example prompts
    example_prompts = [
        "A flying car crossing a futuristic cityscape.",
        "A serene lake surrounded by mountains at sunset.",
        "A photorealistic robot playing chess in a park.",
        "A cyberpunk city at night with neon lights.",
    ]

    example = st.selectbox(
        "Example prompts (or write your own below)",
        options=[""] + example_prompts,
        index=0,
    )

    # Text prompt input
    prompt = st.text_area(
        "Enter your prompt",
        value=example if example else "",
        height=150,
        placeholder="Describe the image you want to generate...",
    )

    # Negative prompt
    negative_prompt = st.text_input(
        "Negative prompt (optional)", placeholder="Elements to avoid in the image"
    )

    # Submit button
    generate_button = st.button(
        "Generate Image",
        type="primary",
        disabled=not prompt or not st.session_state.get("HF_TOKEN", ""),
    )

with col2:
    # Output area
    st.subheader("Generated Image")

    # Display error if API token is not set
    if not st.session_state.get("HF_TOKEN", ""):
        st.error("Please set your API token in the sidebar first!")

    # Placeholder for the image
    image_placeholder = st.empty()

# Generate image when button is clicked
if generate_button and prompt and st.session_state.get("HF_TOKEN", ""):
    with st.spinner("Generating image..."):
        try:
            # Create inference client
            client = InferenceClient(
                model=selected_model,
                provider=provider,
                token=st.session_state["HF_TOKEN"],
            )

            # Different parameters for different providers
            if provider == "hf-inference":
                params = {}
                if negative_prompt:
                    params["negative_prompt"] = negative_prompt
                if num_inference_steps:
                    params["num_inference_steps"] = num_inference_steps
                if guidance_scale:
                    params["guidance_scale"] = guidance_scale

                generated_image = client.text_to_image(
                    prompt,
                    **params,
                    width=width,
                    height=height,
                )
            else:
                # For other providers, parameters may vary
                generated_image = client.text_to_image(
                    prompt,
                    model=selected_model,
                )

            # Display the generated image
            image_placeholder.image(
                generated_image, caption=prompt, use_container_width=True
            )

            # Save image temporarily and provide download button
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                generated_image.save(tmp_file.name)
                st.download_button(
                    label="Download Image",
                    data=open(tmp_file.name, "rb").read(),
                    file_name="generated_image.png",
                    mime="image/png",
                )
                # Clean up temp file
                os.unlink(tmp_file.name)

        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            st.info(
                "Some models or providers may require specific API access. Please check your credentials."
            )
