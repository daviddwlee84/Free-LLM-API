import streamlit as st
from shared_components import api_settings_ui
import os
from huggingface_hub import InferenceClient, HfApi
from PIL import Image
import tempfile
import io
import requests
from io import BytesIO

# Page config
st.set_page_config(page_title="Text + Image to Image", layout="wide")
st.title("Text + Image -> Image")
st.caption("Transform images using text prompts")
st.caption("Text-Guided Image Generation")

st.link_button(
    "Hugging Face Image to Image Task",
    "https://huggingface.co/docs/inference-providers/tasks/image-to-image",
)

# Sidebar for API settings
with st.sidebar:
    api_settings_ui()

    # Provider selection
    st.subheader("Provider Settings")
    provider = st.selectbox(
        "Select Provider",
        options=[
            "hf-inference",
            "replicate",
            "fireworks-ai",
            "black-forest-labs",
            "together",
        ],
        index=0,
        help="Select the provider for image transformation",
    )

    # Get available models
    available_models = {
        "hf-inference": [
            "nitrosocke/Ghibli-Diffusion",
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            "enhanceaiteam/Flux-Uncensored-V2",
            "R1000/Flux.1-dev-Controlnet-Upscaler",
            "R1000/Flux-Super-Realism-LoRA-i2i",
        ],
        "replicate": [
            "stability-ai/sdxl-img2img",
            "cjwbw/dreamshaper-img2img",
            "fofr/face-to-many",
            "cjwbw/controlnet-scribble",
        ],
        "fireworks-ai": [
            "fireworks/stable-diffusion-xl-img2img",
        ],
        "black-forest-labs": [
            "black-forest-labs/FLUX.1-Canny-dev",
        ],
        "together": [
            "stabilityai/stable-diffusion-xl-base-1.0",
        ],
    }

    selected_model = st.selectbox(
        "Select Model",
        options=available_models.get(provider, ["No models available"]),
        index=0,
        help="Select a model for image transformation",
    )

    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        num_inference_steps = st.slider(
            "Number of Steps", min_value=1, max_value=100, value=30
        )
        guidance_scale = st.slider(
            "Guidance Scale", min_value=1.0, max_value=20.0, value=7.5
        )
        target_size = st.radio(
            "Target Size",
            options=["Original", "512x512", "768x768", "1024x1024"],
            index=0,
        )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    # Input area
    st.subheader("Input Image")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image to transform",
        type=["png", "jpg", "jpeg"],
        help="Upload an image that you want to transform",
    )

    # Sample images option
    use_sample = st.checkbox("Use sample image instead", value=False)

    # Display the input image
    if use_sample:
        sample_options = {
            "Sketch": "sample/pig_draft.png",
            "Photograph": (
                "sample/flower.jpg" if os.path.exists("sample/flower.jpg") else None
            ),
            "Landscape": (
                "sample/landscape.jpg"
                if os.path.exists("sample/landscape.jpg")
                else None
            ),
        }
        # Filter out None values
        sample_options = {k: v for k, v in sample_options.items() if v is not None}

        if sample_options:
            sample_choice = st.selectbox(
                "Choose a sample image", options=list(sample_options.keys())
            )
            sample_path = sample_options.get(sample_choice)
            if os.path.exists(sample_path):
                input_image = Image.open(sample_path)
                st.image(
                    input_image,
                    caption=f"Sample: {sample_choice}",
                    use_container_width=True,
                )
            else:
                st.error(f"Sample image not found: {sample_path}")
                input_image = None
        else:
            st.error("No sample images available")
            input_image = None
    elif uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error opening image: {e}")
            input_image = None
    else:
        input_image = None
        st.info("Please upload an image or use a sample image")

    # Prompt for transformation
    st.subheader("Transformation Prompt")
    prompt = st.text_area(
        "Enter prompt to guide the transformation",
        value="Convert this image into a Studio Ghibli style artwork",
        height=100,
        help="Describe how you want to transform the image",
    )

    # Negative prompt
    negative_prompt = st.text_input(
        "Negative prompt (optional)",
        placeholder="Elements to avoid in the transformed image",
    )

    # Submit button
    transform_button = st.button(
        "Transform Image",
        type="primary",
        disabled=not input_image
        or not prompt
        or not st.session_state.get("HF_TOKEN", ""),
    )

with col2:
    # Output area
    st.subheader("Transformed Image")

    # Display error if API token is not set
    if not st.session_state.get("HF_TOKEN", ""):
        st.error("Please set your API token in the sidebar first!")

    # Placeholder for the transformed image
    output_placeholder = st.empty()

# Process the transformation when button is clicked
if transform_button and input_image and prompt and st.session_state.get("HF_TOKEN", ""):
    with st.spinner("Transforming image... (this may take a while)"):
        try:
            # Resize image if needed
            if target_size != "Original":
                width, height = map(int, target_size.split("x"))
                input_image = input_image.resize((width, height))

            # Create inference client
            client = InferenceClient(
                provider=provider,
                token=st.session_state["HF_TOKEN"],
            )

            # Different parameters for different providers
            params = {
                "prompt": prompt,
                "model": selected_model,
            }

            # Add optional parameters
            if negative_prompt:
                params["negative_prompt"] = negative_prompt

            if num_inference_steps:
                params["num_inference_steps"] = num_inference_steps

            if guidance_scale:
                params["guidance_scale"] = guidance_scale

            # Use the image_to_image method
            if provider == "hf-inference":
                # Convert PIL Image to bytes for hf-inference provider
                img_byte_arr = BytesIO()
                # Convert RGBA to RGB if needed
                if input_image.mode == "RGBA":
                    input_image = input_image.convert("RGB")
                input_image.save(img_byte_arr, format="JPEG")
                img_bytes = img_byte_arr.getvalue()

                # Pass bytes instead of PIL Image
                transformed_image = client.image_to_image(image=img_bytes, **params)
            else:
                # Other providers may accept PIL Image directly
                transformed_image = client.image_to_image(image=input_image, **params)

            # Display the transformed image
            output_placeholder.image(
                transformed_image, caption="Transformed Image", use_container_width=True
            )

            # Save image temporarily and provide download button
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                transformed_image.save(tmp_file.name)
                st.download_button(
                    label="Download Transformed Image",
                    data=open(tmp_file.name, "rb").read(),
                    file_name="transformed_image.png",
                    mime="image/png",
                )
                # Clean up temp file
                os.unlink(tmp_file.name)

        except Exception as e:
            st.error(f"Error transforming image: {str(e)}")
            st.info(
                "Try using a different model, provider, or simplifying your prompt."
            )
