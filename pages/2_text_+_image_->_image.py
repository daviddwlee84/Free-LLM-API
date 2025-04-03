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

# BUG: somehow buggy, figuring out is it possible to use text-guided image generation with Hugging Face InferenceClient
#     # Provider selection
#     st.subheader("Provider Settings")
#     provider = st.selectbox(
#         "Select Provider",
#         options=["hf-inference", "replicate"],
#         index=0,
#         help="Select the provider for image generation",
#     )

#     # Model selection based on provider
#     model_options = {
#         "hf-inference": [
#             "stabilityai/stable-diffusion-xl-base-1.0",
#             "runwayml/stable-diffusion-v1-5",
#         ],
#         "replicate": [
#             "black-forest-labs/FLUX.1-canny-dev",
#             "black-forest-labs/FLUX.1-Redux-dev",
#         ],
#     }

#     selected_model = st.selectbox(
#         "Select Model",
#         options=model_options.get(provider, ["default-model"]),
#         index=0,
#         help="Select the model for image generation",
#     )

#     # Advanced options
#     with st.expander("Advanced Options", expanded=False):
#         num_inference_steps = st.slider(
#             "Number of Steps", min_value=1, max_value=100, value=50
#         )
#         guidance_scale = st.slider(
#             "Guidance Scale", min_value=1.0, max_value=40.0, value=30.0
#         )
#         width = st.select_slider("Width", options=[512, 768, 1024], value=1024)
#         height = st.select_slider("Height", options=[512, 768, 1024], value=1024)

# # Main content area
# col1, col2 = st.columns([1, 1])

# with col1:
#     # Input area
#     st.subheader("Control Image")

#     # Image upload
#     uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

#     # Sample image option
#     use_sample = st.checkbox("Use sample image", value=not uploaded_file)

#     # Display uploaded or sample image
#     if uploaded_file:
#         input_image = Image.open(uploaded_file)
#         st.image(input_image, caption="Uploaded Image", use_container_width=True)
#     elif use_sample:
#         sample_path = os.path.join(
#             os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#             "sample",
#             "pig_draft.png",
#         )
#         if os.path.exists(sample_path):
#             input_image = Image.open(sample_path)
#             st.image(input_image, caption="Sample Image", use_container_width=True)
#         else:
#             st.error("Sample image not found at: " + sample_path)
#             input_image = None
#     else:
#         input_image = None
#         st.info("Please upload an image or use the sample image")

#     # Text prompt input
#     st.subheader("Text Prompt")

#     default_prompt = "Please refine this pig to be a more realistic one (make sure it's colorful, and still like a draw)"

#     prompt = st.text_area(
#         "Enter your prompt",
#         value=default_prompt,
#         height=150,
#         placeholder="Describe how you want to transform the image...",
#     )

#     # Negative prompt
#     negative_prompt = st.text_input(
#         "Negative prompt (optional)", placeholder="Elements to avoid in the image"
#     )

#     # Submit button
#     generate_button = st.button(
#         "Generate Image",
#         type="primary",
#         disabled=not prompt
#         or not input_image
#         or not st.session_state.get("HF_TOKEN", ""),
#     )

# with col2:
#     # Output area
#     st.subheader("Generated Image")

#     # Display error if API token is not set
#     if not st.session_state.get("HF_TOKEN", ""):
#         st.error("Please set your API token in the sidebar first!")

#     # Placeholder for the image
#     image_placeholder = st.empty()

# # Generate image when button is clicked
# if generate_button and prompt and input_image and st.session_state.get("HF_TOKEN", ""):
#     with st.spinner("Generating image..."):
#         try:
#             # Convert PIL image to bytes for API
#             img_byte_arr = io.BytesIO()
#             input_image.save(img_byte_arr, format="PNG")
#             img_byte_arr.seek(0)

#             # Create a client that can handle both providers
#             client = InferenceClient(token=st.session_state["HF_TOKEN"])

#             if provider == "hf-inference":
#                 # For Hugging Face models
#                 inputs = {
#                     "prompt": prompt,
#                     "image": input_image,
#                     "num_inference_steps": num_inference_steps,
#                     "guidance_scale": guidance_scale,
#                     "width": width,
#                     "height": height,
#                 }

#                 if negative_prompt:
#                     inputs["negative_prompt"] = negative_prompt

#                 generated_image = client.image_to_image(model=selected_model, **inputs)

#             elif provider == "replicate":
#                 # For Replicate models, use a simpler approach with the HF InferenceClient
#                 # The InferenceClient automatically handles the Replicate integration
#                 try:
#                     # First try using image-to-image endpoint
#                     inputs = {
#                         "prompt": prompt,
#                         "image": input_image,
#                         "num_inference_steps": num_inference_steps,
#                         "guidance_scale": guidance_scale,
#                         "width": width,
#                         "height": height,
#                     }

#                     if negative_prompt:
#                         inputs["negative_prompt"] = negative_prompt

#                     client = InferenceClient(
#                         model=selected_model,
#                         provider="replicate",
#                         token=st.session_state["HF_TOKEN"],
#                     )

#                     generated_image = client.image_to_image(**inputs)

#                 except Exception as e:
#                     st.warning(
#                         f"Image-to-image failed, trying alternate approach: {str(e)}"
#                     )

#                     # Fallback to text-to-image with control_image parameter
#                     inputs = {
#                         "prompt": prompt,
#                         "control_image": input_image,  # Some models use this param name
#                         "num_inference_steps": num_inference_steps,
#                         "guidance_scale": guidance_scale,
#                         "width": width,
#                         "height": height,
#                     }

#                     if negative_prompt:
#                         inputs["negative_prompt"] = negative_prompt

#                     client = InferenceClient(
#                         model=selected_model,
#                         provider="replicate",
#                         token=st.session_state["HF_TOKEN"],
#                     )

#                     generated_image = client.text_to_image(**inputs)

#             # Display the generated image
#             image_placeholder.image(
#                 generated_image, caption=prompt, use_container_width=True
#             )

#             # Save image temporarily and provide download button
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
#                 if isinstance(generated_image, list):
#                     # Some models return a list of images
#                     generated_image[0].save(tmp_file.name)
#                 else:
#                     generated_image.save(tmp_file.name)

#                 st.download_button(
#                     label="Download Image",
#                     data=open(tmp_file.name, "rb").read(),
#                     file_name="generated_image.png",
#                     mime="image/png",
#                 )
#                 # Clean up temp file
#                 os.unlink(tmp_file.name)

#         except Exception as e:
#             st.error(f"Error generating image: {str(e)}")
#             st.info(
#                 "Some models may require specific API access. Please check your credentials and model availability."
#             )
#             st.exception(e)
