import streamlit as st
from huggingface_hub import HfApi
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper function to format numbers as K/M notation
def format_number(num):
    if num is None:
        return "N/A"
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)


# Common NLP and ML tasks from https://huggingface.co/tasks
# TODO: I remember there is the list in huggingface_hub library, we should use that instead of hardcoding it
TASKS = [
    # NLP Tasks
    "text-generation",
    "fill-mask",
    "summarization",
    "question-answering",
    "translation",
    "text-classification",
    "token-classification",
    "sentence-similarity",
    "feature-extraction",
    "text-to-image",
    "image-to-text",
    "zero-shot-classification",
    # Computer Vision Tasks
    "image-classification",
    "object-detection",
    "image-segmentation",
    "visual-question-answering",
    # Audio Tasks
    "automatic-speech-recognition",
    "text-to-speech",
    "audio-classification",
]

# Known inference providers
PROVIDERS = [
    "huggingface",
    "inference-endpoint",
    "together-ai",
    "cohere-ai",
    "fireworks-ai",
    "cerebras",
    "nebius-ai",
    "hyperbolic",
    "fal-ai",
    "black-forest-labs",
    "novita-ai",
    "replicate",
    "sambanova",
]

# Page configuration
st.set_page_config(page_title="HuggingFace Models", layout="wide")
st.title("HuggingFace Inference-Ready Models")

st.link_button("HuggingFace Models", "https://huggingface.co/models")
st.link_button("HuggingFace Tasks", "https://huggingface.co/tasks")

# Initialize HF API
api = HfApi()

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Text search
    search_text = st.text_input("Search models", placeholder="Enter model name...")

    # Task selection
    with st.expander("Task", expanded=True):
        selected_task = st.selectbox("Select Task", options=["Any"] + TASKS)

    # Inference state
    with st.expander("Inference State", expanded=True):
        inference_state = st.radio(
            "Inference API State",
            options=["warm", "cold", "Any"],
            index=0,
            help="Warm models are available for immediate use. Cold models will be loaded on first inference call.",
        )

    # Provider selection - temporarily disabled until we fix the API issue
    show_providers = st.sidebar.checkbox(
        "Show provider information (experimental)", value=False
    )

    selected_provider = "Any"  # Default value
    if show_providers:
        with st.expander("Inference Provider", expanded=True):
            selected_provider = st.selectbox(
                "Provider",
                options=["Any"] + PROVIDERS,
                help="Filter models by available inference provider",
            )

    # Result limiting
    with st.expander("Results", expanded=True):
        limit = st.slider("Max results", 5, 100, 20)

    # Sort options
    with st.expander("Sorting", expanded=True):
        sort_by = st.selectbox(
            "Sort by",
            options=["downloads", "likes", "trending_score", "last_modified"],
            index=0,
        )

        sort_direction = st.radio(
            "Direction", options=["Descending", "Ascending"], index=0, horizontal=True
        )

# Prepare filter arguments
filter_args = {
    "sort": sort_by,
    "direction": -1 if sort_direction == "Descending" else 1,
    "limit": limit,
}

# Conditionally add provider mapping expansion
if show_providers:
    filter_args["full"] = True  # Fetch more details when showing providers

# Add task filter if specified
if selected_task != "Any":
    filter_args["task"] = selected_task

# Add inference state filter if specified
if inference_state != "Any":
    filter_args["inference"] = inference_state

# Add model name search if provided
if search_text:
    filter_args["model_name"] = search_text

# Display loading spinner
with st.spinner(f"Fetching models from HuggingFace Hub..."):
    try:
        # Get models with filters
        models = list(api.list_models(**filter_args))

        if models:
            # Convert to DataFrame
            data = []
            for model in models:
                # Safely extract provider information if available
                providers = []
                provider_mapping = {}

                # Only process provider info if we're showing it
                if show_providers:
                    try:
                        # Try to access inference provider mapping if it exists and is a dictionary
                        if hasattr(model, "inference_provider_mapping"):
                            mapping = model.inference_provider_mapping
                            if isinstance(mapping, dict):
                                provider_mapping = mapping
                                providers = list(mapping.keys())
                            elif isinstance(mapping, list):
                                # Handle the case where it's a list
                                logger.info(
                                    f"Model {model.id} has a list for inference_provider_mapping: {mapping}"
                                )
                                providers = [
                                    p.get("id", "unknown")
                                    for p in mapping
                                    if isinstance(p, dict) and "id" in p
                                ]
                                provider_mapping = {
                                    p.get("id", f"provider_{i}"): p
                                    for i, p in enumerate(mapping)
                                    if isinstance(p, dict)
                                }
                    except Exception as e:
                        logger.error(
                            f"Error processing provider info for {model.id}: {str(e)}"
                        )

                # Skip if filtering by provider and this model doesn't have it
                if selected_provider != "Any" and selected_provider not in providers:
                    continue

                data.append(
                    {
                        "Model ID": model.id,
                        "Pipeline Tag": getattr(model, "pipeline_tag", "N/A"),
                        "Downloads": getattr(model, "downloads", 0),
                        "Likes": getattr(model, "likes", 0),
                        "Last Modified": getattr(model, "last_modified", "N/A"),
                        "Tags": getattr(model, "tags", []),
                        "Task": (
                            selected_task
                            if selected_task != "Any"
                            else getattr(model, "pipeline_tag", "N/A")
                        ),
                        "Inference Status": (
                            inference_state if inference_state != "Any" else "Available"
                        ),
                        "Providers": providers,
                        "Provider Mapping": provider_mapping,
                    }
                )

            df = pd.DataFrame(data)

            # Format values for display
            display_df = df.copy()
            display_df["Downloads"] = display_df["Downloads"].apply(format_number)
            display_df["Likes"] = display_df["Likes"].apply(format_number)

            # Only include Providers column if showing provider info
            column_order = ["Model ID", "Task", "Inference Status"]
            if show_providers:
                display_df["Providers"] = display_df["Providers"].apply(
                    lambda x: ", ".join(x) if x else "None"
                )
                column_order.append("Providers")

            column_order.extend(["Downloads", "Likes", "Last Modified"])

            # Display model count
            st.subheader(f"Found {len(df)} models")

            # Display as dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_order=column_order,
            )

            # Add an expandable section for model details
            with st.expander("Model Details", expanded=False):
                # Select a model to see details
                model_names = df["Model ID"].tolist()
                selected_model = st.selectbox("Select a model for details", model_names)

                if selected_model:
                    model_data = df[df["Model ID"] == selected_model].iloc[0]

                    # Display detailed information
                    st.subheader(model_data["Model ID"])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Task:** ", model_data["Task"])
                        st.write(
                            "**Inference Status:** ", model_data["Inference Status"]
                        )
                        st.write("**Downloads:** ", model_data["Downloads"])
                        st.write("**Likes:** ", model_data["Likes"])

                    with col2:
                        st.write("**Last Modified:** ", model_data["Last Modified"])
                        st.write(
                            "**Tags:** ",
                            (
                                ", ".join(model_data["Tags"])
                                if isinstance(model_data["Tags"], list)
                                else model_data["Tags"]
                            ),
                        )
                        if show_providers:
                            st.write(
                                "**Available Providers:** ",
                                (
                                    ", ".join(model_data["Providers"])
                                    if model_data["Providers"]
                                    else "None"
                                ),
                            )

                    # Provider details if available and requested
                    if show_providers and model_data["Provider Mapping"]:
                        st.subheader("Provider Details")
                        provider_data = []
                        for provider, info in model_data["Provider Mapping"].items():
                            if isinstance(info, dict):
                                provider_data.append(
                                    {
                                        "Provider": provider,
                                        "Status": info.get("status", "Unknown"),
                                        "URL": info.get("url", ""),
                                        "Documentation": info.get("documentation", ""),
                                    }
                                )

                        if provider_data:
                            st.dataframe(
                                pd.DataFrame(provider_data),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No detailed provider information available")

                    # Model URL
                    model_url = f"https://huggingface.co/{selected_model}"
                    st.link_button("View on HuggingFace", model_url)

                    # Inference API usage example
                    st.subheader("Inference API Usage")

                    # Get available providers for example code
                    available_providers = ["huggingface"]
                    if show_providers and model_data["Provider Mapping"]:
                        available_providers = (
                            model_data["Providers"]
                            if model_data["Providers"]
                            else ["huggingface"]
                        )

                    selected_provider_for_code = st.selectbox(
                        "Provider for code example:",
                        options=available_providers,
                        index=0,
                    )

                    code = f"""
from huggingface_hub import InferenceClient

# Initialize the client
client = InferenceClient(model="{selected_model}")

# Run inference on the model
response = client.post(
    json={{"inputs": "YOUR_INPUT_HERE"}},
    model="{selected_model}"{', provider="' + selected_provider_for_code + '"' if selected_provider_for_code != "huggingface" else ''}
)

# Process response
result = response.json()
"""
                    st.code(code, language="python")

        else:
            st.warning(
                "No models found with the selected filters. Try adjusting your criteria."
            )
    except Exception as e:
        st.error(f"An error occurred while fetching models: {str(e)}")
        st.info(
            "Try adjusting your filters or disabling provider information if enabled."
        )
        logger.exception("Error in HuggingFace models page")
