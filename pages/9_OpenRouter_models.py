import streamlit as st
from open_router import OpenRouter


# Helper function to format numbers as K/M notation
def format_number(num):
    if num >= 1000000:
        return f"{num/1000000:.0f}M"
    elif num >= 1000:
        return f"{num/1000:.0f}K"
    else:
        return str(num)


# Page configuration
st.set_page_config(page_title="OpenRouter Models", layout="wide")
st.title("OpenRouter Models")

st.link_button("OpenRouter Models", "https://openrouter.ai/models")
st.link_button(
    "OpenRouter API", "https://openrouter.ai/docs/api-reference/list-available-models"
)

# Initialize OpenRouter API
api = OpenRouter()

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Text search
    search_text = st.text_input("Search models", placeholder="Enter model name...")

    # Input Modalities
    with st.expander("Input Modalities", expanded=True):
        input_text = st.checkbox("text", value=True)
        input_image = st.checkbox("image")

        input_modalities = []
        if input_text:
            input_modalities.append("text")
        if input_image:
            input_modalities.append("image")

    # Output Modalities
    with st.expander("Output Modalities", expanded=True):
        output_text = st.checkbox("text", value=True, key="output_text")
        output_image = st.checkbox("image", key="output_image")

        output_modalities = []
        if output_text:
            output_modalities.append("text")
        if output_image:
            output_modalities.append("image")

    # Context length
    with st.expander("Context length", expanded=True):
        min_context, max_context = st.slider(
            "Context Length Range",
            min_value=0,
            max_value=1000000,
            value=(0, 0),
            step=1000,
            format="%d",
            label_visibility="collapsed",
        )

        # Display formatted values
        if max_context > 0:
            # Display formatted values
            st.caption(
                f"Selected: {format_number(min_context)} - {format_number(max_context)}"
            )
        else:
            st.caption("Selected: All")

    # Prompt pricing
    with st.expander("Prompt pricing", expanded=True):
        min_price, max_price = st.slider(
            "Price Range",
            min_value=0.0,
            max_value=0.01,
            value=(0.0, 0.01),
            step=0.0001,
            format="%.5f",
            label_visibility="collapsed",
        )

        # Display formatted price values
        price_min_text = "FREE" if min_price == 0 else f"${min_price:.7f}"
        price_max_text = f"${max_price:.7f}"
        st.caption(f"Selected: {price_min_text} - {price_max_text}")

    # # Series
    # with st.expander("Series", expanded=True):
    #     series_gpt = st.checkbox("GPT")
    #     series_claude = st.checkbox("Claude")
    #     series_gemini = st.checkbox("Gemini")
    #     series_more = st.checkbox("More...")
    #
    # # Category
    # with st.expander("Category", expanded=True):
    #     category_roleplay = st.checkbox("Roleplay")
    #     category_programming = st.checkbox("Programming")
    #
    # # Supported Parameters
    # # Providers

# Apply filters to get models
filters = {}

if input_modalities:
    filters["input_modalities"] = input_modalities
if output_modalities:
    filters["output_modalities"] = output_modalities
if min_price >= 0:
    filters["min_price_prompt"] = min_price
if max_price < 0.01:
    filters["max_price_prompt"] = max_price

# Add fuzzy name search if provided
if search_text:
    filters["fuzzy_name"] = search_text

# Get filtered models
filtered_models = api.list_models(**filters)

# Convert to DataFrame
df = api.list_models_df(filtered_models)

# Apply additional post-query filters
# Series filtering (post-query)
# series_filters = []
# if series_gpt:
#     series_filters.append(df["name"].str.contains("GPT", case=False))
# if series_claude:
#     series_filters.append(df["name"].str.contains("Claude", case=False))
# if series_gemini:
#     series_filters.append(df["name"].str.contains("Gemini", case=False))
#
# if series_filters:
#     combined_filter = series_filters[0]
#     for filter_condition in series_filters[1:]:
#         combined_filter = combined_filter | filter_condition
#     df = df[combined_filter]

# Context length filtering (post-query)
if max_context > 0:
    df = df[
        ((df["context_length"] >= min_context) & (df["context_length"] <= max_context))
    ]

# Display results
if not df.empty:
    # Clean up price columns
    df["price_prompt"] = df["price_prompt"].str.replace("$", "").astype(float)
    df["price_completion"] = df["price_completion"].str.replace("$", "").astype(float)

    # Display model count
    st.subheader(f"Found {len(df)} models")

    # Format the display
    display_df = df.copy()

    # Format prices for display
    display_df["Price (Prompt)"] = display_df["price_prompt"].apply(
        lambda x: "FREE" if x == 0 else f"${x:.7f}"
    )
    display_df["Price (Completion)"] = display_df["price_completion"].apply(
        lambda x: "FREE" if x == 0 else f"${x:.7f}"
    )

    # Create a clean display columns
    # TODO: input token price, output token price
    display_columns = [
        "name",
        "Price (Prompt)",
        "Price (Completion)",
        "context_length",
        "input_modalities",
        "output_modalities",
    ]

    # Rename columns for display
    renamed_df = display_df[display_columns].rename(
        columns={
            "name": "Model Name",
            "context_length": "Context Length",
            "input_modalities": "Input Modalities",
            "output_modalities": "Output Modalities",
        }
    )

    # Display the dataframe
    st.dataframe(
        renamed_df,
        use_container_width=True,
        hide_index=True,
    )

    # Add an expandable section for model details
    with st.expander("Model Details", expanded=False):
        # Select a model to see details
        model_names = df["name"].tolist()
        selected_model = st.selectbox("Select a model for details", model_names)

        if selected_model:
            model_data = df[df["name"] == selected_model].iloc[0]

            # Display detailed information
            st.subheader(model_data["name"])
            st.write("**ID:** ", model_data["id"])
            st.write("**Description:** ", model_data["description"])
            st.write(
                "**Context Length:** ",
                format_number(model_data["context_length"]),
                " tokens",
            )
            st.write("**Tokenizer:** ", model_data["tokenizer"])
            st.write(
                "**Input Modalities:** ", ", ".join(model_data["input_modalities"])
            )
            st.write(
                "**Output Modalities:** ", ", ".join(model_data["output_modalities"])
            )
            st.write(
                "**Price (Prompt):** ",
                (
                    "FREE"
                    if model_data["price_prompt"] == 0
                    else f"${model_data['price_prompt']:.7f}"
                ),
            )
            st.write(
                "**Price (Completion):** ",
                (
                    "FREE"
                    if model_data["price_completion"] == 0
                    else f"${model_data['price_completion']:.7f}"
                ),
            )

else:
    st.warning(
        "No models found with the selected filters. Try adjusting your criteria."
    )
