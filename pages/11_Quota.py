import streamlit as st
import requests
from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.utils import get_token
import json

# Page configuration
st.set_page_config(page_title="HuggingFace Quota", layout="wide")
st.title("HuggingFace API Quota Information")

# Information section
st.info(
    """
## Monthly Included Credits
Hugging Face provides monthly credits for inference requests based on your account type:
- **Free Users**: Less than $0.10 (subject to change)
- **PRO Users**: $2.00
- **Enterprise Hub Organizations**: $2.00 per seat, shared among members

Once your credits are exhausted, you'll see a '402 Payment Required' error for inference requests.
"""
)

# Links to official quota/billing information
col1, col2 = st.columns(2)
with col1:
    st.link_button("View Billing Settings", "https://huggingface.co/settings/billing")
with col2:
    st.link_button(
        "Pricing Documentation",
        "https://huggingface.co/docs/inference-providers/pricing",
    )

# Check authentication status
token = get_token()
st.subheader("Authentication Status")

if token:
    st.success("✅ HuggingFace token found")

    # Try to get user info to determine account type
    try:
        api = HfApi(token=token)
        user_info = requests.get(
            "https://huggingface.co/api/whoami",
            headers={"Authorization": f"Bearer {token}"},
        ).json()

        # Display basic user info
        st.subheader("Account Information")
        st.write(f"**Username**: {user_info.get('name', 'Unknown')}")

        account_type = "Free"
        if user_info.get("isPro"):
            account_type = "PRO"

        st.write(f"**Account Type**: {account_type}")

        # Show estimated quota based on account type
        st.subheader("Estimated Monthly Credits")
        if account_type == "PRO":
            st.write("Monthly Credits: $2.00")
        else:
            st.write("Monthly Credits: Less than $0.10")

        st.info(
            "For precise usage information, please check your [billing settings](https://huggingface.co/settings/billing)."
        )
    except Exception as e:
        st.warning(f"Could not retrieve account information: {str(e)}")
else:
    st.error("⚠️ No HuggingFace token found")
    st.info(
        """
    To use the HuggingFace API, you need to set up your token:
    1. Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    2. Set the token as an environment variable: `export HUGGINGFACE_TOKEN=your_token`
    3. Or use `huggingface-cli login` in your terminal
    """
    )

# Test quota section
st.subheader("Test Current Quota Status")
st.write("Run a simple inference test to check if you have available quota:")

test_model = st.selectbox(
    "Select model to test:",
    [
        "sentence-transformers/all-MiniLM-L6-v2",  # Small text embedding model (low cost)
        "google/flan-t5-small",  # Small text generation model
        "stabilityai/stable-diffusion-2-1-base",  # Image generation (higher cost)
    ],
)

if st.button("Test Quota"):
    with st.spinner("Testing API quota..."):
        try:
            client = InferenceClient()

            if "sentence-transformers" in test_model:
                # Testing embedding model
                result = client.feature_extraction("Hello world", model=test_model)
                st.success("✅ API call successful! You have available quota.")

            elif "flan-t5" in test_model:
                # Testing text generation
                result = client.text_generation(
                    "Translate to French: Hello, how are you?", model=test_model
                )
                st.success(
                    f"✅ API call successful! You have available quota.\n\nResponse: {result}"
                )

            elif "stable-diffusion" in test_model:
                # Testing image generation (most expensive)
                result = client.text_to_image("a photo of a cat", model=test_model)
                st.success("✅ API call successful! You have available quota.")
                st.image(result)

        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg and "Payment Required" in error_msg:
                st.error("❌ Quota exceeded! You have exhausted your monthly credits.")
                st.info(
                    "Consider upgrading to PRO for more credits or wait until your quota resets next month."
                )
            else:
                st.error(f"❌ Error during API call: {error_msg}")

# Usage tips section
with st.expander("Tips for Managing Quota"):
    st.markdown(
        """
    ### Conserve Your Credits
    - Use smaller models when possible
    - Cache results for repeated queries
    - Set appropriate timeout values
    - Test with small inputs before scaling
    
    ### Billing Organization
    If you're part of an organization with Enterprise Hub:
    ```python
    from huggingface_hub import InferenceClient
    
    # Bill to organization instead of personal account
    client = InferenceClient(provider="provider-name", bill_to="your-org-name")
    ```
    """
    )
