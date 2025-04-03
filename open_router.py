from typing import List, Dict, Optional
from pydantic import BaseModel
import requests
from fuzzywuzzy import process
import pandas as pd


class Architecture(BaseModel):
    input_modalities: List[str]
    output_modalities: List[str]
    tokenizer: str


class TopProvider(BaseModel):
    is_moderated: bool


class Pricing(BaseModel):
    prompt: str
    completion: str
    image: str
    request: str
    input_cache_read: str
    input_cache_write: str
    web_search: str
    internal_reasoning: str


class ModelData(BaseModel):
    id: str
    name: str
    created: int
    description: str
    architecture: Architecture
    top_provider: TopProvider
    pricing: Pricing
    context_length: int
    per_request_limits: Optional[Dict[str, str]] = None


class OpenRouterModelsResponse(BaseModel):
    data: List[ModelData]


class OpenRouter:
    def list_models(
        self,
        min_price_prompt: Optional[float] = None,
        max_price_prompt: Optional[float] = None,
        min_price_completion: Optional[float] = None,
        max_price_completion: Optional[float] = None,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
        fuzzy_name: Optional[str] = None,
        fuzzy_id: Optional[str] = None,
        fuzzy_threshold: int = 70,
    ) -> List[ModelData]:
        """
        Retrieve and filter models from OpenRouter API

        https://openrouter.ai/docs/api-reference/list-available-models

        Args:
            min_price_prompt: Minimum price per prompt token
            max_price_prompt: Maximum price per prompt token
            min_price_completion: Minimum price per completion token
            max_price_completion: Maximum price per completion token
            input_modalities: List of required input modalities
            output_modalities: List of required output modalities
            fuzzy_name: Fuzzy search string for model name
            fuzzy_id: Fuzzy search string for model ID
            fuzzy_threshold: Threshold for fuzzy matches (0-100)

        Returns:
            List of filtered ModelData objects
        """
        response = requests.get("https://openrouter.ai/api/v1/models")
        models = OpenRouterModelsResponse(**response.json()).data

        # Apply filters
        filtered_models = models.copy()

        # Filter by price range for prompt
        if min_price_prompt is not None:
            filtered_models = [
                model
                for model in filtered_models
                if float(model.pricing.prompt.strip("$")) >= min_price_prompt
            ]

        if max_price_prompt is not None:
            filtered_models = [
                model
                for model in filtered_models
                if float(model.pricing.prompt.strip("$")) <= max_price_prompt
            ]

        # Filter by price range for completion
        if min_price_completion is not None:
            filtered_models = [
                model
                for model in filtered_models
                if float(model.pricing.completion.strip("$")) >= min_price_completion
            ]

        if max_price_completion is not None:
            filtered_models = [
                model
                for model in filtered_models
                if float(model.pricing.completion.strip("$")) <= max_price_completion
            ]

        # Filter by input modalities
        if input_modalities:
            filtered_models = [
                model
                for model in filtered_models
                if all(
                    modality in model.architecture.input_modalities
                    for modality in input_modalities
                )
            ]

        # Filter by output modalities
        if output_modalities:
            filtered_models = [
                model
                for model in filtered_models
                if all(
                    modality in model.architecture.output_modalities
                    for modality in output_modalities
                )
            ]

        # Apply fuzzy search on model name
        if fuzzy_name:
            model_names = {model.name: model for model in filtered_models}
            matches = process.extractBests(
                fuzzy_name, model_names.keys(), score_cutoff=fuzzy_threshold
            )
            filtered_models = [model_names[match[0]] for match in matches]

        # Apply fuzzy search on model ID
        if fuzzy_id:
            model_ids = {model.id: model for model in filtered_models}
            matches = process.extractBests(
                fuzzy_id, model_ids.keys(), score_cutoff=fuzzy_threshold
            )
            filtered_models = [model_ids[match[0]] for match in matches]

        return filtered_models

    def list_models_df(self, models: Optional[List[ModelData]] = None) -> pd.DataFrame:
        """
        Convert list of models to pandas DataFrame

        Args:
            models: List of ModelData objects (if None, fetches all models)

        Returns:
            pandas DataFrame with model information
        """
        import pandas as pd

        if models is None:
            models = self.list_models()

        data = []
        for model in models:
            data.append(
                {
                    "id": model.id,
                    "name": model.name,
                    "created": model.created,
                    "description": model.description,
                    "context_length": model.context_length,
                    "input_modalities": model.architecture.input_modalities,
                    "output_modalities": model.architecture.output_modalities,
                    "tokenizer": model.architecture.tokenizer,
                    "price_prompt": model.pricing.prompt,
                    "price_completion": model.pricing.completion,
                    "price_image": model.pricing.image,
                    "is_moderated": model.top_provider.is_moderated,
                }
            )

        return pd.DataFrame(data)


if __name__ == "__main__":
    print(models := OpenRouter().list_models())

    # Example with filtering
    openrouter = OpenRouter()

    # Get models with text input/output
    text_models = openrouter.list_models(
        input_modalities=["text"], output_modalities=["text"]
    )

    # Get models with price range
    affordable_models = openrouter.list_models(
        max_price_prompt=0.001, max_price_completion=0.002
    )

    # Fuzzy search for Claude models
    claude_models = openrouter.list_models(fuzzy_name="claude")

    # Convert to DataFrame and display
    df = openrouter.list_models_df(claude_models)
    print(df[["name", "price_prompt", "price_completion", "context_length"]])

    import ipdb

    ipdb.set_trace()
