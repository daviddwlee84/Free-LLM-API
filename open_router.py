from typing import List, Dict, Optional
from pydantic import BaseModel
import requests


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
    def list_models(self) -> List[ModelData]:
        # TODO: add filter
        response = requests.get("https://openrouter.ai/api/v1/models")
        return OpenRouterModelsResponse(**response.json()).data


if __name__ == "__main__":
    print(models := OpenRouter().list_models())
    import ipdb

    ipdb.set_trace()
