# Free LLM API

Providers

- [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index)
  - [Playground - Hugging Face](https://huggingface.co/playground) (text model only)
  - [Quickstart](https://huggingface.co/docs/huggingface_hub/v0.30.1/quick-start#authentication)
  - [Run Inference on servers](https://huggingface.co/docs/huggingface_hub/v0.30.1/guides/inference)
- [OpenRouter](https://openrouter.ai/)
  - [OpenRouter FAQ | Developer Documentation — OpenRouter | Documentation](https://openrouter.ai/docs/faq#what-free-tier-options-exist) => low rate limits (200 requests per day total)
- [Together AI – The AI Acceleration Cloud - Fast Inference, Fine-Tuning & Training](https://www.together.ai/)

Models

- OpenRouter:
  - [Models | OpenRouter](https://openrouter.ai/models?max_price=0)
- Together AI
  - [Together AI | FLUX.1 [schnell] Free API](https://www.together.ai/models/flux-1-schnell)

## Getting Started

```bash
uv install

streamlit run Overview.py
```

## Todo

- Multi-Modalities
  - [X] list in table
  - [X] list models: [OpenRouter - List available models](https://openrouter.ai/docs/api-reference/list-available-models)
- Limitations
  - quota / usage
