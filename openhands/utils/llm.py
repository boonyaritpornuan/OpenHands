import warnings

import httpx

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm
    from litellm import LlmProviders, ProviderConfigManager, get_llm_provider

from openhands.core.config import LLMConfig, OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm import bedrock

# Hardcoded OpenHands provider models used in self-hosted mode.
# In SaaS mode these are loaded from the database instead.
OPENHANDS_MODELS = [
    'openhands/claude-opus-4-6',
    'openhands/claude-opus-4-5-20251101',
    'openhands/claude-sonnet-4-6',
    'openhands/claude-sonnet-4-5-20250929',
    'openhands/gpt-5.2-codex',
    'openhands/gpt-5.2',
    'openhands/minimax-m2.5',
    'openhands/gemini-3-pro-preview',
    'openhands/gemini-3.1-pro-preview',
    'openhands/gemini-3-flash-preview',
    'openhands/deepseek-chat',
    'openhands/devstral-medium-2512',
    'openhands/kimi-k2-0711-preview',
    'openhands/kimi-k2.5',
    'openhands/qwen3-coder-480b',
    'openhands/qwen3-coder-next',
    'openhands/glm-4.7',
    'openhands/glm-5',
]

CLARIFAI_MODELS = [
    'clarifai/openai.chat-completion.gpt-oss-120b',
    'clarifai/openai.chat-completion.gpt-oss-20b',
    'clarifai/openai.chat-completion.gpt-5',
    'clarifai/openai.chat-completion.gpt-5-mini',
    'clarifai/qwen.qwen3.qwen3-next-80B-A3B-Thinking',
    'clarifai/qwen.qwenLM.Qwen3-30B-A3B-Instruct-2507',
    'clarifai/qwen.qwenLM.Qwen3-30B-A3B-Thinking-2507',
    'clarifai/qwen.qwenLM.Qwen3-14B',
    'clarifai/qwen.qwenCoder.Qwen3-Coder-30B-A3B-Instruct',
    'clarifai/deepseek-ai.deepseek-chat.DeepSeek-R1-0528-Qwen3-8B',
    'clarifai/deepseek-ai.deepseek-chat.DeepSeek-V3_1',
    'clarifai/zai.completion.GLM_4_5',
    'clarifai/moonshotai.kimi.Kimi-K2-Instruct',
]


async def fetch_openrouter_models() -> list[str]:
    """Fetch available models from OpenRouter API.
    
    Returns:
        list[str]: List of OpenRouter model IDs prefixed with 'openrouter/'
                   Free models are marked with ' [FREE]' suffix
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get('https://openrouter.ai/api/v1/models')
            response.raise_for_status()
            data = response.json()
            models = data.get('data', [])
            
            result = []
            for model in models:
                if 'id' not in model:
                    continue
                    
                model_id = f"openrouter/{model['id']}"
                
                # Check if model is free (both pricing and prompt caching pricing)
                pricing = model.get('pricing', {})
                is_free = False
                
                if pricing:
                    # Check if both prompt and completion are free
                    prompt_price = float(pricing.get('prompt', '0'))
                    completion_price = float(pricing.get('completion', '0'))
                    
                    # Also check context pricing
                    context_price = float(pricing.get('context', '0'))
                    write_price = float(pricing.get('write', '0'))
                    
                    if prompt_price == 0 and completion_price == 0 and context_price == 0 and write_price == 0:
                        is_free = True
                
                if is_free:
                    model_id += ' [FREE]'
                
                result.append(model_id)
            
            # Sort: free models first, then alphabetically
            result.sort(key=lambda x: (not x.endswith('[FREE]'), x.lower()))
            return result
    except httpx.HTTPError as e:
        logger.error(f'Error fetching OpenRouter models: {e}')
        return []
    except (KeyError, ValueError) as e:
        logger.error(f'Error parsing OpenRouter models response: {e}')
        return []


def is_openhands_model(model: str | None) -> bool:
    """Check if the model uses the OpenHands provider.

    Args:
        model: The model name to check.

    Returns:
        True if the model starts with 'openhands/', False otherwise.
    """
    return bool(model and model.startswith('openhands/'))


def get_provider_api_base(model: str) -> str | None:
    """Get the API base URL for a model using litellm.

    This function tries multiple approaches to determine the API base URL:
    1. First tries litellm.get_api_base() which handles OpenAI, Gemini, Mistral
    2. Falls back to ProviderConfigManager.get_provider_model_info() for providers
       like Anthropic that have ModelInfo classes with get_api_base() methods

    Args:
        model: The model name (e.g., 'gpt-4', 'anthropic/claude-sonnet-4-5-20250929')

    Returns:
        The API base URL if found, None otherwise.
    """
    # First try get_api_base (handles OpenAI, Gemini with specific URL patterns)
    try:
        api_base = litellm.get_api_base(model, {})
        if api_base:
            return api_base
    except Exception:
        pass

    # Fall back to ProviderConfigManager for providers like Anthropic
    try:
        # Get the provider from the model
        _, provider_name, _, _ = get_llm_provider(model)
        if provider_name:
            # Convert provider name to LlmProviders enum
            try:
                provider_enum = LlmProviders(provider_name)
                model_info = ProviderConfigManager.get_provider_model_info(
                    model, provider_enum
                )
                if model_info and hasattr(model_info, 'get_api_base'):
                    return model_info.get_api_base()
            except ValueError:
                pass  # Provider not in enum
    except Exception:
        pass

    return None


def get_supported_llm_models(
    config: OpenHandsConfig,
    verified_models: list[str] | None = None,
) -> list[str]:
    """Get all models supported by LiteLLM.

    This function combines models from litellm and Bedrock, removing any
    error-prone Bedrock models.

    Args:
        config: The OpenHands configuration.
        verified_models: Optional list of verified model strings from the database
            (SaaS mode). When provided, these replace the hardcoded OPENHANDS_MODELS.

    Returns:
        list[str]: A sorted list of unique model names.
    """
    litellm_model_list = litellm.model_list + list(litellm.model_cost.keys())
    litellm_model_list_without_bedrock = bedrock.remove_error_modelId(
        litellm_model_list
    )
    # TODO: for bedrock, this is using the default config
    llm_config: LLMConfig = config.get_llm_config()
    bedrock_model_list = []
    if (
        llm_config.aws_region_name
        and llm_config.aws_access_key_id
        and llm_config.aws_secret_access_key
    ):
        bedrock_model_list = bedrock.list_foundation_models(
            llm_config.aws_region_name,
            llm_config.aws_access_key_id.get_secret_value(),
            llm_config.aws_secret_access_key.get_secret_value(),
        )
    model_list = litellm_model_list_without_bedrock + bedrock_model_list
    for llm_config in config.llms.values():
        ollama_base_url = llm_config.ollama_base_url
        if llm_config.model.startswith('ollama'):
            if not ollama_base_url:
                ollama_base_url = llm_config.base_url
        if ollama_base_url:
            ollama_url = ollama_base_url.strip('/') + '/api/tags'
            try:
                ollama_models_list = httpx.get(ollama_url, timeout=3).json()['models']  # noqa: ASYNC100
                for model in ollama_models_list:
                    model_list.append('ollama/' + model['name'])
                break
            except httpx.HTTPError as e:
                logger.error(f'Error getting OLLAMA models: {e}')

    # Use database-backed models if provided (SaaS), otherwise use hardcoded list
    openhands_models = verified_models if verified_models else OPENHANDS_MODELS
    model_list = openhands_models + CLARIFAI_MODELS + model_list

    return sorted(set(model_list))
