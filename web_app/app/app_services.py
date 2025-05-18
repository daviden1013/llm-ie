import os
import logging
from typing import Dict, Any
from llm_ie.engines import (
    InferenceEngine, OllamaInferenceEngine, OpenAIInferenceEngine, AzureOpenAIInferenceEngine,
    HuggingFaceHubInferenceEngine, LiteLLMInferenceEngine
)
from llm_ie.chunkers import (
    SentenceUnitChunker,
    WholeDocumentUnitChunker,
    NoContextChunker,
    WholeDocumentContextChunker,
    SlideWindowContextChunker
)
from .extractors import AppDirectFrameExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_llm_engine_from_config(config: Dict[str, Any]) -> InferenceEngine:
    """
    Creates and configures an LLM InferenceEngine instance based on the provided configuration dictionary.

    Args:
        config (Dict[str, Any]): A dictionary containing LLM engine configuration.
                                 Expected keys depend on 'api_type'.

    Returns:
        InferenceEngine: An instance of the configured LLM engine.

    Raises:
        ValueError: If configuration is invalid or api_type is unsupported.
        ImportError: If a required library for an api_type is not installed.
    """
    api_type = config.get('api_type')
    if not api_type:
        raise ValueError("LLM API type ('api_type') is missing in the configuration.")
    logger.info(f"Attempting to create engine for API type: {api_type}")

    try:
        if api_type == "openai_compatible":
            base_url = config.get('llm_base_url')
            model = config.get('llm_model_openai_comp')
            api_key = config.get('openai_compatible_api_key', "EMPTY") # Default to EMPTY if not provided
            if not base_url or not model:
                raise ValueError("Missing 'llm_base_url' or 'llm_model_openai_comp' for OpenAI Compatible.")
            return OpenAIInferenceEngine(model=model, api_key=api_key, base_url=base_url)
        elif api_type == "ollama":
            model = config.get('ollama_model')
            host = config.get('ollama_host') # Defaults to None if not present, Ollama client handles default
            num_ctx = int(config.get('ollama_num_ctx', 4096))
            if not model:
                raise ValueError("Missing 'ollama_model' for Ollama.")
            return OllamaInferenceEngine(model_name=model, host=host, num_ctx=num_ctx)
        elif api_type == "huggingface_hub":
            model_or_endpoint = config.get('hf_model_or_endpoint')
            token = config.get('hf_token') # Defaults to None
            if not model_or_endpoint:
                raise ValueError("Missing 'hf_model_or_endpoint' for HuggingFace Hub.")
            return HuggingFaceHubInferenceEngine(model=model_or_endpoint, token=token)
        elif api_type == "openai":
            model = config.get('openai_model')
            api_key = config.get('openai_api_key') # Defaults to None (OpenAI client will check env var)
            reasoning_model = config.get('openai_reasoning_model', False)
            if not model:
                raise ValueError("Missing 'openai_model' for OpenAI.")
            return OpenAIInferenceEngine(model=model, api_key=api_key, reasoning_model=reasoning_model)
        elif api_type == "azure_openai":
            deployment = config.get('azure_deployment_name')
            api_key = config.get('azure_openai_api_key') # Defaults to None
            endpoint = config.get('azure_endpoint')     # Defaults to None
            api_version = config.get('azure_api_version')
            reasoning_model = config.get('azure_reasoning_model', False)
            if not deployment or not api_version:
                raise ValueError("Missing 'azure_deployment_name' or 'azure_api_version' for Azure OpenAI.")
            return AzureOpenAIInferenceEngine(
                model=deployment, api_key=api_key, azure_endpoint=endpoint, api_version=api_version, reasoning_model=reasoning_model
            )
        elif api_type == "litellm":
            model_str = config.get('litellm_model')
            api_key = config.get('litellm_api_key') # Defaults to None
            base_url = config.get('litellm_base_url') # Defaults to None
            if not model_str:
                raise ValueError("Missing 'litellm_model' string for LiteLLM.")
            return LiteLLMInferenceEngine(model=model_str, api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unsupported LLM API type: {api_type}")
    except KeyError as e:
        logger.error(f"Missing configuration key for {api_type}: {e}")
        raise ValueError(f"Missing configuration key for {api_type}: {e}") from e
    except ImportError as e:
        logger.error(f"Missing library for {api_type}: {e}")
        raise ImportError(f"Required library for {api_type} not installed: {e}") from e
    except Exception as e:
        logger.error(f"Failed to initialize engine for {api_type}: {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize LLM engine for {api_type}.") from e


def get_app_frame_extractor(engine: InferenceEngine, extractor_config: Dict[str, Any]) -> AppDirectFrameExtractor:
    """
    Creates and configures an AppDirectFrameExtractor instance.
    This will use the AppDirectFrameExtractor from app.extractors.

    Args:
        engine (InferenceEngine): An instance of the LLM engine.
        extractor_config (Dict[str, Any]): Configuration for the frame extractor.
                                           Expected keys: 'prompt_template', 'extraction_unit_type', etc.

    Returns:
        AppDirectFrameExtractor: Configured instance.
    """
    from .extractors import AppDirectFrameExtractor # Import here to avoid circular dependency at module load time

    prompt_template = extractor_config.get('prompt_template')
    if not prompt_template:
        raise ValueError("Prompt template is required for frame extractor.")

    extraction_unit_type = extractor_config.get('extraction_unit_type', 'Sentence')
    context_chunker_type = extractor_config.get('context_chunker_type', 'NoContext')
    slide_window_size = int(extractor_config.get('slide_window_size', 2)) # Used if context_chunker_type is SlideWindow

    if extraction_unit_type == "WholeDocument":
        unit_chunker = WholeDocumentUnitChunker()
    elif extraction_unit_type == "Sentence":
        unit_chunker = SentenceUnitChunker()
    elif extraction_unit_type == "TextLine": # Assuming TextLine maps to Sentence for now
        logger.warning("Mapping 'TextLine' extraction unit to SentenceUnitChunker.")
        unit_chunker = SentenceUnitChunker()
    else:
        raise ValueError(f"Unsupported extraction_unit_type: {extraction_unit_type}")

    if context_chunker_type == "NoContext":
        context_chunker = NoContextChunker()
    elif context_chunker_type == "WholeDocument":
        context_chunker = WholeDocumentContextChunker()
    elif context_chunker_type == "SlideWindow":
        context_chunker = SlideWindowContextChunker(window_size=slide_window_size)
    else:
        raise ValueError(f"Unsupported context_chunker_type: {context_chunker_type}")

    logger.info(f"Instantiating AppDirectFrameExtractor with Unit Chunker: {type(unit_chunker).__name__} and Context Chunker: {type(context_chunker).__name__}")

    return AppDirectFrameExtractor(
        inference_engine=engine,
        unit_chunker=unit_chunker,
        context_chunker=context_chunker,
        prompt_template=prompt_template
    )
