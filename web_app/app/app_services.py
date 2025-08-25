import os
import logging
from typing import Dict, Any
from llm_ie.engines import (
    LLMConfig, BasicLLMConfig, ReasoningLLMConfig, OpenAIReasoningLLMConfig, Qwen3LLMConfig,
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
    The specific LLMConfig class to use is determined by 'llm_config_type' in the config.

    Args:
        config (Dict[str, Any]): A dictionary containing LLM engine configuration.
                                 Expected keys: 'api_type', 'llm_config_type',
                                 and other parameters for engine and LLMConfig.

    Returns:
        InferenceEngine: An instance of the configured LLM engine.

    Raises:
        ValueError: If configuration is invalid or api_type/llm_config_type is unsupported.
        ImportError: If a required library for an api_type is not installed.
    """
    api_type = config.get('api_type')
    llm_config_type_str = config.get('llm_config_type', 'BasicLLMConfig') # Default to BasicLLMConfig

    if not api_type:
        raise ValueError("LLM API type ('api_type') is missing in the configuration.")
    logger.info(f"Attempting to create engine for API type: {api_type} with LLMConfig type: {llm_config_type_str}")

    # --- Instantiate the selected LLMConfig ---
    llm_config_instance: LLMConfig
    
    # General parameters for all LLMConfigs that come directly from the top-level config
    # 'max_tokens' from frontend maps to 'max_new_tokens' in BasicLLMConfig
    # Temperature is also a common parameter.
    base_llm_params = {
        'temperature': config.get('temperature', 0.2), 
        'max_new_tokens': config.get('max_tokens', 4096)
    }
    # You might want to add other general params here from 'config.get(...)'
    # and then pass them to the **kwargs of the LLMConfig constructors.

    if llm_config_type_str == "BasicLLMConfig":
        llm_config_instance = BasicLLMConfig(
            max_new_tokens=base_llm_params['max_new_tokens'],
            temperature=base_llm_params['temperature']
        )
    elif llm_config_type_str == "ReasoningLLMConfig":
        llm_config_instance = ReasoningLLMConfig(
            max_new_tokens=base_llm_params['max_new_tokens'],
            temperature=base_llm_params['temperature']
        )

    elif llm_config_type_str == "OpenAIReasoningLLMConfig":
        reasoning_effort = config.get('openai_reasoning_effort', 'low')
        llm_config_instance = OpenAIReasoningLLMConfig(
            reasoning_effort=reasoning_effort,
            max_new_tokens=base_llm_params['max_new_tokens']
        )
    elif llm_config_type_str == "Qwen3LLMConfig":
        thinking_mode = config.get('qwen_thinking_mode', True)
        llm_config_instance = Qwen3LLMConfig(
            thinking_mode=thinking_mode,
            max_new_tokens=base_llm_params['max_new_tokens'], 
            temperature=base_llm_params['temperature'] 
        )
    else:
        raise ValueError(f"Unsupported LLMConfig type: {llm_config_type_str}")

    # --- Instantiate the InferenceEngine with the chosen LLMConfig ---
    try:
        if api_type == "openai_compatible":
            base_url = config.get('llm_base_url')
            model = config.get('llm_model_openai_comp')
            api_key = config.get('openai_compatible_api_key', "EMPTY")
            if not base_url or not model:
                raise ValueError("Missing 'llm_base_url' or 'llm_model_openai_comp' for OpenAI Compatible.")
            return OpenAIInferenceEngine(model=model, api_key=api_key, base_url=base_url, config=llm_config_instance)
        
        elif api_type == "ollama":
            model = config.get('ollama_model')
            host = config.get('ollama_host') 
            num_ctx = int(config.get('ollama_num_ctx', 4096)) # This is an Ollama engine param, not LLMConfig
            if not model:
                raise ValueError("Missing 'ollama_model' for Ollama.")
            return OllamaInferenceEngine(model_name=model, host=host, num_ctx=num_ctx, config=llm_config_instance)
            
        elif api_type == "huggingface_hub":
            model_or_endpoint = config.get('hf_model_or_endpoint')
            token = config.get('hf_token') 
            if not model_or_endpoint:
                raise ValueError("Missing 'hf_model_or_endpoint' for HuggingFace Hub.")
            return HuggingFaceHubInferenceEngine(model=model_or_endpoint, token=token, config=llm_config_instance)
            
        elif api_type == "openai":
            model = config.get('openai_model')
            api_key = config.get('openai_api_key') 
            if not model:
                raise ValueError("Missing 'openai_model' for OpenAI.")
            return OpenAIInferenceEngine(model=model, api_key=api_key, config=llm_config_instance)
            
        elif api_type == "azure_openai":
            deployment = config.get('azure_deployment_name')
            api_key = config.get('azure_openai_api_key') 
            endpoint = config.get('azure_endpoint')     
            api_version = config.get('azure_api_version')
            if not deployment or not api_version:
                raise ValueError("Missing 'azure_deployment_name' or 'azure_api_version' for Azure OpenAI.")
            return AzureOpenAIInferenceEngine(
                model=deployment, api_key=api_key, azure_endpoint=endpoint, api_version=api_version, config=llm_config_instance
            )
            
        elif api_type == "litellm":
            model_str = config.get('litellm_model')
            api_key = config.get('litellm_api_key') 
            base_url = config.get('litellm_base_url') 
            if not model_str:
                raise ValueError("Missing 'litellm_model' string for LiteLLM.")
            return LiteLLMInferenceEngine(model=model_str, api_key=api_key, base_url=base_url, config=llm_config_instance)
        
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
