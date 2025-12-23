from llm_inference_engine import (
    # Configs
    LLMConfig,
    BasicLLMConfig,
    ReasoningLLMConfig,
    Qwen3LLMConfig,
    OpenAIReasoningLLMConfig,
    
    # Base Engine
    InferenceEngine,
    
    # Concrete Engines
    OllamaInferenceEngine,
    OpenAIInferenceEngine,
    HuggingFaceHubInferenceEngine,
    AzureOpenAIInferenceEngine,
    LiteLLMInferenceEngine,
    OpenAICompatibleInferenceEngine,
    VLLMInferenceEngine,
    SGLangInferenceEngine,
    OpenRouterInferenceEngine
)

from llm_inference_engine.utils import MessagesLogger

class LlamaCppInferenceEngine(InferenceEngine):
    """
    Deprecated: This engine is no longer supported. Please run llama.cpp as a server and use OpenAICompatibleInferenceEngine instead.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LlamaCppInferenceEngine has been deprecated. "
            "Please run llama.cpp as a server and use OpenAICompatibleInferenceEngine."
        )

    def chat(self, *args, **kwargs):
        raise NotImplementedError("This engine is deprecated.")