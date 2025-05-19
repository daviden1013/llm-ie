We provide an interface for different LLM inference engines to work in the information extraction workflow. The built-in engines are `LiteLLMInferenceEngine`, `OpenAIInferenceEngine`, `HuggingFaceHubInferenceEngine`, `OllamaInferenceEngine`, and `LlamaCppInferenceEngine`. For customization, see [customize inference engine](#customize-inference-engine). Inference engines accept a [LLMConfig]() class where **sampling parameters** (e.g., temperature, top-p, top-k, maximum new tokens) and **reasoning configuration** (e.g., OpenAI o-series models, Qwen3) can be set.

## LiteLLM
The LiteLLM is an adaptor project that unifies many proprietary and open-source LLM APIs. Popular inferncing servers, including OpenAI, Huggingface Hub, and Ollama are supported via its interface. For more details, refer to [LiteLLM GitHub page](https://github.com/BerriAI/litellm). 

To use LiteLLM with LLM-IE, import the `LiteLLMInferenceEngine` and follow the required model naming.
```python
from llm_ie.engines import LiteLLMInferenceEngine

# Huggingface serverless inferencing
os.environ['HF_TOKEN']
inference_engine = LiteLLMInferenceEngine(model="huggingface/meta-llama/Meta-Llama-3-8B-Instruct")

# OpenAI GPT models
os.environ['OPENAI_API_KEY']
inference_engine = LiteLLMInferenceEngine(model="openai/gpt-4o-mini")

# OpenAI compatible local server
inference_engine = LiteLLMInferenceEngine(model="openai/Llama-3.1-8B-Instruct", base_url="http://localhost:8000/v1", api_key="EMPTY")

# Ollama 
inference_engine = LiteLLMInferenceEngine(model="ollama/llama3.1:8b-instruct-q8_0")
```

## OpenAI API & Compatible Services
In bash, save API key to the environmental variable ```OPENAI_API_KEY```.
```
export OPENAI_API_KEY=<your_API_key>
```

In Python, create inference engine and specify model name. For the available models, refer to [OpenAI webpage](https://platform.openai.com/docs/models). 
For more parameters, see [OpenAI API reference](https://platform.openai.com/docs/api-reference/introduction).

```python
from llm_ie.engines import OpenAIInferenceEngine

inference_engine = OpenAIInferenceEngine(model="gpt-4o-mini")
```

For OpenAI reasoning models (o-series), pass a `OpenAIReasoningLLMConfig` object to `OpenAIInferenceEngine` constructor. 

```python
from llm_ie.engines import OpenAIInferenceEngine, OpenAIReasoningLLMConfig

inference_engine = OpenAIInferenceEngine(model="o1-mini", 
                                         config=OpenAIReasoningLLMConfig(reasoning_effort="low"))
```

For OpenAI compatible services (OpenRouter for example):
```python
from llm_ie.engines import OpenAIInferenceEngine

inference_engine = OpenAIInferenceEngine(base_url="https://openrouter.ai/api/v1", model="meta-llama/llama-4-scout")
```

## Azure OpenAI API
In bash, save the endpoint name and API key to environmental variables `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`.
```
export AZURE_OPENAI_API_KEY="<your_API_key>"
export AZURE_OPENAI_ENDPOINT="<your_endpoint>"
```

In Python, create inference engine and specify model name. For the available models, refer to [OpenAI webpage](https://platform.openai.com/docs/models). 
For more parameters, see [Azure OpenAI reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart).

```python
from llm_ie.engines import AzureOpenAIInferenceEngine

inference_engine = AzureOpenAIInferenceEngine(model="gpt-4o-mini")
```

For reasoning models (o-series), pass a `OpenAIReasoningLLMConfig` object to `OpenAIInferenceEngine` constructor. 

```python
from llm_ie.engines import AzureOpenAIInferenceEngine

inference_engine = AzureOpenAIInferenceEngine(model="o1-mini", 
                                              config=OpenAIReasoningLLMConfig(reasoning_effort="low"))
```

## Huggingface_hub
The ```model``` can be a model id hosted on the Hugging Face Hub or a URL to a deployed Inference Endpoint. Refer to the [Inference Client](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client) documentation for more details. 

```python
from llm_ie.engines import HuggingFaceHubInferenceEngine

inference_engine = HuggingFaceHubInferenceEngine(model="meta-llama/Meta-Llama-3-8B-Instruct")
```

##  Ollama
The ```model_name``` must match the names on the [Ollama library](https://ollama.com/library). Use the command line ```ollama ls``` to check your local model list. ```num_ctx``` determines the context length LLM will consider during text generation. Empirically, longer context length gives better performance, while consuming more memory and increases computation. ```keep_alive``` regulates the lifespan of LLM. It indicates a number of seconds to hold the LLM after the last API call. Default is 5 minutes (300 sec).

```python
from llm_ie.engines import OllamaInferenceEngine

inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0", num_ctx=4096, keep_alive=300)
```

## vLLM
The vLLM support follows the [OpenAI Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html). For more parameters, please refer to the documentation.

Start the server
```cmd
CUDA_VISIBLE_DEVICES=<GPU#> vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --api-key MY_API_KEY --tensor-parallel-size <# of GPUs to use>
```
Use ```CUDA_VISIBLE_DEVICES``` to specify GPUs to use. The ```--tensor-parallel-size``` should be set accordingly. The ```--api-key``` is optional. 
the default port is 8000. ```--port``` sets the port. 

Define inference engine
```python
from llm_ie.engines import OpenAIInferenceEngine
inference_engine = OpenAIInferenceEngine(base_url="http://localhost:8000/v1",
                               api_key="MY_API_KEY",
                               model="meta-llama/Meta-Llama-3.1-8B-Instruct")
```
The ```model``` must match the repo name specified in the server.

## Llama-cpp-python
The ```repo_id``` and ```gguf_filename``` must match the ones on the Huggingface repo to ensure the correct model is loaded. ```n_ctx``` determines the context length LLM will consider during text generation. Empirically, longer context length gives better performance, while consuming more memory and increases computation. Note that when ```n_ctx``` is less than the prompt length, Llama.cpp throws exceptions. ```n_gpu_layers``` indicates a number of model layers to offload to GPU. Default is -1 for all layers (entire LLM). Flash attention ```flash_attn``` is supported by Llama.cpp. The ```verbose``` indicates whether model information should be displayed. For more input parameters, see 🦙 [Llama-cpp-python](https://github.com/abetlen/llama-cpp-python). 

```python
from llm_ie.engines import LlamaCppInferenceEngine

inference_engine = LlamaCppInferenceEngine(repo_id="bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF",
                                           gguf_filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                                           n_ctx=4096,
                                           n_gpu_layers=-1,
                                           flash_attn=True,
                                           verbose=False)
```

## Test inference engine configuration
To test the inference engine, use the ```chat()``` method. 

```python
from llm_ie.engines import OllamaInferenceEngine

inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")
inference_engine.chat(messages=[{"role": "user", "content":"Hi"}], verbose=True)
```
The output should be something like (might vary by LLMs and versions)

```python
'How can I help you today?'
```

## Customize inference engine
The abstract class ```InferenceEngine``` defines the interface and required method ```chat()```. Inherit this class for customized API. 
```python
class InferenceEngine:
    @abc.abstractmethod
    def __init__(self, config:LLMConfig, **kwrs):
        """
        This is an abstract class to provide interfaces for LLM inference engines. 
        Children classes that inherts this class can be used in extrators. Must implement chat() method.

        Parameters:
        ----------
        config : LLMConfig
            the LLM configuration. Must be a child class of LLMConfig.
        """
        return NotImplemented


    @abc.abstractmethod
    def chat(self, messages:List[Dict[str,str]], 
             verbose:bool=False, stream:bool=False) -> Union[str, Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.  
        """
        return NotImplemented

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 

        Return : Dict[str, Any]
            the config parameters.
        """
        return NotImplemented
```