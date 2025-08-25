In some cases, it is helpful to adjust **LLM sampling parameters** (e.g., temperature, top-p, top-k, maximum new tokens) or use **reasoning models** (e.g., OpenAI o-series models, Qwen3) which requires special treatments in system prompt, user prompt, and sampling parameters. For example, OpenAI o-series reasoning models disallow passing a system prompt or setting custom temperature. Another example is Qwen3 hybrid thinking mode. Special tokens "/think" and "/no_think" should be appended to user prompts to control for the reasoning behavior. 


| Config class | LLMs |
|----------|----------|
| `BasicLLMConfig` | Most non-reasoning LLMs <br> <li>Llama4 <li>Qwen3-30B-A3B-Instruct-2507 <li>... |
| `ReasoningLLMConfig` | Most reasoning LLMs <br> <li>Qwen3-30B-A3B-Thinking-2507 <li>gpt-oss-120b <li>... |
| `Qwen3LLMConfig` | Qwen3 **hybrid thinking** <br> <li>Qwen3-30B-A3B <li>Qwen3-32B <li>... |
| `OpenAIReasoningLLMConfig` | OpenAI API reasoning models <br> <li>"o" series (o1, o3, o4) <li>... |

## Setting sampling parameters
LLM sampling parameters such as temperature, top-p, top-k, and maximum new tokens can be set by passing a `LLMConfig` class to the `InferenceEngine` constructor.

```python
from llm_ie.engines import OpenAIInferenceEngine, BasicLLMConfig

config = BasicLLMConfig(temperature=0.2, max_new_tokens=4096)
inference_engine = OpenAIInferenceEngine(model="gpt-4o-mini", config=config)
```

## Reasoning models
To use reasoning models such as OpenAI o-series (e.g., o1, o3, o3-mini, o4-mini), some special processing is required. We provide dedicated configuration classes for them.

### General reasoning models
Most reasoning models can be configured with `ReasoningLLMConfig`. By specifying the start and end thinking tags, the reasoning tokens will be excluded from model response while stored in messages log (if `return_messages_log=True`).

```python
from llm_ie.engines import OpenAIInferenceEngine, ReasoningLLMConfig

llm = OpenAIInferenceEngine(base_url="http://localhost:8000/v1", 
                            model="Qwen/Qwen3-30B-A3B-Thinking-2507", 
                            api_key="EMPTY", 
                            config=ReasoningLLMConfig(thinking_token_start="<think>", thinking_token_end="</think>", temperature=0.8, max_tokens=8192))
```


### OpenAI o-series reasoning models
OpenAI o-series reasoning model API does not allow setting system prompts. Contents in the system should be included in user prompts. Also, custom temperature is not allowed. We provide a dedicated configuration class `OpenAIReasoningLLMConfig` for these models. 

```python
from llm_ie.engines import OpenAIInferenceEngine, OpenAIReasoningLLMConfig

inference_engine = OpenAIInferenceEngine(model="o1-mini", 
                                         config=OpenAIReasoningLLMConfig(reasoning_effort="low"))
```

### Qwen3 (hybrid thinking mode)
**Does NOT work for Qwen3 2507 models**. Use `ReasoningLLMConfig` and `BasicLLMConfig` instead.

Qwen3 has a special way to manage reasoning behavior. The same models have *thinking mode* and *non-thinking mode*, controled by the prompting template. When a special token "/think" is appended to the user prompt, the models generate thinking tokens in a `<think>... </think>` block. When 
a special token "/no_think" is appended to the user prompt, the models generate an empty `<think>... </think>` block. We provide a dedicated configuration class `Qwen3LLMConfig` for these models. 

```python
from llm_ie.engines import OpenAIInferenceEngine, Qwen3LLMConfig

# Thinking mode
llm = OpenAIInferenceEngine(base_url="http://localhost:8000/v1", 
                            model="Qwen/Qwen3-30B-A3B", 
                            api_key="EMPTY", 
                            config=Qwen3LLMConfig(thinking_mode=True, temperature=0.8, max_tokens=8192))

# Non-thinking mode
llm = OpenAIInferenceEngine(base_url="http://localhost:8000/v1", 
                            model="Qwen/Qwen3-30B-A3B", 
                            api_key="EMPTY", 
                            config=Qwen3LLMConfig(thinking_mode=False, temperature=0.0, max_tokens=2048))
```
