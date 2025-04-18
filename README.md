<div align="center"><img src=doc_asset/readme_img/LLM-IE.png width=500 ></div>

![Python Version](https://img.shields.io/pypi/pyversions/llm-ie)
![PyPI](https://img.shields.io/pypi/v/llm-ie)


An LLM-powered tool that transforms everyday language into robust information extraction pipelines. 

| Features | Support |
|----------|----------|
| **LLM Agent for prompt writing** | :white_check_mark: Interactive chat, Python functions |
| **Named Entity Recognition (NER)** | :white_check_mark: Document-level, Sentence-level |
| **Entity Attributes Extraction** | :white_check_mark: Flexible formats |
| **Relation Extraction (RE)** | :white_check_mark: Binary & Multiclass relations |
| **Visualization** | :white_check_mark: Built-in entity & relation visualization |

## Recent Updates
- [v0.3.0](https://github.com/daviden1013/llm-ie/releases/tag/v0.3.0) (Oct 17, 2024): Interactive chat to Prompt editor LLM agent.
- [v0.3.1](https://github.com/daviden1013/llm-ie/releases/tag/v0.3.1) (Oct 26, 2024): Added Sentence Review Frame Extractor and Sentence CoT Frame Extractor
- [v0.3.4](https://github.com/daviden1013/llm-ie/releases/tag/v0.3.4) (Nov 24, 2024): Added entity fuzzy search.
- [v0.3.5](https://github.com/daviden1013/llm-ie/releases/tag/v0.3.5) (Nov 27, 2024): Adopted `json_repair` to fix broken JSON from LLM outputs.
- [v0.4.0](https://github.com/daviden1013/llm-ie/releases/tag/v0.4.0) (Jan 4, 2025): 
    - Concurrent LLM inferencing to speed up frame and relation extraction. 
    - Support for LiteLLM.
- [v0.4.1](https://github.com/daviden1013/llm-ie/releases/tag/v0.4.1) (Jan 25, 2025): Added filters, table view, and some new features to visualization tool (make sure to update [ie-viz](https://github.com/daviden1013/ie-viz)).
- [v0.4.3](https://github.com/daviden1013/llm-ie/releases/tag/v0.4.3) (Feb 7, 2025): Added Azure OpenAI support. 
- [v0.4.5](https://github.com/daviden1013/llm-ie/releases/tag/v0.4.5) (Feb 16, 2025): 
    - Added option to adjust number of context sentences in sentence-based extractors.
    - Added support for OpenAI reasoning models ("o" series).
- [v0.4.6](https://github.com/daviden1013/llm-ie/releases/tag/v0.4.6) (Mar 1, 2025): Allow LLM to output overlapping frames.

## Table of Contents
- [Overview](#overview)
- [Prerequisite](#prerequisite)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [User Guide](#user-guide)
    - [LLM Inference Engine](#llm-inference-engine)
    - [Prompt Template](#prompt-template)
    - [Prompt Editor LLM Agent](#prompt-editor-llm-agent)
    - [Extractor](#extractor)
        - [FrameExtractor](#frameextractor)
        - [RelationExtractor](#relationextractor)
- [Visualization](#visualization)
- [Benchmarks](#benchmarks)
- [Citation](#citation)

## Overview
LLM-IE is a toolkit that provides robust information extraction utilities for named entity, entity attributes, and entity relation extraction. Since prompt design has a significant impact on generative information extraction with LLMs, it has a built-in LLM agent ("editor") to help with prompt writing. The flowchart below demonstrates the workflow starting from a casual language request to output visualization.

<div align="center"><img src="doc_asset/readme_img/LLM-IE flowchart.png" width=800 ></div>

## Prerequisite
At least one LLM inference engine is required. There are built-in supports for 🚅 [LiteLLM](https://github.com/BerriAI/litellm), 🦙 [Llama-cpp-python](https://github.com/abetlen/llama-cpp-python), <img src="doc_asset/readme_img/ollama_icon.png" alt="Icon" width="22"/> [Ollama](https://github.com/ollama/ollama), 🤗 [Huggingface_hub](https://github.com/huggingface/huggingface_hub), <img src=doc_asset/readme_img/openai-logomark_white.png width=16 /> [OpenAI API](https://platform.openai.com/docs/api-reference/introduction), and <img src=doc_asset/readme_img/vllm-logo_small.png width=20 /> [vLLM](https://github.com/vllm-project/vllm). For installation guides, please refer to those projects. Other inference engines can be configured through the [InferenceEngine](src/llm_ie/engines.py) abstract class. See [LLM Inference Engine](#llm-inference-engine) section below.

## Installation
The Python package is available on PyPI. 
```
pip install llm-ie 
```
Note that this package does not check LLM inference engine installation nor install them. See [prerequisite](#prerequisite) section for details. 

## Quick Start
We use a [synthesized medical note](demo/document/synthesized_note.txt) by ChatGPT to demo the information extraction process. Our task is to extract diagnosis names, spans, and corresponding attributes (i.e., diagnosis datetime, status).

#### Choose an LLM inference engine
Choose one of the built-in engines below.

<details>
<summary>🚅 LiteLLM</summary>

```python
from llm_ie.engines import LiteLLMInferenceEngine

inference_engine = LiteLLMInferenceEngine(model="openai/Llama-3.3-70B-Instruct", base_url="http://localhost:8000/v1", api_key="EMPTY")
```
</details>

<details>
<summary><img src=doc_asset/readme_img/openai-logomark_white.png width=16 /> OpenAI API</summary>

Follow the [Best Practices for API Key Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) to set up API key.
```python
from llm_ie.engines import OpenAIInferenceEngine

inference_engine = OpenAIInferenceEngine(model="gpt-4o-mini")
```
</details>

<details>
<summary><img src=doc_asset/readme_img/Azure_icon.png width=32 /> Azure OpenAI API</summary>

Follow the [Azure AI Services Quickstart](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Ckeyless%2Ctypescript-keyless%2Cpython-new&pivots=programming-language-python) to set up Endpoint and API key.

```python
from llm_ie.engines import AzureOpenAIInferenceEngine

inference_engine = AzureOpenAIInferenceEngine(model="gpt-4o-mini", 
                                              api_version="<your api version>")
```

</details>

<details>
<summary>🤗 Huggingface_hub</summary>

```python
from llm_ie.engines import HuggingFaceHubInferenceEngine

inference_engine = HuggingFaceHubInferenceEngine(model="meta-llama/Meta-Llama-3-8B-Instruct")
```
</details>

<details>
<summary><img src="doc_asset/readme_img/ollama_icon.png" alt="Icon" width="22"/> Ollama</summary>

```python 
from llm_ie.engines import OllamaInferenceEngine

inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")
```
</details>

<details>
<summary><img src=doc_asset/readme_img/vllm-logo_small.png width=20 /> vLLM</summary>

The vLLM support follows the [OpenAI Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html). For more parameters, please refer to the documentation.

Start the server
```cmd
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct
```
Define inference engine
```python
from llm_ie.engines import OpenAIInferenceEngine
inference_engine = OpenAIInferenceEngine(base_url="http://localhost:8000/v1",
                                         api_key="EMPTY",
                                         model="meta-llama/Meta-Llama-3.1-8B-Instruct")
```
</details>

<details>
<summary>🦙 Llama-cpp-python</summary>

```python
from llm_ie.engines import LlamaCppInferenceEngine

inference_engine = LlamaCppInferenceEngine(repo_id="bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF",
                                           gguf_filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
```
</details>

In this quick start demo, we use Ollama to run Llama-3.1-8B with int8 quantization.
The outputs might be slightly different with other inference engines, LLMs, or quantization. 

#### Casual language as prompt 
We start with a casual description: 

*"Extract diagnosis from the clinical note. Make sure to include diagnosis date and status."* 

Define the AI prompt editor.
```python
from llm_ie import OllamaInferenceEngine, PromptEditor, SentenceFrameExtractor

# Define a LLM inference engine
inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")
# Define LLM prompt editor
editor = PromptEditor(inference_engine, SentenceFrameExtractor)
# Start chat
editor.chat()
```

This opens an interactive session:
<div align="left"><img src=doc_asset/readme_img/terminal_chat.PNG width=1000 ></div>


The ```PromptEditor``` drafts a prompt template following the schema required by the ```SentenceFrameExtractor```:

```
# Task description
The paragraph below contains a clinical note with diagnoses listed. Please carefully review it and extract the diagnoses, including the diagnosis date and status.

# Schema definition
Your output should contain: 
    "Diagnosis" which is the name of the diagnosis,
    "Date" which is the date when the diagnosis was made,
    "Status" which is the current status of the diagnosis (e.g. active, resolved, etc.)

# Output format definition
Your output should follow JSON format, for example:
[
    {"Diagnosis": "<Diagnosis text>", "Date": "<date in YYYY-MM-DD format>", "Status": "<status>"},
    {"Diagnosis": "<Diagnosis text>", "Date": "<date in YYYY-MM-DD format>", "Status": "<status>"}
]

# Additional hints
Your output should be 100% based on the provided content. DO NOT output fake information.
If there is no specific date or status, just omit those keys.

# Input placeholder
Below is the clinical note:
{{input}}
```


#### Information extraction pipeline
Now we apply the prompt template to build an information extraction pipeline.

```python
# Load synthesized medical note
with open("./demo/document/synthesized_note.txt", 'r') as f:
    note_text = f.read()

# Define extractor
extractor = SentenceFrameExtractor(inference_engine, prompt_template)

# Extract
# To stream the extraction process, use concurrent=False, stream=True:
frames =  extractor.extract_frames(note_text, entity_key="Diagnosis", concurrent=False, stream=True)
# For faster extraction, use concurrent=True to enable asynchronous prompting
frames =  extractor.extract_frames(note_text, entity_key="Diagnosis", concurrent=True)

# Check extractions
for frame in frames:
    print(frame.to_dict())
```
The output is a list of frames. Each frame has a ```entity_text```, ```start```, ```end```, and a dictionary of ```attr```. 

```python
{'frame_id': '0', 'start': 537, 'end': 549, 'entity_text': 'hypertension', 'attr': {'Date': '2010-01-01', 'Status': 'Active'}}
{'frame_id': '1', 'start': 551, 'end': 565, 'entity_text': 'hyperlipidemia', 'attr': {'Date': '2015-01-01', 'Status': 'Active'}}
{'frame_id': '2', 'start': 571, 'end': 595, 'entity_text': 'Type 2 diabetes mellitus', 'attr': {'Date': '2018-01-01', 'Status': 'Active'}}
{'frame_id': '3', 'start': 660, 'end': 670, 'entity_text': 'chest pain', 'attr': {'Date': 'July 18, 2024'}}
{'frame_id': '4', 'start': 991, 'end': 1003, 'entity_text': 'Hypertension', 'attr': {'Date': '2010-01-01'}}
{'frame_id': '5', 'start': 1026, 'end': 1040, 'entity_text': 'Hyperlipidemia', 'attr': {'Date': '2015-01-01'}}
{'frame_id': '6', 'start': 1063, 'end': 1087, 'entity_text': 'Type 2 Diabetes Mellitus', 'attr': {'Date': '2018-01-01'}}
{'frame_id': '7', 'start': 1926, 'end': 1947, 'entity_text': 'ST-segment depression', 'attr': None}
{'frame_id': '8', 'start': 2049, 'end': 2066, 'entity_text': 'acute infiltrates', 'attr': None}
{'frame_id': '9', 'start': 2117, 'end': 2150, 'entity_text': 'Mild left ventricular hypertrophy', 'attr': None}
{'frame_id': '10', 'start': 2402, 'end': 2425, 'entity_text': 'acute coronary syndrome', 'attr': {'Date': 'July 20, 2024', 'Status': 'Active'}}
```

We can save the frames to a document object for better management. The document holds ```text``` and ```frames```. The ```add_frame()``` method performs validation and (if passed) adds a frame to the document.
The ```valid_mode``` controls how frame validation should be performed. For example, the ```valid_mode = "span"``` will prevent new frames from being added if the frame spans (```start```, ```end```) has already exist. The ```create_id = True``` allows the document to assign unique frame IDs.  

```python
from llm_ie.data_types import LLMInformationExtractionDocument

# Define document
doc = LLMInformationExtractionDocument(doc_id="Synthesized medical note",
                                       text=note_text)
# Add frames to a document
doc.add_frames(frames, create_id=True)

# Save document to file (.llmie)
doc.save("<your filename>.llmie")
```

To visualize the extracted frames, we use the ```viz_serve()``` method. 
```python
doc.viz_serve()
```
A Flask App starts at port 5000 (default).
```
* Serving Flask app 'ie_viz.utilities'
* Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
127.0.0.1 - - [03/Oct/2024 23:36:22] "GET / HTTP/1.1" 200 -
```

<div align="left"><img src="doc_asset/readme_img/llm-ie_demo.PNG" width=1000 ></div>


## Examples
  - [Interactive chat with LLM prompt editors](demo/prompt_template_writing_via_chat.ipynb)
  - [Write prompt templates with LLM prompt editors](demo/prompt_template_writing.ipynb)
  - [NER + RE for Drug, Strength, Frequency](demo/medication_relation_extraction.ipynb)

## User Guide
This package is comprised of some key classes:
- LLM Inference Engine
- Prompt Template
- Prompt Editor
- Extractors

### LLM Inference Engine
Provides an interface for different LLM inference engines to work in the information extraction workflow. The built-in engines are `LiteLLMInferenceEngine`, `OpenAIInferenceEngine`, `HuggingFaceHubInferenceEngine`, `OllamaInferenceEngine`, and `LlamaCppInferenceEngine`. 

#### 🚅 LiteLLM
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

#### <img src=doc_asset/readme_img/openai-logomark_white.png width=16 /> OpenAI API
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

For reasoning models ("o" series), use the `reasoning_model=True` flag. The `max_completion_tokens` will be used instead of the `max_tokens`. `temperature` will be ignored.

```python
from llm_ie.engines import OpenAIInferenceEngine

inference_engine = OpenAIInferenceEngine(model="o1-mini", reasoning_model=True)
```

#### <img src=doc_asset/readme_img/Azure_icon.png width=32 /> Azure OpenAI API
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

For reasoning models ("o" series), use the `reasoning_model=True` flag. The `max_completion_tokens` will be used instead of the `max_tokens`. `temperature` will be ignored. 

```python
from llm_ie.engines import AzureOpenAIInferenceEngine

inference_engine = AzureOpenAIInferenceEngine(model="o1-mini", reasoning_model=True)
```

#### 🤗 huggingface_hub
The ```model``` can be a model id hosted on the Hugging Face Hub or a URL to a deployed Inference Endpoint. Refer to the [Inference Client](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client) documentation for more details. 

```python
from llm_ie.engines import HuggingFaceHubInferenceEngine

inference_engine = HuggingFaceHubInferenceEngine(model="meta-llama/Meta-Llama-3-8B-Instruct")
```

####  <img src="doc_asset/readme_img/ollama_icon.png" alt="Icon" width="22"/> Ollama
The ```model_name``` must match the names on the [Ollama library](https://ollama.com/library). Use the command line ```ollama ls``` to check your local model list. ```num_ctx``` determines the context length LLM will consider during text generation. Empirically, longer context length gives better performance, while consuming more memory and increases computation. ```keep_alive``` regulates the lifespan of LLM. It indicates a number of seconds to hold the LLM after the last API call. Default is 5 minutes (300 sec).

```python
from llm_ie.engines import OllamaInferenceEngine

inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0", num_ctx=4096, keep_alive=300)
```

#### <img src=doc_asset/readme_img/vllm-logo_small.png width=20 /> vLLM
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

#### 🦙 Llama-cpp-python
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

#### Test inference engine configuration
To test the inference engine, use the ```chat()``` method. 

```python
from llm_ie.engines import OllamaInferenceEngine

inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")
inference_engine.chat(messages=[{"role": "user", "content":"Hi"}], stream=True)
```
The output should be something like (might vary by LLMs and versions)

```python
'How can I help you today?'
```

#### Customize inference engine
The abstract class ```InferenceEngine``` defines the interface and required method ```chat()```. Inherit this class for customized API. 
```python
class InferenceEngine:
    @abc.abstractmethod
    def __init__(self):
        """
        This is an abstract class to provide interfaces for LLM inference engines. 
        Children classes that inherits this class can be used in extractors. Must implement chat() method.
        """
        return NotImplemented

    @abc.abstractmethod
    def chat(self, messages:List[Dict[str,str]], max_new_tokens:int=2048, temperature:float=0.0, stream:bool=False, **kwrs) -> str:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        max_new_tokens : str, Optional
            the max number of new tokens LLM can generate. 
        temperature : float, Optional
            the temperature for token sampling. 
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        """
        return NotImplemented
```

### Prompt Template
A prompt template is a string with one or many placeholders ```{{<placeholder_name>}}```. When input to an extractor, the ```text_content``` will be inserted into the placeholders to construct a prompt. Below is a demo:

```python
prompt_template = """
    Below is a medical note. Your task is to extract diagnosis information. 
    Your output should include: 
        "Diagnosis": extract diagnosis names, 
        "Datetime": date/ time of diagnosis, 
        "Status": status of present, history, or family history

    Your output should follow a JSON format:
    [
        {"Diagnosis": <exact words as in the document>, "Datetime": <diagnosis datetime>, "Status": <one of "present", "history">},
        {"Diagnosis": <exact words as in the document>, "Datetime": <diagnosis datetime>, "Status": <one of "present", "history">},
        ...
    ]

    Below is the medical note:
    "{{input}}"
"""
# Define a inference engine
ollama = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")

# Define an extractor 
extractor = BasicFrameExtractor(ollama, prompt_template)

# Apply text content to prompt template
prompt_text = extractor._get_user_prompt(text_content="<some text...>")
print(prompt_text)
```

The ```prompt_text``` is the text content filled in the placeholder spot. 

```
Below is a medical note. Your task is to extract diagnosis information. 
Your output should include: 
    "Diagnosis": extract diagnosis names, 
    "Datetime": date/ time of diagnosis, 
    "Status": status of present, history, or family history
Your output should follow a JSON format:
[
    {"Diagnosis": <exact words as in the document>, "Datetime": <diagnosis datetime>, "Status": <one of "present", "history">},
    {"Diagnosis": <exact words as in the document>, "Datetime": <diagnosis datetime>, "Status": <one of "present", "history">},
    ...
]
Below is the medical note:
"<some text...>"
```

#### Placeholder
When only one placeholder is defined in the prompt template, the ```text_content``` can be a string or a dictionary with one key (regardless of the key name). When multiple placeholders are defined in the prompt template, the ```text_content``` should be a dictionary with:

```python
{"<placeholder 1>": "<some text>", "<placeholder 2>": "<some text>"...}
```
For example,

```python
prompt_template = """
    Below is a medical note. Your task is to extract diagnosis information. 

    # Backgound knowledge
    {{knowledge}}
    Your output should include: 
        "Diagnosis": extract diagnosis names, 
        "Datetime": date/ time of diagnosis, 
        "Status": status of present, history, or family history

    Your output should follow a JSON format:
    [
        {"Diagnosis": <exact words as in the document>, "Datetime": <diagnosis datetime>, "Status": <one of "present", "history">},
        {"Diagnosis": <exact words as in the document>, "Datetime": <diagnosis datetime>, "Status": <one of "present", "history">},
        ...
    ]

    Below is the medical note:
    "{{note}}"
"""
inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")
extractor = BasicFrameExtractor(inference_engine, prompt_template)
prompt_text = extractor._get_user_prompt(text_content={"knowledge": "<some text...>", 
                                                       "note": "<some text...>")
print(prompt_text)
```
Note that the keys in ```text_content``` must match the placeholder names defined in ```{{}}```.

#### Prompt writing guide
The quality of the prompt template can significantly impact the performance of information extraction. Also, the schema defined in prompt templates is dependent on the choice of extractors. When designing a prompt template schema, it is important to consider which extractor will be used. 

The ```Extractor``` class provides documentation and examples for prompt template writing. 

```python
from llm_ie.extractors import BasicFrameExtractor

print(BasicFrameExtractor.get_prompt_guide())
```

### Prompt Editor LLM Agent
The prompt editor is an LLM agent that help users write prompt templates following the defined schema and guideline of each extractor. Chat with the promtp editor:

```python
from llm_ie.prompt_editor import PromptEditor
from llm_ie.extractors import BasicFrameExtractor
from llm_ie.engines import OllamaInferenceEngine

# Define an LLM inference engine
inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")

# Define editor
editor = PromptEditor(inference_engine, BasicFrameExtractor)

editor.chat()
```

In a terminal environment, an interactive chat session will start:
<div align="left"><img src=doc_asset/readme_img/terminal_chat.PNG width=1000 ></div>

In the Jupyter/IPython environment, an ipywidgets session will start:
<div align="left"><img src=doc_asset/readme_img/IPython_chat.PNG width=1000 ></div>


We can also use the `rewrite()` and `comment()` methods to programmingly interact with the prompt editor: 
1. start with a casual description of the task
2. have the prompt editor generate a prompt template as the starting point
3. manually revise the prompt template
4. have the prompt editor to comment/ rewrite it

```python
from llm_ie.prompt_editor import PromptEditor
from llm_ie.extractors import BasicFrameExtractor
from llm_ie.engines import OllamaInferenceEngine

# Define an LLM inference engine
inference_engine = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")

# Define editor
editor = PromptEditor(inference_engine, BasicFrameExtractor)

# Have editor to generate initial prompt template
initial_version = editor.rewrite("Extract treatment events from the discharge summary.")
print(initial_version)
```
The editor generated a ```initial_version``` as below:

```
# Task description
The paragraph below contains information about treatment events in a patient's discharge summary. Please carefully review it and extract the treatment events, including any relevant details such as medications or procedures. Note that each treatment event may be nested under a specific section of the discharge summary.

# Schema definition
Your output should contain: 
    "TreatmentEvent" which is the name of the treatment,
    If applicable, "Medication" which is the medication used for the treatment,
    If applicable, "Procedure" which is the procedure performed during the treatment,
    "Evidence" which is the EXACT sentence in the text where you found the TreatmentEvent from

# Output format definition
Your output should follow JSON format, for example:
[
    {"TreatmentEvent": "<Treatment event name>", "Medication": "<name of medication>", "Procedure": "<name of procedure>", "Evidence": "<exact sentence from the text>"},
    {"TreatmentEvent": "<Treatment event name>", "Medication": "<name of medication>", "Procedure": "<name of procedure>", "Evidence": "<exact sentence from the text>"} 
]

# Additional hints
Your output should be 100% based on the provided content. DO NOT output fake information.
If there is no specific medication or procedure, just omit the corresponding key.

# Input placeholder
Below is the discharge summary:
{{input}}
```
Manually reviewing it and thinking about our needs, we found certain issues:
1. The task description is not specific enough. This is expected since the editor does not have access to the real document. 
2. Depending on the project, we might not need evidence text. Outputing it consumes more output tokens. 

Therefore, we manually revised the prompt template as below:

```python
manually_revised = """
# Task description
The paragraph below is a patient's discharge summary. Please carefully review it and extract the treatment events, including any relevant details such as medications or procedures. Note that each treatment event may be nested under a specific section of the discharge summary.

# Schema definition
Your output should contain: 
    "TreatmentEvent" which is the name of the treatment,
    If applicable, "Medication" which is the medication used for the treatment,
    If applicable, "Procedure" which is the procedure performed during the treatment

# Output format definition
Your output should follow JSON format, for example:
[
    {"TreatmentEvent": "<Treatment event name>", "Medication": "<name of medication>", "Procedure": "<name of procedure>"},
    {"TreatmentEvent": "<Treatment event name>", "Medication": "<name of medication>", "Procedure": "<name of procedure>"} 
]

# Additional hints
Your output should be 100% based on the provided content. DO NOT output fake information.
If there is no specific medication or procedure, just omit the corresponding key.

# Input placeholder
Below is the discharge summary:
{{input}}
"""
```

Now we have the editor to comment our revision.

```python
comment = editor.comment(manually_revised)

print(comment)
```
The comments from the editor are:

```
Here's an analysis of the draft prompt based on the prompt guideline:

**1. Task description**: The task description is clear and concise, explaining that the goal is to extract treatment events from a patient's discharge summary.

**2. Schema definition**: The schema definition is well-defined, specifying three keys: "TreatmentEvent", "Medication", and "Procedure". However, it would be helpful to provide more context on what constitutes a "treatment event" and how to distinguish between medication and procedure information.

**3. Output format definition**: The output format is correctly specified as JSON, with an example provided for clarity.

**4. Additional hints**: The additional hints are clear and concise, emphasizing the importance of extracting only real information from the text and omitting fake data.

**5. Input placeholder**: The input placeholder is present, but it would be helpful to provide a more detailed description of what type of discharge summary is expected (e.g., medical history, treatment plan, etc.).

Overall, the draft prompt is well-structured and easy to follow. However, providing more context and clarity on certain aspects, such as the definition of "treatment event" and the distinction between medication and procedure information, would make it even more effective.

Rating: 8/10

Recommendations:

* Provide a more detailed description of what constitutes a "treatment event".
* Clarify how to distinguish between medication and procedure information.
* Consider adding an example of a discharge summary to help illustrate the task.
```

After a few iterations of revision, we will have a high-quality prompt template for the information extraction pipeline. 

### Extractor
An extractor implements a prompting method for information extraction. There are two extractor families: ```FrameExtractor``` and ```RelationExtractor```. 
The ```FrameExtractor``` extracts named entities with attributes ("frames"). The ```RelationExtractor``` extracts the relations (and relation types) between frames.

#### FrameExtractor
The ```BasicFrameExtractor``` directly prompts LLM to generate a list of dictionaries. Each dictionary is then post-processed into a frame. The ```ReviewFrameExtractor``` is based on the ```BasicFrameExtractor``` but adds a review step after the initial extraction to boost sensitivity and improve performance. ```SentenceFrameExtractor``` gives LLM the entire document upfront as a reference, then prompts LLM sentence by sentence and collects per-sentence outputs. ```SentenceReviewFrameExtractor``` is the combined version of ```ReviewFrameExtractor``` and ```SentenceFrameExtractor``` which each sentence is extracted and reviewed. The ```SentenceCoTFrameExtractor``` implements chain of thoughts (CoT). It first analyzes a sentence, then extract frames based on the CoT. To learn about an extractor, use the class method ```get_prompt_guide()``` to print out the prompt guide. 

Since the output entity text from LLMs might not be consistent with the original text due to the limitations of LLMs, we apply fuzzy search in post-processing to find the accurate entity span. In the `FrameExtractor.extract_frames()` method, setting parameter `fuzzy_match=True` applies Jaccard similarity matching. 

<details>
<summary>BasicFrameExtractor</summary>

The ```BasicFrameExtractor``` directly prompts LLM to generate a list of dictionaries. Each dictionary is then post-processed into a frame. The ```text_content``` holds the input text as a string, or as a dictionary (if prompt template has multiple input placeholders). The ```entity_key``` defines which JSON key should be used as entity text. It must be consistent with the prompt template. 

```python
from llm_ie.extractors import BasicFrameExtractor

extractor = BasicFrameExtractor(inference_engine, prompt_temp)
frames = extractor.extract_frames(text_content=text, entity_key="Diagnosis", case_sensitive=False, fuzzy_match=True, stream=True)
```

Use the ```get_prompt_guide()``` method to inspect the prompt template guideline for ```BasicFrameExtractor```. 

```python
from llm_ie.extractors import BasicFrameExtractor

print(BasicFrameExtractor.get_prompt_guide())
```

```
Prompt Template Design:

1. Task Description:  
   Provide a detailed description of the task, including the background and the type of task (e.g., named entity recognition).

2. Schema Definition:  
   List the key concepts that should be extracted, and provide clear definitions for each one.

3. Output Format Definition:  
   The output should be a JSON list, where each element is a dictionary representing a frame (an entity along with its attributes). Each dictionary must include a key that holds the entity text. This key can be named "entity_text" or anything else depend on the context. The attributes can either be flat (e.g., {"entity_text": "<entity_text>", "attr1": "<attr1>", "attr2": "<attr2>"}) or nested (e.g., {"entity_text": "<entity_text>", "attributes": {"attr1": "<attr1>", "attr2": "<attr2>"}}).

4. Optional: Hints:  
   Provide itemized hints for the information extractors to guide the extraction process.

5. Optional: Examples:  
   Include examples in the format:  
    Input: ...  
    Output: ...

6. Input Placeholder:  
   The template must include a placeholder in the format {{<placeholder_name>}} for the input text. The placeholder name can be customized as needed.

......
```
</details>

<details>
<summary>ReviewFrameExtractor</summary>

The ```ReviewFrameExtractor``` is based on the ```BasicFrameExtractor``` but adds a review step after the initial extraction to boost sensitivity and improve performance. The ```review_prompt``` and ```review_mode``` are required when constructing the ```ReviewFrameExtractor```.

There are two review modes:
1. **Addition mode**: add more frames while keeping current. This is efficient for boosting recall. 
2. **Revision mode**: regenerate frames (add new and delete existing). 

Under the **Addition mode**, the ```review_prompt``` needs to instruct the LLM not to regenerate existing extractions:

*... You should ONLY add new diagnoses. DO NOT regenerate the entire answer.*

The ```review_mode``` should be set to ```review_mode="addition"```

Under the **Revision mode**, the ```review_prompt``` needs to instruct the LLM to regenerate:

*... Regenerate your output.*

The ```review_mode``` should be set to ```review_mode="revision"```

 ```python 
review_prompt = "Review the input and your output again. If you find some diagnosis was missed, add them to your output. Regenerate your output."

extractor = ReviewFrameExtractor(inference_engine, prompt_temp, review_prompt, review_mode="revision")
frames = extractor.extract_frames(text_content=text, entity_key="Diagnosis", stream=True)
 ```
</details>

<details>
<summary>SentenceFrameExtractor</summary>

The ```SentenceFrameExtractor``` instructs the LLM to extract sentence by sentence. The reason is to ensure the accuracy of frame spans. It also prevents LLMs from overseeing sections/ sentences. Empirically, this extractor results in better recall than the ```BasicFrameExtractor``` in complex tasks. 

For concurrent extraction (recommended), the `async/await` feature is used to speed up inferencing. The `concurrent_batch_size` sets the batch size of sentences to be processed in cocurrent.

```python
from llm_ie.extractors import SentenceFrameExtractor

extractor = SentenceFrameExtractor(inference_engine, prompt_temp)
frames = extractor.extract_frames(text_content=text, entity_key="Diagnosis", case_sensitive=False, fuzzy_match=True, concurrent=True, concurrent_batch_size=32)
```

The `context_sentences` sets number of sentences before and after the sentence of interest to provide additional context. When `context_sentences=2`, 2 sentences before and 2 sentences after are included in the user prompt as context. When `context_sentences="all"`, the entire document is included as context. When `context_sentences=0`, no context is provided and LLM will only extract based on the current sentence of interest.

```python
from llm_ie.extractors import SentenceFrameExtractor

extractor = SentenceFrameExtractor(inference_engine=inference_engine, 
                                   prompt_template=prompt_temp, 
                                   context_sentences=2)
frames = extractor.extract_frames(text_content=text, entity_key="Diagnosis", case_sensitive=False, fuzzy_match=True, stream=True)
```

For the sentence:

*The patient has a history of hypertension, hyperlipidemia, and Type 2 diabetes mellitus.*

The context is "previous sentence 2" "previous sentence 1" "the sentence of interest" "proceeding sentence 1" "proceeding sentence 2":

*Emily Brown, MD (Cardiology), Dr. Michael Green, MD (Pulmonology)

*#### Reason for Admission*
*John Doe, a 49-year-old male, was admitted to the hospital with complaints of chest pain, shortness of breath, and dizziness. The patient has a history of hypertension, hyperlipidemia, and Type 2 diabetes mellitus. #### History of Present Illness*
*The patient reported that the chest pain started two days prior to admission. The pain was described as a pressure-like sensation in the central chest, radiating to the left arm and jaw.*

</details>

<details>
<summary>SentenceReviewFrameExtractor</summary>

The `SentenceReviewFrameExtractor` performs sentence-level extraction and review. 

```python
from llm_ie.extractors import SentenceReviewFrameExtractor

extractor = SentenceReviewFrameExtractor(inference_engine, prompt_temp, review_mode="revision")
frames = extractor.extract_frames(text_content=note_text, entity_key="Diagnosis", stream=True)
```

```
Sentence: 
#### History of Present Illness
The patient reported that the chest pain started two days prior to admission.

Initial Output:
[
  {"Diagnosis": "chest pain", "Date": "two days prior to admission", "Status": "reported"}
]
Review:
[
  {"Diagnosis": "admission", "Date": null, "Status": null}
]
```

</details>

<details>
<summary>SentenceCoTFrameExtractor</summary>

The `SentenceCoTFrameExtractor` processes document sentence-by-sentence. For each sentence, it first generate an analysis paragraph in `<Analysis>... </Analysis>`(chain-of-thought). Then output extraction in JSON in `<Outputs>... </Outputs>`, similar to `SentenceFrameExtractor`.

```python
from llm_ie.extractors import SentenceCoTFrameExtractor

extractor = SentenceCoTFrameExtractor(inference_engine, CoT_prompt_temp)
frames = extractor.extract_frames(text_content=note_text, entity_key="Diagnosis", stream=True)
```

```
Sentence: 
#### Discharge Medications
- Aspirin 81 mg daily
- Clopidogrel 75 mg daily
- Atorvastatin 40 mg daily
- Metoprolol 50 mg twice daily
- Lisinopril 20 mg daily
- Metformin 1000 mg twice daily

#### Discharge Instructions
John Doe was advised to follow a heart-healthy diet, engage in regular physical activity, and monitor his blood glucose levels.

CoT:
<Analysis>
The given text does not explicitly mention a diagnosis, but rather lists the discharge medications and instructions for the patient. However, we can infer that the patient has been diagnosed with conditions that require these medications, such as high blood pressure, high cholesterol, and diabetes.

</Analysis>

<Outputs>
[
  {"Diagnosis": "hypertension", "Date": null, "Status": "confirmed"},
  {"Diagnosis": "hyperlipidemia", "Date": null, "Status": "confirmed"},
  {"Diagnosis": "Type 2 diabetes mellitus", "Date": null, "Status": "confirmed"}
]
</Outputs>
```

</details>

#### RelationExtractor
Relation extractors prompt LLM with combinations of two frames from a document (```LLMInformationExtractionDocument```) and extract relations.
The ```BinaryRelationExtractor``` extracts binary relations (yes/no) between two frames. The ```MultiClassRelationExtractor``` extracts relations and assign relation types ("multi-class"). 

An important feature of the relation extractors is that users are required to define a ```possible_relation_func``` or ```possible_relation_types_func``` function for the extractors. The reason is, there are too many possible combinations of two frames (N choose 2 combinations). The ```possible_relation_func``` helps rule out impossible combinations and therefore, reduce the LLM inferencing burden.

<details>
<summary>BinaryRelationExtractor</summary>

Use the get_prompt_guide() method to inspect the prompt template guideline for BinaryRelationExtractor.
```python
from llm_ie.extractors import BinaryRelationExtractor

print(BinaryRelationExtractor.get_prompt_guide())
```

```
Prompt Template Design:

1. Task description:
   Provide a detailed description of the task, including the background and the type of task (e.g., binary relation extraction). Mention the region of interest (ROI) text. 
2. Schema definition: 
   List the criterion for relation (True) and for no relation (False).

3. Output format definition:
   The ouptut must be a dictionary with a key "Relation" (i.e., {"Relation": "<True or False>"}).

4. (optional) Hints:
   Provide itemized hints for the information extractors to guide the extraction process.

5. (optional) Examples:
   Include examples in the format:  
    Input: ...  
    Output: ...

6. Entity 1 full information:
   Include a placeholder in the format {{<frame_1>}}

7. Entity 2 full information:
   Include a placeholder in the format {{<frame_2>}}

8. Input placeholders:
   The template must include a placeholder "{{roi_text}}" for the ROI text.


Example:

    # Task description
    This is a binary relation extraction task. Given a region of interest (ROI) text and two entities from a medical note, indicate the relation existence between the two entities.

    # Schema definition
        True: if there is a relationship between a medication name (one of the entities) and its strength or frequency (the other entity).
        False: Otherwise.

    # Output format definition
    Your output should follow the JSON format:
    {"Relation": "<True or False>"}

    I am only interested in the content between []. Do not explain your answer. 

    # Hints
        1. Your input always contains one medication entity and 1) one strength entity or 2) one frequency entity.
        2. Pay attention to the medication entity and see if the strength or frequency is for it.
        3. If the strength or frequency is for another medication, output False. 
        4. If the strength or frequency is for the same medication but at a different location (span), output False.

    # Entity 1 full information:
    {{frame_1}}

    # Entity 2 full information:
    {{frame_2}}

    # Input placeholders
    ROI Text with the two entities annotated with <entity_1> and <entity_2>:
    "{{roi_text}}"
```

As an example, we define the ```possible_relation_func``` function:
  - if the two frames are > 500 characters apart, we assume no relation (False)
  - if the two frames are "Medication" and "Strength", or "Medication" and "Frequency", there could be relations (True)

```python
def possible_relation_func(frame_1, frame_2) -> bool:
    """
    This function pre-process two frames and outputs a bool indicating whether the two frames could be related.
    """
    # if the distance between the two frames are > 500 characters, assume no relation.
    if abs(frame_1.start - frame_2.start) > 500:
        return False
    
    # if the entity types are "Medication" and "Strength", there could be relations.
    if (frame_1.attr["entity_type"] == "Medication" and frame_2.attr["entity_type"] == "Strength") or \
        (frame_2.attr["entity_type"] == "Medication" and frame_1.attr["entity_type"] == "Strength"):
        return True
    
    # if the entity types are "Medication" and "Frequency", there could be relations.
    if (frame_1.attr["entity_type"] == "Medication" and frame_2.attr["entity_type"] == "Frequency") or \
        (frame_2.attr["entity_type"] == "Medication" and frame_1.attr["entity_type"] == "Frequency"):
        return True

    # Otherwise, no relation.
    return False
```

In the ```BinaryRelationExtractor``` constructor, we pass in the prompt template and ```possible_relation_func```.

```python
from llm_ie.extractors import BinaryRelationExtractor

extractor = BinaryRelationExtractor(inference_engine, prompt_template=prompt_template, possible_relation_func=possible_relation_func)
# Extract binary relations with concurrent mode (faster)
relations = extractor.extract_relations(doc, concurrent=True)

# To print out the step-by-step, use the `concurrent=False` and `stream=True` options
relations = extractor.extract_relations(doc, concurrent=False, stream=True)
```

</details>


<details>
<summary>MultiClassRelationExtractor</summary>

The main difference from ```BinaryRelationExtractor``` is that the ```MultiClassRelationExtractor``` allows specifying relation types. The prompt template guideline has an additional placeholder for possible relation types ```{{pos_rel_types}}```. 

```python
print(MultiClassRelationExtractor.get_prompt_guide())
```

```
Prompt Template Design:

1. Task description:
   Provide a detailed description of the task, including the background and the type of task (e.g., binary relation extraction). Mention the region of interest (ROI) text. 
2. Schema definition: 
   List the criterion for relation (True) and for no relation (False).

3. Output format definition:
   This section must include a placeholder "{{pos_rel_types}}" for the possible relation types.
   The ouptut must be a dictionary with a key "RelationType" (i.e., {"RelationType": "<relation type or No Relation>"}).

4. (optional) Hints:
   Provide itemized hints for the information extractors to guide the extraction process.

5. (optional) Examples:
   Include examples in the format:  
    Input: ...  
    Output: ...

6. Entity 1 full information:
   Include a placeholder in the format {{<frame_1>}}

7. Entity 2 full information:
   Include a placeholder in the format {{<frame_2>}}

8. Input placeholders:
   The template must include a placeholder "{{roi_text}}" for the ROI text.



Example:

    # Task description
    This is a multi-class relation extraction task. Given a region of interest (ROI) text and two frames from a medical note, classify the relation types between the two frames. 

    # Schema definition
        Strength-Drug: this is a relationship between the drug strength and its name. 
        Dosage-Drug: this is a relationship between the drug dosage and its name.
        Duration-Drug: this is a relationship between a drug duration and its name.
        Frequency-Drug: this is a relationship between a drug frequency and its name.
        Form-Drug: this is a relationship between a drug form and its name.
        Route-Drug: this is a relationship between the route of administration for a drug and its name.
        Reason-Drug: this is a relationship between the reason for which a drug was administered (e.g., symptoms, diseases, etc.) and a drug name.
        ADE-Drug: this is a relationship between an adverse drug event (ADE) and a drug name.

    # Output format definition
    Choose one of the relation types listed below or choose "No Relation":
    {{pos_rel_types}}

    Your output should follow the JSON format:
    {"RelationType": "<relation type or No Relation>"}

    I am only interested in the content between []. Do not explain your answer. 

    # Hints
        1. Your input always contains one medication entity and 1) one strength entity or 2) one frequency entity.
        2. Pay attention to the medication entity and see if the strength or frequency is for it.
        3. If the strength or frequency is for another medication, output "No Relation". 
        4. If the strength or frequency is for the same medication but at a different location (span), output "No Relation".

    # Entity 1 full information:
    {{frame_1}}

    # Entity 2 full information:
    {{frame_2}}

    # Input placeholders
    ROI Text with the two entities annotated with <entity_1> and <entity_2>:
    "{{roi_text}}"
```

As an example, we define the ```possible_relation_types_func``` :
  - if the two frames are > 500 characters apart, we assume "No Relation" (output [])
  - if the two frames are "Medication" and "Strength", the only possible relation types are "Strength-Drug" or "No Relation"
  - if the two frames are "Medication" and "Frequency", the only possible relation types are "Frequency-Drug" or "No Relation"

```python 
def possible_relation_types_func(frame_1, frame_2) -> List[str]:
    # If the two frames are > 500 characters apart, we assume "No Relation"
    if abs(frame_1.start - frame_2.start) > 500:
        return []
    
    # If the two frames are "Medication" and "Strength", the only possible relation types are "Strength-Drug" or "No Relation"
    if (frame_1.attr["entity_type"] == "Medication" and frame_2.attr["entity_type"] == "Strength") or \
        (frame_2.attr["entity_type"] == "Medication" and frame_1.attr["entity_type"] == "Strength"):
        return ['Strength-Drug']
    
    # If the two frames are "Medication" and "Frequency", the only possible relation types are "Frequency-Drug" or "No Relation"
    if (frame_1.attr["entity_type"] == "Medication" and frame_2.attr["entity_type"] == "Frequency") or \
        (frame_2.attr["entity_type"] == "Medication" and frame_1.attr["entity_type"] == "Frequency"):
        return ['Frequency-Drug']

    return []
```


```python
from llm_ie.extractors import MultiClassRelationExtractor

extractor = MultiClassRelationExtractor(inference_engine, prompt_template=re_prompt_template,
                                        possible_relation_types_func=possible_relation_types_func)

# Extract multi-class relations with concurrent mode (faster)
relations = extractor.extract_relations(doc, concurrent=True)

# To print out the step-by-step, use the `concurrent=False` and `stream=True` options
relations = extractor.extract_relations(doc, concurrent=False, stream=True)
```

</details>

### Visualization

<div align="center"><img src="doc_asset/readme_img/visualization.PNG" width=95% ></div>

The `LLMInformationExtractionDocument` class supports named entity, entity attributes, and relation visualization. The implementation is through our plug-in package [ie-viz](https://github.com/daviden1013/ie-viz). Check the example Jupyter Notebook [NER + RE for Drug, Strength, Frequency](demo/medication_relation_extraction.ipynb) for a working demo.

```cmd
pip install ie-viz
```

The `viz_serve()` method starts a Flask App on localhost port 5000 by default. 
```python
from llm_ie.data_types import LLMInformationExtractionDocument

# Define document
doc = LLMInformationExtractionDocument(doc_id="Medical note",
                                       text=note_text)
# Add extracted frames and relations to document
doc.add_frames(frames)
doc.add_relations(relations)
# Visualize the document
doc.viz_serve()
```

Alternatively, the `viz_render()` method returns a self-contained (HTML + JS + CSS) string. Save it to file and open with a browser.
```python
html = doc.viz_render()

with open("Medical note.html", "w") as f:
    f.write(html)
```

To customize colors for different entities, use `color_attr_key` (simple) or `color_map_func` (advanced). 

The `color_attr_key` automatically assign colors based on the specified attribute key. For example, "EntityType".
```python
doc.viz_serve(color_attr_key="EntityType")
```

The `color_map_func` allow users to define a custom entity-color mapping function. For example,
```python
def color_map_func(entity) -> str:
    if entity['attr']['<attribute key>'] == "<a certain value>":
        return "#7f7f7f"
    else:
        return "#03A9F4"

doc.viz_serve(color_map_func=color_map_func)
```

## Benchmarks
We benchmarked the frame and relation extractors on biomedical information extraction tasks. The results and experiment code is available on [this page](https://github.com/daviden1013/LLM-IE_Benchmark).


## Citation
For more information and benchmarks, please check our paper:
```bibtex
@article{hsu2025llm,
  title={LLM-IE: a python package for biomedical generative information extraction with large language models},
  author={Hsu, Enshuo and Roberts, Kirk},
  journal={JAMIA open},
  volume={8},
  number={2},
  pages={ooaf012},
  year={2025},
  publisher={Oxford University Press}
}
```