<div align="center"><img src=asset/LLM-IE.png width=500 ></div>

An LLM-powered tool that transforms everyday language into robust information extraction pipelines. 

## Table of Contents
- [Overview](#overview)
- [Prerequisite](#prerequisite)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [User Guide](#user-guide)
    - [LLM Inference Engine](#llm-inference-engine)
    - [Prompt Template](#prompt-template)
    - [Prompt Editor](#prompt-editor)
    - [Extractor](#extractor)

## Overview
LLM-IE is a toolkit that provides robust information extraction utilities for frame-based information extraction. Since prompt design has a significant impact on generative information extraction with LLMs, it also provides a built-in LLM editor to help with prompt writing. The flowchart below demonstrates the workflow starting from a casual language request.

<div align="center"><img src="asset/LLM-IE flowchart.png" width=800 ></div>

## Prerequisite
At least one LLM inference engine is required. We provide built-in support for 🦙 [Llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and <img src="https://avatars.githubusercontent.com/u/151674099?s=48&v=4" alt="Icon" width="20"/> [Ollama](https://github.com/ollama/ollama). For installation guides, please refer to those projects. Other inference engines can be configured through the [InferenceEngine](src/llm_ie/engines.py) abstract class. See [LLM Inference Engine](#llm-inference-engine) section below.

## Installation
The Python package is available on PyPI. 
```
pip install llm-ie 
```
Note that this package does not check LLM inference engine installation nor install them. See [prerequisite](#prerequisite) section for details. 

## Quick Start
We use a [synthesized medical note](demo/document/synthesized_note.txt) by ChatGPT to demo the information extraction process. Our task is to extract diagnosis names, spans, and corresponding attributes (i.e., diagnosis datetime, status).

#### Choose an LLM inference engine
We use one of the built-in engines.

<details>
<summary><img src="https://avatars.githubusercontent.com/u/151674099?s=48&v=4" alt="Icon" width="20"/> Ollama</summary>

```python 
from llm_ie.engines import OllamaInferenceEngine

llm = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")
```
</details>
<details>
<summary>🦙 Llama-cpp-python</summary>

```python
from llm_ie.engines import LlamaCppInferenceEngine

llama_cpp = LlamaCppInferenceEngine(repo_id="bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF",
                                    gguf_filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
```
</details>


#### Casual language as prompt 
We start with a casual description: 

*"Extract diagnosis from the clinical note. Make sure to include diagnosis date and status."* 

The ```PromptEditor``` rewrites it following the schema required by the ```BasicFrameExtractor```. 

```python 
from llm_ie.extractors import BasicFrameExtractor
from llm_ie.prompt_editor import PromptEditor

# Describe the task in casual language
prompt_draft = "Extract diagnosis from the clinical note. Make sure to include diagnosis date and status."

# Use LLM editor to generate a formal prompt template with standard extraction schema
editor = PromptEditor(llm, BasicFrameExtractor)
prompt_template = editor.rewrite(prompt_draft)
```

The editor generates a prompt template as below:
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
extractor = BasicFrameExtractor(llm, prompt_template)

# Extract
frames =  extractor.extract_frames(note_text, entity_key="Diagnosis", stream=True)

# Check extractions
for frame in frames:
    print(frame.to_dict())
```
The output is a list of frames. Each frame has a ```entity_text```, ```start```, ```end```, and a dictionary of ```attr```. 

```python
{'frame_id': '0', 'start': 537, 'end': 549, 'entity_text': 'Hypertension', 'attr': {'Datetime': '2010', 'Status': 'history'}}
{'frame_id': '1', 'start': 551, 'end': 565, 'entity_text': 'Hyperlipidemia', 'attr': {'Datetime': '2015', 'Status': 'history'}}
{'frame_id': '2', 'start': 571, 'end': 595, 'entity_text': 'Type 2 Diabetes Mellitus', 'attr': {'Datetime': '2018', 'Status': 'history'}}
{'frame_id': '3', 'start': 2402, 'end': 2431, 'entity_text': 'Acute Coronary Syndrome (ACS)', 'attr': {'Datetime': 'July 20, 2024', 'Status': 'present'}}
```

We can save the frames to a document object for better management. The document holds ```text``` and ```frames```. The ```add_frame()``` method performs validation and (if passed) adds a frame to the document.
The ```valid_mode``` controls how frame validation should be performed. For example, the ```valid_mode = "span"``` will prevent new frames from being added if the frame spans (```start```, ```end```) has already exist. The ```create_id = True``` allows the document to assign unique frame IDs.  

```python
from llm_ie.data_types import LLMInformationExtractionDocument

# Define document
doc = LLMInformationExtractionDocument(doc_id="Synthesized medical note",
                                       text=note_text)
# Add frames to a document
for frame in frames:
    doc.add_frame(frame, valid_mode="span", create_id=True)

# Save document to file (.llmie)
doc.save("<your filename>.llmie")
```

## User Guide
This package is comprised of some key classes:
- LLM Inference Engine
- Prompt Template
- Prompt Editor
- Extractors

### LLM Inference Engine
Provides an interface for different LLM inference engines to work in the information extraction workflow. The built-in engines are ```LlamaCppInferenceEngine``` and ```OllamaInferenceEngine```. 

#### 🦙 Llama-cpp-python
The ```repo_id``` and ```gguf_filename``` must match the ones on the Huggingface repo to ensure the correct model is loaded. ```n_ctx``` determines the context length LLM will consider during text generation. Empirically, longer context length gives better performance, while consuming more memory and increases computation. Note that when ```n_ctx``` is less than the prompt length, Llama.cpp throws exceptions. ```n_gpu_layers``` indicates a number of model layers to offload to GPU. Default is -1 for all layers (entire LLM). Flash attention ```flash_attn``` is supported by Llama.cpp. The ```verbose``` indicates whether model information should be displayed. For more input parameters, see 🦙 [Llama-cpp-python](https://github.com/abetlen/llama-cpp-python). 

```python
from llm_ie.engines import LlamaCppInferenceEngine

llama_cpp = LlamaCppInferenceEngine(repo_id="bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF",
                                    gguf_filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                                    n_ctx=4096,
                                    n_gpu_layers=-1,
                                    flash_attn=True,
                                    verbose=False)
```
####  <img src="https://avatars.githubusercontent.com/u/151674099?s=48&v=4" alt="Icon" width="20"/> Ollama
The ```model_name``` must match the names on the [Ollama library](https://ollama.com/library). Use the command line ```ollama ls``` to check your local model list. ```num_ctx``` determines the context length LLM will consider during text generation. Empirically, longer context length gives better performance, while consuming more memory and increases computation. ```keep_alive``` regulates the lifespan of LLM. It indicates a number of seconds to hold the LLM after the last API call. Default is 5 minutes (300 sec).

```python
from llm_ie.engines import OllamaInferenceEngine

ollama = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0",
                               num_ctx=4096,
                               keep_alive=300)
```

#### Test inference engine configuration
To test the inference engine, use the ```chat()``` method. 

```python
from llm_ie.engines import OllamaInferenceEngine

ollama = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")
engine.chat(messages=[{"role": "user", "content":"Hi"}], stream=True)
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
ollama = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")
extractor = BasicFrameExtractor(ollama, prompt_template)
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

### Prompt Editor
The prompt editor is an LLM agent that reviews, comments and rewrites a prompt following the defined schema of each extractor. It is recommended to use prompt editor iteratively: 
1. start with a casual description of the task
2. have the prompt editor generate a prompt template as the starting point
3. manually revise the prompt template
4. have the prompt editor to comment/ rewrite it

```python
from llm_ie.prompt_editor import PromptEditor
from llm_ie.extractors import BasicFrameExtractor
from llm_ie.engines import OllamaInferenceEngine

# Define an LLM inference engine
ollama = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")

# Define editor
editor = PromptEditor(ollama, BasicFrameExtractor)

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
An extractor implements a prompting method for information extraction. The ```BasicFrameExtractor``` directly prompts LLM to generate a list of dictionaries. Each dictionary is then post-processed into a frame. The ```ReviewFrameExtractor``` is based on the ```BasicFrameExtractor``` but adds a review step after the initial extraction to boost sensitivity and improve performance. ```SentenceFrameExtractor``` gives LLM the entire document upfront as a reference, then prompts LLM sentence by sentence and collects per-sentence outputs. To learn about an extractor, use the class method ```get_prompt_guide()``` to print out the prompt guide. 

<details>
<summary>BasicFrameExtractor</summary>

The ```BasicFrameExtractor``` directly prompts LLM to generate a list of dictionaries. Each dictionary is then post-processed into a frame.

```python
from llm_ie.extractors import BasicFrameExtractor

print(BasicFrameExtractor.get_prompt_guide())
```

```
Prompt template design:
    1. Task description
    2. Schema definition
    3. Output format definition
    4. Additional hints
    5. Input placeholder

Example:

    # Task description
    The paragraph below is from the Food and Drug Administration (FDA) Clinical Pharmacology Section of Labeling for Human Prescription Drug and Biological Products, Adverse reactions section. Please carefully review it and extract the adverse reactions and percentages. Note that each adverse reaction is nested under a clinical trial and potentially an arm. Your output should take that into consideration.

    # Schema definition
    Your output should contain: 
        "ClinicalTrial" which is the name of the trial, 
        If applicable, "Arm" which is the arm within the clinical trial, 
        "AdverseReaction" which is the name of the adverse reaction,
        If applicable, "Percentage" which is the occurance of the adverse reaction within the trial and arm,
        "Evidence" which is the EXACT sentence in the text where you found the AdverseReaction from

    # Output format definition
    Your output should follow JSON format, for example:
    [
        {"ClinicalTrial": "<Clinical trial name or number>", "Arm": "<name of arm>", "AdverseReaction": "<Adverse reaction text>", "Percentage": "<a percent>", "Evidence": "<exact sentence from the text>"},
        {"ClinicalTrial": "<Clinical trial name or number>", "Arm": "<name of arm>", "AdverseReaction": "<Adverse reaction text>", "Percentage": "<a percent>", "Evidence": "<exact sentence from the text>"} 
    ]

    # Additional hints
    Your output should be 100% based on the provided content. DO NOT output fake numbers. 
    If there is no specific arm, just omit the "Arm" key. If the percentage is not reported, just omit the "Percentage" key. The "Evidence" should always be provided.

    # Input placeholder
    Below is the Adverse reactions section:
    {{input}}
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

extractor = ReviewFrameExtractor(llm, prompt_temp, review_prompt, review_mode="revision")
frames = extractor.extract_frames(text_content=text, entity_key="Diagnosis", stream=True)
 ```
</details>

<details>
<summary>SentenceFrameExtractor</summary>

The ```SentenceFrameExtractor``` instructs the LLM to extract sentence by sentence. The reason is to ensure the accuracy of frame spans. It also prevents LLMs from overseeing sections/ sentences. Empirically, this extractor results in better sensitivity than the ```BasicFrameExtractor``` in complex tasks. 

```python
from llm_ie.extractors import SentenceFrameExtractor

extractor = SentenceFrameExtractor(llm, prompt_temp)
frames = extractor.extract_frames(text_content=text, entity_key="Diagnosis", stream=True)
```
</details>


