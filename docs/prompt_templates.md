A prompt template is a string with one or many placeholders ```{{<placeholder_name>}}```. When input to an extractor, the ```text_content``` will be inserted into the placeholders to construct a prompt. Below is a demo:

```text
### Task description
The paragraph below contains a clinical note with diagnoses listed. Please carefully review it and extract the diagnoses, including the diagnosis date and status.

### Schema definition
Your output should contain: 
    "entity_text" which is the name of the diagnosis spelled as it appears in the text,
    "Date" which is the date when the diagnosis was made,
    "Status" which is the current status of the diagnosis (e.g. active, resolved, etc.)

### Output format definition
Your output should follow JSON format, for example:
[
    {"entity_text": "<Diagnosis text>", "attr": {"Date": "<date in YYYY-MM-DD format>", "Status": "<status>"}},
    {"entity_text": "<Diagnosis text>", "attr": {"Date": "<date in YYYY-MM-DD format>", "Status": "<status>"}}
]

### Additional hints
- Your output should be 100% based on the provided content. DO NOT output fake information.
- If there is no specific date or status, just omit those keys.

### Context
The text below is from the clinical note:
"{{input}}"
```

## Placeholder
When only one placeholder is defined in the prompt template, the ```text_content``` can be a string or a dictionary with one key (regardless of the key name). When multiple placeholders are defined in the prompt template, the ```text_content``` should be a dictionary with:

```python
{"<placeholder 1>": "<some text>", "<placeholder 2>": "<some text>"...}
```
This is commonly used for injecting external knowledge, for example,

```text

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
```

## Prompt writing guide
The quality of the prompt template can significantly impact the performance of information extraction. Also, the schema defined in prompt templates is dependent on the choice of extractors. When designing a prompt template schema, it is important to consider which extractor will be used. 

The ```Extractor``` class provides documentation and examples for prompt template writing. This is used by the [Pormpt Editor](./prompt_editor.md). 

```python
from llm_ie.extractors import DirectFrameExtractor

print(DirectFrameExtractor.get_prompt_guide())
```
