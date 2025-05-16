An extractor implements a prompting algorithm for information extraction. There are two main extractor families: `FrameExtractor` and `RelationExtractor`. 
The `FrameExtractor` extracts named entities with attributes ("frames"). The `RelationExtractor` extracts the relations (and relation types) between frames. Under `FrameExtractor`, we made pre-packaged extractors that does not require much configuation and are often sufficient for regular use case ([Convenience FrameExtractor](#convenience-frameextractor)).

## FrameExtractor
Frame extractors in general adopts an **unit-context schema**. The purpose is to avoid having LLM to process long context and suffer from *needle in the haystack* challenge. We split an input document into multiple units. LLM only process a unit of text at a time. 

- **Unit:** a text snippet that LLM extrator will process at a time. It could be a sentence, a line of text, or a paragraph. 
- **Context:** the context around the unit. For exapmle, a slidewindow of 2 sentences before and after. Context is optional. 

![unit-context schema](/readme_img/unit_context_schema.png)

### DirectFrameExtractor
The `DirectFrameExtractor` implements the unit-context schema. We start by defining the unit using one of the `UnitChunker`. The `SentenceUnitChunker` chunks the input document into sentences. Then, we define how context should be provided by choosing one of the `ContextChunker`. The `SlideWindowContextChunker` parse 2 units (sentences in this case) before and after each unit as context. For more options, see [Chunkers](./api/chunkers.md).

```python
from llm_ie import DirectFrameExtractor, SentenceUnitChunker, SlideWindowContextChunker

unit_chunker = SentenceUnitChunker()
context_chunker = SlideWindowContextChunker(window_size=2)
extractor = DirectFrameExtractor(inference_engine=llm, 
                                 unit_chunker=unit_chunker,
                                 context_chunker=context_chunker,
                                 prompt_template=prompt_template)
```

### ReviewFrameExtractor
The `ReviewFrameExtractor` is a child of `DirectFrameExtractor`. It adds a review step after the initial output.
There are two review modes:

1. **Addition mode**: add more frames while keeping current. This is efficient for boosting recall. 
2. **Revision mode**: regenerate frames (add new and delete existing). 

Under the **Addition mode** (`review_mode="addition"`), the `review_prompt` needs to instruct the LLM not to regenerate existing extractions:

*... You should ONLY add new diagnoses. DO NOT regenerate the entire answer.*

Under the **Revision mode** (`review_mode="revision"`), the `review_prompt` needs to instruct the LLM to regenerate:

*... Regenerate your output.*

It is recommended to leave the `review_prompt=None` and use the default, unless there are special needs. 

```python 
from llm_ie import ReviewFrameExtractor, SentenceUnitChunker, SlideWindowContextChunker

unit_chunker = SentenceUnitChunker()
context_chunker = SlideWindowContextChunker(window_size=2)
extractor = ReviewFrameExtractor(inference_engine=llm, 
                                 unit_chunker=unit_chunker,
                                 context_chunker=context_chunker,
                                 prompt_template=prompt_template,
                                 review_mode="addition")
```

### Post-processing 
Since the output entity text from LLMs might not be consistent with the original text due to the limitations of LLMs, we apply JSON repair, case-sensitive, fuzzy search, and entity overlap settings in post-processing to find the accurate entity span. 

#### JSON repair
Automatically detect and fix broken JSON format with [json_repair](https://github.com/mangiucugna/json_repair).

#### Case sensitive
set `case_sensitive=False` to allow matching even when LLM generates inconsistent upper/lower cases. 

#### Fussy match
In the `extract_frames()` method, setting parameter `fuzzy_match=True` applies Jaccard similarity matching. The most likely spans will be returned as entity text.

#### Entity overlap
Set `allow_overlap_entities=True` to cpature overlapping entities. Note that this can cause multiple frames to be generated on the same entity span if they have same entity text.

### Concurrent Optimization
For concurrent extraction (recommended), set `concurrent=True` in `FrameExtractor.extract_frames`. The `concurrent_batch_size` sets the batch size of units to be processed in cocurrent.

## Convenience FrameExtractor
The `DirectFrameExtractor` and `ReviewFrameExtractor` provide flexible interfaces for all settings. However, in most use cases, simple interface is preferred. We pre-package some common (and high performance) settings for convenience. 

### BasicFrameExtractor
The ```BasicFrameExtractor``` prompts LLM with the entire document.

```python
from llm_ie import BasicFrameExtractor

extractor = BasicFrameExtractor(inference_engine, prompt_template)
```

It is equivalent to:

```python
from llm_ie import DirectFrameExtractor, WholeDocumentUnitChunker, NoContextChunker

unit_chunker = WholeDocumentUnitChunker()
context_chunker = NoContextChunker()
extractor = DirectFrameExtractor(inference_engine=llm, 
                                 unit_chunker=unit_chunker,
                                 context_chunker=context_chunker,
                                 prompt_template=prompt_template)
```

### BasicReviewFrameExtractor
Similar to the `BasicFrameExtractor`, but adds a revieww step after the initial outputs.

```python 
from llm_ie import BasicReviewFrameExtractor

extractor = BasicReviewFrameExtractor(inference_engine, prompt_template, review_mode="revision")
```

This is equivalent to:

```python
from llm_ie import ReviewFrameExtractor, WholeDocumentUnitChunker, NoContextChunker

unit_chunker = WholeDocumentUnitChunker()
context_chunker = NoContextChunker()
extractor = ReviewFrameExtractor(inference_engine=llm, 
                                 unit_chunker=unit_chunker,
                                 context_chunker=context_chunker,
                                 prompt_template=prompt_template,
                                 review_mode="revision")
```

### SentenceFrameExtractor

The ```SentenceFrameExtractor``` prompts LLMs to extract sentence-by-sentence. The `context_sentences` sets number of sentences before and after the sentence of interest to provide additional context. When `context_sentences=2`, 2 sentences before and 2 sentences after are included in the user prompt as context. When `context_sentences="all"`, the entire document is included as context. When `context_sentences=0`, no context is provided and LLM will only extract based on the current sentence of interest.

```python
from llm_ie import SentenceFrameExtractor

# slide window of 2 sentences as context
extractor = SentenceFrameExtractor(inference_engine, prompt_template, context_sentences=2)
```

It is equivalent to:

```python 
from llm_ie import DirectFrameExtractor, SentenceUnitChunker, SlideWindowContextChunker

unit_chunker = SentenceUnitChunker()
context_chunker = SlideWindowContextChunker(window_size=2)
extractor = DirectFrameExtractor(inference_engine=llm, 
                                 unit_chunker=unit_chunker,
                                 context_chunker=context_chunker,
                                 prompt_template=prompt_template)
```

### SentenceReviewFrameExtractor

The `SentenceReviewFrameExtractor` performs sentence-level extraction and review. The example below use no context, revision review mode.

```python
from llm_ie import SentenceReviewFrameExtractor

extractor = SentenceReviewFrameExtractor(inference_engine, prompt_temp, context_sentences=0, review_mode="revision")
```

It is equivalent to:

```python 
from llm_ie import ReviewFrameExtractor, SentenceUnitChunker, NoContextChunker

unit_chunker = SentenceUnitChunker()
context_chunker = NoContextChunker()
extractor = ReviewFrameExtractor(inference_engine=llm, 
                                 unit_chunker=unit_chunker,
                                 context_chunker=context_chunker,
                                 prompt_template=prompt_template,
                                 review_mode="revision")
```

## RelationExtractor
Relation extractors prompt LLM with combinations of two frames from a document (```LLMInformationExtractionDocument```) and extract relations.
The ```BinaryRelationExtractor``` extracts binary relations (yes/no) between two frames. The ```MultiClassRelationExtractor``` extracts relations and assign relation types ("multi-class"). 

An important feature of the relation extractors is that users are required to define a ```possible_relation_func``` or ```possible_relation_types_func``` function for the extractors. The reason is, there are too many possible combinations of two frames (N choose 2 combinations). The ```possible_relation_func``` helps rule out impossible combinations and therefore, reduce the LLM inferencing burden.


### BinaryRelationExtractor

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


### MultiClassRelationExtractor

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

### Concurrent Optimization
For concurrent extraction (recommended), the `async/await` feature is used to speed up inferencing. Set `concurrent=True` in `RelationExtractor.extract_relations`. The `concurrent_batch_size` sets the batch size of frame pairs to be processed in cocurrent.