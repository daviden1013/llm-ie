Chunkers chunk document text into smaller pieces (units and contexts) for easier processing by large language models. Their outputs are used in the [Extractor](./extractors.md).

The general usage pattern is as follows:

```python
from llm_ie import SentenceUnitChunker, SlideWindowContextChunker, DirectFrameExtractor
# Define unit chunker. Prompts sentences-by-sentence.
unit_chunker = SentenceUnitChunker()
# Define context chunker. Provides context for units.
context_chunker = SlideWindowContextChunker(window_size=2)
# Define extractor
extractor = DirectFrameExtractor(inference_engine=extractor_llm, 
                                 unit_chunker=unit_chunker,
                                 context_chunker=context_chunker,
                                 prompt_template=prompt_template)
```
The unit chunker breaks the document into sentences, and the context chunker provides 2 sentences around each unit as additional context. The extractor then processes these units and contexts using a large language model.

## Unit Chunkers
Unit Chunkers break text into smaller units, such as sentences or paragraphs. 

### WholeDocumentUnitChunker
This chunker treats the entire document as a single unit. When used, the entire document text is returned as one unit for LLM processing. This is suitable for tasks that requires the entire document to be processed at once, such as paragraph extraction.

```python
from llm_ie import WholeDocumentUnitChunker

chunker = WholeDocumentUnitChunker()
units = chunker.chunk(text)
print(units)
```

The unit is represented as:

```python
[FrameExtractionUnit(doc_id=f584157e-82a5-49f9-a8e0-2a730c790b62, start=0, end=3626, status=pending, text='# Clinical Note

**Patient Name**: Mary Johnson  
**Medical Record Number**: 987654321  
**Date of V...')]
```

### SeparatorUnitChunker
This chunker splits the document into units based on a specified separator, such as double newlines ("\n\n") for paragraphs.
```python
from llm_ie import SeparatorUnitChunker

chunker = SeparatorUnitChunker(sep="\n\n")
units = chunker.chunk(text)
```

The units are represented as:
```python
[
    FrameExtractionUnit(doc_id=12d12de1-7209-45fd-9f22-7f2ae71d79b3, start=0, end=15, status=pending, text='# Clinical Note...'), 
    FrameExtractionUnit(doc_id=12d12de1-7209-45fd-9f22-7f2ae71d79b3, start=17, end=159, status=pending, text='**Patient Name**: Mary Johnson  
**Medical Record Number**: 987654321  
**Date of Visit**: January 5...'), 
    rameExtractionUnit(doc_id=12d12de1-7209-45fd-9f22-7f2ae71d79b3, start=161, end=164, status=pending, text='---...'), 
    FrameExtractionUnit(doc_id=12d12de1-7209-45fd-9f22-7f2ae71d79b3, start=166, end=233, status=pending, text='## Chief Complaint  
Severe abdominal pain and nausea for 48 hours....')
...
]
``` 

### SentenceUnitChunker
This chunker splits the document into individual sentences using nltk package.
```python
from llm_ie import SentenceUnitChunker

chunker = SentenceUnitChunker()
units = chunker.chunk(text)
```

The units are represented as:
```python
[
    FrameExtractionUnit(doc_id=f2718b10-9fa9-4010-93a7-13affa1a64d6, start=0, end=143, status=pending, text='# Clinical Note

**Patient Name**: Mary Johnson  
**Medical Record Number**: 987654321  
**Date of V...'), 
    FrameExtractionUnit(doc_id=f2718b10-9fa9-4010-93a7-13affa1a64d6, start=144, end=233, status=pending, text='David Lee, DO  

---

## Chief Complaint  
Severe abdominal pain and nausea for 48 hours....'), 
    FrameExtractionUnit(doc_id=f2718b10-9fa9-4010-93a7-13affa1a64d6, start=235, end=420, status=pending, text='---

## History of Present Illness  
Mary Johnson is a 34-year-old female presenting with a 48-hour ...')
...
]
```

### TextLineUnitChunker
This chunker splits the document into lines based on newline characters ("\n").
```python
from llm_ie import TextLineUnitChunker

chunker = TextLineUnitChunker()
units = chunker.chunk(text)
```

The units are represented as:
```python
[
    FrameExtractionUnit(doc_id=f8a5e51a-3329-48e5-bc6c-59253141cb4f, start=0, end=15, status=pending, text='# Clinical Note...'), 
    FrameExtractionUnit(doc_id=f8a5e51a-3329-48e5-bc6c-59253141cb4f, start=16, end=16, status=pending, text='...'), 
    FrameExtractionUnit(doc_id=f8a5e51a-3329-48e5-bc6c-59253141cb4f, start=17, end=49, status=pending, text='**Patient Name**: Mary Johnson  ...'), 
    FrameExtractionUnit(doc_id=f8a5e51a-3329-48e5-bc6c-59253141cb4f, start=50, end=88, status=pending, text='**Medical Record Number**: 987654321  ...')
    ...
]
```

### LLMUnitChunker
This chunker uses LLM to intelligently segment the document into units based on sementic boundaries.
```python
from llm_ie import VLLMInferenceEngine, ReasoningLLMConfig, LLMUnitChunker

inference_engine = VLLMInferenceEngine(model="openai/gpt-oss-120b", config=ReasoningLLMConfig(reasoning_effort="medium"))
chunker = LLMUnitChunker(inference_engine=inference_engine)
units = chunker.chunk(text)
```

The units are represented as:
```python
[
    FrameExtractionUnit(doc_id=01a1e9f7-8aaa-4df3-8b26-df71c279550c, start=0, end=166, status=pending, text='# Clinical Note

**Patient Name**: Mary Johnson  
**Medical Record Number**: 987654321  
**Date of V...'), 
    FrameExtractionUnit(doc_id=01a1e9f7-8aaa-4df3-8b26-df71c279550c, start=166, end=240, status=pending, text='## Chief Complaint  
Severe abdominal pain and nausea for 48 hours.

---

...'), 
    FrameExtractionUnit(doc_id=01a1e9f7-8aaa-4df3-8b26-df71c279550c, start=240, end=907, status=pending, text='## History of Present Illness  
Mary Johnson is a 34-year-old female presenting with a 48-hour histo...'), 
    FrameExtractionUnit(doc_id=01a1e9f7-8aaa-4df3-8b26-df71c279550c, start=907, end=1053, status=pending, text='## Past Medical History  
- Migraines, managed with sumatriptan as needed.  
- No history of gastroi...'), 
    FrameExtractionUnit(doc_id=01a1e9f7-8aaa-4df3-8b26-df71c279550c, start=1053, end=1117, status=pending, text='## Past Surgical History  
- Cholecystectomy at age 29.  

---

...')
...
]
``` 

## Context Chunkers
Context chunkers intake units and return contexts (text before and after each unit) to provide additional information for language models during processing. 

### NoContextChunker
This chunker does not provide any additional context for the units. Each unit is processed independently without any surrounding text.
```python
from llm_ie import NoContextChunker

context_chunker = NoContextChunker()
context_chunker.chunk(units[0])
```

### WholeDocumentContextChunker
This chunker provides the entire document as context for each unit. Each unit is processed with access to the full document text.
```python
from llm_ie import WholeDocumentContextChunker

context_chunker = WholeDocumentContextChunker()
context_chunker.chunk(units[0])
``` 

### SlideWindowContextChunker
This chunker provides a sliding window of surrounding units as context for each unit. The window size can be specified to include a certain number of units before and after the target unit.
```python
from llm_ie import SlideWindowContextChunker

context_chunker = SlideWindowContextChunker(window_size=2)
context_chunker.chunk(units[0])
```
