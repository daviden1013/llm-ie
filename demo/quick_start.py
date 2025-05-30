import os
from llm_ie import OpenAIInferenceEngine, BasicLLMConfig, DirectFrameExtractor, PromptEditor, SentenceUnitChunker, SlideWindowContextChunker

# Load synthesized medical note
with open("/home/daviden1013/David_projects/llm-ie/demo/document/synthesized_note.txt", 'r') as f:
    note_text = f.read()

# Define a LLM inference engine for the prompt editor
prompt_editor_llm = OpenAIInferenceEngine(base_url="https://openrouter.ai/api/v1", 
                                          model="meta-llama/llama-4-scout", 
                                          api_key=os.getenv("OPENROUTER_API_KEY"),
                                          config=BasicLLMConfig(temperature=0.4, 
                                                                max_new_tokens=4096))

# Use LLM to generrate a prompt template
editor = PromptEditor(prompt_editor_llm, DirectFrameExtractor)
editor.chat()

prompt_template = """
### Task description
The paragraph below contains a clinical note with diagnoses listed. Please carefully review it and extract the diagnoses, including the diagnosis date and status.

### Schema definition
Your output should contain: 
    "entity_text" which is the diagnosis spelled as it appears in the text,
    "Date" which is the date when the diagnosis was made,
    "Status" which is the current status of the diagnosis (e.g. active, resolved, etc.)

### Output format definition
Your output should follow JSON format, for example:
[
    {"entity_text": "<Diagnosis>", "attr": {"Date": "<date in YYYY-MM-DD format>", "Status": "<status>"}},
    {"entity_text": "<Diagnosis>", "attr": {"Date": "<date in YYYY-MM-DD format>", "Status": "<status>"}}
]
If there are no diagnoses, please output an empty list: [].

Do not explain your answer, just output the JSON.

### Additional hints
- Your output should be 100% based on the provided content. DO NOT output fake information.
- If there is no specific date or status, just omit those keys.

### Context
The text below is from the clinical note:
"{{input}}"
"""

# Define a LLM inference engine for the extractor
extractor_llm = OpenAIInferenceEngine(base_url="https://openrouter.ai/api/v1", 
                                      model="meta-llama/llama-3.1-70b-instruct", 
                                      api_key=os.getenv("OPENROUTER_API_KEY"),
                                      config=BasicLLMConfig(temperature=0.0, 
                                                            max_new_tokens=1024))
# Define unit chunker. Prompt sentences-by-sentence.
unit_chunker = SentenceUnitChunker()
# Define context chunker. Provides context for units.
context_chunker = SlideWindowContextChunker(window_size=2)
# Define extractor
extractor = DirectFrameExtractor(inference_engine=extractor_llm, 
                                 unit_chunker=unit_chunker,
                                 context_chunker=context_chunker,
                                 prompt_template=prompt_template)

# Extract
frames =  extractor.extract_frames(text_content=note_text, verbose=True)

# Check extractions
len(frames)

for frame in frames:
    print(frame.to_dict())


from llm_ie.data_types import LLMInformationExtractionDocument

# Define document
doc = LLMInformationExtractionDocument(doc_id="Meidcal note",
                                       text=note_text)

# Add frames to document
doc.add_frames(frames, create_id=True)

# Save document to file (.llmie)
doc.save("<your filename>.llmie")

# Visualize the document
doc.viz_serve()

