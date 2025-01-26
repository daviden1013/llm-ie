import sys
sys.path.insert(0, r"/home/daviden1013/David_projects/llm-ie/src")

from llm_ie.engines import OllamaInferenceEngine
from llm_ie.extractors import SentenceFrameExtractor
from llm_ie.prompt_editor import PromptEditor


# Load synthesized medical note
with open("/home/daviden1013/David_projects/llm-ie/demo/document/synthesized_note.txt", 'r') as f:
    note_text = f.read()

# Define a LLM inference engine
llm = OllamaInferenceEngine(model_name="llama3.1:8b-instruct-q8_0")

# Describe the task in casual language
prompt_draft = "Extract diagnosis from the clinical note. Make sure to include diagnosis date and status."

# Use LLM to generrate a prompt template
editor = PromptEditor(llm, SentenceFrameExtractor)
prompt_template = editor.rewrite(prompt_draft)

# Alternatively, you can chat with the AI editor
editor.chat()

# Define extractor
extractor = SentenceFrameExtractor(llm, prompt_template)

# Extract
frames =  extractor.extract_frames(note_text, entity_key="Diagnosis", concurrent=True)

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

