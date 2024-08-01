import os
from llm_ie.engines import InferenceEngine
from llm_ie.extractors import FrameExtractor

class PromptEditor:
    def __init__(self, inference_engine:InferenceEngine, extractor:FrameExtractor):
        self.inference_engine = inference_engine
        self.prompt_guide = extractor.get_prompt_guide()

    def rewrite(self, draft:str) -> str:
        with open(os.path.join('/home/daviden1013/David_projects/llm-ie', 'asset', 'PromptEditor_prompts', 'rewrite.txt'), 'r') as f:
            prompt = f.read()

        prompt = prompt.replace("{{draft}}", draft).replace("{{prompt_guideline}}", self.prompt_guide)
        messages = [{"role": "user", "content": prompt}]
        res = self.inference_engine.chat(messages, stream=True)
        return res
    
    def comment(self, draft:str) -> str:
        with open(os.path.join('/home/daviden1013/David_projects/llm-ie', 'asset', 'PromptEditor_prompts', 'comment.txt'), 'r') as f:
            prompt = f.read()

        prompt = prompt.replace("{{draft}}", draft).replace("{{prompt_guideline}}", self.prompt_guide)
        messages = [{"role": "user", "content": prompt}]
        res = self.inference_engine.chat(messages, stream=True)
        return res