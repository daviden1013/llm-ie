import os
import importlib.resources
from llm_ie.engines import InferenceEngine
from llm_ie.extractors import FrameExtractor

class PromptEditor:
    def __init__(self, inference_engine:InferenceEngine, extractor:FrameExtractor):
        """
        This class is a LLM agent that rewrite or comment a prompt draft based on the prompt guide of an extractor.

        Parameters
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        extractor : FrameExtractor
            a FrameExtractor. 
        """
        self.inference_engine = inference_engine
        self.prompt_guide = extractor.get_prompt_guide()

    def rewrite(self, draft:str) -> str:
        """
        This method inputs a prompt draft and rewrites it following the extractor's guideline.
        """
        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('rewrite.txt')
        with open(file_path, 'r') as f:
            prompt = f.read()

        prompt = prompt.replace("{{draft}}", draft).replace("{{prompt_guideline}}", self.prompt_guide)
        messages = [{"role": "user", "content": prompt}]
        res = self.inference_engine.chat(messages, stream=True)
        return res
    
    def comment(self, draft:str) -> str:
        """
        This method inputs a prompt draft and comment following the extractor's guideline.
        """
        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('comment.txt')
        with open(file_path, 'r') as f:
            prompt = f.read()

        prompt = prompt.replace("{{draft}}", draft).replace("{{prompt_guideline}}", self.prompt_guide)
        messages = [{"role": "user", "content": prompt}]
        res = self.inference_engine.chat(messages, stream=True)
        return res