from typing import Dict, Union
import importlib.resources
from llm_ie.engines import InferenceEngine
from llm_ie.extractors import FrameExtractor
import re
import colorama
from colorama import Fore, Style

    
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

        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('system.txt')
        with open(file_path, 'r') as f:
            self.system_prompt =  f.read()
        

    def _apply_prompt_template(self, text_content:Dict[str,str], prompt_template:str) -> str:
        """
        This method applies text_content to prompt_template and returns a prompt.

        Parameters
        ----------
        text_content : Dict[str,str]
            the input text content to put in prompt template. 
            all the keys must be included in the prompt template placeholder {{<placeholder name>}}.

        Returns : str
            a prompt.
        """
        pattern = re.compile(r'{{(.*?)}}')
        placeholders = pattern.findall(prompt_template)
        if len(placeholders) != len(text_content):
            raise ValueError(f"Expect text_content ({len(text_content)}) and prompt template placeholder ({len(placeholders)}) to have equal size.")
        if not all([k in placeholders for k, _ in text_content.items()]):
            raise ValueError(f"All keys in text_content ({text_content.keys()}) must match placeholders in prompt template ({placeholders}).")

        prompt = pattern.sub(lambda match: re.sub(r'\\', r'\\\\', text_content[match.group(1)]), prompt_template)

        return prompt
    

    def rewrite(self, draft:str) -> str:
        """
        This method inputs a prompt draft and rewrites it following the extractor's guideline.
        """
        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('rewrite.txt')
        with open(file_path, 'r') as f:
            rewrite_prompt_template = f.read()

        prompt = self._apply_prompt_template(text_content={"draft": draft, "prompt_guideline": self.prompt_guide}, 
                                             prompt_template=rewrite_prompt_template)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
        res = self.inference_engine.chat(messages, stream=True)
        return res
    
    def comment(self, draft:str) -> str:
        """
        This method inputs a prompt draft and comment following the extractor's guideline.
        """
        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('comment.txt')
        with open(file_path, 'r') as f:
            comment_prompt_template = f.read()

        prompt = self._apply_prompt_template(text_content={"draft": draft, "prompt_guideline": self.prompt_guide}, 
                                             prompt_template=comment_prompt_template)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
        res = self.inference_engine.chat(messages, stream=True)
        return res
    

    def chat(self):
        """
        This method runs an interactive chat session in the terminal to help users write prompt templates.
        """
        colorama.init(autoreset=True)
        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('chat.txt')
        with open(file_path, 'r') as f:
            chat_prompt_template = f.read()

        prompt = self._apply_prompt_template(text_content={"prompt_guideline": self.prompt_guide}, 
                                             prompt_template=chat_prompt_template)

        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
        
        print(f'Welcome to the interactive chat! Type "{Fore.RED}exit{Style.RESET_ALL}" or {Fore.YELLOW}control + C{Style.RESET_ALL} to end the conversation.')

        while True:
            # Get user input
            user_input = input(f"{Fore.GREEN}\nUser: {Style.RESET_ALL}")
            
            # Exit condition
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Chat
            messages.append({"role": "user", "content": user_input})
            print(f"{Fore.BLUE}Assistant: {Style.RESET_ALL}", end="")
            response = self.inference_engine.chat(messages, stream=True)
            messages.append({"role": "assistant", "content": response})
            