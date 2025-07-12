import sys
import warnings
from typing import List, Dict, Generator
import importlib.resources
from llm_ie.engines import InferenceEngine
from llm_ie.extractors import FrameExtractor
import re
import json
from colorama import Fore, Style

    
class PromptEditor:
    def __init__(self, inference_engine:InferenceEngine, extractor:FrameExtractor, prompt_guide:str=None):
        """
        This class is a LLM agent that rewrite or comment a prompt draft based on the prompt guide of an extractor.

        Parameters
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        extractor : FrameExtractor
            a FrameExtractor. 
        prompt_guide : str, optional
            the prompt guide for the extractor. 
            All built-in extractors have a prompt guide in the asset folder. Passing values to this parameter 
            will override the built-in prompt guide which is not recommended.
            For custom extractors, this parameter must be provided.
        """
        self.inference_engine = inference_engine

        # if prompt_guide is provided, use it anyways
        if prompt_guide:
            self.prompt_guide = prompt_guide
        # if prompt_guide is not provided, get it from the extractor
        else:
            self.prompt_guide = extractor.get_prompt_guide()
            # when extractor does not have a prompt guide (e.g. custom extractor), ValueError
            if self.prompt_guide is None:
                raise ValueError(f"Prompt guide for {extractor.__class__.__name__} is not available. Use `prompt_guide` parameter to provide a prompt guide.")
        
        # get system prompt
        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('system.txt')
        with open(file_path, 'r') as f:
            self.system_prompt =  f.read()

        # internal memory (history messages) for the `chat` method
        self.messages = []

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
        This method is stateless.
        """
        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('rewrite.txt')
        with open(file_path, 'r') as f:
            rewrite_prompt_template = f.read()

        prompt = self._apply_prompt_template(text_content={"draft": draft, "prompt_guideline": self.prompt_guide}, 
                                             prompt_template=rewrite_prompt_template)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
        res = self.inference_engine.chat(messages, verbose=True)
        return res
    
    def comment(self, draft:str) -> str:
        """
        This method inputs a prompt draft and comment following the extractor's guideline.
        This method is stateless.
        """
        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('comment.txt')
        with open(file_path, 'r') as f:
            comment_prompt_template = f.read()

        prompt = self._apply_prompt_template(text_content={"draft": draft, "prompt_guideline": self.prompt_guide}, 
                                             prompt_template=comment_prompt_template)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
        res = self.inference_engine.chat(messages, verbose=True)
        return res
    
    def clear_messages(self):
        """
        Clears the current chat history.
        """
        self.messages = []

    def export_chat(self, file_path: str):
        """
        Exports the current chat history to a JSON file.

        Parameters
        ----------
        file_path : str
            path to the file where the chat history will be saved.
            Should have a .json extension.
        """
        if not self.messages:
            raise ValueError("Chat history is empty. Nothing to export.")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, indent=4)

    def import_chat(self, file_path: str):
        """
        Imports a chat history from a JSON file, overwriting the current history.

        Parameters
        ----------
        file_path : str
            The path to the .json file containing the chat history.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_messages = json.load(f)

        # Validate the loaded messages format.
        if not isinstance(loaded_messages, list):
            raise TypeError("Invalid format: The file should contain a JSON list of messages.")
        for message in loaded_messages:
            if not (isinstance(message, dict) and 'role' in message and 'content' in message):
                raise ValueError("Invalid format: Each message must be a dictionary with 'role' and 'content' keys.")
        
        self.messages = loaded_messages


    def _terminal_chat(self):
        """
        This method runs an interactive chat session in the terminal to help users write prompt templates.
        """
        print(f'Welcome to the interactive chat! Type "{Fore.RED}exit{Style.RESET_ALL}" or {Fore.YELLOW}control + C{Style.RESET_ALL} to end the conversation.')
        if len(self.messages) > 0:
            print(f"\nPrevious conversation:")
            for message in self.messages:
                if message["role"] == "user":
                    print(f"{Fore.GREEN}\nUser: {Style.RESET_ALL}")
                    print(message["content"])
                elif message["role"] == "assistant":
                    print(f"{Fore.BLUE}Assistant: {Style.RESET_ALL}", end="")
                    print(message["content"])

        while True:
            # Get user input
            user_input = input(f"{Fore.GREEN}\nUser: {Style.RESET_ALL}")
            
            # Exit condition
            if user_input.lower() == 'exit':
                print(f"{Fore.YELLOW}Interactive chat ended. Goodbye!{Style.RESET_ALL}")
                break
            
            # Chat
            self.messages.append({"role": "user", "content": user_input})
            print(f"{Fore.BLUE}Assistant: {Style.RESET_ALL}", end="")
            response = self.inference_engine.chat(self.messages, verbose=True)
            self.messages.append({"role": "assistant", "content": response})
            

    def _IPython_chat(self):
        """
        This method runs an interactive chat session in Jupyter/IPython using ipywidgets to help users write prompt templates.
        """
        # Check if ipywidgets is installed
        if importlib.util.find_spec("ipywidgets") is None:
            raise ImportError("ipywidgets not found. Please install ipywidgets (```pip install ipywidgets```).")
        import ipywidgets as widgets

        # Check if IPython is installed
        if importlib.util.find_spec("IPython") is None:
            raise ImportError("IPython not found. Please install IPython (```pip install ipython```).")
        from IPython.display import display, HTML

        # Widgets for user input and chat output
        input_box = widgets.Text(placeholder="Type your message here...")
        output_area = widgets.Output()

        # Display initial instructions
        with output_area:
            display(HTML('Welcome to the interactive chat! Type "<span style="color: red;">exit</span>" to end the conversation.'))
            if len(self.messages) > 0:
                display(HTML(f'<p style="color: red;">Previous messages:</p>'))
                for message in self.messages:
                    if message["role"] == "user":
                        display(HTML(f'<p style="color: green;">User: {message["content"]}</p>'))
                    elif message["role"] == "assistant":
                        display(HTML(f'<p style="color: blue;">Assistant: {message["content"]}</p>'))

        def handle_input(sender):
            user_input = input_box.value
            input_box.value = ''  # Clear the input box after submission

            # Exit condition
            if user_input.strip().lower() == 'exit':
                with output_area:
                    display(HTML('<p style="color: orange;">Interactive chat ended. Goodbye!</p>'))
                input_box.disabled = True  # Disable the input box after exiting
                return

            # Append user message to conversation
            self.messages.append({"role": "user", "content": user_input})
            print(f"User: {user_input}")
            
            # Display the user message
            with output_area:
                display(HTML(f'<pre><span style="color: green;">User: </span>{user_input}</pre>'))

            # Get assistant's response and append it to conversation
            print("Assistant: ", end="")
            response = self.inference_engine.chat(self.messages, verbose=True)
            self.messages.append({"role": "assistant", "content": response})

            # Display the assistant's response
            with output_area:
                display(HTML(f'<pre><span style="color: blue;">Assistant: </span>{response}</pre>'))

        # Bind the user input to the handle_input function
        input_box.on_submit(handle_input)

        # Display the input box and output area
        display(input_box)
        display(output_area)

    def chat(self):
        """
        External method that detects the environment and calls the appropriate chat method.
        This method use and updates the `messages` list (internal memory).
        This method is stateful.
        """
        # Check if the conversation is empty, if so, load the initial chat prompt template.
        if len(self.messages) == 0:
            file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('chat.txt')
            with open(file_path, 'r') as f:
                chat_prompt_template = f.read()

            guideline = self._apply_prompt_template(text_content={"prompt_guideline": self.prompt_guide}, 
                                                    prompt_template=chat_prompt_template)

            self.messages = [{"role": "system", "content": self.system_prompt + guideline}]

        if 'ipykernel' in sys.modules:
            self._IPython_chat()
        else:
            self._terminal_chat()

    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        This method processes messages and yields response chunks from the inference engine.
        This is for frontend App.
        This method is stateless.

        Parameters:
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries (e.g., [{"role": "user", "content": "Hi"}]).

        Yields:
        -------
            Chunks of the assistant's response.
        """
        # Validate messages
        if not isinstance(messages, list) or not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
             raise ValueError("Messages must be a list of dictionaries with 'role' and 'content' keys.")
        
        # Always append system prompt and initial user message
        file_path = importlib.resources.files('llm_ie.asset.PromptEditor_prompts').joinpath('chat.txt')
        with open(file_path, 'r') as f:
            chat_prompt_template = f.read()

        guideline = self._apply_prompt_template(text_content={"prompt_guideline": self.prompt_guide}, 
                                                prompt_template=chat_prompt_template)

        messages = [{"role": "system", "content": self.system_prompt + guideline}] + messages

        stream_generator = self.inference_engine.chat(messages, stream=True)
        yield from stream_generator