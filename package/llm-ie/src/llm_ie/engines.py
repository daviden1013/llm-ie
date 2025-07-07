import abc
import re
import warnings
import importlib.util
from typing import Any, Tuple, List, Dict, Union, Generator


class LLMConfig(abc.ABC):
    def __init__(self, **kwargs):
        """
        This is an abstract class to provide interfaces for LLM configuration. 
        Children classes that inherts this class can be used in extrators and prompt editor.
        Common LLM parameters: max_new_tokens, temperature, top_p, top_k, min_p.
        """
        self.params = kwargs.copy()


    @abc.abstractmethod
    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        This method preprocesses the input messages before passing them to the LLM.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        return NotImplemented

    @abc.abstractmethod
    def postprocess_response(self, response:Union[str, Generator[str, None, None]]) -> Union[str, Generator[str, None, None]]:
        """
        This method postprocesses the LLM response after it is generated.

        Parameters:
        ----------
        response : Union[str, Generator[str, None, None]]
            the LLM response. Can be a string or a generator.
        
        Returns:
        -------
        response : str
            the postprocessed LLM response
        """
        return NotImplemented


class BasicLLMConfig(LLMConfig):
    def __init__(self, max_new_tokens:int=2048, temperature:float=0.0, **kwargs):
        """
        The basic LLM configuration for most non-reasoning models.
        """
        super().__init__(**kwargs)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.params["max_new_tokens"] = self.max_new_tokens
        self.params["temperature"] = self.temperature

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        This method preprocesses the input messages before passing them to the LLM.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        return messages

    def postprocess_response(self, response:Union[str, Generator[str, None, None]]) -> Union[str, Generator[Dict[str, str], None, None]]:
        """
        This method postprocesses the LLM response after it is generated.

        Parameters:
        ----------
        response : Union[str, Generator[str, None, None]]
            the LLM response. Can be a string or a generator.
        
        Returns: Union[str, Generator[Dict[str, str], None, None]]
            the postprocessed LLM response. 
            if input is a generator, the output will be a generator {"data": <content>}.
        """
        if isinstance(response, str):
            return response

        def _process_stream():
            for chunk in response:
                yield {"type": "response", "data": chunk}

        return _process_stream()

class Qwen3LLMConfig(LLMConfig):
    def __init__(self, thinking_mode:bool=True, **kwargs):
        """
        The Qwen3 LLM configuration for reasoning models.

        Parameters:
        ----------
        thinking_mode : bool, Optional
            if True, a special token "/think" will be placed after each system and user prompt. Otherwise, "/no_think" will be placed.
        """
        super().__init__(**kwargs)
        self.thinking_mode = thinking_mode

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        Append a special token to the system and user prompts.
        The token is "/think" if thinking_mode is True, otherwise "/no_think".

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        thinking_token = "/think" if self.thinking_mode else "/no_think"
        new_messages = []
        for message in messages:
            if message['role'] in ['system', 'user']:
                new_message = {'role': message['role'], 'content': f"{message['content']} {thinking_token}"}
            else:
                new_message = {'role': message['role'], 'content': message['content']}

            new_messages.append(new_message)

        return new_messages

    def postprocess_response(self, response:Union[str, Generator[str, None, None]]) -> Union[str, Generator[Dict[str,str], None, None]]:
        """
        If input is a generator, tag contents in <think> and </think> as {"type": "reasoning", "data": <content>},
        and the rest as {"type": "response", "data": <content>}.
        If input is a string, drop contents in <think> and </think>.

        Parameters:
        ----------
        response : Union[str, Generator[str, None, None]]
            the LLM response. Can be a string or a generator.
        
        Returns:
        -------
        response : Union[str, Generator[str, None, None]]
            the postprocessed LLM response.
            if input is a generator, the output will be a generator {"type": <reasoning or response>, "data": <content>}.
        """
        if isinstance(response, str):
            return re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()

        if isinstance(response, Generator):
            def _process_stream():
                think_flag = False
                buffer = ""
                for chunk in response:
                    if isinstance(chunk, str):
                        buffer += chunk
                        # switch between reasoning and response
                        if "<think>" in buffer:
                            think_flag = True
                            buffer = buffer.replace("<think>", "")
                        elif "</think>" in buffer:
                            think_flag = False
                            buffer = buffer.replace("</think>", "")
                        
                        # if chunk is in thinking block, tag it as reasoning; else tag it as response
                        if chunk not in ["<think>", "</think>"]:
                            if think_flag:
                                yield {"type": "reasoning", "data": chunk}
                            else:
                                yield {"type": "response", "data": chunk}

            return _process_stream()


class OpenAIReasoningLLMConfig(LLMConfig):
    def __init__(self, reasoning_effort:str=None, **kwargs):
        """
        The OpenAI "o" series configuration.
        1. The reasoning effort as one of {"low", "medium", "high"}.
            For models that do not support setting reasoning effort (e.g., o1-mini, o1-preview), set to None.
        2. The temperature parameter is not supported and will be ignored.
        3. The system prompt is not supported and will be concatenated to the next user prompt.

        Parameters:
        ----------
        reasoning_effort : str, Optional
            the reasoning effort. Must be one of {"low", "medium", "high"}. Default is "low".
        """
        super().__init__(**kwargs)
        if reasoning_effort is not None:
            if reasoning_effort not in ["low", "medium", "high"]:
                raise ValueError("reasoning_effort must be one of {'low', 'medium', 'high'}.")

            self.reasoning_effort = reasoning_effort
            self.params["reasoning_effort"] = self.reasoning_effort

        if "temperature" in self.params:
            warnings.warn("Reasoning models do not support temperature parameter. Will be ignored.", UserWarning)
            self.params.pop("temperature")

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        Concatenate system prompts to the next user prompt.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        system_prompt_holder = ""
        new_messages = []
        for i, message in enumerate(messages):
            # if system prompt, store it in system_prompt_holder
            if message['role'] == 'system':
                system_prompt_holder = message['content']
            # if user prompt, concatenate it with system_prompt_holder
            elif message['role'] == 'user':
                if system_prompt_holder:
                    new_message = {'role': message['role'], 'content': f"{system_prompt_holder} {message['content']}"}
                    system_prompt_holder = ""
                else:
                    new_message = {'role': message['role'], 'content': message['content']}

                new_messages.append(new_message)
            # if assistant/other prompt, do nothing
            else:
                new_message = {'role': message['role'], 'content': message['content']}
                new_messages.append(new_message)

        return new_messages

    def postprocess_response(self, response:Union[str, Generator[str, None, None]]) -> Union[str, Generator[Dict[str, str], None, None]]:
        """
        This method postprocesses the LLM response after it is generated.

        Parameters:
        ----------
        response : Union[str, Generator[str, None, None]]
            the LLM response. Can be a string or a generator.
        
        Returns: Union[str, Generator[Dict[str, str], None, None]]
            the postprocessed LLM response. 
            if input is a generator, the output will be a generator {"type": "response", "data": <content>}.
        """
        if isinstance(response, str):
            return response

        def _process_stream():
            for chunk in response:
                yield {"type": "response", "data": chunk}

        return _process_stream()


class InferenceEngine:
    @abc.abstractmethod
    def __init__(self, config:LLMConfig, **kwrs):
        """
        This is an abstract class to provide interfaces for LLM inference engines. 
        Children classes that inherts this class can be used in extrators. Must implement chat() method.

        Parameters:
        ----------
        config : LLMConfig
            the LLM configuration. Must be a child class of LLMConfig.
        """
        return NotImplemented


    @abc.abstractmethod
    def chat(self, messages:List[Dict[str,str]], 
             verbose:bool=False, stream:bool=False) -> Union[str, Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.  
        """
        return NotImplemented

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 

        Return : Dict[str, Any]
            the config parameters.
        """
        return NotImplemented


class LlamaCppInferenceEngine(InferenceEngine):
    def __init__(self, repo_id:str, gguf_filename:str, n_ctx:int=4096, n_gpu_layers:int=-1, config:LLMConfig=None, **kwrs):
        """
        The Llama.cpp inference engine.

        Parameters:
        ----------
        repo_id : str
            the exact name as shown on Huggingface repo
        gguf_filename : str
            the exact name as shown in Huggingface repo -> Files and versions. 
            If multiple gguf files are needed, use the first.
        n_ctx : int, Optional
            context length that LLM will evaluate. 
        n_gpu_layers : int, Optional
            number of layers to offload to GPU. Default is all layers (-1).
        config : LLMConfig
            the LLM configuration. 
        """
        from llama_cpp import Llama
        self.repo_id = repo_id
        self.gguf_filename = gguf_filename
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.config = config if config else BasicLLMConfig()
        self.formatted_params = self._format_config()

        self.model = Llama.from_pretrained(
            repo_id=self.repo_id,
            filename=self.gguf_filename,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            **kwrs
        )

    def __del__(self):
        """
        When the inference engine is deleted, release memory for model.
        """
        del self.model

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["max_tokens"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False) -> str:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        """
        processed_messages = self.config.preprocess_messages(messages)

        response = self.model.create_chat_completion(
                    messages=processed_messages,
                    stream=verbose,
                    **self.formatted_params
                )

        if verbose:
            res = ''
            for chunk in response:
                out_dict = chunk['choices'][0]['delta']
                if 'content' in out_dict:
                    res += out_dict['content']
                    print(out_dict['content'], end='', flush=True)
            print('\n')
            return self.config.postprocess_response(res)
        
        res = response['choices'][0]['message']['content']
        return self.config.postprocess_response(res)


class OllamaInferenceEngine(InferenceEngine):
    def __init__(self, model_name:str, num_ctx:int=4096, keep_alive:int=300, config:LLMConfig=None, **kwrs):
        """
        The Ollama inference engine.

        Parameters:
        ----------
        model_name : str
            the model name exactly as shown in >> ollama ls
        num_ctx : int, Optional
            context length that LLM will evaluate.
        keep_alive : int, Optional
            seconds to hold the LLM after the last API call.
        config : LLMConfig
            the LLM configuration. 
        """
        if importlib.util.find_spec("ollama") is None:
            raise ImportError("ollama-python not found. Please install ollama-python (```pip install ollama```).")
        
        from ollama import Client, AsyncClient
        self.client = Client(**kwrs)
        self.async_client = AsyncClient(**kwrs)
        self.model_name = model_name
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.config = config if config else BasicLLMConfig()
        self.formatted_params = self._format_config()
    
    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["num_predict"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params

    def chat(self, messages:List[Dict[str,str]], 
             verbose:bool=False, stream:bool=False) -> Union[str, Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs VLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, VLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.
        """
        processed_messages = self.config.preprocess_messages(messages)

        options={'num_ctx': self.num_ctx, **self.formatted_params}
        if stream:
            def _stream_generator():
                response_stream = self.client.chat(
                    model=self.model_name, 
                    messages=processed_messages, 
                    options=options,
                    stream=True, 
                    keep_alive=self.keep_alive
                )
                for chunk in response_stream:
                    content_chunk = chunk.get('message', {}).get('content')
                    if content_chunk:
                        yield content_chunk

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            response = self.client.chat(
                            model=self.model_name, 
                            messages=processed_messages, 
                            options=options,
                            stream=True,
                            keep_alive=self.keep_alive
                        )
            
            res = ''
            for chunk in response:
                content_chunk = chunk.get('message', {}).get('content')
                print(content_chunk, end='', flush=True)
                res += content_chunk
            print('\n')
            return self.config.postprocess_response(res)
        
        else:
            response = self.client.chat(
                                model=self.model_name, 
                                messages=processed_messages, 
                                options=options,
                                stream=False,
                                keep_alive=self.keep_alive
                            )
            res = response.get('message', {}).get('content')
            return self.config.postprocess_response(res)
        

    async def chat_async(self, messages:List[Dict[str,str]]) -> str:
        """
        Async version of chat method. Streaming is not supported.
        """
        processed_messages = self.config.preprocess_messages(messages)

        response = await self.async_client.chat(
                            model=self.model_name, 
                            messages=processed_messages, 
                            options={'num_ctx': self.num_ctx, **self.formatted_params},
                            stream=False,
                            keep_alive=self.keep_alive
                        )
        
        res = response['message']['content']
        return self.config.postprocess_response(res)


class HuggingFaceHubInferenceEngine(InferenceEngine):
    def __init__(self, model:str=None, token:Union[str, bool]=None, base_url:str=None, api_key:str=None, config:LLMConfig=None, **kwrs):
        """
        The Huggingface_hub InferenceClient inference engine.
        For parameters and documentation, refer to https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client

        Parameters:
        ----------
        model : str
            the model name exactly as shown in Huggingface repo
        token : str, Optional
            the Huggingface token. If None, will use the token in os.environ['HF_TOKEN'].
        base_url : str, Optional
            the base url for the LLM server. If None, will use the default Huggingface Hub URL.
        api_key : str, Optional
            the API key for the LLM server. 
        config : LLMConfig
            the LLM configuration. 
        """
        if importlib.util.find_spec("huggingface_hub") is None:
            raise ImportError("huggingface-hub not found. Please install huggingface-hub (```pip install huggingface-hub```).")
        
        from huggingface_hub import InferenceClient, AsyncInferenceClient
        self.model = model
        self.base_url = base_url
        self.client = InferenceClient(model=model, token=token, base_url=base_url, api_key=api_key, **kwrs)
        self.client_async = AsyncInferenceClient(model=model, token=token, base_url=base_url, api_key=api_key, **kwrs)
        self.config = config if config else BasicLLMConfig()
        self.formatted_params = self._format_config()

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["max_tokens"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params


    def chat(self, messages:List[Dict[str,str]], 
             verbose:bool=False, stream:bool=False) -> Union[str, Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, VLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.
        """
        processed_messages = self.config.preprocess_messages(messages)

        if stream:
            def _stream_generator():
                response_stream = self.client.chat.completions.create(
                                    messages=processed_messages,
                                    stream=True,
                                    **self.formatted_params
                                )
                for chunk in response_stream:
                    content_chunk = chunk.get('choices')[0].get('delta').get('content')
                    if content_chunk:
                        yield content_chunk

            return self.config.postprocess_response(_stream_generator())
        
        elif verbose:
            response = self.client.chat.completions.create(
                            messages=processed_messages,
                            stream=True,
                            **self.formatted_params
                        )
            
            res = ''
            for chunk in response:
                content_chunk = chunk.get('choices')[0].get('delta').get('content')
                if content_chunk:
                    res += content_chunk
                    print(content_chunk, end='', flush=True)
            return self.config.postprocess_response(res)
        
        else:
            response = self.client.chat.completions.create(
                                messages=processed_messages,
                                stream=False,
                                **self.formatted_params
                            )
            res = response.choices[0].message.content
            return self.config.postprocess_response(res)
    
    async def chat_async(self, messages:List[Dict[str,str]]) -> str:
        """
        Async version of chat method. Streaming is not supported.
        """
        processed_messages = self.config.preprocess_messages(messages)

        response = await self.client_async.chat.completions.create(
                    messages=processed_messages,
                    stream=False,
                    **self.formatted_params
                )
    
        res = response.choices[0].message.content
        return self.config.postprocess_response(res)
        

class OpenAIInferenceEngine(InferenceEngine):
    def __init__(self, model:str, config:LLMConfig=None, **kwrs):
        """
        The OpenAI API inference engine. Supports OpenAI models and OpenAI compatible servers:
        - vLLM OpenAI compatible server (https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
        - Llama.cpp OpenAI compatible server (https://llama-cpp-python.readthedocs.io/en/latest/server/)

        For parameters and documentation, refer to https://platform.openai.com/docs/api-reference/introduction

        Parameters:
        ----------
        model_name : str
            model name as described in https://platform.openai.com/docs/models
        """
        if importlib.util.find_spec("openai") is None:
            raise ImportError("OpenAI Python API library not found. Please install OpanAI (```pip install openai```).")
        
        from openai import OpenAI, AsyncOpenAI
        self.client = OpenAI(**kwrs)
        self.async_client = AsyncOpenAI(**kwrs)
        self.model = model
        self.config = config if config else BasicLLMConfig()
        self.formatted_params = self._format_config()

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["max_completion_tokens"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False) -> Union[str, Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, VLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.
        """
        processed_messages = self.config.preprocess_messages(messages)

        if stream:
            def _stream_generator():
                response_stream = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=processed_messages,
                                        stream=True,
                                        **self.formatted_params
                                    )
                for chunk in response_stream:
                    if len(chunk.choices) > 0:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                        if chunk.choices[0].finish_reason == "length":
                            warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=True,
                **self.formatted_params
            )
            res = ''
            for chunk in response:
                if len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content is not None:
                        res += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end="", flush=True)
                    if chunk.choices[0].finish_reason == "length":
                        warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

            print('\n')
            return self.config.postprocess_response(res)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=False,
                **self.formatted_params
            )
            res = response.choices[0].message.content
            return self.config.postprocess_response(res)
    

    async def chat_async(self, messages:List[Dict[str,str]]) -> str:
        """
        Async version of chat method. Streaming is not supported.
        """
        processed_messages = self.config.preprocess_messages(messages)

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=processed_messages,
            stream=False,
            **self.formatted_params
        )
        
        if response.choices[0].finish_reason == "length":
            warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

        res = response.choices[0].message.content
        return self.config.postprocess_response(res)
    

class AzureOpenAIInferenceEngine(OpenAIInferenceEngine):
    def __init__(self, model:str, api_version:str, config:LLMConfig=None, **kwrs):
        """
        The Azure OpenAI API inference engine.
        For parameters and documentation, refer to 
        - https://azure.microsoft.com/en-us/products/ai-services/openai-service
        - https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart
        
        Parameters:
        ----------
        model : str
            model name as described in https://platform.openai.com/docs/models
        api_version : str
            the Azure OpenAI API version
        config : LLMConfig
            the LLM configuration.
        """
        if importlib.util.find_spec("openai") is None:
            raise ImportError("OpenAI Python API library not found. Please install OpanAI (```pip install openai```).")
        
        from openai import AzureOpenAI, AsyncAzureOpenAI
        self.model = model
        self.api_version = api_version
        self.client = AzureOpenAI(api_version=self.api_version, 
                                  **kwrs)
        self.async_client = AsyncAzureOpenAI(api_version=self.api_version, 
                                             **kwrs)
        self.config = config if config else BasicLLMConfig()
        self.formatted_params = self._format_config()

    
class LiteLLMInferenceEngine(InferenceEngine):
    def __init__(self, model:str=None, base_url:str=None, api_key:str=None, config:LLMConfig=None):
        """
        The LiteLLM inference engine. 
        For parameters and documentation, refer to https://github.com/BerriAI/litellm?tab=readme-ov-file

        Parameters:
        ----------
        model : str
            the model name
        base_url : str, Optional
            the base url for the LLM server
        api_key : str, Optional
            the API key for the LLM server
        config : LLMConfig
            the LLM configuration.
        """
        if importlib.util.find_spec("litellm") is None:
            raise ImportError("litellm not found. Please install litellm (```pip install litellm```).")
        
        import litellm 
        self.litellm = litellm
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.config = config if config else BasicLLMConfig()
        self.formatted_params = self._format_config()

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["max_tokens"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False) -> Union[str, Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"} 
        verbose : bool, Optional
            if True, VLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.
        """
        processed_messages = self.config.preprocess_messages(messages)
        
        if stream:
            def _stream_generator():
                response_stream = self.litellm.completion(
                    model=self.model,
                    messages=processed_messages,
                    stream=True,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    **self.formatted_params
                )

                for chunk in response_stream:
                    chunk_content = chunk.get('choices')[0].get('delta').get('content')
                    if chunk_content:
                        yield chunk_content

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            response = self.litellm.completion(
                model=self.model,
                messages=processed_messages,
                stream=True,
                base_url=self.base_url,
                api_key=self.api_key,
                **self.formatted_params
            )

            res = ''
            for chunk in response:
                chunk_content = chunk.get('choices')[0].get('delta').get('content')
                if chunk_content:
                    res += chunk_content
                    print(chunk_content, end='', flush=True)

            return self.config.postprocess_response(res)
        
        else:
            response = self.litellm.completion(
                    model=self.model,
                    messages=processed_messages,
                    stream=False,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    **self.formatted_params
                )
            res = response.choices[0].message.content
            return self.config.postprocess_response(res)
    
    async def chat_async(self, messages:List[Dict[str,str]]) -> str:
        """
        Async version of chat method. Streaming is not supported.
        """
        processed_messages = self.config.preprocess_messages(messages)

        response = await self.litellm.acompletion(
            model=self.model,
            messages=processed_messages,
            stream=False,
            base_url=self.base_url,
            api_key=self.api_key,
            **self.formatted_params
        )
        
        res = response.get('choices')[0].get('message').get('content')
        return self.config.postprocess_response(res)