import abc
import os
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
    def postprocess_response(self, response:Union[str, Dict[str, str], Generator[str, None, None]]) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
        """
        This method postprocesses the LLM response after it is generated.

        Parameters:
        ----------
        response : Union[str, Dict[str, str], Generator[Dict[str, str], None, None]]
            the LLM response. Can be a dict or a generator. 
        
        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
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
        return messages.copy()

    def postprocess_response(self, response:Union[str, Dict[str, str], Generator[str, None, None]]) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
        """
        This method postprocesses the LLM response after it is generated.

        Parameters:
        ----------
        response : Union[str, Dict[str, str], Generator[str, None, None]]
            the LLM response. Can be a string or a generator.
        
        Returns: Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            the postprocessed LLM response. 
            If input is a string, the output will be a dict {"response": <response>}. 
            if input is a generator, the output will be a generator {"type": "response", "data": <content>}.
        """
        if isinstance(response, str):
            return {"response": response}
        
        elif isinstance(response, dict):
            if "response" in response:
                return response
            else:
                warnings.warn(f"Invalid response dict keys: {response.keys()}. Returning default empty dict.", UserWarning)
                return {"response": ""}

        elif isinstance(response, Generator):
            def _process_stream():
                for chunk in response:
                    if isinstance(chunk, dict):
                        yield chunk
                    elif isinstance(chunk, str):
                        yield {"type": "response", "data": chunk}

            return _process_stream()

        else:
            warnings.warn(f"Invalid response type: {type(response)}. Returning default empty dict.", UserWarning)
            return {"response": ""}

class ReasoningLLMConfig(LLMConfig):
    def __init__(self, thinking_token_start="<think>", thinking_token_end="</think>", **kwargs):
        """
        The general LLM configuration for reasoning models.
        """
        super().__init__(**kwargs)
        self.thinking_token_start = thinking_token_start
        self.thinking_token_end = thinking_token_end

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
        return messages.copy()

    def postprocess_response(self, response:Union[str, Dict[str, str], Generator[str, None, None]]) -> Union[Dict[str,str], Generator[Dict[str,str], None, None]]:
        """
        This method postprocesses the LLM response after it is generated.
        1. If input is a string, it will extract the reasoning and response based on the thinking tokens.
        2. If input is a dict, it should contain keys "reasoning" and "response". This is for inference engines that already parse reasoning and response.
        3. If input is a generator, 
            a. if the chunk is a dict, it should contain keys "type" and "data". This is for inference engines that already parse reasoning and response.
            b. if the chunk is a string, it will yield dicts with keys "type" and "data" based on the thinking tokens.

        Parameters:
        ----------
        response : Union[str, Generator[str, None, None]]
            the LLM response. Can be a string or a generator.
        
        Returns:
        -------
        response : Union[str, Generator[str, None, None]]
            the postprocessed LLM response as a dict {"reasoning": <reasoning>, "response": <content>}
            if input is a generator, the output will be a generator {"type": <reasoning or response>, "data": <content>}.
        """
        if isinstance(response, str):
            # get contents between thinking_token_start and thinking_token_end
            pattern = f"{re.escape(self.thinking_token_start)}(.*?){re.escape(self.thinking_token_end)}"
            match = re.search(pattern, response, re.DOTALL)
            reasoning = match.group(1) if match else ""
            # get response AFTER thinking_token_end
            response = re.sub(f".*?{self.thinking_token_end}", "", response, flags=re.DOTALL).strip()
            return {"reasoning": reasoning, "response": response}

        elif isinstance(response, dict):
            if "reasoning" in response and "response" in response:
                return response
            else:
                warnings.warn(f"Invalid response dict keys: {response.keys()}. Returning default empty dict.", UserWarning)
                return {"reasoning": "", "response": ""}

        elif isinstance(response, Generator):
            def _process_stream():
                think_flag = False
                buffer = ""
                for chunk in response:
                    if isinstance(chunk, dict):
                        yield chunk

                    elif isinstance(chunk, str):
                        buffer += chunk
                        # switch between reasoning and response
                        if self.thinking_token_start in buffer:
                            think_flag = True
                            buffer = buffer.replace(self.thinking_token_start, "")
                        elif self.thinking_token_end in buffer:
                            think_flag = False
                            buffer = buffer.replace(self.thinking_token_end, "")
                        
                        # if chunk is in thinking block, tag it as reasoning; else tag it as response
                        if chunk not in [self.thinking_token_start, self.thinking_token_end]:
                            if think_flag:
                                yield {"type": "reasoning", "data": chunk}
                            else:
                                yield {"type": "response", "data": chunk}

            return _process_stream()
        
        else:
            warnings.warn(f"Invalid response type: {type(response)}. Returning default empty dict.", UserWarning)
            return {"reasoning": "", "response": ""}

class Qwen3LLMConfig(ReasoningLLMConfig):
    def __init__(self, thinking_mode:bool=True, **kwargs):
        """
        The Qwen3 **hybrid thinking** LLM configuration. 
        For Qwen3 thinking 2507, use ReasoningLLMConfig instead; for Qwen3 Instruct, use BasicLLMConfig instead.

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


class OpenAIReasoningLLMConfig(ReasoningLLMConfig):
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


class MessagesLogger:
    def __init__(self):
        """
        This class is used to log the messages for InferenceEngine.chat().
        """
        self.messages_log = []

    def log_messages(self, messages : List[Dict[str,str]]):
        """
        This method logs the messages to a list.
        """
        self.messages_log.append(messages)

    def get_messages_log(self) -> List[List[Dict[str,str]]]:
        """
        This method returns a copy of the current messages log
        """
        return self.messages_log.copy()
    
    def clear_messages_log(self):
        """
        This method clears the current messages log
        """
        self.messages_log.clear()


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

    def get_messages_log(self) -> List[List[Dict[str,str]]]:
        return self.messages_log.copy()

    def clear_messages_log(self):
        self.messages_log = []


    @abc.abstractmethod
    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, 
             messages_logger:MessagesLogger=None) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
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
        Messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
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
        super().__init__(config)
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

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, messages_logger:MessagesLogger=None) -> Dict[str,str]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.
        """
        # Preprocess messages
        processed_messages = self.config.preprocess_messages(messages)
        # Generate response
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
        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict


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
        super().__init__(config)
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

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, 
             messages_logger:MessagesLogger=None) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
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
        Messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
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
                res_text = ""
                for chunk in response_stream:
                    content_chunk = chunk.get('message', {}).get('content')
                    if content_chunk:
                        res_text += content_chunk
                        yield content_chunk
                
                # Postprocess response
                res_dict = self.config.postprocess_response(res_text)
                # Write to messages log
                if messages_logger:
                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "reasoning": res_dict.get("reasoning", "")})
                    messages_logger.log_messages(processed_messages)

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

        else:
            response = self.client.chat(
                                model=self.model_name, 
                                messages=processed_messages, 
                                options=options,
                                stream=False,
                                keep_alive=self.keep_alive
                            )
            res = response.get('message', {}).get('content')
        
        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict
        

    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None) -> Dict[str,str]:
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
        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                        "content": res_dict.get("response", ""), 
                                        "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict


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
        super().__init__(config)
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


    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, 
             messages_logger:MessagesLogger=None) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
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
        messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.
            
        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
        """
        processed_messages = self.config.preprocess_messages(messages)

        if stream:
            def _stream_generator():
                response_stream = self.client.chat.completions.create(
                                    messages=processed_messages,
                                    stream=True,
                                    **self.formatted_params
                                )
                res_text = ""
                for chunk in response_stream:
                    content_chunk = chunk.get('choices')[0].get('delta').get('content')
                    if content_chunk:
                        res_text += content_chunk
                        yield content_chunk

                # Postprocess response
                res_dict = self.config.postprocess_response(res_text)
                # Write to messages log
                if messages_logger:
                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "reasoning": res_dict.get("reasoning", "")})
                    messages_logger.log_messages(processed_messages)

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

        
        else:
            response = self.client.chat.completions.create(
                                messages=processed_messages,
                                stream=False,
                                **self.formatted_params
                            )
            res = response.choices[0].message.content

        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict
    
    
    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None) -> Dict[str,str]:
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
        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                        "content": res_dict.get("response", ""), 
                                        "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict


class OpenAICompatibleInferenceEngine(InferenceEngine):
    def __init__(self, model:str, api_key:str, base_url:str, config:LLMConfig=None, **kwrs):
        """
        General OpenAI-compatible server inference engine.
        https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

        For parameters and documentation, refer to https://platform.openai.com/docs/api-reference/introduction

        Parameters:
        ----------
        model_name : str
            model name as shown in the vLLM server
        api_key : str
            the API key for the vLLM server.
        base_url : str
            the base url for the vLLM server. 
        config : LLMConfig
            the LLM configuration.
        """
        if importlib.util.find_spec("openai") is None:
            raise ImportError("OpenAI Python API library not found. Please install OpanAI (```pip install openai```).")
        
        from openai import OpenAI, AsyncOpenAI
        from openai.types.chat import ChatCompletionChunk
        self.ChatCompletionChunk = ChatCompletionChunk
        super().__init__(config)
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwrs)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwrs)
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
    
    @abc.abstractmethod
    def _format_response(self, response: Any) -> Dict[str, str]:
        """
        This method format the response from OpenAI API to a dict with keys "reasoning" and "response".

        Parameters:
        ----------
        response : Any
            the response from OpenAI-compatible API. Could be a dict, generator, or object.
        """
        return NotImplemented

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, messages_logger:MessagesLogger=None) -> Union[Dict[str, str], Generator[Dict[str, str], None, None]]:
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
        messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
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
                res_text = ""
                for chunk in response_stream:
                    if len(chunk.choices) > 0:
                        chunk_dict = self._format_response(chunk)
                        yield chunk_dict

                        res_text += chunk_dict["data"]
                        if chunk.choices[0].finish_reason == "length":
                            warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

                # Postprocess response
                res_dict = self.config.postprocess_response(res_text)
                # Write to messages log
                if messages_logger:
                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "reasoning": res_dict.get("reasoning", "")})
                    messages_logger.log_messages(processed_messages)

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=True,
                **self.formatted_params
            )
            res = {"reasoning": "", "response": ""}
            for chunk in response:
                if len(chunk.choices) > 0:
                    chunk_dict = self._format_response(chunk)
                    chunk_text = chunk_dict["data"]
                    res[chunk_dict["type"]] += chunk_text

                    print(chunk_text, end="", flush=True)
                    if chunk.choices[0].finish_reason == "length":
                        warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

            print('\n')

        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=False,
                **self.formatted_params
            )
            res = self._format_response(response)

            if response.choices[0].finish_reason == "length":
                warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)
            
        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict
    

    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None) -> Dict[str,str]:
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

        res = self._format_response(response)

        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                        "content": res_dict.get("response", ""), 
                                        "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict


class VLLMInferenceEngine(OpenAICompatibleInferenceEngine):
    def __init__(self, model:str, api_key:str="", base_url:str="http://localhost:8000/v1", config:LLMConfig=None, **kwrs):
        """
        vLLM OpenAI compatible server inference engine.
        https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

        For parameters and documentation, refer to https://platform.openai.com/docs/api-reference/introduction

        Parameters:
        ----------
        model_name : str
            model name as shown in the vLLM server
        api_key : str, Optional
            the API key for the vLLM server.
        base_url : str, Optional
            the base url for the vLLM server. 
        config : LLMConfig
            the LLM configuration.
        """
        super().__init__(model, api_key, base_url, config, **kwrs)


    def _format_response(self, response: Any) -> Dict[str, str]:
        """
        This method format the response from OpenAI API to a dict with keys "reasoning" and "response".

        Parameters:
        ----------
        response : Any
            the response from OpenAI-compatible API. Could be a dict, generator, or object.
        """
        if isinstance(response, self.ChatCompletionChunk):
            if hasattr(response.choices[0].delta, "reasoning_content") and getattr(response.choices[0].delta, "reasoning_content") is not None:
                chunk_text = getattr(response.choices[0].delta, "reasoning_content", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "reasoning", "data": chunk_text}
            else:
                chunk_text = getattr(response.choices[0].delta, "content", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "response", "data": chunk_text}

        return {"reasoning": getattr(response.choices[0].message, "reasoning_content", ""),
                "response": getattr(response.choices[0].message, "content", "")}
        

class OpenRouterInferenceEngine(OpenAICompatibleInferenceEngine):
    def __init__(self, model:str, api_key:str=None, base_url:str="https://openrouter.ai/api/v1", config:LLMConfig=None, **kwrs):
        """
        OpenRouter OpenAI-compatible server inference engine.

        Parameters:
        ----------
        model_name : str
            model name as shown in the vLLM server
        api_key : str, Optional
            the API key for the vLLM server. If None, will use the key in os.environ['OPENROUTER_API_KEY'].
        base_url : str, Optional
            the base url for the vLLM server. 
        config : LLMConfig
            the LLM configuration.
        """
        super().__init__(model, api_key, base_url, config, **kwrs)
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = os.getenv("OPENROUTER_API_KEY")

    def _format_response(self, response: Any) -> Dict[str, str]:
        """
        This method format the response from OpenAI API to a dict with keys "reasoning" and "response".

        Parameters:
        ----------
        response : Any
            the response from OpenAI-compatible API. Could be a dict, generator, or object.
        """
        if isinstance(response, self.ChatCompletionChunk):
            if hasattr(response.choices[0].delta, "reasoning") and getattr(response.choices[0].delta, "reasoning") is not None:
                chunk_text = getattr(response.choices[0].delta, "reasoning", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "reasoning", "data": chunk_text}
            else:
                chunk_text = getattr(response.choices[0].delta, "content", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "response", "data": chunk_text}

        return {"reasoning": getattr(response.choices[0].message, "reasoning", ""),
                "response": getattr(response.choices[0].message, "content", "")}


class OpenAIInferenceEngine(InferenceEngine):
    def __init__(self, model:str, config:LLMConfig=None, **kwrs):
        """
        The OpenAI API inference engine. 
        For parameters and documentation, refer to https://platform.openai.com/docs/api-reference/introduction

        Parameters:
        ----------
        model_name : str
            model name as described in https://platform.openai.com/docs/models
        """
        if importlib.util.find_spec("openai") is None:
            raise ImportError("OpenAI Python API library not found. Please install OpanAI (```pip install openai```).")
        
        from openai import OpenAI, AsyncOpenAI
        super().__init__(config)
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

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, messages_logger:MessagesLogger=None) -> Union[Dict[str, str], Generator[Dict[str, str], None, None]]:
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
        messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
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
                res_text = ""
                for chunk in response_stream:
                    if len(chunk.choices) > 0:
                        chunk_text = chunk.choices[0].delta.content
                        if chunk_text is not None:
                            res_text += chunk_text
                            yield chunk_text
                        if chunk.choices[0].finish_reason == "length":
                            warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

                # Postprocess response
                res_dict = self.config.postprocess_response(res_text)
                # Write to messages log
                if messages_logger:
                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "reasoning": res_dict.get("reasoning", "")})
                    messages_logger.log_messages(processed_messages)

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

        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=False,
                **self.formatted_params
            )
            res = response.choices[0].message.content
            
        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict
    

    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None) -> Dict[str,str]:
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
        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict
    

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
        super().__init__(config)
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

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, messages_logger:MessagesLogger=None) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
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
        messages_logger: MessagesLogger, Optional
            a messages logger that logs the messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
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
                res_text = ""
                for chunk in response_stream:
                    chunk_content = chunk.get('choices')[0].get('delta').get('content')
                    if chunk_content:
                        res_text += chunk_content
                        yield chunk_content

                # Postprocess response
                res_dict = self.config.postprocess_response(res_text)
                # Write to messages log
                if messages_logger:
                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "reasoning": res_dict.get("reasoning", "")})
                    messages_logger.log_messages(processed_messages)

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

        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                        "content": res_dict.get("response", ""), 
                                        "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict
    
    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None) -> Dict[str,str]:
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

        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)
        return res_dict
