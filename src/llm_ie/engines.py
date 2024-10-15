import abc
from typing import List, Dict

class InferenceEngine:
    @abc.abstractmethod
    def __init__(self):
        """
        This is an abstract class to provide interfaces for LLM inference engines. 
        Children classes that inherts this class can be used in extrators. Must implement chat() method.
        """
        return NotImplemented


    @abc.abstractmethod
    def chat(self, messages:List[Dict[str,str]], max_new_tokens:int=2048, temperature:float=0.0, stream:bool=False, **kwrs) -> str:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        max_new_tokens : str, Optional
            the max number of new tokens LLM can generate. 
        temperature : float, Optional
            the temperature for token sampling. 
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        """
        return NotImplemented


class LlamaCppInferenceEngine(InferenceEngine):
    def __init__(self, repo_id:str, gguf_filename:str, n_ctx:int=4096, n_gpu_layers:int=-1, **kwrs):
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
        """
        from llama_cpp import Llama
        self.repo_id = repo_id
        self.gguf_filename = gguf_filename
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

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


    def chat(self, messages:List[Dict[str,str]], max_new_tokens:int=2048, temperature:float=0.0, stream:bool=False, **kwrs) -> str:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        max_new_tokens : str, Optional
            the max number of new tokens LLM can generate. 
        temperature : float, Optional
            the temperature for token sampling. 
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        """
        response = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_new_tokens, 
                    temperature=temperature,
                    stream=stream,
                    **kwrs
                )

        if stream:
            res = ''
            for chunk in response:
                out_dict = chunk['choices'][0]['delta']
                if 'content' in out_dict:
                    res += out_dict['content']
                    print(out_dict['content'], end='', flush=True)
            print('\n')
            return res
        
        return response['choices'][0]['message']['content']


class OllamaInferenceEngine(InferenceEngine):
    def __init__(self, model_name:str, num_ctx:int=4096, keep_alive:int=300, **kwrs):
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
        """
        import ollama
        self.ollama = ollama
        self.model_name = model_name
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive

    def chat(self, messages:List[Dict[str,str]], max_new_tokens:int=2048, temperature:float=0.0, stream:bool=False, **kwrs) -> str:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        max_new_tokens : str, Optional
            the max number of new tokens LLM can generate. 
        temperature : float, Optional
            the temperature for token sampling. 
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        """
        response = self.ollama.chat(
                            model=self.model_name, 
                            messages=messages, 
                            options={'temperature':temperature, 'num_ctx': self.num_ctx, 'num_predict': max_new_tokens, **kwrs},
                            stream=stream,
                            keep_alive=self.keep_alive
                        )
        if stream:
            res = ''
            for chunk in response:
                res += chunk['message']['content']
                print(chunk['message']['content'], end='', flush=True)
            print('\n')
            return res
        
        return response['message']['content']


class HuggingFaceHubInferenceEngine(InferenceEngine):
    def __init__(self, **kwrs):
        """
        The Huggingface_hub InferenceClient inference engine.
        For parameters and documentation, refer to https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client
        """
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(**kwrs)

    def chat(self, messages:List[Dict[str,str]], max_new_tokens:int=2048, temperature:float=0.0, stream:bool=False, **kwrs) -> str:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        max_new_tokens : str, Optional
            the max number of new tokens LLM can generate. 
        temperature : float, Optional
            the temperature for token sampling. 
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        """
        response = self.client.chat.completions.create(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stream=stream,
                    **kwrs
                )
        
        if stream:
            res = ''
            for chunk in response:
                res += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end='', flush=True)
            return res
        
        return response.choices[0].message.content
        

class OpenAIInferenceEngine(InferenceEngine):
    def __init__(self, model:str, **kwrs):
        """
        The OpenAI API inference engine.
        For parameters and documentation, refer to https://platform.openai.com/docs/api-reference/introduction

        Parameters:
        ----------
        model_name : str
            model name as described in https://platform.openai.com/docs/models
        """
        from openai import OpenAI
        self.client = OpenAI(**kwrs)
        self.model = model

    def chat(self, messages:List[Dict[str,str]], max_new_tokens:int=2048, temperature:float=0.0, stream:bool=False, **kwrs) -> str:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        max_new_tokens : str, Optional
            the max number of new tokens LLM can generate. 
        temperature : float, Optional
            the temperature for token sampling. 
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=stream,
            **kwrs
        )

        if stream:
            res = ''
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    res += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)
            return res
        
        return response.choices[0].message.content