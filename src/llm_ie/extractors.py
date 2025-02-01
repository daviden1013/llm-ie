import abc
import re
import copy
import json
import json_repair
import inspect
import importlib.resources
import warnings
import itertools
import asyncio
import nest_asyncio
from typing import Set, List, Dict, Tuple, Union, Callable
from llm_ie.data_types import LLMInformationExtractionFrame, LLMInformationExtractionDocument
from llm_ie.engines import InferenceEngine
from colorama import Fore, Style
from nltk.tokenize import RegexpTokenizer


class Extractor:
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None, **kwrs):
        """
        This is the abstract class for (frame and relation) extractors.
        Input LLM inference engine, system prompt (optional), prompt template (with instruction, few-shot examples).

        Parameters:
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        system_prompt : str, Optional
            system prompt.
        """
        self.inference_engine = inference_engine
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt


    @classmethod
    def get_prompt_guide(cls) -> str:
        """
        This method returns the pre-defined prompt guideline for the extractor from the package asset.
        """
        # Check if the prompt guide is available
        file_path = importlib.resources.files('llm_ie.asset.prompt_guide').joinpath(f"{cls.__name__}_prompt_guide.txt")
        try:     
            with open(file_path, 'r', encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            warnings.warn(f"Prompt guide for {cls.__name__} is not available. Is it a customed extractor?", UserWarning)
            return None

    def _get_user_prompt(self, text_content:Union[str, Dict[str,str]]) -> str:
        """
        This method applies text_content to prompt_template and returns a prompt.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}. All values must be str.

        Returns : str
            a user prompt.
        """
        pattern = re.compile(r'{{(.*?)}}')
        if isinstance(text_content, str):
            matches = pattern.findall(self.prompt_template)
            if len(matches) != 1:
                raise ValueError("When text_content is str, the prompt template must has exactly 1 placeholder {{<placeholder name>}}.")
            text = re.sub(r'\\', r'\\\\', text_content)
            prompt = pattern.sub(text, self.prompt_template)

        elif isinstance(text_content, dict):
            # Check if all values are str
            if not all([isinstance(v, str) for v in text_content.values()]):
                raise ValueError("All values in text_content must be str.")
            # Check if all keys are in the prompt template
            placeholders = pattern.findall(self.prompt_template)
            if len(placeholders) != len(text_content):
                raise ValueError(f"Expect text_content ({len(text_content)}) and prompt template placeholder ({len(placeholders)}) to have equal size.")
            if not all([k in placeholders for k, _ in text_content.items()]):
                raise ValueError(f"All keys in text_content ({text_content.keys()}) must match placeholders in prompt template ({placeholders}).")

            prompt = pattern.sub(lambda match: re.sub(r'\\', r'\\\\', text_content[match.group(1)]), self.prompt_template)

        return prompt
    
    def _find_dict_strings(self, text: str) -> List[str]:
        """
        Extracts balanced JSON-like dictionaries from a string, even if nested.

        Parameters:
        -----------
        text : str
            the input text containing JSON-like structures.

        Returns : List[str]
            A list of valid JSON-like strings representing dictionaries.
        """
        open_brace = 0
        start = -1
        json_objects = []

        for i, char in enumerate(text):
            if char == '{':
                if open_brace == 0:
                    # start of a new JSON object
                    start = i 
                open_brace += 1
            elif char == '}':
                open_brace -= 1
                if open_brace == 0 and start != -1:
                    json_objects.append(text[start:i + 1])
                    start = -1

        return json_objects
    
    
    def _extract_json(self, gen_text:str) -> List[Dict[str, str]]:
        """ 
        This method inputs a generated text and output a JSON of information tuples
        """
        out = []
        dict_str_list = self._find_dict_strings(gen_text)
        for dict_str in dict_str_list:
            try:
                dict_obj = json.loads(dict_str)
                out.append(dict_obj)
            except json.JSONDecodeError:
                dict_obj = json_repair.repair_json(dict_str, skip_json_loads=True, return_objects=True)
                if dict_obj:
                    warnings.warn(f'JSONDecodeError detected, fixed with repair_json:\n{dict_str}', RuntimeWarning)
                    out.append(dict_obj)
                else:
                    warnings.warn(f'JSONDecodeError could not be fixed:\n{dict_str}', RuntimeWarning)
        return out
    

class FrameExtractor(Extractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None, **kwrs):
        """
        This is the abstract class for frame extraction.
        Input LLM inference engine, system prompt (optional), prompt template (with instruction, few-shot examples).

        Parameters:
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine,
                         prompt_template=prompt_template,
                         system_prompt=system_prompt,
                         **kwrs)
        self.tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')


    def _jaccard_score(self, s1:Set[str], s2:Set[str]) -> float:
        """
        This method calculates the Jaccard score between two sets of word tokens.
        """
        return len(s1.intersection(s2)) / len(s1.union(s2))


    def _get_word_tokens(self, text) -> Tuple[List[str], List[Tuple[int]]]:
        """
        This method tokenizes the input text into a list of word tokens and their spans.
        """
        tokens = []
        spans = []
        for span in self.tokenizer.span_tokenize(text):
            spans.append(span)
            start, end = span
            tokens.append(text[start:end])
        return tokens, spans


    def _get_closest_substring(self, text:str, pattern:str, buffer_size:float=0.2) -> Tuple[Tuple[int, int], float]:
        """
        This method finds the closest (highest Jaccard score) substring in text that matches the pattern.
        the substring must start with the same word token as the pattern. This is due to the observation that 
        LLM often generate the first few words consistently. 

        Parameters:
        ----------
        text : str
            the input text.
        pattern : str
            the pattern to match.
        buffer_size : float, Optional
            the buffer size for the matching window. Default is 20% of pattern length.

        Returns : Tuple[Tuple[int, int], float]
            a tuple of 2-tuple span and Jaccard score.
        """
        if not text or not pattern:
            return None, 0

        text_tokens, text_spans = self._get_word_tokens(text)
        pattern_tokens, _ = self._get_word_tokens(pattern)
        pattern_tokens_set = set(pattern_tokens)
        window_size = len(pattern_tokens)
        window_size_min = max(1, int(window_size * (1 - buffer_size)))
        window_size_max = int(window_size * (1 + buffer_size)) + 1
        closest_substring_span = None
        best_score = 0
        
        for i in range(len(text_tokens) - window_size_max):
            for w in range(window_size_min, window_size_max): 
                sub_str_tokens = text_tokens[i:i + w]
                if len(sub_str_tokens) > 0 and sub_str_tokens[0] == pattern_tokens[0]:
                    score = self._jaccard_score(set(sub_str_tokens), pattern_tokens_set)
                    if score > best_score:
                        best_score = score
                        sub_string_word_spans = text_spans[i:i + w]
                        closest_substring_span = (sub_string_word_spans[0][0], sub_string_word_spans[-1][-1])

        return closest_substring_span, best_score


    def _find_entity_spans(self, text: str, entities: List[str], case_sensitive:bool=False, 
                           fuzzy_match:bool=True, fuzzy_buffer_size:float=0.2, fuzzy_score_cutoff:float=0.8) -> List[Tuple[int]]:
        """
        This function inputs a text and a list of entity text, 
        outputs a list of spans (2-tuple) for each entity.
        Entities that are not found in the text will be None from output.

        Parameters:
        ----------
        text : str
            text that contains entities
        entities : List[str]
            a list of entity text to find in the text
        case_sensitive : bool, Optional
            if True, entity text matching will be case-sensitive.
        fuzzy_match : bool, Optional
            if True, fuzzy matching will be applied to find entity text.
        fuzzy_buffer_size : float, Optional
            the buffer size for fuzzy matching. Default is 20% of entity text length.
        fuzzy_score_cutoff : float, Optional
            the Jaccard score cutoff for fuzzy matching. 
            Matched entity text must have a score higher than this value or a None will be returned.
        """
        # Handle case sensitivity
        if not case_sensitive:
            text = text.lower()

        # Match entities
        entity_spans = []
        for entity in entities:   
            if not isinstance(entity, str):  
                entity_spans.append(None) 
                continue    
            if not case_sensitive:
                entity = entity.lower()

            # Exact match  
            match = re.search(re.escape(entity), text)
            if match and entity:
                start, end = match.span()
                entity_spans.append((start, end))
                # Replace the found entity with spaces to avoid finding the same instance again
                text = text[:start] + ' ' * (end - start) + text[end:]
            # Fuzzy match
            elif fuzzy_match:
                closest_substring_span, best_score = self._get_closest_substring(text, entity, buffer_size=fuzzy_buffer_size)
                if closest_substring_span and best_score >= fuzzy_score_cutoff:
                    entity_spans.append(closest_substring_span)
                    # Replace the found entity with spaces to avoid finding the same instance again
                    text = text[:closest_substring_span[0]] + ' ' * (closest_substring_span[1] - closest_substring_span[0]) + text[closest_substring_span[1]:]
                else:
                    entity_spans.append(None)

            # No match
            else:
                entity_spans.append(None)

        return entity_spans

    @abc.abstractmethod
    def extract(self, text_content:Union[str, Dict[str,str]], max_new_tokens:int=2048, **kwrs) -> str:
        """
        This method inputs text content and outputs a string generated by LLM

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        max_new_tokens : str, Optional
            the max number of new tokens LLM can generate. 

        Return : str
            the output from LLM. Need post-processing.
        """
        return NotImplemented
    
    
    @abc.abstractmethod
    def extract_frames(self, text_content:Union[str, Dict[str,str]], entity_key:str, max_new_tokens:int=2048, 
                       document_key:str=None, **kwrs) -> List[LLMInformationExtractionFrame]:
        """
        This method inputs text content and outputs a list of LLMInformationExtractionFrame
        It use the extract() method and post-process outputs into frames.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        entity_key : str
            the key (in ouptut JSON) for entity text. Any extraction that does not include entity key will be dropped.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.

        Return : str
            a list of frames.
        """
        return NotImplemented
    

class BasicFrameExtractor(FrameExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None, **kwrs):
        """
        This class diretly prompt LLM for frame extraction.
        Input system prompt (optional), prompt template (with instruction, few-shot examples), 
        and specify a LLM.

        Parameters:
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine, 
                         prompt_template=prompt_template, 
                         system_prompt=system_prompt, 
                         **kwrs)
        

    def extract(self, text_content:Union[str, Dict[str,str]], max_new_tokens:int=2048, 
                temperature:float=0.0, stream:bool=False, **kwrs) -> str:
        """
        This method inputs a text and outputs a string generated by LLM.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        max_new_tokens : str, Optional
            the max number of new tokens LLM can generate. 
        temperature : float, Optional
            the temperature for token sampling. 
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : str
            the output from LLM. Need post-processing.
        """
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': self._get_user_prompt(text_content)})
        response = self.inference_engine.chat(
                    messages=messages,
                    max_new_tokens=max_new_tokens, 
                    temperature=temperature,
                    stream=stream,
                    **kwrs
                )
        
        return response
    

    def extract_frames(self, text_content:Union[str, Dict[str,str]], entity_key:str, max_new_tokens:int=2048, 
                       temperature:float=0.0, document_key:str=None, stream:bool=False,
                       case_sensitive:bool=False, fuzzy_match:bool=True, fuzzy_buffer_size:float=0.2, 
                       fuzzy_score_cutoff:float=0.8, **kwrs) -> List[LLMInformationExtractionFrame]:
        """
        This method inputs a text and outputs a list of LLMInformationExtractionFrame
        It use the extract() method and post-process outputs into frames.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        entity_key : str
            the key (in ouptut JSON) for entity text. Any extraction that does not include entity key will be dropped.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time.
        case_sensitive : bool, Optional
            if True, entity text matching will be case-sensitive.
        fuzzy_match : bool, Optional
            if True, fuzzy matching will be applied to find entity text.
        fuzzy_buffer_size : float, Optional
            the buffer size for fuzzy matching. Default is 20% of entity text length.
        fuzzy_score_cutoff : float, Optional
            the Jaccard score cutoff for fuzzy matching. 
            Matched entity text must have a score higher than this value or a None will be returned.

        Return : str
            a list of frames.
        """
        if isinstance(text_content, str):
            text = text_content
        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            text = text_content[document_key]
        
        frame_list = []
        gen_text = self.extract(text_content=text_content, 
                                max_new_tokens=max_new_tokens, 
                                temperature=temperature, 
                                stream=stream,
                                **kwrs)

        entity_json = []
        for entity in self._extract_json(gen_text=gen_text):
            if entity_key in entity:
                entity_json.append(entity)
            else:
                warnings.warn(f'Extractor output "{entity}" does not have entity_key ("{entity_key}"). This frame will be dropped.', RuntimeWarning)

        spans = self._find_entity_spans(text=text, 
                                        entities=[e[entity_key] for e in entity_json], 
                                        case_sensitive=case_sensitive,
                                        fuzzy_match=fuzzy_match,
                                        fuzzy_buffer_size=fuzzy_buffer_size,
                                        fuzzy_score_cutoff=fuzzy_score_cutoff)
        
        for i, (ent, span) in enumerate(zip(entity_json, spans)):
            if span is not None:
                start, end = span
                frame = LLMInformationExtractionFrame(frame_id=f"{i}", 
                            start=start,
                            end=end,
                            entity_text=text[start:end],
                            attr={k: v for k, v in ent.items() if k != entity_key and v != ""})
                frame_list.append(frame)
        return frame_list
        

class ReviewFrameExtractor(BasicFrameExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, 
                 review_mode:str, review_prompt:str=None,system_prompt:str=None, **kwrs):
        """
        This class add a review step after the BasicFrameExtractor.
        The Review process asks LLM to review its output and:
            1. add more frames while keep current. This is efficient for boosting recall. 
            2. or, regenerate frames (add new and delete existing). 
        Use the review_mode parameter to specify. Note that the review_prompt should instruct LLM accordingly.

        Parameters:
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        review_prompt : str: Optional
            the prompt text that ask LLM to review. Specify addition or revision in the instruction.
            if not provided, a default review prompt will be used. 
        review_mode : str
            review mode. Must be one of {"addition", "revision"}
            addition mode only ask LLM to add new frames, while revision mode ask LLM to regenerate.
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine, prompt_template=prompt_template, 
                         system_prompt=system_prompt, **kwrs)
        if review_mode not in {"addition", "revision"}: 
            raise ValueError('review_mode must be one of {"addition", "revision"}.')
        self.review_mode = review_mode

        if review_prompt:
            self.review_prompt = review_prompt
        else:
            file_path = importlib.resources.files('llm_ie.asset.default_prompts').\
                joinpath(f"{self.__class__.__name__}_{self.review_mode}_review_prompt.txt")
            with open(file_path, 'r', encoding="utf-8") as f:
                self.review_prompt = f.read()

            warnings.warn(f'Custom review prompt not provided. The default review prompt is used:\n"{self.review_prompt}"', UserWarning)


    def extract(self, text_content:Union[str, Dict[str,str]], 
                max_new_tokens:int=4096, temperature:float=0.0, stream:bool=False, **kwrs) -> str:
        """
        This method inputs a text and outputs a string generated by LLM.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        max_new_tokens : str, Optional
            the max number of new tokens LLM can generate. 
        temperature : float, Optional
            the temperature for token sampling. 
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : str
            the output from LLM. Need post-processing.
        """
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': self._get_user_prompt(text_content)})
        # Initial output
        if stream:
            print(f"{Fore.BLUE}Initial Output:{Style.RESET_ALL}")

        initial = self.inference_engine.chat(
                        messages=messages,
                        max_new_tokens=max_new_tokens, 
                        temperature=temperature,
                        stream=stream,
                        **kwrs
                    )

        # Review
        messages.append({'role': 'assistant', 'content': initial})
        messages.append({'role': 'user', 'content': self.review_prompt})

        if stream:
            print(f"\n{Fore.YELLOW}Review:{Style.RESET_ALL}")
        review = self.inference_engine.chat(
                        messages=messages, 
                        max_new_tokens=max_new_tokens, 
                        temperature=temperature,
                        stream=stream,
                        **kwrs
                    )

        # Output
        if self.review_mode == "revision":
            return review
        elif self.review_mode == "addition":
            return initial + '\n' + review


class SentenceFrameExtractor(FrameExtractor):
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None, **kwrs):
        """
        This class performs sentence-by-sentence information extraction.
        The process is as follows:
            1. system prompt (optional)
            2. user prompt with instructions (schema, background, full text, few-shot example...)
            3. feed a sentence (start with first sentence)
            4. LLM extract entities and attributes from the sentence
            5. repeat #3 and #4

        Input system prompt (optional), prompt template (with user instructions), 
        and specify a LLM.

        Parameters:
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine, prompt_template=prompt_template, 
                         system_prompt=system_prompt, **kwrs)
        
    def _get_sentences(self, text:str) -> List[Dict[str,str]]:
        """
        This method sentence tokenize the input text into a list of sentences 
        as dict of {start, end, sentence_text}

        Parameters:
        ----------
        text : str
            text to sentence tokenize.

        Returns : List[Dict[str,str]]
            a list of sentences as dict with keys: {"sentence_text", "start", "end"}. 
        """
        sentences = []
        for start, end in self.PunktSentenceTokenizer().span_tokenize(text):
            sentences.append({"sentence_text": text[start:end],
                            "start": start,
                            "end": end})    
        return sentences
    
    
    def extract(self, text_content:Union[str, Dict[str,str]], max_new_tokens:int=512, 
                document_key:str=None, multi_turn:bool=False, temperature:float=0.0, stream:bool=False, **kwrs) -> List[Dict[str,str]]:
        """
        This method inputs a text and outputs a list of outputs per sentence. 

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        multi_turn : bool, Optional
            multi-turn conversation prompting. 
            If True, sentences and LLM outputs will be appended to the input message and carry-over. 
            If False, only the current sentence is prompted. 
            For LLM inference engines that supports prompt cache (e.g., Llama.Cpp, Ollama), use multi-turn conversation prompting
            can better utilize the KV caching. 
        temperature : float, Optional
            the temperature for token sampling.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : str
            the output from LLM. Need post-processing.
        """
        # define output
        output = []
        # sentence tokenization
        if isinstance(text_content, str):
            sentences = self._get_sentences(text_content)
        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            sentences = self._get_sentences(text_content[document_key])
        # construct chat messages
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': self._get_user_prompt(text_content)})
        messages.append({'role': 'assistant', 'content': 'Sure, please start with the first sentence.'})

        # generate sentence by sentence
        for sent in sentences:
            messages.append({'role': 'user', 'content': sent['sentence_text']})
            if stream:
                print(f"\n\n{Fore.GREEN}Sentence: {Style.RESET_ALL}\n{sent['sentence_text']}\n")
                print(f"{Fore.BLUE}Extraction:{Style.RESET_ALL}")

            gen_text = self.inference_engine.chat(
                            messages=messages, 
                            max_new_tokens=max_new_tokens, 
                            temperature=temperature,
                            stream=stream,
                            **kwrs
                        )
            
            if multi_turn:
                # update chat messages with LLM outputs
                messages.append({'role': 'assistant', 'content': gen_text})
            else:
                # delete sentence so that message is reset
                del messages[-1]

            # add to output
            output.append({'sentence_start': sent['start'],
                            'sentence_end': sent['end'],
                            'sentence_text': sent['sentence_text'],
                            'gen_text': gen_text})
        return output
    

    async def extract_async(self, text_content:Union[str, Dict[str,str]], max_new_tokens:int=512, 
                document_key:str=None, temperature:float=0.0, concurrent_batch_size:int=32, **kwrs) -> List[Dict[str,str]]:
        """
        The asynchronous version of the extract() method.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        temperature : float, Optional
            the temperature for token sampling.
        concurrent_batch_size : int, Optional
            the number of sentences to process in concurrent.
        """
        # Check if self.inference_engine.chat_async() is implemented
        if not hasattr(self.inference_engine, 'chat_async'):
            raise NotImplementedError(f"{self.inference_engine.__class__.__name__} does not have chat_async() method.")

        # define output
        output = []
        # sentence tokenization
        if isinstance(text_content, str):
            sentences = self._get_sentences(text_content)
        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            sentences = self._get_sentences(text_content[document_key])
        # construct chat messages
        base_messages = []
        if self.system_prompt:
            base_messages.append({'role': 'system', 'content': self.system_prompt})

        base_messages.append({'role': 'user', 'content': self._get_user_prompt(text_content)})
        base_messages.append({'role': 'assistant', 'content': 'Sure, please start with the first sentence.'})

        # generate sentence by sentence
        tasks = []
        for i in range(0, len(sentences), concurrent_batch_size):
            batch = sentences[i:i + concurrent_batch_size]
            for sent in batch:
                messages = copy.deepcopy(base_messages)
                messages.append({'role': 'user', 'content': sent['sentence_text']})
                task = asyncio.create_task(
                    self.inference_engine.chat_async(
                                messages=messages, 
                                max_new_tokens=max_new_tokens, 
                                temperature=temperature,
                                **kwrs
                            )
                )
                tasks.append(task)

            # Wait until the batch is done, collect results and move on to next batch
            responses = await asyncio.gather(*tasks)

        # Collect outputs
        for gen_text, sent in zip(responses, sentences):
            output.append({'sentence_start': sent['start'],
                            'sentence_end': sent['end'],
                            'sentence_text': sent['sentence_text'],
                            'gen_text': gen_text})
        return output
    

    def extract_frames(self, text_content:Union[str, Dict[str,str]], entity_key:str, max_new_tokens:int=512, 
                            document_key:str=None, multi_turn:bool=False, temperature:float=0.0, stream:bool=False, 
                            concurrent:bool=False, concurrent_batch_size:int=32,
                            case_sensitive:bool=False, fuzzy_match:bool=True, fuzzy_buffer_size:float=0.2, fuzzy_score_cutoff:float=0.8,
                            **kwrs) -> List[LLMInformationExtractionFrame]:
        """
        This method inputs a text and outputs a list of LLMInformationExtractionFrame
        It use the extract() method and post-process outputs into frames.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        entity_key : str
            the key (in ouptut JSON) for entity text.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        multi_turn : bool, Optional
            multi-turn conversation prompting. 
            If True, sentences and LLM outputs will be appended to the input message and carry-over. 
            If False, only the current sentence is prompted. 
            For LLM inference engines that supports prompt cache (e.g., Llama.Cpp, Ollama), use multi-turn conversation prompting
            can better utilize the KV caching. 
        temperature : float, Optional
            the temperature for token sampling.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        concurrent : bool, Optional
            if True, the sentences will be extracted in concurrent.
        concurrent_batch_size : int, Optional
            the number of sentences to process in concurrent. Only used when `concurrent` is True.
        case_sensitive : bool, Optional
            if True, entity text matching will be case-sensitive.
        fuzzy_match : bool, Optional
            if True, fuzzy matching will be applied to find entity text.
        fuzzy_buffer_size : float, Optional
            the buffer size for fuzzy matching. Default is 20% of entity text length.
        fuzzy_score_cutoff : float, Optional
            the Jaccard score cutoff for fuzzy matching. 
            Matched entity text must have a score higher than this value or a None will be returned.

        Return : str
            a list of frames.
        """
        if concurrent:
            if stream:
                warnings.warn("stream=True is not supported in concurrent mode.", RuntimeWarning)
            if multi_turn:
                warnings.warn("multi_turn=True is not supported in concurrent mode.", RuntimeWarning)

            nest_asyncio.apply() # For Jupyter notebook. Terminal does not need this.
            llm_output_sentences = asyncio.run(self.extract_async(text_content=text_content, 
                                        max_new_tokens=max_new_tokens, 
                                        document_key=document_key,
                                        temperature=temperature, 
                                        concurrent_batch_size=concurrent_batch_size,
                                        **kwrs)
                                        )
        else:
            llm_output_sentences = self.extract(text_content=text_content, 
                                            max_new_tokens=max_new_tokens, 
                                            document_key=document_key,
                                            multi_turn=multi_turn, 
                                            temperature=temperature, 
                                            stream=stream,
                                            **kwrs)
        frame_list = []
        for sent in llm_output_sentences:
            entity_json = []
            for entity in self._extract_json(gen_text=sent['gen_text']):
                if entity_key in entity:
                    entity_json.append(entity)
                else:
                    warnings.warn(f'Extractor output "{entity}" does not have entity_key ("{entity_key}"). This frame will be dropped.', RuntimeWarning)

            spans = self._find_entity_spans(text=sent['sentence_text'], 
                                            entities=[e[entity_key] for e in entity_json], 
                                            case_sensitive=case_sensitive,
                                            fuzzy_match=fuzzy_match,
                                            fuzzy_buffer_size=fuzzy_buffer_size,
                                            fuzzy_score_cutoff=fuzzy_score_cutoff)
            for ent, span in zip(entity_json, spans):
                if span is not None:
                    start, end = span
                    entity_text = sent['sentence_text'][start:end]
                    start += sent['sentence_start']
                    end += sent['sentence_start']
                    frame = LLMInformationExtractionFrame(frame_id=f"{len(frame_list)}", 
                                start=start,
                                end=end,
                                entity_text=entity_text,
                                attr={k: v for k, v in ent.items() if k != entity_key and v != ""})
                    frame_list.append(frame)
        return frame_list


class SentenceReviewFrameExtractor(SentenceFrameExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str,  
                 review_mode:str, review_prompt:str=None, system_prompt:str=None, **kwrs):
        """
        This class adds a review step after the SentenceFrameExtractor.
        For each sentence, the review process asks LLM to review its output and:
            1. add more frames while keeping current. This is efficient for boosting recall. 
            2. or, regenerate frames (add new and delete existing). 
        Use the review_mode parameter to specify. Note that the review_prompt should instruct LLM accordingly.

        Parameters:
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        review_prompt : str: Optional
            the prompt text that ask LLM to review. Specify addition or revision in the instruction.
            if not provided, a default review prompt will be used. 
        review_mode : str
            review mode. Must be one of {"addition", "revision"}
            addition mode only ask LLM to add new frames, while revision mode ask LLM to regenerate.
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine, prompt_template=prompt_template, 
                         system_prompt=system_prompt, **kwrs)

        if review_mode not in {"addition", "revision"}: 
            raise ValueError('review_mode must be one of {"addition", "revision"}.')
        self.review_mode = review_mode

        if review_prompt:
            self.review_prompt = review_prompt
        else:
            file_path = importlib.resources.files('llm_ie.asset.default_prompts').\
                joinpath(f"{self.__class__.__name__}_{self.review_mode}_review_prompt.txt")
            with open(file_path, 'r', encoding="utf-8") as f:
                self.review_prompt = f.read()

            warnings.warn(f'Custom review prompt not provided. The default review prompt is used:\n"{self.review_prompt}"', UserWarning)


    def extract(self, text_content:Union[str, Dict[str,str]], max_new_tokens:int=512, 
                document_key:str=None, multi_turn:bool=False, temperature:float=0.0, stream:bool=False, **kwrs) -> List[Dict[str,str]]:
        """
        This method inputs a text and outputs a list of outputs per sentence. 

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        multi_turn : bool, Optional
            multi-turn conversation prompting. 
            If True, sentences and LLM outputs will be appended to the input message and carry-over. 
            If False, only the current sentence is prompted. 
            For LLM inference engines that supports prompt cache (e.g., Llama.Cpp, Ollama), use multi-turn conversation prompting
            can better utilize the KV caching. 
        temperature : float, Optional
            the temperature for token sampling.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : str
            the output from LLM. Need post-processing.
        """
        # define output
        output = []
        # sentence tokenization
        if isinstance(text_content, str):
            sentences = self._get_sentences(text_content)
        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            sentences = self._get_sentences(text_content[document_key])
        # construct chat messages
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': self._get_user_prompt(text_content)})
        messages.append({'role': 'assistant', 'content': 'Sure, please start with the first sentence.'})

        # generate sentence by sentence
        for sent in sentences:
            messages.append({'role': 'user', 'content': sent['sentence_text']})
            if stream:
                print(f"\n\n{Fore.GREEN}Sentence: {Style.RESET_ALL}\n{sent['sentence_text']}\n")
                print(f"{Fore.BLUE}Initial Output:{Style.RESET_ALL}")

            initial = self.inference_engine.chat(
                            messages=messages, 
                            max_new_tokens=max_new_tokens, 
                            temperature=temperature,
                            stream=stream,
                            **kwrs
                        )
            
            # Review
            if stream:
                print(f"\n{Fore.YELLOW}Review:{Style.RESET_ALL}")
            messages.append({'role': 'assistant', 'content': initial})
            messages.append({'role': 'user', 'content': self.review_prompt})

            review = self.inference_engine.chat(
                            messages=messages, 
                            max_new_tokens=max_new_tokens, 
                            temperature=temperature,
                            stream=stream,
                            **kwrs
                        )
            
            # Output
            if self.review_mode == "revision":
                gen_text = review
            elif self.review_mode == "addition":
                gen_text = initial + '\n' + review
            
            if multi_turn:
                # update chat messages with LLM outputs
                messages.append({'role': 'assistant', 'content': review})
            else:
                # delete sentence and review so that message is reset
                del messages[-3:]

            # add to output
            output.append({'sentence_start': sent['start'],
                            'sentence_end': sent['end'],
                            'sentence_text': sent['sentence_text'],
                            'gen_text': gen_text})
        return output
    
    async def extract_async(self, text_content:Union[str, Dict[str,str]], max_new_tokens:int=512, 
                document_key:str=None, temperature:float=0.0, concurrent_batch_size:int=32, **kwrs) -> List[Dict[str,str]]:
        """
        The asynchronous version of the extract() method.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        temperature : float, Optional
            the temperature for token sampling.
        concurrent_batch_size : int, Optional
            the number of sentences to process in concurrent.

        Return : str
            the output from LLM. Need post-processing.
        """
        # Check if self.inference_engine.chat_async() is implemented
        if not hasattr(self.inference_engine, 'chat_async'):
            raise NotImplementedError(f"{self.inference_engine.__class__.__name__} does not have chat_async() method.")
        
        # define output
        output = []
        # sentence tokenization
        if isinstance(text_content, str):
            sentences = self._get_sentences(text_content)
        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            sentences = self._get_sentences(text_content[document_key])
        # construct chat messages
        base_messages = []
        if self.system_prompt:
            base_messages.append({'role': 'system', 'content': self.system_prompt})

        base_messages.append({'role': 'user', 'content': self._get_user_prompt(text_content)})
        base_messages.append({'role': 'assistant', 'content': 'Sure, please start with the first sentence.'})

        # generate initial outputs sentence by sentence
        initials = []
        tasks = []
        message_list = []
        for i in range(0, len(sentences), concurrent_batch_size):
            batch = sentences[i:i + concurrent_batch_size]
            for sent in batch:
                messages = copy.deepcopy(base_messages)
                messages.append({'role': 'user', 'content': sent['sentence_text']})
                message_list.append(messages)
                task = asyncio.create_task(
                    self.inference_engine.chat_async(
                                messages=messages, 
                                max_new_tokens=max_new_tokens, 
                                temperature=temperature,
                                **kwrs
                            )
                )
                tasks.append(task)

        # Wait until the batch is done, collect results and move on to next batch
        responses = await asyncio.gather(*tasks)
        # Collect initials
        for gen_text, sent, message in zip(responses, sentences, message_list):
            initials.append({'sentence_start': sent['start'],
                            'sentence_end': sent['end'],
                            'sentence_text': sent['sentence_text'],
                            'gen_text': gen_text,
                            'messages': message})
            
        # Review
        reviews = []
        tasks = []
        for i in range(0, len(initials), concurrent_batch_size):
            batch = initials[i:i + concurrent_batch_size]
            for init in batch:
                messages = init["messages"]
                initial = init["gen_text"]
                messages.append({'role': 'assistant', 'content': initial})
                messages.append({'role': 'user', 'content': self.review_prompt})
                task = asyncio.create_task(
                                self.inference_engine.chat_async(
                                messages=messages, 
                                max_new_tokens=max_new_tokens, 
                                temperature=temperature,
                                **kwrs
                                )
                            )
                tasks.append(task) 
            
            responses = await asyncio.gather(*tasks)

        # Collect reviews
        for gen_text, sent in zip(responses, sentences):
            reviews.append({'sentence_start': sent['start'],
                            'sentence_end': sent['end'],
                            'sentence_text': sent['sentence_text'],
                            'gen_text': gen_text})

        for init, rev in zip(initials, reviews):
            if self.review_mode == "revision":
                gen_text = rev['gen_text']
            elif self.review_mode == "addition":
                gen_text = init['gen_text'] + '\n' + rev['gen_text']

            # add to output
            output.append({'sentence_start': init['sentence_start'],
                            'sentence_end': init['sentence_end'],
                            'sentence_text': init['sentence_text'],
                            'gen_text': gen_text})
        return output
        

class SentenceCoTFrameExtractor(SentenceFrameExtractor):
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None, **kwrs):
        """
        This class performs sentence-based Chain-of-thoughts (CoT) information extraction.
        A simulated chat follows this process:
            1. system prompt (optional)
            2. user instructions (schema, background, full text, few-shot example...)
            3. user input first sentence
            4. assistant analyze the sentence
            5. assistant extract outputs
            6. repeat #3, #4, #5

        Input system prompt (optional), prompt template (with user instructions), 
        and specify a LLM.

        Parameters
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine, prompt_template=prompt_template, 
                         system_prompt=system_prompt, **kwrs)
        

    def extract(self, text_content:Union[str, Dict[str,str]], max_new_tokens:int=512, 
                document_key:str=None, multi_turn:bool=False, temperature:float=0.0, stream:bool=False, **kwrs) -> List[Dict[str,str]]:
        """
        This method inputs a text and outputs a list of outputs per sentence. 

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        multi_turn : bool, Optional
            multi-turn conversation prompting. 
            If True, sentences and LLM outputs will be appended to the input message and carry-over. 
            If False, only the current sentence is prompted. 
            For LLM inference engines that supports prompt cache (e.g., Llama.Cpp, Ollama), use multi-turn conversation prompting
            can better utilize the KV caching. 
        temperature : float, Optional
            the temperature for token sampling.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : str
            the output from LLM. Need post-processing.
        """
        # define output
        output = []
        # sentence tokenization
        if isinstance(text_content, str):
            sentences = self._get_sentences(text_content)
        elif isinstance(text_content, dict):
            sentences = self._get_sentences(text_content[document_key])
        # construct chat messages
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': self._get_user_prompt(text_content)})
        messages.append({'role': 'assistant', 'content': 'Sure, please start with the first sentence.'})

        # generate sentence by sentence
        for sent in sentences:
            messages.append({'role': 'user', 'content': sent['sentence_text']})
            if stream:
                print(f"\n\n{Fore.GREEN}Sentence: {Style.RESET_ALL}\n{sent['sentence_text']}\n")
                print(f"{Fore.BLUE}CoT:{Style.RESET_ALL}")

            gen_text = self.inference_engine.chat(
                            messages=messages, 
                            max_new_tokens=max_new_tokens, 
                            temperature=temperature,
                            stream=stream,
                            **kwrs
                        )
            
            if multi_turn:
                # update chat messages with LLM outputs
                messages.append({'role': 'assistant', 'content': gen_text})
            else:
                # delete sentence so that message is reset
                del messages[-1]

            # add to output
            output.append({'sentence_start': sent['start'],
                            'sentence_end': sent['end'],
                            'sentence_text': sent['sentence_text'],
                            'gen_text': gen_text})
        return output
    

class RelationExtractor(Extractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None, **kwrs):
        """
        This is the abstract class for relation extraction.
        Input LLM inference engine, system prompt (optional), prompt template (with instruction, few-shot examples).

        Parameters
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine,
                         prompt_template=prompt_template,
                         system_prompt=system_prompt,
                         **kwrs)
        
    def _get_ROI(self, frame_1:LLMInformationExtractionFrame, frame_2:LLMInformationExtractionFrame, 
                 text:str, buffer_size:int=100) -> str:
        """
        This method returns the Region of Interest (ROI) that covers the two frames. Leaves a buffer_size of characters before and after.
        The returned text has the two frames inline annotated with <entity_1>, <entity_2>.

        Parameters:
        -----------
        frame_1 : LLMInformationExtractionFrame
            a frame
        frame_2 : LLMInformationExtractionFrame
            the other frame
        text : str
            the entire document text
        buffer_size : int, Optional
            the number of characters before and after the two frames in the ROI text.

        Return : str
            the ROI text with the two frames inline annotated with <entity_1>, <entity_2>.
        """
        left_frame, right_frame = sorted([frame_1, frame_2], key=lambda f: f.start)
        left_frame_name = "entity_1" if left_frame == frame_1 else "entity_2"
        right_frame_name = "entity_1" if right_frame == frame_1 else "entity_2"

        start = max(left_frame.start - buffer_size, 0)
        end = min(right_frame.end + buffer_size, len(text))
        roi = text[start:end]

        roi_annotated = roi[0:left_frame.start - start] + \
                f'<{left_frame_name}>' + \
                roi[left_frame.start - start:left_frame.end - start] + \
                f"</{left_frame_name}>" + \
                roi[left_frame.end - start:right_frame.start - start] + \
                f'<{right_frame_name}>' + \
                roi[right_frame.start - start:right_frame.end - start] + \
                f"</{right_frame_name}>" + \
                roi[right_frame.end - start:end - start]

        if start > 0:
            roi_annotated = "..." + roi_annotated
        if end < len(text):
            roi_annotated = roi_annotated + "..."
        return roi_annotated
    

    @abc.abstractmethod
    def extract_relations(self, doc:LLMInformationExtractionDocument, buffer_size:int=100, max_new_tokens:int=128, 
                         temperature:float=0.0, stream:bool=False, **kwrs) -> List[Dict]:
        """
        This method considers all combinations of two frames. 

        Parameters:
        -----------
        doc : LLMInformationExtractionDocument
            a document with frames.
        buffer_size : int, Optional
            the number of characters before and after the two frames in the ROI text.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : List[Dict]
            a list of dict with {"frame_1", "frame_2"} for all relations.
        """
        return NotImplemented
    

class BinaryRelationExtractor(RelationExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, possible_relation_func: Callable, 
                 system_prompt:str=None, **kwrs):
        """
        This class extracts binary (yes/no) relations between two entities.
        Input LLM inference engine, system prompt (optional), prompt template (with instruction, few-shot examples).

        Parameters
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        possible_relation_func : Callable, Optional
            a function that inputs 2 frames and returns a bool indicating possible relations between them.
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine,
                         prompt_template=prompt_template,
                         system_prompt=system_prompt,
                         **kwrs)
        
        if possible_relation_func:
            # Check if possible_relation_func is a function
            if not callable(possible_relation_func):
                raise TypeError(f"Expect possible_relation_func as a function, received {type(possible_relation_func)} instead.")
            
            sig = inspect.signature(possible_relation_func)
            # Check if frame_1, frame_2 are in input parameters
            if len(sig.parameters) != 2:
                raise ValueError("The possible_relation_func must have exactly frame_1 and frame_2 as parameters.")
            if "frame_1" not in sig.parameters.keys():
                raise ValueError("The possible_relation_func is missing frame_1 as a parameter.")
            if "frame_2" not in sig.parameters.keys():
                raise ValueError("The possible_relation_func is missing frame_2 as a parameter.")
            # Check if output is a bool
            if sig.return_annotation != bool:
                raise ValueError(f"Expect possible_relation_func to output a bool, current type hint suggests {sig.return_annotation} instead.")

            self.possible_relation_func = possible_relation_func


    def _post_process(self, rel_json:str) -> bool:
        if len(rel_json) > 0:
            if "Relation" in rel_json[0]:
                rel = rel_json[0]["Relation"]
                if isinstance(rel, bool):
                    return rel
                elif isinstance(rel, str) and rel in {"True", "False"}:
                    return eval(rel)
                else:
                    warnings.warn('Extractor output JSON "Relation" key does not have bool or {"True", "False"} as value.' + \
                                'Following default, relation = False.', RuntimeWarning)
            else:
                warnings.warn('Extractor output JSON without "Relation" key. Following default, relation = False.', RuntimeWarning)
        else:
            warnings.warn('Extractor did not output a JSON list. Following default, relation = False.', RuntimeWarning)
        return False
    

    def extract(self, doc:LLMInformationExtractionDocument, buffer_size:int=100, max_new_tokens:int=128, 
                temperature:float=0.0, stream:bool=False, **kwrs) -> List[Dict]:
        """
        This method considers all combinations of two frames. Use the possible_relation_func to filter impossible pairs.
        Outputs pairs that are related.

        Parameters:
        -----------
        doc : LLMInformationExtractionDocument
            a document with frames.
        buffer_size : int, Optional
            the number of characters before and after the two frames in the ROI text.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : List[Dict]
            a list of dict with {"frame_1_id", "frame_2_id"}.
        """
        pairs = itertools.combinations(doc.frames, 2)
        output = []
        for frame_1, frame_2 in pairs:
            pos_rel = self.possible_relation_func(frame_1, frame_2)

            if pos_rel:
                roi_text = self._get_ROI(frame_1, frame_2, doc.text, buffer_size=buffer_size)
                if stream:
                    print(f"\n\n{Fore.GREEN}ROI text:{Style.RESET_ALL} \n{roi_text}\n")
                    print(f"{Fore.BLUE}Extraction:{Style.RESET_ALL}")
                messages = []
                if self.system_prompt:
                    messages.append({'role': 'system', 'content': self.system_prompt})

                messages.append({'role': 'user', 'content': self._get_user_prompt(text_content={"roi_text":roi_text, 
                                                                                                "frame_1": str(frame_1.to_dict()),
                                                                                                "frame_2": str(frame_2.to_dict())}
                                                                                                )})

                gen_text = self.inference_engine.chat(
                                messages=messages, 
                                max_new_tokens=max_new_tokens, 
                                temperature=temperature,
                                stream=stream,
                                **kwrs
                            )
                rel_json = self._extract_json(gen_text)
                if self._post_process(rel_json):
                    output.append({'frame_1':frame_1.frame_id, 'frame_2':frame_2.frame_id})

        return output
    
    
    async def extract_async(self, doc:LLMInformationExtractionDocument, buffer_size:int=100, max_new_tokens:int=128, 
                            temperature:float=0.0, concurrent_batch_size:int=32, **kwrs) -> List[Dict]:
        """
        This is the asynchronous version of the extract() method.

        Parameters:
        -----------
        doc : LLMInformationExtractionDocument
            a document with frames.
        buffer_size : int, Optional
            the number of characters before and after the two frames in the ROI text.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        concurrent_batch_size : int, Optional
            the number of frame pairs to process in concurrent.

        Return : List[Dict]
            a list of dict with {"frame_1", "frame_2"}. 
        """
        # Check if self.inference_engine.chat_async() is implemented
        if not hasattr(self.inference_engine, 'chat_async'):
            raise NotImplementedError(f"{self.inference_engine.__class__.__name__} does not have chat_async() method.")
        
        pairs = itertools.combinations(doc.frames, 2)
        n_frames = len(doc.frames)
        num_pairs = (n_frames * (n_frames-1)) // 2
        rel_pair_list = []
        tasks = []
        for i in range(0, num_pairs, concurrent_batch_size):
            batch = list(itertools.islice(pairs, concurrent_batch_size))
            for frame_1, frame_2 in batch:
                pos_rel = self.possible_relation_func(frame_1, frame_2)
    
                if pos_rel:
                    rel_pair_list.append({'frame_1_id':frame_1.frame_id, 'frame_2_id':frame_2.frame_id})
                    roi_text = self._get_ROI(frame_1, frame_2, doc.text, buffer_size=buffer_size)
                    messages = []
                    if self.system_prompt:
                        messages.append({'role': 'system', 'content': self.system_prompt})

                    messages.append({'role': 'user', 'content': self._get_user_prompt(text_content={"roi_text":roi_text, 
                                                                                                    "frame_1": str(frame_1.to_dict()),
                                                                                                    "frame_2": str(frame_2.to_dict())}
                                                                                                    )})
                    task = asyncio.create_task(
                        self.inference_engine.chat_async(
                            messages=messages, 
                            max_new_tokens=max_new_tokens, 
                            temperature=temperature,
                            **kwrs
                        )
                    )
                    tasks.append(task)

            responses = await asyncio.gather(*tasks)

        output = []
        for d, response in zip(rel_pair_list, responses):
            rel_json = self._extract_json(response)
            if self._post_process(rel_json):
                output.append(d)

        return output


    def extract_relations(self, doc:LLMInformationExtractionDocument, buffer_size:int=100, max_new_tokens:int=128, 
                         temperature:float=0.0, concurrent:bool=False, concurrent_batch_size:int=32, stream:bool=False, **kwrs) -> List[Dict]:
        """
        This method considers all combinations of two frames. Use the possible_relation_func to filter impossible pairs.

        Parameters:
        -----------
        doc : LLMInformationExtractionDocument
            a document with frames.
        buffer_size : int, Optional
            the number of characters before and after the two frames in the ROI text.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        concurrent: bool, Optional
            if True, the extraction will be done in concurrent.
        concurrent_batch_size : int, Optional
            the number of frame pairs to process in concurrent.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : List[Dict]
            a list of dict with {"frame_1", "frame_2"} for all relations.
        """
        if not doc.has_frame():
            raise ValueError("Input document must have frames.")

        if doc.has_duplicate_frame_ids():
            raise ValueError("All frame_ids in the input document must be unique.")

        if concurrent:
            if stream:
                warnings.warn("stream=True is not supported in concurrent mode.", RuntimeWarning)

            nest_asyncio.apply() # For Jupyter notebook. Terminal does not need this.
            return asyncio.run(self.extract_async(doc=doc, 
                                                  buffer_size=buffer_size, 
                                                  max_new_tokens=max_new_tokens,
                                                  temperature=temperature,
                                                  concurrent_batch_size=concurrent_batch_size, 
                                                  **kwrs)
                                )
        else:
            return self.extract(doc=doc, 
                                buffer_size=buffer_size, 
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                stream=stream,
                                **kwrs)
            

class MultiClassRelationExtractor(RelationExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, possible_relation_types_func: Callable, 
                 system_prompt:str=None, **kwrs):
        """
        This class extracts relations with relation types.
        Input LLM inference engine, system prompt (optional), prompt template (with instruction, few-shot examples).

        Parameters
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        possible_relation_types_func : Callable
            a function that inputs 2 frames and returns a List of possible relation types between them. 
            If the two frames must not have relations, this function should return an empty list [].
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine,
                         prompt_template=prompt_template,
                         system_prompt=system_prompt,
                         **kwrs)
        
        if possible_relation_types_func:
            # Check if possible_relation_types_func is a function
            if not callable(possible_relation_types_func):
                raise TypeError(f"Expect possible_relation_types_func as a function, received {type(possible_relation_types_func)} instead.")
            
            sig = inspect.signature(possible_relation_types_func)
            # Check if frame_1, frame_2 are in input parameters
            if len(sig.parameters) != 2:
                raise ValueError("The possible_relation_types_func must have exactly frame_1 and frame_2 as parameters.")
            if "frame_1" not in sig.parameters.keys():
                raise ValueError("The possible_relation_types_func is missing frame_1 as a parameter.")
            if "frame_2" not in sig.parameters.keys():
                raise ValueError("The possible_relation_types_func is missing frame_2 as a parameter.")
            # Check if output is a List
            if sig.return_annotation not in {inspect._empty, List, List[str]}:
                raise ValueError(f"Expect possible_relation_types_func to output a List of string, current type hint suggests {sig.return_annotation} instead.")

            self.possible_relation_types_func = possible_relation_types_func


    def _post_process(self, rel_json:List[Dict], pos_rel_types:List[str]) -> Union[str, None]:
        """
        This method post-processes the extracted relation JSON.

        Parameters:
        -----------
        rel_json : List[Dict]
            the extracted relation JSON.
        pos_rel_types : List[str]
            possible relation types by the possible_relation_types_func.

        Return : Union[str, None]
            the relation type (str) or None for no relation.
        """
        if len(rel_json) > 0:
            if "RelationType" in rel_json[0]:
                if rel_json[0]["RelationType"] in pos_rel_types:
                    return rel_json[0]["RelationType"]
            else:
                warnings.warn('Extractor output JSON without "RelationType" key. Following default, relation = "No Relation".', RuntimeWarning)
        else:
            warnings.warn('Extractor did not output a JSON. Following default, relation = "No Relation".', RuntimeWarning)
        return None
    

    def extract(self, doc:LLMInformationExtractionDocument, buffer_size:int=100, max_new_tokens:int=128, 
                temperature:float=0.0, stream:bool=False, **kwrs) -> List[Dict]:
        """
        This method considers all combinations of two frames. Use the possible_relation_types_func to filter impossible pairs.

        Parameters:
        -----------
        doc : LLMInformationExtractionDocument
            a document with frames.
        buffer_size : int, Optional
            the number of characters before and after the two frames in the ROI text.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : List[Dict]
            a list of dict with {"frame_1", "frame_2", "relation"} for all frame pairs.
        """
        pairs = itertools.combinations(doc.frames, 2)
        output = []
        for frame_1, frame_2 in pairs:
            pos_rel_types = self.possible_relation_types_func(frame_1, frame_2)

            if pos_rel_types:
                roi_text = self._get_ROI(frame_1, frame_2, doc.text, buffer_size=buffer_size)
                if stream:
                    print(f"\n\n{Fore.GREEN}ROI text:{Style.RESET_ALL} \n{roi_text}\n")
                    print(f"{Fore.BLUE}Extraction:{Style.RESET_ALL}")
                messages = []
                if self.system_prompt:
                    messages.append({'role': 'system', 'content': self.system_prompt})

                messages.append({'role': 'user', 'content': self._get_user_prompt(text_content={"roi_text":roi_text, 
                                                                                                "frame_1": str(frame_1.to_dict()),
                                                                                                "frame_2": str(frame_2.to_dict()),
                                                                                                "pos_rel_types":str(pos_rel_types)}
                                                                                                )})

                gen_text = self.inference_engine.chat(
                                messages=messages, 
                                max_new_tokens=max_new_tokens, 
                                temperature=temperature,
                                stream=stream,
                                **kwrs
                            )
                rel_json = self._extract_json(gen_text)
                rel = self._post_process(rel_json, pos_rel_types)
                if rel:
                    output.append({'frame_1':frame_1.frame_id, 'frame_2':frame_2.frame_id, 'relation':rel})

        return output   
    
    
    async def extract_async(self, doc:LLMInformationExtractionDocument, buffer_size:int=100, max_new_tokens:int=128, 
                            temperature:float=0.0, concurrent_batch_size:int=32, **kwrs) -> List[Dict]:
        """
        This is the asynchronous version of the extract() method.

        Parameters:
        -----------
        doc : LLMInformationExtractionDocument
            a document with frames.
        buffer_size : int, Optional
            the number of characters before and after the two frames in the ROI text.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        concurrent_batch_size : int, Optional
            the number of frame pairs to process in concurrent.

        Return : List[Dict]
            a list of dict with {"frame_1", "frame_2", "relation"} for all frame pairs. 
        """
        # Check if self.inference_engine.chat_async() is implemented
        if not hasattr(self.inference_engine, 'chat_async'):
            raise NotImplementedError(f"{self.inference_engine.__class__.__name__} does not have chat_async() method.")
        
        pairs = itertools.combinations(doc.frames, 2)
        n_frames = len(doc.frames)
        num_pairs = (n_frames * (n_frames-1)) // 2
        rel_pair_list = []
        tasks = []
        for i in range(0, num_pairs, concurrent_batch_size):
            batch = list(itertools.islice(pairs, concurrent_batch_size))
            for frame_1, frame_2 in batch:
                pos_rel_types = self.possible_relation_types_func(frame_1, frame_2)
    
                if pos_rel_types:
                    rel_pair_list.append({'frame_1':frame_1.frame_id, 'frame_2':frame_2.frame_id, 'pos_rel_types':pos_rel_types})
                    roi_text = self._get_ROI(frame_1, frame_2, doc.text, buffer_size=buffer_size)
                    messages = []
                    if self.system_prompt:
                        messages.append({'role': 'system', 'content': self.system_prompt})

                    messages.append({'role': 'user', 'content': self._get_user_prompt(text_content={"roi_text":roi_text, 
                                                                                                    "frame_1": str(frame_1.to_dict()),
                                                                                                    "frame_2": str(frame_2.to_dict()),
                                                                                                    "pos_rel_types":str(pos_rel_types)}
                                                                                                    )})
                    task = asyncio.create_task(
                        self.inference_engine.chat_async(
                            messages=messages, 
                            max_new_tokens=max_new_tokens, 
                            temperature=temperature,
                            **kwrs
                        )
                    )
                    tasks.append(task)

            responses = await asyncio.gather(*tasks)

        output = []
        for d, response in zip(rel_pair_list, responses):
            rel_json = self._extract_json(response)
            rel = self._post_process(rel_json, d['pos_rel_types'])
            if rel:
                output.append({'frame_1':d['frame_1'], 'frame_2':d['frame_2'], 'relation':rel})

        return output


    def extract_relations(self, doc:LLMInformationExtractionDocument, buffer_size:int=100, max_new_tokens:int=128, 
                         temperature:float=0.0, concurrent:bool=False, concurrent_batch_size:int=32, stream:bool=False, **kwrs) -> List[Dict]:
        """
        This method considers all combinations of two frames. Use the possible_relation_types_func to filter impossible pairs.

        Parameters:
        -----------
        doc : LLMInformationExtractionDocument
            a document with frames.
        buffer_size : int, Optional
            the number of characters before and after the two frames in the ROI text.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        concurrent: bool, Optional
            if True, the extraction will be done in concurrent.
        concurrent_batch_size : int, Optional
            the number of frame pairs to process in concurrent.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : List[Dict]
            a list of dict with {"frame_1", "frame_2", "relation"} for all relations.
        """
        if not doc.has_frame():
            raise ValueError("Input document must have frames.")

        if doc.has_duplicate_frame_ids():
            raise ValueError("All frame_ids in the input document must be unique.")

        if concurrent:
            if stream:
                warnings.warn("stream=True is not supported in concurrent mode.", RuntimeWarning)

            nest_asyncio.apply() # For Jupyter notebook. Terminal does not need this.
            return asyncio.run(self.extract_async(doc=doc, 
                                                  buffer_size=buffer_size, 
                                                  max_new_tokens=max_new_tokens,
                                                  temperature=temperature,
                                                  concurrent_batch_size=concurrent_batch_size, 
                                                  **kwrs)
                                )
        else:
            return self.extract(doc=doc, 
                                buffer_size=buffer_size, 
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                stream=stream,
                                **kwrs)
            