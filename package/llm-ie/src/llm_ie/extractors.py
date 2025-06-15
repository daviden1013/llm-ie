import abc
import re
import json
import json_repair
import inspect
import importlib.resources
import warnings
import itertools
import asyncio
import nest_asyncio
from typing import Any, Set, List, Dict, Tuple, Union, Callable, Generator, Optional
from llm_ie.data_types import FrameExtractionUnit, FrameExtractionUnitResult, LLMInformationExtractionFrame, LLMInformationExtractionDocument
from llm_ie.chunkers import UnitChunker, WholeDocumentUnitChunker, SentenceUnitChunker
from llm_ie.chunkers import ContextChunker, NoContextChunker, WholeDocumentContextChunker, SlideWindowContextChunker
from llm_ie.engines import InferenceEngine
from colorama import Fore, Style        


class Extractor:
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None):
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
        It searches for a guide specific to the current class first, if not found, it will search
        for the guide in its ancestors by traversing the class's method resolution order (MRO).
        """
        original_class_name = cls.__name__ 

        for current_class_in_mro in cls.__mro__:
            if current_class_in_mro is object: 
                continue

            current_class_name = current_class_in_mro.__name__
            
            try:
                file_path_obj = importlib.resources.files('llm_ie.asset.prompt_guide').joinpath(f"{current_class_name}_prompt_guide.txt")
                
                with open(file_path_obj, 'r', encoding="utf-8") as f:
                    prompt_content = f.read()
                    # If the guide was found for an ancestor, not the original class, issue a warning.
                    if cls is not current_class_in_mro:
                        warnings.warn(
                            f"Prompt guide for '{original_class_name}' not found. "
                            f"Using guide from ancestor: '{current_class_name}_prompt_guide.txt'.",
                            UserWarning
                        )
                    return prompt_content
            except FileNotFoundError:
                pass

            except Exception as e:
                warnings.warn(
                    f"Error attempting to read prompt guide for '{current_class_name}' "
                    f"from '{str(file_path_obj)}': {e}. Trying next in MRO.",
                    UserWarning
                )
                continue 

        # If the loop completes, no prompt guide was found for the original class or any of its ancestors.
        raise FileNotFoundError(
            f"Prompt guide for '{original_class_name}' not found in the package asset. "
            f"Is it a custom extractor?"
        )

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
    from nltk.tokenize import RegexpTokenizer
    def __init__(self, inference_engine:InferenceEngine, unit_chunker:UnitChunker, 
                 prompt_template:str, system_prompt:str=None, context_chunker:ContextChunker=None):
        """
        This is the abstract class for frame extraction.
        Input LLM inference engine, system prompt (optional), prompt template (with instruction, few-shot examples).

        Parameters:
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        unit_chunker : UnitChunker
            the unit chunker object that determines how to chunk the document text into units.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        system_prompt : str, Optional
            system prompt.
        context_chunker : ContextChunker
            the context chunker object that determines how to get context for each unit.
        """
        super().__init__(inference_engine=inference_engine,
                         prompt_template=prompt_template,
                         system_prompt=system_prompt)
                
        self.unit_chunker = unit_chunker
        if context_chunker is None:
            self.context_chunker = NoContextChunker()
        else:
            self.context_chunker = context_chunker
        
        self.tokenizer = self.RegexpTokenizer(r'\w+|[^\w\s]')


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
                           fuzzy_match:bool=True, fuzzy_buffer_size:float=0.2, fuzzy_score_cutoff:float=0.8,
                           allow_overlap_entities:bool=False) -> List[Tuple[int]]:
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
        allow_overlap_entities : bool, Optional
            if True, entities can overlap in the text. 
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
                if not allow_overlap_entities:
                    # Replace the found entity with spaces to avoid finding the same instance again
                    text = text[:start] + ' ' * (end - start) + text[end:]
            # Fuzzy match
            elif fuzzy_match:
                closest_substring_span, best_score = self._get_closest_substring(text, entity, buffer_size=fuzzy_buffer_size)
                if closest_substring_span and best_score >= fuzzy_score_cutoff:
                    entity_spans.append(closest_substring_span)
                    if not allow_overlap_entities:
                        # Replace the found entity with spaces to avoid finding the same instance again
                        text = text[:closest_substring_span[0]] + ' ' * (closest_substring_span[1] - closest_substring_span[0]) + text[closest_substring_span[1]:]
                else:
                    entity_spans.append(None)

            # No match
            else:
                entity_spans.append(None)

        return entity_spans

    @abc.abstractmethod
    def extract(self, text_content:Union[str, Dict[str,str]], return_messages_log:bool=False, **kwrs) -> str:
        """
        This method inputs text content and outputs a string generated by LLM

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.

        Return : str
            the output from LLM. Need post-processing.
        """
        return NotImplemented
    
    
    @abc.abstractmethod
    def extract_frames(self, text_content:Union[str, Dict[str,str]], entity_key:str, 
                       document_key:str=None, return_messages_log:bool=False, **kwrs) -> List[LLMInformationExtractionFrame]:
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
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.

        Return : str
            a list of frames.
        """
        return NotImplemented
    

class DirectFrameExtractor(FrameExtractor):
    def __init__(self, inference_engine:InferenceEngine, unit_chunker:UnitChunker, 
                 prompt_template:str, system_prompt:str=None, context_chunker:ContextChunker=None):
        """
        This class is for general unit-context frame extraction.
        Input LLM inference engine, system prompt (optional), prompt template (with instruction, few-shot examples).

        Parameters:
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        unit_chunker : UnitChunker
            the unit chunker object that determines how to chunk the document text into units.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        system_prompt : str, Optional
            system prompt.
        context_chunker : ContextChunker
            the context chunker object that determines how to get context for each unit.
        """
        super().__init__(inference_engine=inference_engine,
                         unit_chunker=unit_chunker,
                         prompt_template=prompt_template,
                         system_prompt=system_prompt,
                         context_chunker=context_chunker)


    def extract(self, text_content:Union[str, Dict[str,str]], 
                document_key:str=None, verbose:bool=False, return_messages_log:bool=False) -> List[FrameExtractionUnitResult]:
        """
        This method inputs a text and outputs a list of outputs per unit.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.

        Return : List[FrameExtractionUnitResult]
            the output from LLM for each unit. Contains the start, end, text, and generated text.
        """
        # define output
        output = []
        # unit chunking
        if isinstance(text_content, str):
            doc_text = text_content

        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            doc_text = text_content[document_key]
        
        units = self.unit_chunker.chunk(doc_text)
        # context chunker init
        self.context_chunker.fit(doc_text, units)
        # messages log
        if return_messages_log:
            messages_log = []

        # generate unit by unit
        for i, unit in enumerate(units):
            # construct chat messages
            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})

            context = self.context_chunker.chunk(unit)
            
            if context == "":
                # no context, just place unit in user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit.text)})
                else:
                    unit_content = text_content.copy()
                    unit_content[document_key] = unit.text
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit_content)})
            else:
                # insert context to user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context)})
                else:
                    context_content = text_content.copy()
                    context_content[document_key] = context
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context_content)})
                # simulate conversation where assistant confirms
                messages.append({'role': 'assistant', 'content': 'Sure, please provide the unit text (e.g., sentence, line, chunk) of interest.'})
                # place unit of interest
                messages.append({'role': 'user', 'content': unit.text})

            if verbose:
                print(f"\n\n{Fore.GREEN}Unit {i}:{Style.RESET_ALL}\n{unit.text}\n")
                if context != "":
                    print(f"{Fore.YELLOW}Context:{Style.RESET_ALL}\n{context}\n")
                
                print(f"{Fore.BLUE}Extraction:{Style.RESET_ALL}")

            
            gen_text = self.inference_engine.chat(
                            messages=messages, 
                            verbose=verbose,
                            stream=False
                        )

            if return_messages_log:
                messages.append({"role": "assistant", "content": gen_text})
                messages_log.append(messages)

            # add to output
            result = FrameExtractionUnitResult(
                            start=unit.start,
                            end=unit.end,
                            text=unit.text,
                            gen_text=gen_text)
            output.append(result)
            
        if return_messages_log:
            return output, messages_log
        
        return output
    
    def stream(self, text_content: Union[str, Dict[str, str]], 
               document_key: str = None) -> Generator[Dict[str, Any], None, List[FrameExtractionUnitResult]]:
        """
        Streams LLM responses per unit with structured event types,
        and returns collected data for post-processing.

        Yields:
        -------
        Dict[str, Any]: (type, data)
            - {"type": "info", "data": str_message}: General informational messages.
            - {"type": "unit", "data": dict_unit_info}: Signals start of a new unit. dict_unit_info contains {'id', 'text', 'start', 'end'}
            - {"type": "context", "data": str_context}: Context string for the current unit.
            - {"type": "reasoning", "data": str_chunk}: A reasoning model thinking chunk from the LLM.
            - {"type": "response", "data": str_chunk}: A response/answer chunk from the LLM.

        Returns:
        --------
        List[FrameExtractionUnitResult]:
            A list of FrameExtractionUnitResult objects, each containing the
            original unit details and the fully accumulated 'gen_text' from the LLM.
        """
        collected_results: List[FrameExtractionUnitResult] = []

        if isinstance(text_content, str):
            doc_text = text_content
        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            if document_key not in text_content:
                raise ValueError(f"document_key '{document_key}' not found in text_content.")
            doc_text = text_content[document_key]
        else:
            raise TypeError("text_content must be a string or a dictionary.")

        units: List[FrameExtractionUnit] = self.unit_chunker.chunk(doc_text)
        self.context_chunker.fit(doc_text, units)

        yield {"type": "info", "data": f"Starting LLM processing for {len(units)} units."}

        for i, unit in enumerate(units):
            unit_info_payload = {"id": i, "text": unit.text, "start": unit.start, "end": unit.end}
            yield {"type": "unit", "data": unit_info_payload}

            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})

            context_str = self.context_chunker.chunk(unit)

            # Construct prompt input based on whether text_content was str or dict
            if context_str:
                yield {"type": "context", "data": context_str}
                prompt_input_for_context = context_str
                if isinstance(text_content, dict):
                    context_content_dict = text_content.copy()
                    context_content_dict[document_key] = context_str
                    prompt_input_for_context = context_content_dict
                messages.append({'role': 'user', 'content': self._get_user_prompt(prompt_input_for_context)})
                messages.append({'role': 'assistant', 'content': 'Sure, please provide the unit text (e.g., sentence, line, chunk) of interest.'})
                messages.append({'role': 'user', 'content': unit.text})
            else: # No context
                prompt_input_for_unit = unit.text
                if isinstance(text_content, dict):
                    unit_content_dict = text_content.copy()
                    unit_content_dict[document_key] = unit.text
                    prompt_input_for_unit = unit_content_dict
                messages.append({'role': 'user', 'content': self._get_user_prompt(prompt_input_for_unit)})

            current_gen_text = ""

            response_stream = self.inference_engine.chat(
                messages=messages,
                stream=True
            )
            for chunk in response_stream:
                yield chunk
                current_gen_text += chunk
           
            # Store the result for this unit
            result_for_unit = FrameExtractionUnitResult(
                start=unit.start,
                end=unit.end,
                text=unit.text,
                gen_text=current_gen_text
            )
            collected_results.append(result_for_unit)

        yield {"type": "info", "data": "All units processed by LLM."}
        return collected_results

    async def extract_async(self, text_content:Union[str, Dict[str,str]], document_key:str=None, 
                            concurrent_batch_size:int=32, return_messages_log:bool=False) -> List[FrameExtractionUnitResult]:
        """
        This is the asynchronous version of the extract() method.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        concurrent_batch_size : int, Optional
            the batch size for concurrent processing. 
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.

        Return : List[FrameExtractionUnitResult]
            the output from LLM for each unit. Contains the start, end, text, and generated text.
        """
        if isinstance(text_content, str):
            doc_text = text_content
        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            if document_key not in text_content:
                 raise ValueError(f"document_key '{document_key}' not found in text_content dictionary.")
            doc_text = text_content[document_key]
        else:
            raise TypeError("text_content must be a string or a dictionary.")

        units = self.unit_chunker.chunk(doc_text)

        # context chunker init 
        self.context_chunker.fit(doc_text, units)

        # Prepare inputs for all units first
        tasks_input = []
        for i, unit in enumerate(units):
            # construct chat messages
            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})

            context = self.context_chunker.chunk(unit)

            if context == "":
                 # no context, just place unit in user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit.text)})
                else:
                    unit_content = text_content.copy()
                    unit_content[document_key] = unit.text
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit_content)})
            else:
                # insert context to user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context)})
                else:
                    context_content = text_content.copy()
                    context_content[document_key] = context
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context_content)})
                # simulate conversation where assistant confirms
                messages.append({'role': 'assistant', 'content': 'Sure, please provide the unit text (e.g., sentence, line, chunk) of interest.'})
                # place unit of interest
                messages.append({'role': 'user', 'content': unit.text})

            # Store unit and messages together for the task
            tasks_input.append({"unit": unit, "messages": messages, "original_index": i})

        # Process units concurrently with asyncio.Semaphore
        semaphore = asyncio.Semaphore(concurrent_batch_size)

        async def semaphore_helper(task_data: Dict, **kwrs):
            unit = task_data["unit"]
            messages = task_data["messages"]
            original_index = task_data["original_index"]

            async with semaphore:
                gen_text = await self.inference_engine.chat_async(
                    messages=messages
                )
            return {"original_index": original_index, "unit": unit, "gen_text": gen_text, "messages": messages}
           
        # Create and gather tasks
        tasks = []
        for task_inp in tasks_input:
            task = asyncio.create_task(semaphore_helper(
                task_inp
            ))
            tasks.append(task)

        results_raw = await asyncio.gather(*tasks)

        # Sort results back into original order using the index stored
        results_raw.sort(key=lambda x: x["original_index"])

        # Restructure the results
        output: List[FrameExtractionUnitResult] = []
        messages_log: Optional[List[List[Dict[str, str]]]] = [] if return_messages_log else None

        for result_data in results_raw:
            unit = result_data["unit"]
            gen_text = result_data["gen_text"]

            # Create result object
            result = FrameExtractionUnitResult(
                start=unit.start,
                end=unit.end,
                text=unit.text,
                gen_text=gen_text
            )
            output.append(result)

            # Append to messages log if requested
            if return_messages_log:
                final_messages = result_data["messages"] + [{"role": "assistant", "content": gen_text}]
                messages_log.append(final_messages)

        if return_messages_log:
            return output, messages_log
        else:
            return output


    def extract_frames(self, text_content:Union[str, Dict[str,str]], document_key:str=None, 
                       verbose:bool=False, concurrent:bool=False, concurrent_batch_size:int=32,
                        case_sensitive:bool=False, fuzzy_match:bool=True, fuzzy_buffer_size:float=0.2, fuzzy_score_cutoff:float=0.8,
                        allow_overlap_entities:bool=False, return_messages_log:bool=False) -> List[LLMInformationExtractionFrame]:
        """
        This method inputs a text and outputs a list of LLMInformationExtractionFrame
        It use the extract() method and post-process outputs into frames.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        verbose : bool, Optional
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
        allow_overlap_entities : bool, Optional
            if True, entities can overlap in the text. 
            Note that this can cause multiple frames to be generated on the same entity span if they have same entity text.
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.

        Return : str
            a list of frames.
        """
        ENTITY_KEY = "entity_text"
        if concurrent:
            if verbose:
                warnings.warn("verbose=True is not supported in concurrent mode.", RuntimeWarning)

            nest_asyncio.apply() # For Jupyter notebook. Terminal does not need this.
            extraction_results = asyncio.run(self.extract_async(text_content=text_content, 
                                                document_key=document_key,
                                                concurrent_batch_size=concurrent_batch_size,
                                                return_messages_log=return_messages_log)
                                            )
        else:
            extraction_results = self.extract(text_content=text_content, 
                                                document_key=document_key,
                                                verbose=verbose,
                                                return_messages_log=return_messages_log)
            
        llm_output_results, messages_log = extraction_results if return_messages_log else (extraction_results, None)
        
        frame_list = []
        for res in llm_output_results:
            entity_json = []
            for entity in self._extract_json(gen_text=res.gen_text):
                if ENTITY_KEY in entity:
                    entity_json.append(entity)
                else:
                    warnings.warn(f'Extractor output "{entity}" does not have entity_key ("{ENTITY_KEY}"). This frame will be dropped.', RuntimeWarning)

            spans = self._find_entity_spans(text=res.text, 
                                            entities=[e[ENTITY_KEY] for e in entity_json], 
                                            case_sensitive=case_sensitive,
                                            fuzzy_match=fuzzy_match,
                                            fuzzy_buffer_size=fuzzy_buffer_size,
                                            fuzzy_score_cutoff=fuzzy_score_cutoff,
                                            allow_overlap_entities=allow_overlap_entities)
            for ent, span in zip(entity_json, spans):
                if span is not None:
                    start, end = span
                    entity_text = res.text[start:end]
                    start += res.start
                    end += res.start
                    attr = {}
                    if "attr" in ent and ent["attr"] is not None:
                        attr = ent["attr"]
                       
                    frame = LLMInformationExtractionFrame(frame_id=f"{len(frame_list)}", 
                                start=start,
                                end=end,
                                entity_text=entity_text,
                                attr=attr)
                    frame_list.append(frame)

        if return_messages_log:
            return frame_list, messages_log
        return frame_list
        

class ReviewFrameExtractor(DirectFrameExtractor):
    def __init__(self, unit_chunker:UnitChunker, context_chunker:ContextChunker, inference_engine:InferenceEngine, 
                 prompt_template:str, review_mode:str, review_prompt:str=None, system_prompt:str=None):
        """
        This class add a review step after the DirectFrameExtractor.
        The Review process asks LLM to review its output and:
            1. add more frames while keep current. This is efficient for boosting recall. 
            2. or, regenerate frames (add new and delete existing). 
        Use the review_mode parameter to specify. Note that the review_prompt should instruct LLM accordingly.

        Parameters:
        ----------
        unit_chunker : UnitChunker
            the unit chunker object that determines how to chunk the document text into units.
        context_chunker : ContextChunker
            the context chunker object that determines how to get context for each unit.
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
        super().__init__(inference_engine=inference_engine, 
                         unit_chunker=unit_chunker, 
                         prompt_template=prompt_template, 
                         system_prompt=system_prompt, 
                         context_chunker=context_chunker)
        # check review mode
        if review_mode not in {"addition", "revision"}: 
            raise ValueError('review_mode must be one of {"addition", "revision"}.')
        self.review_mode = review_mode
        # assign review prompt
        if review_prompt:
            self.review_prompt = review_prompt
        else:
            self.review_prompt = None
            original_class_name = self.__class__.__name__

            current_class_name = original_class_name
            for current_class_in_mro in self.__class__.__mro__:
                if current_class_in_mro is object: 
                    continue

                current_class_name = current_class_in_mro.__name__
                try:
                    file_path = importlib.resources.files('llm_ie.asset.default_prompts').\
                        joinpath(f"{self.__class__.__name__}_{self.review_mode}_review_prompt.txt")
                    with open(file_path, 'r', encoding="utf-8") as f:
                        self.review_prompt = f.read()
                except FileNotFoundError:
                    pass

                except Exception as e:
                    warnings.warn(
                        f"Error attempting to read default review prompt for '{current_class_name}' "
                        f"from '{str(file_path)}': {e}. Trying next in MRO.",
                        UserWarning
                    )
                    continue 
            
        if self.review_prompt is None:
            raise ValueError(f"Cannot find review prompt for {self.__class__.__name__} in the package. Please provide a review_prompt.")

    def extract(self, text_content:Union[str, Dict[str,str]], document_key:str=None, 
                verbose:bool=False, return_messages_log:bool=False) -> List[FrameExtractionUnitResult]:
        """
        This method inputs a text and outputs a list of outputs per unit.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.

        Return : List[FrameExtractionUnitResult]
            the output from LLM for each unit. Contains the start, end, text, and generated text.
        """
        # define output
        output = []
        # unit chunking
        if isinstance(text_content, str):
            doc_text = text_content

        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            doc_text = text_content[document_key]
        
        units = self.unit_chunker.chunk(doc_text)
        # context chunker init
        self.context_chunker.fit(doc_text, units)
        # messages log
        if return_messages_log:
            messages_log = []

        # generate unit by unit
        for i, unit in enumerate(units):
            # <--- Initial generation step --->
            # construct chat messages
            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})

            context = self.context_chunker.chunk(unit)
            
            if context == "":
                # no context, just place unit in user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit.text)})
                else:
                    unit_content = text_content.copy()
                    unit_content[document_key] = unit.text
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit_content)})
            else:
                # insert context to user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context)})
                else:
                    context_content = text_content.copy()
                    context_content[document_key] = context
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context_content)})
                # simulate conversation where assistant confirms
                messages.append({'role': 'assistant', 'content': 'Sure, please provide the unit text (e.g., sentence, line, chunk) of interest.'})
                # place unit of interest
                messages.append({'role': 'user', 'content': unit.text})

            if verbose:
                print(f"\n\n{Fore.GREEN}Unit {i}:{Style.RESET_ALL}\n{unit.text}\n")
                if context != "":
                    print(f"{Fore.YELLOW}Context:{Style.RESET_ALL}\n{context}\n")
                
                print(f"{Fore.BLUE}Extraction:{Style.RESET_ALL}")
            

            initial = self.inference_engine.chat(
                            messages=messages, 
                            verbose=verbose,
                            stream=False
                        )

            if return_messages_log:
                messages.append({"role": "assistant", "content": initial})
                messages_log.append(messages)

            # <--- Review step --->
            if verbose:
                print(f"\n{Fore.YELLOW}Review:{Style.RESET_ALL}")

            messages.append({'role': 'assistant', 'content': initial})
            messages.append({'role': 'user', 'content': self.review_prompt})
            
            review = self.inference_engine.chat(
                            messages=messages, 
                            verbose=verbose,
                            stream=False
                        )

            # Output
            if self.review_mode == "revision":
                gen_text = review
            elif self.review_mode == "addition":
                gen_text = initial + '\n' + review

            if return_messages_log:
                messages.append({"role": "assistant", "content": review})
                messages_log.append(messages)

            # add to output
            result = FrameExtractionUnitResult(
                            start=unit.start,
                            end=unit.end,
                            text=unit.text,
                            gen_text=gen_text)
            output.append(result)
            
        if return_messages_log:
            return output, messages_log
        
        return output


    def stream(self, text_content:Union[str, Dict[str,str]], document_key:str=None) -> Generator[str, None, None]:
        """
        This method inputs a text and outputs a list of outputs per unit.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        document_key : str, Optional
            specify the key in text_content where document text is. 
            If text_content is str, this parameter will be ignored.

        Return : List[FrameExtractionUnitResult]
            the output from LLM for each unit. Contains the start, end, text, and generated text.
        """
        # unit chunking
        if isinstance(text_content, str):
            doc_text = text_content

        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            doc_text = text_content[document_key]
        
        units = self.unit_chunker.chunk(doc_text)
        # context chunker init
        self.context_chunker.fit(doc_text, units)

        # generate unit by unit
        for i, unit in enumerate(units):
            # <--- Initial generation step --->
            # construct chat messages
            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})

            context = self.context_chunker.chunk(unit)
            
            if context == "":
                # no context, just place unit in user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit.text)})
                else:
                    unit_content = text_content.copy()
                    unit_content[document_key] = unit.text
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit_content)})
            else:
                # insert context to user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context)})
                else:
                    context_content = text_content.copy()
                    context_content[document_key] = context
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context_content)})
                # simulate conversation where assistant confirms
                messages.append({'role': 'assistant', 'content': 'Sure, please provide the unit text (e.g., sentence, line, chunk) of interest.'})
                # place unit of interest
                messages.append({'role': 'user', 'content': unit.text})


            yield f"\n\n{Fore.GREEN}Unit {i}:{Style.RESET_ALL}\n{unit.text}\n"
            if context != "":
                yield f"{Fore.YELLOW}Context:{Style.RESET_ALL}\n{context}\n"
            
            yield f"{Fore.BLUE}Extraction:{Style.RESET_ALL}\n"

            response_stream = self.inference_engine.chat(
                            messages=messages, 
                            stream=True
                        )
            
            initial = ""
            for chunk in response_stream:
                initial += chunk
                yield chunk

            # <--- Review step --->
            yield f"\n{Fore.YELLOW}Review:{Style.RESET_ALL}"

            messages.append({'role': 'assistant', 'content': initial})
            messages.append({'role': 'user', 'content': self.review_prompt})

            response_stream = self.inference_engine.chat(
                            messages=messages, 
                            stream=True
                        )
            
            for chunk in response_stream:
                yield chunk

    async def extract_async(self, text_content:Union[str, Dict[str,str]], document_key:str=None,
                            concurrent_batch_size:int=32, return_messages_log:bool=False, **kwrs) -> List[FrameExtractionUnitResult]:
        """
        This is the asynchronous version of the extract() method with the review step.

        Parameters:
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template.
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.
        document_key : str, Optional
            specify the key in text_content where document text is.
            If text_content is str, this parameter will be ignored.
        concurrent_batch_size : int, Optional
            the batch size for concurrent processing.
        return_messages_log : bool, Optional
            if True, a list of messages will be returned, including review steps.

        Return : List[FrameExtractionUnitResult]
            the output from LLM for each unit after review. Contains the start, end, text, and generated text.
        """
        if isinstance(text_content, str):
            doc_text = text_content
        elif isinstance(text_content, dict):
            if document_key is None:
                raise ValueError("document_key must be provided when text_content is dict.")
            if document_key not in text_content:
                 raise ValueError(f"document_key '{document_key}' not found in text_content dictionary.")
            doc_text = text_content[document_key]
        else:
            raise TypeError("text_content must be a string or a dictionary.")

        units = self.unit_chunker.chunk(doc_text)

        # context chunker init
        self.context_chunker.fit(doc_text, units)

        # <--- Initial generation step --->
        initial_tasks_input = []
        for i, unit in enumerate(units):
            # construct chat messages for initial generation
            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})

            context = self.context_chunker.chunk(unit)

            if context == "":
                 # no context, just place unit in user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit.text)})
                else:
                    unit_content = text_content.copy()
                    unit_content[document_key] = unit.text
                    messages.append({'role': 'user', 'content': self._get_user_prompt(unit_content)})
            else:
                # insert context to user prompt
                if isinstance(text_content, str):
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context)})
                else:
                    context_content = text_content.copy()
                    context_content[document_key] = context
                    messages.append({'role': 'user', 'content': self._get_user_prompt(context_content)})
                # simulate conversation where assistant confirms
                messages.append({'role': 'assistant', 'content': 'Sure, please provide the unit text (e.g., sentence, line, chunk) of interest.'})
                # place unit of interest
                messages.append({'role': 'user', 'content': unit.text})

            # Store unit and messages together for the initial task
            initial_tasks_input.append({"unit": unit, "messages": messages, "original_index": i})

        semaphore = asyncio.Semaphore(concurrent_batch_size)

        async def initial_semaphore_helper(task_data: Dict):
            unit = task_data["unit"]
            messages = task_data["messages"]
            original_index = task_data["original_index"]

            async with semaphore:
                gen_text = await self.inference_engine.chat_async(
                    messages=messages
                )
            # Return initial generation result along with the messages used and the unit
            return {"original_index": original_index, "unit": unit, "initial_gen_text": gen_text, "initial_messages": messages}

        # Create and gather initial generation tasks
        initial_tasks = [
            asyncio.create_task(initial_semaphore_helper(
                task_inp
            ))
            for task_inp in initial_tasks_input
        ]

        initial_results_raw = await asyncio.gather(*initial_tasks)

        # Sort initial results back into original order
        initial_results_raw.sort(key=lambda x: x["original_index"])

        # <--- Review step --->
        review_tasks_input = []
        for result_data in initial_results_raw:
            # Prepare messages for the review step
            initial_messages = result_data["initial_messages"]
            initial_gen_text = result_data["initial_gen_text"]
            review_messages = initial_messages + [
                {'role': 'assistant', 'content': initial_gen_text},
                {'role': 'user', 'content': self.review_prompt}
            ]
            # Store data needed for review task
            review_tasks_input.append({
                "unit": result_data["unit"],
                "initial_gen_text": initial_gen_text,
                "messages": review_messages, 
                "original_index": result_data["original_index"],
                "full_initial_log": initial_messages + [{'role': 'assistant', 'content': initial_gen_text}] if return_messages_log else None # Log up to initial generation
            })


        async def review_semaphore_helper(task_data: Dict, **kwrs):
            messages = task_data["messages"] 
            original_index = task_data["original_index"]

            async with semaphore:
                review_gen_text = await self.inference_engine.chat_async(
                    messages=messages
                )
            # Combine initial and review results
            task_data["review_gen_text"] = review_gen_text
            if return_messages_log:
                # Log for the review call itself
                 task_data["full_review_log"] = messages + [{'role': 'assistant', 'content': review_gen_text}]
            return task_data # Return the augmented dictionary

        # Create and gather review tasks
        review_tasks = [
             asyncio.create_task(review_semaphore_helper(
                task_inp
            ))
           for task_inp in review_tasks_input
        ]

        final_results_raw = await asyncio.gather(*review_tasks)

        # Sort final results back into original order (although gather might preserve order for tasks added sequentially)
        final_results_raw.sort(key=lambda x: x["original_index"])

        # <--- Process final results --->
        output: List[FrameExtractionUnitResult] = []
        messages_log: Optional[List[List[Dict[str, str]]]] = [] if return_messages_log else None

        for result_data in final_results_raw:
            unit = result_data["unit"]
            initial_gen = result_data["initial_gen_text"]
            review_gen = result_data["review_gen_text"]

            # Combine based on review mode
            if self.review_mode == "revision":
                final_gen_text = review_gen
            elif self.review_mode == "addition":
                final_gen_text = initial_gen + '\n' + review_gen
            else: # Should not happen due to init check
                final_gen_text = review_gen # Default to revision if mode is somehow invalid

            # Create final result object
            result = FrameExtractionUnitResult(
                start=unit.start,
                end=unit.end,
                text=unit.text,
                gen_text=final_gen_text # Use the combined/reviewed text
            )
            output.append(result)

            # Append full conversation log if requested
            if return_messages_log:
                full_log_for_unit = result_data.get("full_initial_log", []) + [{'role': 'user', 'content': self.review_prompt}] + [{'role': 'assistant', 'content': review_gen}]
                messages_log.append(full_log_for_unit)

        if return_messages_log:
            return output, messages_log
        else:
            return output


class BasicFrameExtractor(DirectFrameExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None):
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
                         unit_chunker=WholeDocumentUnitChunker(),
                         prompt_template=prompt_template, 
                         system_prompt=system_prompt, 
                         context_chunker=NoContextChunker())
        
class BasicReviewFrameExtractor(ReviewFrameExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, review_mode:str, review_prompt:str=None, system_prompt:str=None):
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
        super().__init__(inference_engine=inference_engine, 
                         unit_chunker=WholeDocumentUnitChunker(),
                         prompt_template=prompt_template, 
                         review_mode=review_mode,
                         review_prompt=review_prompt,
                         system_prompt=system_prompt, 
                         context_chunker=NoContextChunker())
        

class SentenceFrameExtractor(DirectFrameExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None,
                 context_sentences:Union[str, int]="all"):
        """
        This class performs sentence-by-sentence information extraction.
        The process is as follows:
            1. system prompt (optional)
            2. user prompt with instructions (schema, background, full text, few-shot example...)
            3. feed a sentence (start with first sentence)
            4. LLM extract entities and attributes from the sentence
            5. iterate to the next sentence and repeat steps 3-4 until all sentences are processed.

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
        context_sentences : Union[str, int], Optional
            number of sentences before and after the given sentence to provide additional context. 
            if "all", the full text will be provided in the prompt as context. 
            if 0, no additional context will be provided.
                This is good for tasks that does not require context beyond the given sentence. 
            if > 0, the number of sentences before and after the given sentence to provide as context.
                This is good for tasks that require context beyond the given sentence. 
        """
        if not isinstance(context_sentences, int) and context_sentences != "all":
            raise ValueError('context_sentences must be an integer (>= 0) or "all".')
        
        if isinstance(context_sentences, int) and context_sentences < 0:
            raise ValueError("context_sentences must be a positive integer.")
            
        if isinstance(context_sentences, int):
            context_chunker = SlideWindowContextChunker(window_size=context_sentences)
        elif context_sentences == "all":
            context_chunker = WholeDocumentContextChunker()

        super().__init__(inference_engine=inference_engine, 
                         unit_chunker=SentenceUnitChunker(),
                         prompt_template=prompt_template, 
                         system_prompt=system_prompt, 
                         context_chunker=context_chunker)
        

class SentenceReviewFrameExtractor(ReviewFrameExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str,  
                 review_mode:str, review_prompt:str=None, system_prompt:str=None,
                 context_sentences:Union[str, int]="all"):
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
        context_sentences : Union[str, int], Optional
            number of sentences before and after the given sentence to provide additional context. 
            if "all", the full text will be provided in the prompt as context. 
            if 0, no additional context will be provided.
                This is good for tasks that does not require context beyond the given sentence. 
            if > 0, the number of sentences before and after the given sentence to provide as context.
                This is good for tasks that require context beyond the given sentence. 
        """
        if not isinstance(context_sentences, int) and context_sentences != "all":
            raise ValueError('context_sentences must be an integer (>= 0) or "all".')
        
        if isinstance(context_sentences, int) and context_sentences < 0:
            raise ValueError("context_sentences must be a positive integer.")
        
        if isinstance(context_sentences, int):
            context_chunker = SlideWindowContextChunker(window_size=context_sentences)
        elif context_sentences == "all":
            context_chunker = WholeDocumentContextChunker()

        super().__init__(inference_engine=inference_engine, 
                         unit_chunker=SentenceUnitChunker(),
                         prompt_template=prompt_template,
                         review_mode=review_mode,
                         review_prompt=review_prompt, 
                         system_prompt=system_prompt, 
                         context_chunker=context_chunker)
    

class AttributeExtractor(Extractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None):
        """
        This class is for attribute extraction for frames. Though FrameExtractors can also extract attributes, when
        the number of attribute increases, it is more efficient to use a dedicated AttributeExtractor.

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
                         system_prompt=system_prompt)
        # validate prompt template
        if "{{context}}" not in self.prompt_template or "{{frame}}" not in self.prompt_template:
            raise ValueError("prompt_template must contain both {{context}} and {{frame}} placeholders.")
        
    def _get_context(self, frame:LLMInformationExtractionFrame, text:str, context_size:int=256) -> str:
        """
        This method returns the context that covers the frame. Leaves a context_size of characters before and after.
        The returned text has the frame inline annotated with <entity>.

        Parameters:
        -----------
        frame : LLMInformationExtractionFrame
            a frame
        text : str
            the entire document text
        context_size : int, Optional
            the number of characters before and after the frame in the context text.

        Return : str
            the context text with the frame inline annotated with <entity>.
        """
        start = max(frame.start - context_size, 0)
        end = min(frame.end + context_size, len(text))
        context = text[start:end]

        context_annotated = context[0:frame.start - start] + \
                f"<entity> " + \
                context[frame.start - start:frame.end - start] + \
                f" </entity>" + \
                context[frame.end - start:end - start]

        if start > 0:
            context_annotated = "..." + context_annotated
        if end < len(text):
            context_annotated = context_annotated + "..."
        return context_annotated
    
    def _extract_from_frame(self, frame:LLMInformationExtractionFrame, text:str,
                            context_size:int=256, verbose:bool=False, return_messages_log:bool=False) -> Dict[str, Any]:
        """
        This method extracts attributes from a single frame.

        Parameters:
        -----------
        frame : LLMInformationExtractionFrame
            a frame to extract attributes from.
        text : str
            the entire document text.
        context_size : int, Optional
            the number of characters before and after the frame in the context text.
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time.
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.

        Return : Dict[str, Any]
            a dictionary of attributes extracted from the frame.
            If return_messages_log is True, a list of messages will be returned as well.
        """
        # construct chat messages
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        context = self._get_context(frame, text, context_size)
        messages.append({'role': 'user', 'content': self._get_user_prompt({"context": context, "frame": str(frame.to_dict())})})

        if verbose:
            print(f"\n\n{Fore.GREEN}Frame: {frame.frame_id}{Style.RESET_ALL}\n{frame.to_dict()}\n")
            if context != "":
                print(f"{Fore.YELLOW}Context:{Style.RESET_ALL}\n{context}\n")
            
            print(f"{Fore.BLUE}Extraction:{Style.RESET_ALL}")

        get_text = self.inference_engine.chat(
                            messages=messages,
                            verbose=verbose,
                            stream=False
                        )
        if return_messages_log:
            messages.append({"role": "assistant", "content": get_text})

        attribute_list = self._extract_json(gen_text=get_text)
        if isinstance(attribute_list, list) and len(attribute_list) > 0:
            attributes = attribute_list[0]
            if return_messages_log:
                return attributes, messages
            return attributes


    def extract(self, frames:List[LLMInformationExtractionFrame], text:str, context_size:int=256, verbose:bool=False, 
                return_messages_log:bool=False, inplace:bool=True) -> Union[None, List[LLMInformationExtractionFrame]]:
        """
        This method extracts attributes from the document.

        Parameters:
        -----------
        frames : List[LLMInformationExtractionFrame]
            a list of frames to extract attributes from.
        text : str
            the entire document text.
        context_size : int, Optional
            the number of characters before and after the frame in the context text.
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.
        inplace : bool, Optional
            if True, the method will modify the frames in-place.
        
        Return : Union[None, List[LLMInformationExtractionFrame]]
            if inplace is True, the method will modify the frames in-place.
            if inplace is False, the method will return a list of frames with attributes extracted.
        """
        for frame in frames:
            if not isinstance(frame, LLMInformationExtractionFrame):
                raise TypeError(f"Expect frame as LLMInformationExtractionFrame, received {type(frame)} instead.")
        if not isinstance(text, str):
            raise TypeError(f"Expect text as str, received {type(text)} instead.")
        
        new_frames = []
        messages_log = [] if return_messages_log else None

        for frame in frames:
            if return_messages_log:
                attr, messages = self._extract_from_frame(frame=frame, text=text, context_size=context_size,
                                                          verbose=verbose, return_messages_log=return_messages_log)
                messages_log.append(messages)
            else: 
                attr = self._extract_from_frame(frame=frame, text=text, context_size=context_size,
                                                verbose=verbose, return_messages_log=return_messages_log)
            
            if inplace:
                frame.attr.update(attr)
            else:
                new_frame = frame.copy()
                new_frame.attr.update(attr)
                new_frames.append(new_frame)

        if inplace:
            return messages_log if return_messages_log else None
        else:
            return (new_frames, messages_log) if return_messages_log else new_frames


    async def extract_async(self, frames:List[LLMInformationExtractionFrame], text:str, context_size:int=256,
                            concurrent_batch_size:int=32, inplace:bool=True, return_messages_log:bool=False) -> Union[None, List[LLMInformationExtractionFrame]]:
        """
        This method extracts attributes from the document asynchronously.

        Parameters:
        -----------
        frames : List[LLMInformationExtractionFrame]
            a list of frames to extract attributes from.
        text : str
            the entire document text.
        context_size : int, Optional
            the number of characters before and after the frame in the context text.
        concurrent_batch_size : int, Optional
            the batch size for concurrent processing. 
        inplace : bool, Optional
            if True, the method will modify the frames in-place.
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.
        
        Return : Union[None, List[LLMInformationExtractionFrame]]
            if inplace is True, the method will modify the frames in-place.
            if inplace is False, the method will return a list of frames with attributes extracted.
        """
        # validation
        for frame in frames:
            if not isinstance(frame, LLMInformationExtractionFrame):
                raise TypeError(f"Expect frame as LLMInformationExtractionFrame, received {type(frame)} instead.")
        if not isinstance(text, str):
            raise TypeError(f"Expect text as str, received {type(text)} instead.")

        # async helper
        semaphore = asyncio.Semaphore(concurrent_batch_size)
        
        async def semaphore_helper(frame:LLMInformationExtractionFrame, text:str, context_size:int) -> dict:
            async with semaphore:
                messages = []
                if self.system_prompt:
                    messages.append({'role': 'system', 'content': self.system_prompt})

                context = self._get_context(frame, text, context_size)
                messages.append({'role': 'user', 'content': self._get_user_prompt({"context": context, "frame": str(frame.to_dict())})})

                gen_text = await self.inference_engine.chat_async(messages=messages)
                
                if return_messages_log:
                    messages.append({"role": "assistant", "content": gen_text})

                attribute_list = self._extract_json(gen_text=gen_text)
                attributes = attribute_list[0] if isinstance(attribute_list, list) and len(attribute_list) > 0 else {}
                return {"frame": frame, "attributes": attributes, "messages": messages}

        # create tasks
        tasks = [asyncio.create_task(semaphore_helper(frame, text, context_size)) for frame in frames]
        results = await asyncio.gather(*tasks)

        # process results
        new_frames = []
        messages_log = [] if return_messages_log else None

        for result in results:
            if return_messages_log:
                messages_log.append(result["messages"])

            if inplace:
                result["frame"].attr.update(result["attributes"])
            else:
                new_frame = result["frame"].copy()
                new_frame.attr.update(result["attributes"])
                new_frames.append(new_frame)

        # output
        if inplace:
            return messages_log if return_messages_log else None
        else:
            return (new_frames, messages_log) if return_messages_log else new_frames

    def extract_attributes(self, frames:List[LLMInformationExtractionFrame], text:str, context_size:int=256, 
                           concurrent:bool=False, concurrent_batch_size:int=32, verbose:bool=False, 
                           return_messages_log:bool=False, inplace:bool=True) -> Union[None, List[LLMInformationExtractionFrame]]:
        """
        This method extracts attributes from the document.

        Parameters:
        -----------
        frames : List[LLMInformationExtractionFrame]
            a list of frames to extract attributes from.
        text : str
            the entire document text.
        context_size : int, Optional
            the number of characters before and after the frame in the context text.
        concurrent : bool, Optional
            if True, the method will run in concurrent mode with batch size concurrent_batch_size.
        concurrent_batch_size : int, Optional
            the batch size for concurrent processing.
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 
        return_messages_log : bool, Optional
            if True, a list of messages will be returned.
        inplace : bool, Optional
            if True, the method will modify the frames in-place.
        
        Return : Union[None, List[LLMInformationExtractionFrame]]
            if inplace is True, the method will modify the frames in-place.
            if inplace is False, the method will return a list of frames with attributes extracted.
        """
        if concurrent:
            if verbose:
                warnings.warn("verbose=True is not supported in concurrent mode.", RuntimeWarning)

            nest_asyncio.apply() # For Jupyter notebook. Terminal does not need this.

            return asyncio.run(self.extract_async(frames=frames, text=text, context_size=context_size,
                                                  concurrent_batch_size=concurrent_batch_size, 
                                                  inplace=inplace, return_messages_log=return_messages_log))
        else:
            return self.extract(frames=frames, text=text, context_size=context_size, 
                                verbose=verbose, return_messages_log=return_messages_log, inplace=inplace)


class RelationExtractor(Extractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None):
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
                         system_prompt=system_prompt)
        
    def _get_ROI(self, frame_1:LLMInformationExtractionFrame, frame_2:LLMInformationExtractionFrame, 
                 text:str, buffer_size:int=128) -> str:
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
        left_frame_name = "entity_1" if left_frame.frame_id == frame_1.frame_id else "entity_2"
        right_frame_name = "entity_1" if right_frame.frame_id == frame_1.frame_id else "entity_2"

        start = max(left_frame.start - buffer_size, 0)
        end = min(right_frame.end + buffer_size, len(text))
        roi = text[start:end]

        roi_annotated = roi[0:left_frame.start - start] + \
                f"<{left_frame_name}> " + \
                roi[left_frame.start - start:left_frame.end - start] + \
                f" </{left_frame_name}>" + \
                roi[left_frame.end - start:right_frame.start - start] + \
                f"<{right_frame_name}> " + \
                roi[right_frame.start - start:right_frame.end - start] + \
                f" </{right_frame_name}>" + \
                roi[right_frame.end - start:end - start]

        if start > 0:
            roi_annotated = "..." + roi_annotated
        if end < len(text):
            roi_annotated = roi_annotated + "..."
        return roi_annotated
    
    @abc.abstractmethod
    def _get_task_if_possible(self, frame_1: LLMInformationExtractionFrame, frame_2: LLMInformationExtractionFrame, 
                              text: str, buffer_size: int) -> Optional[Dict[str, Any]]:
        """Checks if a relation is possible and constructs the task payload."""
        raise NotImplementedError

    @abc.abstractmethod
    def _post_process_result(self, gen_text: str, pair_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes the LLM output for a single pair and returns the final relation dictionary."""
        raise NotImplementedError

    def _extract(self, doc: LLMInformationExtractionDocument, buffer_size: int = 128, verbose: bool = False, 
                 return_messages_log: bool = False) -> Union[List[Dict], Tuple[List[Dict], List]]:
        pairs = itertools.combinations(doc.frames, 2)
        relations = []
        messages_log = [] if return_messages_log else None

        for frame_1, frame_2 in pairs:
            task_payload = self._get_task_if_possible(frame_1, frame_2, doc.text, buffer_size)
            if task_payload:
                if verbose:
                    print(f"\n\n{Fore.GREEN}Evaluating pair:{Style.RESET_ALL} ({frame_1.frame_id}, {frame_2.frame_id})")
                    print(f"{Fore.YELLOW}ROI Text:{Style.RESET_ALL}\n{task_payload['roi_text']}\n")
                    print(f"{Fore.BLUE}Extraction:{Style.RESET_ALL}")

                gen_text = self.inference_engine.chat(
                    messages=task_payload['messages'],
                    verbose=verbose
                )
                relation = self._post_process_result(gen_text, task_payload)
                if relation:
                    relations.append(relation)

                if return_messages_log:
                    task_payload['messages'].append({"role": "assistant", "content": gen_text})
                    messages_log.append(task_payload['messages'])

        return (relations, messages_log) if return_messages_log else relations

    async def _extract_async(self, doc: LLMInformationExtractionDocument, buffer_size: int = 128, concurrent_batch_size: int = 32, return_messages_log: bool = False) -> Union[List[Dict], Tuple[List[Dict], List]]:
        pairs = list(itertools.combinations(doc.frames, 2))
        tasks_input = [self._get_task_if_possible(f1, f2, doc.text, buffer_size) for f1, f2 in pairs]
        # Filter out impossible pairs
        tasks_input = [task for task in tasks_input if task is not None] 

        relations = []
        messages_log = [] if return_messages_log else None
        semaphore = asyncio.Semaphore(concurrent_batch_size)

        async def semaphore_helper(task_payload: Dict):
            async with semaphore:
                gen_text = await self.inference_engine.chat_async(messages=task_payload['messages'])
                return gen_text, task_payload

        tasks = [asyncio.create_task(semaphore_helper(payload)) for payload in tasks_input]
        results = await asyncio.gather(*tasks)

        for gen_text, task_payload in results:
            relation = self._post_process_result(gen_text, task_payload)
            if relation:
                relations.append(relation)

            if return_messages_log:
                task_payload['messages'].append({"role": "assistant", "content": gen_text})
                messages_log.append(task_payload['messages'])

        return (relations, messages_log) if return_messages_log else relations

    def extract_relations(self, doc: LLMInformationExtractionDocument, buffer_size: int = 128, concurrent: bool = False, concurrent_batch_size: int = 32, verbose: bool = False, return_messages_log: bool = False) -> List[Dict]:
        if not doc.has_frame():
            raise ValueError("Input document must have frames.")
        if doc.has_duplicate_frame_ids():
            raise ValueError("All frame_ids in the input document must be unique.")

        if concurrent:
            if verbose:
                warnings.warn("verbose=True is not supported in concurrent mode.", RuntimeWarning)
            nest_asyncio.apply()
            return asyncio.run(self._extract_async(doc, buffer_size, concurrent_batch_size, return_messages_log))
        else:
            return self._extract(doc, buffer_size, verbose, return_messages_log)
    

class BinaryRelationExtractor(RelationExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, possible_relation_func: Callable,
                 system_prompt:str=None):
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
        super().__init__(inference_engine, prompt_template, system_prompt)
        if not callable(possible_relation_func):
            raise TypeError(f"Expect possible_relation_func as a function, received {type(possible_relation_func)} instead.")
        
        sig = inspect.signature(possible_relation_func)
        if len(sig.parameters) != 2:
            raise ValueError("The possible_relation_func must have exactly two parameters.")
        
        if sig.return_annotation not in {bool, inspect.Signature.empty}:
            warnings.warn(f"Expected possible_relation_func return annotation to be bool, but got {sig.return_annotation}.")
        
        self.possible_relation_func = possible_relation_func

    def _get_task_if_possible(self, frame_1: LLMInformationExtractionFrame, frame_2: LLMInformationExtractionFrame, 
                              text: str, buffer_size: int) -> Optional[Dict[str, Any]]:
        if self.possible_relation_func(frame_1, frame_2):
            roi_text = self._get_ROI(frame_1, frame_2, text, buffer_size)
            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})

            messages.append({'role': 'user', 'content': self._get_user_prompt(
                text_content={"roi_text": roi_text, "frame_1": str(frame_1.to_dict()), "frame_2": str(frame_2.to_dict())}
            )})
            return {"frame_1": frame_1, "frame_2": frame_2, "messages": messages, "roi_text": roi_text}
        return None

    def _post_process_result(self, gen_text: str, pair_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rel_json = self._extract_json(gen_text)
        if len(rel_json) > 0 and "Relation" in rel_json[0]:
            rel = rel_json[0]["Relation"]
            if (isinstance(rel, bool) and rel) or (isinstance(rel, str) and rel.lower() == 'true'):
                return {'frame_1_id': pair_data['frame_1'].frame_id, 'frame_2_id': pair_data['frame_2'].frame_id}
        return None
            

class MultiClassRelationExtractor(RelationExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, possible_relation_types_func: Callable, 
                 system_prompt:str=None):
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
                         system_prompt=system_prompt)
        
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


    def _get_task_if_possible(self, frame_1: LLMInformationExtractionFrame, frame_2: LLMInformationExtractionFrame, 
                              text: str, buffer_size: int) -> Optional[Dict[str, Any]]:
        pos_rel_types = self.possible_relation_types_func(frame_1, frame_2)
        if pos_rel_types:
            roi_text = self._get_ROI(frame_1, frame_2, text, buffer_size)
            messages = []
            if self.system_prompt:
                messages.append({'role': 'system', 'content': self.system_prompt})
            messages.append({'role': 'user', 'content': self._get_user_prompt(
                text_content={"roi_text": roi_text, "frame_1": str(frame_1.to_dict()), "frame_2": str(frame_2.to_dict()), "pos_rel_types": str(pos_rel_types)}
            )})
            return {"frame_1": frame_1, "frame_2": frame_2, "messages": messages, "pos_rel_types": pos_rel_types, "roi_text": roi_text}
        return None

    def _post_process_result(self, gen_text: str, pair_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rel_json = self._extract_json(gen_text)
        pos_rel_types = pair_data['pos_rel_types']
        if len(rel_json) > 0 and "RelationType" in rel_json[0]:
            rel_type = rel_json[0]["RelationType"]
            if rel_type in pos_rel_types:
                return {'frame_1_id': pair_data['frame_1'].frame_id, 'frame_2_id': pair_data['frame_2'].frame_id, 'relation': rel_type}
        return None