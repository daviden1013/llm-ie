import abc
import re
import json
import inspect
import importlib.resources
import warnings
import itertools
from typing import List, Dict, Tuple, Union, Callable
from llm_ie.data_types import LLMInformationExtractionFrame, LLMInformationExtractionDocument
from llm_ie.engines import InferenceEngine


class Extractor:
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None, **kwrs):
        """
        This is the abstract class for (frame and relation) extractors.
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
        self.inference_engine = inference_engine
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt


    @classmethod
    def get_prompt_guide(cls) -> str:
        """
        This method returns the pre-defined prompt guideline for the extractor from the package asset.
        """
        file_path = importlib.resources.files('llm_ie.asset.prompt_guide').joinpath(f"{cls.__name__}_prompt_guide.txt")
        with open(file_path, 'r') as f:
            return f.read()


    def _get_user_prompt(self, text_content:Union[str, Dict[str,str]]) -> str:
        """
        This method applies text_content to prompt_template and returns a prompt.

        Parameters
        ----------
        text_content : Union[str, Dict[str,str]]
            the input text content to put in prompt template. 
            If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
            If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}.

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
            placeholders = pattern.findall(self.prompt_template)
            if len(placeholders) != len(text_content):
                raise ValueError(f"Expect text_content ({len(text_content)}) and prompt template placeholder ({len(placeholders)}) to have equal size.")
            if not all([k in placeholders for k, _ in text_content.items()]):
                raise ValueError(f"All keys in text_content ({text_content.keys()}) must match placeholders in prompt template ({placeholders}).")

            prompt = pattern.sub(lambda match: re.sub(r'\\', r'\\\\', text_content[match.group(1)]), self.prompt_template)

        return prompt
    
    def _extract_json(self, gen_text:str) -> List[Dict[str, str]]:
        """ 
        This method inputs a generated text and output a JSON of information tuples
        """
        pattern = r'\{.*?\}'
        out = []
        for match in re.findall(pattern, gen_text, re.DOTALL):
            try:
                tup_dict = json.loads(match)
                out.append(tup_dict)
            except json.JSONDecodeError:
                print(f'Post-processing failed at:\n{match}')
        return out
    

class FrameExtractor(Extractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, system_prompt:str=None, **kwrs):
        """
        This is the abstract class for frame extraction.
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
    

    def _find_entity_spans(self, text: str, entities: List[str], case_sensitive:bool=False) -> List[Tuple[int]]:
        """
        This function inputs a text and a list of entity text, 
        outputs a list of spans (2-tuple) for each entity.
        Entities that are not found in the text will be None from output.

        Parameters
        ----------
        text : str
            text that contains entities
        """
        entity_spans = []
        for entity in entities:            
            if case_sensitive:
                match = re.search(re.escape(entity), text)
            else: 
                match = re.search(re.escape(entity), text, re.IGNORECASE)
                
            if match:
                start, end = match.span()
                entity_spans.append((start, end))
                # Replace the found entity with spaces to avoid finding the same instance again
                text = text[:start] + ' ' * (end - start) + text[end:]
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
                       temperature:float=0.0, document_key:str=None, **kwrs) -> List[LLMInformationExtractionFrame]:
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

        Return : str
            a list of frames.
        """
        frame_list = []
        gen_text = self.extract(text_content=text_content, 
                                max_new_tokens=max_new_tokens, temperature=temperature, **kwrs)
        entity_json = self._extract_json(gen_text=gen_text)
        if isinstance(text_content, str):
            text = text_content
        elif isinstance(text_content, dict):
            text = text_content[document_key]

        spans = self._find_entity_spans(text=text, 
                                        entities=[e[entity_key] for e in entity_json], 
                                        case_sensitive=False)
        
        for i, (ent, span) in enumerate(zip(entity_json, spans)):
            if span is not None:
                start, end = span
                frame = LLMInformationExtractionFrame(frame_id=f"{i}", 
                            start=start,
                            end=end,
                            entity_text=ent[entity_key],
                            attr={k: v for k, v in ent.items() if k != entity_key and v != ""})
                frame_list.append(frame)
        return frame_list
        

class ReviewFrameExtractor(BasicFrameExtractor):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str, review_prompt:str, 
                 review_mode:str, system_prompt:str=None, **kwrs):
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
        review_prompt : str
            the prompt text that ask LLM to review. Specify addition or revision in the instruction. 
        review_mode : str
            review mode. Must be one of {"addition", "revision"}
            addition mode only ask LLM to add new frames, while revision mode ask LLM to regenerate.
        system_prompt : str, Optional
            system prompt.
        """
        super().__init__(inference_engine=inference_engine, prompt_template=prompt_template, 
                         system_prompt=system_prompt, **kwrs)
        self.review_prompt = review_prompt
        if review_mode not in {"addition", "revision"}: 
            raise ValueError('review_mode must be one of {"addition", "revision"}.')
        self.review_mode = review_mode


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
        # Pormpt extraction
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': self._get_user_prompt(text_content)})
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
        This class performs sentence-based information extraction.
        A simulated chat follows this process:
            1. system prompt (optional)
            2. user instructions (schema, background, full text, few-shot example...)
            3. user input first sentence
            4. assistant extract outputs
            5. repeat #3 and #4

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
        
    def _get_sentences(self, text:str) -> List[Dict[str,str]]:
        """
        This method sentence tokenize the input text into a list of sentences 
        as dict of {start, end, sentence_text}

        Parameters
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
                document_key:str=None, multi_turn:bool=True, temperature:float=0.0, stream:bool=False, **kwrs) -> List[Dict[str,str]]:
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
                print(f"\n\nSentence: \n{sent['sentence_text']}\n")
                print("Extraction:")

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
    

    def extract_frames(self, text_content:Union[str, Dict[str,str]], entity_key:str, max_new_tokens:int=512, 
                       document_key:str=None, multi_turn:bool=True, temperature:float=0.0, stream:bool=False, **kwrs) -> List[LLMInformationExtractionFrame]:
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

        Return : str
            a list of frames.
        """
        llm_output_sentence = self.extract(text_content=text_content, 
                                           max_new_tokens=max_new_tokens, 
                                           document_key=document_key,
                                           multi_turn=multi_turn, 
                                           temperature=temperature, 
                                           stream=stream,
                                           **kwrs)
        frame_list = []
        for sent in llm_output_sentence:
            entity_json = self._extract_json(gen_text=sent['gen_text'])
            spans = self._find_entity_spans(text=sent['sentence_text'], 
                                                entities=[e[entity_key] for e in entity_json], case_sensitive=False)
            for ent, span in zip(entity_json, spans):
                if span is not None:
                    start, end = span
                    start += sent['sentence_start']
                    end += sent['sentence_start']
                    frame = LLMInformationExtractionFrame(frame_id=f"{len(frame_list)}", 
                                start=start,
                                end=end,
                                entity_text=ent[entity_key],
                                attr={k: v for k, v in ent.items() if k != entity_key and v != ""})
                    frame_list.append(frame)
        return frame_list


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


    def _extract_relation(self, frame_1:LLMInformationExtractionFrame, frame_2:LLMInformationExtractionFrame, 
                      text:str, buffer_size:int=100, max_new_tokens:int=128, temperature:float=0.0, stream:bool=False, **kwrs) -> bool:
        """
        This method inputs two frames and a ROI text, extracts the binary relation.

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
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : bool
            a relation indicator
        """
        roi_text = self._get_ROI(frame_1, frame_2, text, buffer_size=buffer_size)
        if stream:
            print(f"\n\nROI text: \n{roi_text}\n")
            print("Extraction:")

        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': self._get_user_prompt(text_content={"roi_text":roi_text, 
                                                                                        "frame_1": str(frame_1.to_dict()),
                                                                                        "frame_2": str(frame_2.to_dict())}
                                                                                        )})
        response = self.inference_engine.chat(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stream=stream,
                    **kwrs
                )
        
        rel_json = self._extract_json(response)
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
            warnings.warn("Extractor did not output a JSON. Following default, relation = False.", RuntimeWarning)

        return False
    
    
    def extract_relations(self, doc:LLMInformationExtractionDocument, buffer_size:int=100, max_new_tokens:int=128, 
                         temperature:float=0.0, stream:bool=False, **kwrs) -> List[Dict]:
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
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : List[Dict]
            a list of dict with {"frame_1", "frame_2"} for all relations.
        """
        if not doc.has_frame():
            raise ValueError("Input document must have frames.")

        if doc.has_duplicate_frame_ids():
            raise ValueError("All frame_ids in the input document must be unique.")

        pairs = itertools.combinations(doc.frames, 2)
        rel_pair_list = []
        for frame_1, frame_2 in pairs:
            pos_rel = self.possible_relation_func(frame_1, frame_2)
            if pos_rel:
                rel = self._extract_relation(frame_1=frame_1, frame_2=frame_2, text=doc.text, buffer_size=buffer_size, 
                                         max_new_tokens=max_new_tokens, temperature=temperature, stream=stream, **kwrs)
                if rel:
                    rel_pair_list.append({'frame_1':frame_1.frame_id, 'frame_2':frame_2.frame_id})

        return rel_pair_list



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

        
    def _extract_relation(self, frame_1:LLMInformationExtractionFrame, frame_2:LLMInformationExtractionFrame, 
                      pos_rel_types:List[str], text:str, buffer_size:int=100, max_new_tokens:int=128, temperature:float=0.0, stream:bool=False, **kwrs) -> str:
        """
        This method inputs two frames and a ROI text, extracts the relation.

        Parameters:
        -----------
        frame_1 : LLMInformationExtractionFrame
            a frame
        frame_2 : LLMInformationExtractionFrame
            the other frame
        pos_rel_types : List[str]
            possible relation types.
        text : str
            the entire document text
        buffer_size : int, Optional
            the number of characters before and after the two frames in the ROI text.
        max_new_tokens : str, Optional
            the max number of new tokens LLM should generate. 
        temperature : float, Optional
            the temperature for token sampling.
        stream : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time. 

        Return : str
            a relation type 
        """
        roi_text = self._get_ROI(frame_1, frame_2, text, buffer_size=buffer_size)
        if stream:
            print(f"\n\nROI text: \n{roi_text}\n")
            print("Extraction:")

        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})

        messages.append({'role': 'user', 'content': self._get_user_prompt(text_content={"roi_text":roi_text, 
                                                                                        "frame_1": str(frame_1.to_dict()),
                                                                                        "frame_2": str(frame_2.to_dict()),
                                                                                        "pos_rel_types":str(pos_rel_types)})})
        response = self.inference_engine.chat(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stream=stream,
                    **kwrs
                )
        
        rel_json = self._extract_json(response)
        if len(rel_json) > 0:
            if "RelationType" in rel_json[0]:
                rel = rel_json[0]["RelationType"]
                if rel in pos_rel_types:
                    return rel_json[0]["RelationType"]
                else:
                    warnings.warn(f'Extracted relation type "{rel}", which is not in the return of possible_relation_types_func: {pos_rel_types}.'+ \
                                  'Following default, relation = "No Relation".', RuntimeWarning)
            
            else:
                warnings.warn('Extractor output JSON without "RelationType" key. Following default, relation = "No Relation".', RuntimeWarning)
        
        else:
            warnings.warn('Extractor did not output a JSON. Following default, relation = "No Relation".', RuntimeWarning)

        return "No Relation"


    def extract_relations(self, doc:LLMInformationExtractionDocument, buffer_size:int=100, max_new_tokens:int=128, 
                         temperature:float=0.0, stream:bool=False, **kwrs) -> List[Dict]:
        """
        This method considers all combinations of two frames. Use the possible_relation_types_func to filter impossible pairs 
        and to provide possible relation types between two frames. 

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
            a list of dict with {"frame_1", "frame_2", "relation"} for all relations.
        """
        if not doc.has_frame():
            raise ValueError("Input document must have frames.")

        if doc.has_duplicate_frame_ids():
            raise ValueError("All frame_ids in the input document must be unique.")

        pairs = itertools.combinations(doc.frames, 2)
        rel_pair_list = []
        for frame_1, frame_2 in pairs:
            pos_rel_types = self.possible_relation_types_func(frame_1, frame_2)
            if pos_rel_types:
                rel = self._extract_relation(frame_1=frame_1, frame_2=frame_2, pos_rel_types=pos_rel_types, text=doc.text, 
                                         buffer_size=buffer_size, max_new_tokens=max_new_tokens, temperature=temperature, stream=stream, **kwrs)
            
                if rel != "No Relation":
                    rel_pair_list.append({'frame_1':frame_1.frame_id, 'frame_2':frame_2.frame_id, "relation":rel})

        return rel_pair_list
