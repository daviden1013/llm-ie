import abc
import re
import json
import importlib.resources
from typing import List, Dict, Tuple, Union
from llm_ie.data_types import LLMInformationExtractionFrame
from llm_ie.engines import InferenceEngine


class FrameExtractor:
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
        self.inference_engine = inference_engine
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

    @classmethod
    def get_prompt_guide(cls) -> str:
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
            assert len(matches) == 1, \
                "When text_content is str, the prompt template must has only 1 placeholder {{<placeholder name>}}."
            prompt = pattern.sub(text_content, self.prompt_template)

        elif isinstance(text_content, dict):
            placeholders = pattern.findall(self.prompt_template)
            assert len(placeholders) == len(text_content), \
                f"Expect text_content ({len(text_content)}) and prompt template placeholder ({len(placeholders)}) to have equal size."
            assert all([k in placeholders for k, _ in text_content.items()]), \
                f"All keys in text_content ({text_content.keys()}) must match placeholders in prompt template ({placeholders})."

            prompt = pattern.sub(lambda match: text_content[match.group(1)], self.prompt_template)

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
        assert review_mode in {"addition", "revision"}, 'review_mode must be one of {"addition", "revision"}.'
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
