import warnings
from llm_ie import DirectFrameExtractor, LLMInformationExtractionFrame
from llm_ie.engines import InferenceEngine
from llm_ie.data_types import FrameExtractionUnitResult, FrameExtractionUnit
from llm_ie.chunkers import UnitChunker, WholeDocumentUnitChunker, SentenceUnitChunker
from llm_ie.chunkers import ContextChunker, NoContextChunker, WholeDocumentContextChunker, SlideWindowContextChunker
from typing import Any, Union, Dict, List, Generator, Optional
import json

class AppDirectFrameExtractor(DirectFrameExtractor):
    """
    Extends DirectFrameExtractor to provide a specific streaming behavior
    for the frontend application. It streams raw LLM outputs during generation,
    then collects all generated text, performs post-processing at the end,
    and then streams the final processed frames.

    Parameters:
        ----------
        unit_chunker : FrameExtractionUnitChunker
            the unit chunker object that determines how to chunk the document text into units.
        context_chunker : ContextChunker
            the context chunker object that determines how to get context for each unit.
        inference_engine : InferenceEngine
            the LLM inferencing engine object. Must implements the chat() method.
        prompt_template : str
            prompt template with "{{<placeholder name>}}" placeholder.
        system_prompt : str, Optional
            system prompt.
    """

    def __init__(self, inference_engine:InferenceEngine, unit_chunker:UnitChunker, 
                 prompt_template:str, system_prompt:str=None, context_chunker:ContextChunker=None, **kwrs):
        super().__init__(inference_engine=inference_engine,
                         unit_chunker=unit_chunker,
                         prompt_template=prompt_template,
                         system_prompt=system_prompt,
                         context_chunker=context_chunker,
                         **kwrs)

    def stream(self, text_content: Union[str, Dict[str, str]], document_key: str = None) -> Generator[Dict[str, Any], None, List[FrameExtractionUnitResult]]:
        """
        Streams LLM responses per unit with structured event types,
        and returns collected data for post-processing.

        Yields:
        -------
        Dict[str, Any]: (type, data)
            - {"type": "info", "data": str_message}: General informational messages.
            - {"type": "unit", "data": dict_unit_info}: Signals start of a new unit. dict_unit_info contains {'id', 'text', 'start', 'end'}
            - {"type": "context", "data": str_context}: Context string for the current unit.
            - {"type": "llm_chunk", "data": str_chunk}: A raw chunk from the LLM.

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
            # chunk is a generator Dict[str, str]. {"type": "response", "data": <token>} or {"type": "reasoning", "data": <token>}
            for chunk in response_stream:
                yield chunk
                # only collect the response chunks (not reasoning)
                if chunk.get("type") == "response":
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

    def post_process_frames(self, extraction_results:List[FrameExtractionUnitResult],
                            case_sensitive:bool=False, fuzzy_match:bool=True, fuzzy_buffer_size:float=0.2, fuzzy_score_cutoff:float=0.8,
                            allow_overlap_entities:bool=False, return_messages_log:bool=False) -> List[LLMInformationExtractionFrame]:
        """
        This method inputs a text and outputs a list of LLMInformationExtractionFrame
        It use the extract() method and post-process outputs into frames.

        Parameters:
        ----------
        extraction_results : List[FrameExtractionUnitResult]
            a list of FrameExtractionUnitResult objects.
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