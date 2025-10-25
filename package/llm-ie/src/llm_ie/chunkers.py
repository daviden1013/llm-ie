import abc
from typing import List
import asyncio
import uuid
import importlib.resources
from llm_ie.utils import extract_json, apply_prompt_template
from llm_ie.data_types import FrameExtractionUnit
from llm_ie.engines import InferenceEngine


class UnitChunker(abc.ABC):
    def __init__(self):
        """
        This is the abstract class for frame extraction unit chunker.
        It chunks a document into units (e.g., sentences). LLMs process unit by unit. 
        """
        pass

    @abc.abstractmethod
    def chunk(self, text:str, doc_id:str=None) -> List[FrameExtractionUnit]:
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        return NotImplemented

    async def chunk_async(self, text:str, doc_id:str=None, executor=None) -> List[FrameExtractionUnit]:
        """
        asynchronous version of chunk method.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.chunk, text, doc_id)

class WholeDocumentUnitChunker(UnitChunker):
    def __init__(self):
        """
        This class chunks the whole document into a single unit (no chunking).
        """
        super().__init__()

    def chunk(self, text:str, doc_id:str=None) -> List[FrameExtractionUnit]:
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        return [FrameExtractionUnit(
            doc_id=doc_id if doc_id is not None else str(uuid.uuid4()),
            start=0,
            end=len(text),
            text=text
        )]
    
class SeparatorUnitChunker(UnitChunker):
    def __init__(self, sep:str):
        """
        This class chunks a document by separator provided.

        Parameters:
        ----------
        sep : str
            a separator string.
        """
        super().__init__()
        if not isinstance(sep, str):
            raise ValueError("sep must be a string")

        self.sep = sep

    def chunk(self, text:str, doc_id:str=None) -> List[FrameExtractionUnit]:
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        doc_id = doc_id if doc_id is not None else str(uuid.uuid4())
        paragraphs = text.split(self.sep)
        paragraph_units = []
        start = 0
        for paragraph in paragraphs:
            end = start + len(paragraph)
            paragraph_units.append(FrameExtractionUnit(
                doc_id=doc_id,
                start=start,
                end=end,
                text=paragraph
            ))
            start = end + len(self.sep)
        return paragraph_units


class SentenceUnitChunker(UnitChunker):
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    def __init__(self):
        """
        This class uses the NLTK PunktSentenceTokenizer to chunk a document into sentences.
        """
        super().__init__()

    def chunk(self, text:str, doc_id:str=None) -> List[FrameExtractionUnit]:
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        doc_id = doc_id if doc_id is not None else str(uuid.uuid4())
        sentences = []
        for start, end in self.PunktSentenceTokenizer().span_tokenize(text):
            sentences.append(FrameExtractionUnit(
                doc_id=doc_id,
                start=start,
                end=end,
                text=text[start:end]
            ))    
        return sentences
    

class TextLineUnitChunker(UnitChunker):
    def __init__(self):
        """
        This class chunks a document into lines.
        """
        super().__init__()

    def chunk(self, text:str, doc_id:str=None) -> List[FrameExtractionUnit]:
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        doc_id = doc_id if doc_id is not None else str(uuid.uuid4())
        lines = text.split('\n')
        line_units = []
        start = 0
        for line in lines:
            end = start + len(line)
            line_units.append(FrameExtractionUnit(
                doc_id=doc_id,
                start=start,
                end=end,
                text=line
            ))
            start = end + 1 
        return line_units
    
class LLMUnitChunker(UnitChunker):
    def __init__(self, inference_engine:InferenceEngine, prompt_template:str=None, system_prompt:str=None):
        """
        This class prompt an LLM for document segmentation (e.g., sections, paragraphs).

        Parameters:
        ----------
        inference_engine : InferenceEngine
            the LLM inferencing engine object.
        prompt_template : str
            the prompt template that defines how to chunk the document. Must define a JSON schema with 
            ```json
            [
                {
                    "title": "<your title here>",
                    "anchor_text": "<the anchor text of the chunk here>"
                },
                {
                    "title": "<your title here>",
                    "anchor_text": "<the anchor text of the chunk here>"
                }
            ]
            ```
        system_prompt : str, optional
            The system prompt.
        """
        self.inference_engine = inference_engine

        if prompt_template is None:
            file_path = importlib.resources.files('llm_ie.asset.default_prompts').joinpath("LLMUnitChunker_user_prompt.txt")
            with open(file_path, 'r', encoding="utf-8") as f:
                self.prompt_template = f.read()
        else:
            self.prompt_template = prompt_template
        
        self.system_prompt = system_prompt

    def chunk(self, text, doc_id=None) -> List[FrameExtractionUnit]:
        """
        Parameters:
        -----------
        text : str
            the document text.
        doc_id : str, optional
            the document id.
        """
        doc_id = doc_id if doc_id is not None else str(uuid.uuid4())
        user_prompt = apply_prompt_template(prompt_template=self.prompt_template, text_content=text)
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        gen_text = self.inference_engine.chat(messages=messages)

        header_list = extract_json(gen_text=gen_text["response"])
        units = []
        start = 0
        prev_end = 0
        for header in header_list:
            if "anchor_text" not in header:
                Warning.warn(f"Missing anchor_text in header: {header}. Skipping this header.")
                continue
            if not isinstance(header["anchor_text"], str):
                Warning.warn(f"Invalid anchor_text: {header['anchor_text']}. Skipping this header.")
                continue

            start = prev_end
            # find the first instandce of the leading sentence in the rest of the text
            end = text.find(header["anchor_text"], start)
            # if not found, skip this header
            if end == -1:
                continue
            # if start == end (empty text), skip this header
            if start == end:
                continue
            # create a frame extraction unit
            units.append(FrameExtractionUnit(
                doc_id=doc_id,
                start=start,
                end=end,
                text=text[start:end]
            ))
            prev_end = end
        # add the last section
        if prev_end < len(text):
            units.append(FrameExtractionUnit(
                doc_id=doc_id,
                start=prev_end,
                end=len(text),
                text=text[prev_end:]
            ))
        return units


class ContextChunker(abc.ABC):
    def __init__(self):
        """
        This is the abstract class for context chunker. Given a frame extraction unit,
        it returns the context for it.
        """
        pass

    @abc.abstractmethod
    def fit(self, text:str, units:List[FrameExtractionUnit]):
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        pass

    async def fit_async(self, text:str, units:List[FrameExtractionUnit], executor=None):
        """
        asynchronous version of fit method.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.fit, text, units)

    @abc.abstractmethod
    def chunk(self, unit:FrameExtractionUnit) -> str:
        """
        Parameters:
        ----------
        unit : FrameExtractionUnit
            The frame extraction unit.

        Return : str 
            The context for the frame extraction unit.
        """
        return NotImplemented
    
    async def chunk_async(self, unit:FrameExtractionUnit, executor=None) -> str:
        """
        asynchronous version of chunk method.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.chunk, unit)
    

class NoContextChunker(ContextChunker):
    def __init__(self):
        """
        This class does not provide any context.
        """
        super().__init__()

    def fit(self, text:str, units:List[FrameExtractionUnit]):
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        pass

    def chunk(self, unit:FrameExtractionUnit) -> str:
        return ""
    
class WholeDocumentContextChunker(ContextChunker):
    def __init__(self):
        """
        This class provides the whole document as context.
        """
        super().__init__()
        self.text = None

    def fit(self, text:str, units:List[FrameExtractionUnit]):
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        self.text = text

    def chunk(self, unit:FrameExtractionUnit) -> str:
        if self.text is None:
            raise ValueError("The context chunker has not been fitted yet. Please call fit() before chunk().")
        return self.text
    
class SlideWindowContextChunker(ContextChunker):
    def __init__(self, window_size:int):
        """
        This class provides a sliding window context. For example, +-2 sentences around a unit sentence. 
        """
        super().__init__()
        self.window_size = window_size
        self.units = None

    def fit(self, text:str, units:List[FrameExtractionUnit]):
        """
        Parameters:
        ----------
        units : List[FrameExtractionUnit]
            The list of frame extraction units.
        """
        self.units = sorted(units)


    def chunk(self, unit:FrameExtractionUnit) -> str:
        if self.units is None:
            raise ValueError("The context chunker has not been fitted yet. Please call fit() before chunk().")
        
        index = self.units.index(unit)
        start = max(0, index - self.window_size)
        end = min(len(self.units), index + self.window_size + 1)
        context = []
        for i in range(start, end):
            context.append(self.units[i].text)

        return " ".join(context)
    
