import abc
from typing import Set, List, Dict, Tuple, Union, Callable
from llm_ie.data_types import FrameExtractionUnit


class UnitChunker(abc.ABC):
    def __init__(self):
        """
        This is the abstract class for frame extraction unit chunker.
        It chunks a document into units (e.g., sentences). LLMs process unit by unit. 
        """
        pass

    def chunk(self, text:str) -> List[FrameExtractionUnit]:
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        return NotImplemented


class WholeDocumentUnitChunker(UnitChunker):
    def __init__(self):
        """
        This class chunks the whole document into a single unit (no chunking).
        """
        super().__init__()

    def chunk(self, text:str) -> List[FrameExtractionUnit]:
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        return [FrameExtractionUnit(
            start=0,
            end=len(text),
            text=text
        )]
    

class SentenceUnitChunker(UnitChunker):
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    def __init__(self):
        """
        This class uses the NLTK PunktSentenceTokenizer to chunk a document into sentences.
        """
        super().__init__()

    def chunk(self, text:str) -> List[FrameExtractionUnit]:
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        sentences = []
        for start, end in self.PunktSentenceTokenizer().span_tokenize(text):
            sentences.append(FrameExtractionUnit(
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

    def chunk(self, text:str) -> List[FrameExtractionUnit]:
        """
        Parameters:
        ----------
        text : str
            The document text.
        """
        lines = text.split('\n')
        line_units = []
        start = 0
        for line in lines:
            end = start + len(line)
            line_units.append(FrameExtractionUnit(
                start=start,
                end=end,
                text=line
            ))
            start = end + 1 
        return line_units
    

class ContextChunker(abc.ABC):
    def __init__(self):
        """
        This is the abstract class for context chunker. Given a frame extraction unit,
        it returns the context for it.
        """
        pass

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
    
