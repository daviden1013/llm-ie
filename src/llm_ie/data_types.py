from typing import List, Dict
import yaml


class LLMInformationExtractionFrame:
    def __init__(self, frame_id:str, start:int, end:int, entity_text:str, attr:Dict[str,str]):
        """
        This class holds a frame (entity) extracted by LLM. 
        A frame contains the span (start and end character positions), a entity text, and 
        a set of attributes. 

        Parameters
        ----------
        frame_id : str
            unique identiifier for the entity
        start : int
            entity start character position
        end : int
            entity end character position
        entity_text : str
            entity string. Should be the exact string by [start:end]
        attr : Dict[str,str]
            dict of attributes
        """
        assert isinstance(frame_id, str), "frame_id must be a string."
        self.frame_id = frame_id
        self.start = start
        self.end = end
        self.entity_text = entity_text
        self.attr = attr.copy()

    def is_equal(self, frame:"LLMInformationExtractionFrame") -> bool:
        """ 
        This method checks if an external frame holds the same information as self.
        This can be used in evaluation against gold standard. 
        """
        return self.start == frame.start and self.end == frame.end

    def is_overlap(self, frame:"LLMInformationExtractionFrame") -> bool:
        """ 
        This method checks if an external frame overlaps with self.
        This can be used in evaluation against gold standard. 
        """
        if self.end < frame.start or self.start > frame.end:
            return False
        return True

    def to_dict(self) -> Dict[str,str]:
        """
        This method outputs the frame contents to a dictionary.
        """
        return {"frame_id": self.frame_id,
                "start": self.start,
                "end": self.end,
                "entity_text": self.entity_text,
                "attr": self.attr}
    
    @classmethod
    def from_dict(cls, d: Dict[str,str]) -> "LLMInformationExtractionFrame":
        """ 
        This method defines a LLMInformationExtractionFrame from dictionary.
        """
        return cls(frame_id=d['frame_id'],
                    start=d['start'],
                    end=d['end'],
                    entity_text=d['entity_text'],
                    attr=d['attr'])

    def copy(self) -> "LLMInformationExtractionFrame":
        return LLMInformationExtractionFrame(frame_id=self.frame_id,
                                            start=self.start,
                                            end=self.end,
                                            entity_text=self.entity_text,
                                            attr=self.attr)


class LLMInformationExtractionDocument:
    def __init__(self, doc_id:str=None, filename:str=None, text:str=None, frames:List[LLMInformationExtractionFrame]=None):
        """
        This class holds LLM-extracted frames, handles save/ load.

        Parameters
        ----------
        doc_id : str, Optional
            document ID. Must be a string
        filename : str, Optional
            the directory to a yaml file of a saved LLMInformationExtractionDocument
        text : str, Optional
            document text
        frames : List[LLMInformationExtractionFrame], Optional
            a list of LLMInformationExtractionFrame
        """
        assert doc_id or filename, "Either doc_id (create from raw inputs) or filename (create from file) must be provided."
        # if create object from file
        if filename:
            with open(filename) as yaml_file:
                llm_ie = yaml.safe_load(yaml_file)
            if 'doc_id' in llm_ie.keys():
                self.doc_id = llm_ie['doc_id']
            if 'text' in llm_ie.keys():
                self.text = llm_ie['text']
            if 'frames' in llm_ie.keys():
                self.frames = [LLMInformationExtractionFrame.from_dict(d) for d in llm_ie['frames']]

        # create object from raw inputs
        else:
            assert isinstance(doc_id, str), "doc_id must be a string."
            self.doc_id = doc_id
            self.text = text
            self.frames = frames.copy() if frames is not None else []


    def has_frame(self) -> bool:
        return bool(self.frames)
    
    def add_frame(self, frame:LLMInformationExtractionFrame, valid_mode:str=None, create_id:bool=False) -> bool:
        """
        This method add a new frame to the frames (list).

        Parameters
        ----------
        frame : LLMInformationExtractionFrame
            the new frame to add.
        valid_mode : str, Optional
            one of {None, "span", "attr"}
            if None, no validation will be done, add frame.
            if "span", if the new frame's span is equal to an existing frame, add will fail.
            if "attr", if the new frame's span and all attributes is equal to an existing frame, add will fail.
        create_id : bool, Optional
            Assign a sequential frame ID.
        """
        assert valid_mode in {None, "span", "attr"}, 'valid_mode must be one of {None, "span", "attr"}'

        if valid_mode == "span":
            for exist_frame in self.frames:
                if exist_frame.is_equal(frame):
                    return False

        if valid_mode == "attr":
            for exist_frame in self.frames:
                if exist_frame.is_equal(frame) and exist_frame.attr == frame.attr:
                    return False
        
        # Add frame
        frame_clone = frame.copy()
        if create_id:
            frame_clone.doc_id = f"{self.doc_id}_{len(self.frames)}"

        self.frames.append(frame_clone)
        return True


    def __repr__(self, N_top_chars:int=100) -> str:
        text_to_print = self.text[0:N_top_chars]
        frame_count = len(self.frames)
        return ''.join((f'LLMInformationExtractionDocument(doc_id="{self.doc_id}...")\n',
                        f'text="{text_to_print}...",\n',
                        f'frames={frame_count}'))

    def save(self, filename:str):
        with open(filename, 'w') as yaml_file:
            yaml.safe_dump({'doc_id':self.doc_id, 
                            'text':self.text, 
                            'frames':[frame.to_dict() for frame in self.frames]}, 
                            yaml_file, sort_keys=False)
            yaml_file.flush()
            
