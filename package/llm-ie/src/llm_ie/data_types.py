from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Callable
import importlib.util
import warnings
import json


@dataclass
class FrameExtractionUnit:
    def __init__(self, start:int, end:int, text:str):
        """
        This class holds the unit text for frame extraction, for example, a sentence. 
        FrameExtractor prompt it one at a time to extract frames. 
        
        Parameters
        ----------
        start : int
            start character position of the unit text, relative to the whole document
        end : int
            end character position of the unit text, relative to the whole document
        text : str
            the unit text. Should be the exact string by [start:end]
        """
        self.start = start
        self.end = end
        self.text = text

    def __eq__(self, other):
            if not isinstance(other, FrameExtractionUnit):
                return NotImplemented
            return (self.start == other.start and self.end == other.end)

    def __hash__(self):
        return hash((self.start, self.end))
        
    def __lt__(self, other):
        if not isinstance(other, FrameExtractionUnit):
            return NotImplemented
        return self.start < other.start
        
    def __repr__(self):
        return f"FrameExtractionUnit(start={self.start}, end={self.end}, text='{self.text[:100]}...')"


@dataclass
class FrameExtractionUnitResult:
    def __init__(self, start:int, end:int, text:str, gen_text:str):
        """
        This class holds the unit text for frame extraction, for example, a sentence. 
        FrameExtractor prompt it one at a time to extract frames. 
        
        Parameters
        ----------
        start : int
            start character position of the unit text, relative to the whole document
        end : int
            end character position of the unit text, relative to the whole document
        text : str
            the unit text. Should be the exact string by [start:end]
        gen_text : str
            the generated text by LLM (ideally) following '[{"entity_text": "xxx", "attr": {"key": "value"}}]' format. Does not contain spans (start/end).
        """
        self.start = start
        self.end = end
        self.text = text
        self.gen_text = gen_text
        
    def __eq__(self, other):
            if not isinstance(other, FrameExtractionUnit):
                return NotImplemented
            return (self.start == other.start and self.end == other.end and self.text == other.text and self.gen_text == other.gen_text)

    def __hash__(self):
        return hash((self.start, self.end, self.text, self.gen_text))

    def __repr__(self):
        return f"FrameExtractionUnitResult(start={self.start}, end={self.end}, text='{self.text[:100]}...', gen_text='{self.gen_text[:100]}...')"
    

@dataclass
class LLMInformationExtractionFrame:
    def __init__(self, frame_id:str, start:int, end:int, entity_text:str, attr:Dict[str,str]=None):
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
        attr : Dict[str,str], Optional
            dict of attributes
        """
        if not isinstance(frame_id, str):
            raise TypeError("frame_id must be a string.")
        self.frame_id = frame_id
        self.start = start
        self.end = end
        self.entity_text = entity_text
        if attr:
            self.attr = attr.copy()
        else:
            self.attr = {}

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
    def __init__(self, doc_id:str=None, filename:str=None, text:str=None, 
                 frames:List[LLMInformationExtractionFrame]=None, relations:List[Dict[str,str]]=None):
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
        relations : List[Dict[str,str]], Optional
            a list of dictionary of {"frame_1", "frame_2", "relation"}. 
            If binary relation (no relation type), there is no "relation" key. 
        """
        if doc_id is None and filename is None:
            raise ValueError("Either doc_id (create from raw inputs) or filename (create from file) must be provided.")
        # if create object from file
        if filename:
            with open(filename) as json_file:
                llm_ie = json.load(json_file)
            if 'doc_id' in llm_ie.keys():
                self.doc_id = llm_ie['doc_id']
            if 'text' in llm_ie.keys():
                self.text = llm_ie['text']
            if 'frames' in llm_ie.keys():
                self.frames = [LLMInformationExtractionFrame.from_dict(d) for d in llm_ie['frames']]
            if 'relations' in llm_ie.keys():
                self.relations = llm_ie['relations']

        # create object from raw inputs
        else:
            if not isinstance(doc_id, str):
                raise TypeError("doc_id must be a string.")
            self.doc_id = doc_id
            self.text = text
            self.frames = frames.copy() if frames is not None else []
            self.relations = relations.copy() if relations is not None else []


    def has_frame(self) -> bool:
        """
        This method checks if there is any frames.
        """
        return bool(self.frames)
    
    def has_relation(self) -> bool:
        """
        This method checks if there is any relations.
        """
        return bool(self.relations)
    
    def has_duplicate_frame_ids(self) -> bool:
        """
        This method checks for duplicate frame ids.
        """
        frame_id_set = set()
        for frame in self.frames:
            if frame.frame_id in frame_id_set:
                return True
            frame_id_set.add(frame.frame_id)

        return False
    
    def get_frame_by_id(self, frame_id:str) -> LLMInformationExtractionFrame:
        """
        This method use frame_id to search for a frame. 
        If there are redundent frame_ids, the first will be returned

        Parameters:
        -----------
        frame_id : str
            frame id to retrieve

        Returns : LLMInformationExtractionFrame
            a frame (if found) or None (not found).
        """
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame

        return None

    
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
        if not isinstance(frame, LLMInformationExtractionFrame):
            raise TypeError(f"Expect frame to be LLMInformationExtractionFrame, received {type(frame)} instead.")

        if valid_mode not in {None, "span", "attr"}:
            raise ValueError(f'Expect valid_mode to be one of {{None, "span", "attr"}}, received {valid_mode}')

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
            frame_clone.frame_id = str(len(self.frames))

        self.frames.append(frame_clone)
        return True


    def add_frames(self, frames:List[LLMInformationExtractionFrame], valid_mode:str=None, create_id:bool=False):
        """
        This method adds a list of frames.
        """
        if not isinstance(frames, Iterable):
            raise TypeError("frames must be a list or Interable.")
        
        for frame in frames:
            self.add_frame(frame=frame, valid_mode=valid_mode, create_id=create_id)


    def add_relation(self, relation:Dict[str,str]) -> bool:
        """
        This method add a relation to the relations (list).

        Parameters:
        -----------
        relation : Dict[str,str]
            the relation to add. Must be a dict with {"frame_1", "frame_2", ("relation")}. 
            Could have an optional "relation" key for relation type. 

        Returns : bool
            sucess addition.
        """
        if not isinstance(relation, Dict):
            raise TypeError(f"Expect relation to be a Dict, received {type(relation)} instead.")

        required_keys = {"frame_1", "frame_2"}
        if not required_keys.issubset(relation.keys()):
            raise ValueError('relation missing "frame_1" or "frame_2" keys.')
        
        allowed_keys = {"frame_1", "frame_2", "relation"}
        if not set(relation.keys()).issubset(allowed_keys):
            raise ValueError('Only keys {"frame_1", "frame_2", "relation"} are allowed.')
        
        if not self.get_frame_by_id(relation["frame_1"]):
            raise ValueError(f'frame_id: {relation["frame_1"]} not found in frames.')
        
        if not self.get_frame_by_id(relation["frame_2"]):
            raise ValueError(f'frame_id: {relation["frame_2"]} not found in frames.')

        self.relations.append(relation)
        return True

    def add_relations(self, relations:List[Dict[str,str]]):
        """
        This method adds a list of relations.
        """
        if not isinstance(relations, Iterable):
            raise TypeError("relations must be a list or Interable.")
        for relation in relations:
            self.add_relation(relation)


    def __repr__(self, N_top_chars:int=100) -> str:
        text_to_print = self.text[0:N_top_chars]
        frame_count = len(self.frames)
        relation_count = len(self.relations)
        return ''.join((f'LLMInformationExtractionDocument(doc_id: "{self.doc_id}"\n',
                        f'text: "{text_to_print}...",\n',
                        f'frames: {frame_count}\n',
                        f'relations: {relation_count}'))


    def save(self, filename:str):
        with open(filename, 'w') as json_file:
            json.dump({'doc_id': self.doc_id, 
                        'text': self.text, 
                        'frames': [frame.to_dict() for frame in self.frames],
                        'relations': self.relations}, 
                        json_file, indent=4)
            json_file.flush()
            

    def _viz_preprocess(self) -> Tuple:
        """
        This method preprocesses the entities and relations for visualization.
        """
        if importlib.util.find_spec("ie_viz") is None:
            raise ImportError("ie_viz not found. Please install ie_viz (```pip install ie-viz```).")

        if self.has_frame():
            entities = [{"entity_id": frame.frame_id, "start": frame.start, "end": frame.end, "attr": frame.attr} for frame in self.frames]
        else:
            raise ValueError("No frames in the document.")
        
        if self.has_relation():
            relations = []
            for relation in self.relations:
                rel = {"entity_1_id": relation['frame_1'], "entity_2_id": relation['frame_2']}
                relations.append(rel)        
        else:
            relations = None

        return entities, relations


    def viz_serve(self, host: str = '0.0.0.0', port: int = 5000, theme:str = "light", title:str="Frames Visualization",
                  color_attr_key:str=None, color_map_func:Callable=None):
        """
        This method serves a visualization App of the document.

        Parameters:
        -----------
        host : str, Optional
            The host address to run the server on.
        port : int, Optional
            The port number to run the server on.
        theme : str, Optional
            The theme of the visualization. Must be either "light" or "dark".
        title : str, Optional
            the title of the HTML.
        color_attr_key : str, Optional
            The attribute key to be used for coloring the entities.
        color_map_func : Callable, Optional
            The function to be used for mapping the entity attributes to colors. When provided, the color_attr_key and 
            theme will be overwritten. The function must take an entity dictionary as input and return a color string (hex).
        """
        entities, relations = self._viz_preprocess()
        from ie_viz import serve

        try:
            serve(text=self.text,
                    entities=entities,
                    relations=relations,
                    host=host,
                    port=port,
                    theme=theme,
                    title=title,
                    color_attr_key=color_attr_key,
                    color_map_func=color_map_func)
        except TypeError:
            warnings.warn("The version of ie_viz is not the latest. Please update to the latest version (pip install --upgrade ie-viz) for complete features.", UserWarning)
            serve(text=self.text,
                    entities=entities,
                    relations=relations,
                    host=host,
                    port=port,
                    theme=theme,
                    color_attr_key=color_attr_key,
                    color_map_func=color_map_func)
    
    def viz_render(self, theme:str = "light", color_attr_key:str=None, color_map_func:Callable=None,
                   title:str="Frames Visualization") -> str:
        """
        This method renders visualization html of the document.

        Parameters:
        -----------
        theme : str, Optional
            The theme of the visualization. Must be either "light" or "dark".
        color_attr_key : str, Optional
            The attribute key to be used for coloring the entities.
        color_map_func : Callable, Optional
            The function to be used for mapping the entity attributes to colors. When provided, the color_attr_key and 
            theme will be overwritten. The function must take an entity dictionary as input and return a color string (hex).
        title : str, Optional
            the title of the HTML.
        """
        entities, relations = self._viz_preprocess()
        from ie_viz import render

        try:
            return render(text=self.text,
                        entities=entities,
                        relations=relations,
                        theme=theme,
                        title=title,
                        color_attr_key=color_attr_key,
                        color_map_func=color_map_func)
        except TypeError:
                warnings.warn("The version of ie_viz is not the latest. Please update to the latest version (pip install --upgrade ie-viz) for complete features.", UserWarning)
                return render(text=self.text,
                        entities=entities,
                        relations=relations,
                        theme=theme,
                        color_attr_key=color_attr_key,
                        color_map_func=color_map_func)