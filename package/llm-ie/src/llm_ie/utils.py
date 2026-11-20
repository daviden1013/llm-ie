from typing import List, Dict, Union, Tuple, Optional
import re
import json
import warnings
import json_repair

def _find_dict_strings(text: str) -> List[str]:
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
    
    
def extract_json(gen_text:str) -> List[Dict[str, str]]:
    """ 
    This method inputs a generated text and output a JSON of information tuples
    """
    out = []
    dict_str_list = _find_dict_strings(gen_text)
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


def apply_prompt_template(prompt_template:str, text_content:Union[str, Dict[str,str]]) -> str:
    """
    This method applies text_content to prompt_template and returns a prompt.

    Parameters:
    ----------
    prompt_template : str
        the prompt template with placeholders {{<placeholder name>}}.
    text_content : Union[str, Dict[str,str]]
        the input text content to put in prompt template. 
        If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
        If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}. All values must be str.

    Returns : str
        a user prompt.
    """
    pattern = re.compile(r'{{(.*?)}}')
    if isinstance(text_content, str):
        matches = pattern.findall(prompt_template)
        if len(matches) != 1:
            raise ValueError("When text_content is str, the prompt template must has exactly 1 placeholder {{<placeholder name>}}.")
        text = re.sub(r'\\', r'\\\\', text_content)
        prompt = pattern.sub(text, prompt_template)

    elif isinstance(text_content, dict):
        # Check if all values are str
        if not all([isinstance(v, str) for v in text_content.values()]):
            raise ValueError("All values in text_content must be str.")
        # Check if all keys are in the prompt template
        placeholders = pattern.findall(prompt_template)
        if len(placeholders) != len(text_content):
            raise ValueError(f"Expect text_content ({len(text_content)}) and prompt template placeholder ({len(placeholders)}) to have equal size.")
        if not all([k in placeholders for k, _ in text_content.items()]):
            raise ValueError(f"All keys in text_content ({text_content.keys()}) must match placeholders in prompt template ({placeholders}).")

        prompt = pattern.sub(lambda match: re.sub(r'\\', r'\\\\', text_content[match.group(1)]), prompt_template)

    return prompt


class TextTagger:
    TAG_LEVELS = ("paragraph", "sentence", "line", "word")
    # matches any tag of any level, e.g., "<sentence 12>", plus one trailing whitespace character
    TAG_PATTERN = re.compile(r"<(?:paragraph|sentence|line|word) \d+>\s?")

    def __init__(self, tag_level:str="sentence"):
        """
        This class inserts numbered tags (e.g., "<sentence 1>", "<sentence 2>") into text for the "tagged" input mode.
        Each tag is placed at the start of a segment (paragraph, sentence, line, or word).
        Tags are position markers. Spans are always computed against the original (untagged) text.

        Parameters:
        ----------
        tag_level : str, Optional
            the segment level to tag. Must be one of {"paragraph", "sentence", "line", "word"}.
            paragraph segments are separated by "\\n\\n", lines by "\\n", and words by " ".
        """
        if tag_level not in self.TAG_LEVELS:
            raise ValueError(f'tag_level must be one of {set(self.TAG_LEVELS)}.')
        self.tag_level = tag_level

        # lazy import to avoid circular import (chunkers imports utils)
        from llm_ie.chunkers import SeparatorUnitChunker, SentenceUnitChunker, TextLineUnitChunker
        if tag_level == "paragraph":
            self.chunker = SeparatorUnitChunker(sep="\n\n")
        elif tag_level == "sentence":
            self.chunker = SentenceUnitChunker()
        elif tag_level == "line":
            self.chunker = TextLineUnitChunker()
        elif tag_level == "word":
            self.chunker = SeparatorUnitChunker(sep=" ")

    def format_tag(self, tag_id:int) -> str:
        """
        This method returns the tag string for a tag_id. For example, 3 -> "<sentence 3>".
        """
        return f"<{self.tag_level} {tag_id}>"

    def get_tag_starts(self, text:str) -> List[int]:
        """
        This method returns the character positions where tags are placed in the original text.
        The i-th position (0-based) corresponds to tag_id i+1. 
        Whitespace-only segments are not tagged. Leading whitespace of a segment is skipped, 
        so each tag is placed right before the first non-whitespace character of its segment.
        """
        starts = []
        for unit in self.chunker.chunk(text, doc_id=""):
            stripped = unit.text.lstrip()
            if not stripped:
                continue
            starts.append(unit.start + (len(unit.text) - len(stripped)))
        return starts

    def tag(self, text:str) -> Tuple[str, List[int]]:
        """
        This method inserts tags into text.

        Returns : Tuple[str, List[int]]
            the tagged text, and the tag start positions in the original text (see get_tag_starts).
        """
        starts = self.get_tag_starts(text)
        pieces = []
        prev = 0
        for i, start in enumerate(starts):
            pieces.append(text[prev:start])
            pieces.append(f"{self.format_tag(i + 1)} ")
            prev = start
        pieces.append(text[prev:])
        return "".join(pieces), starts

    def has_tag_collision(self, text:str) -> bool:
        """
        This method checks whether the original text already contains tag-like strings (e.g., "<sentence 3>").
        Such strings can confuse the LLM. Span resolution is not affected since it does not parse tags from text.
        """
        return self.TAG_PATTERN.search(text) is not None

    def strip_tags(self, text:str) -> str:
        """
        This method removes tags that LLM copied into entity text. 
        """
        return self.TAG_PATTERN.sub("", text).strip()

    def parse_tag_id(self, value) -> Optional[int]:
        """
        This method parses a tag_id generated by LLM. Accepts 3, "3", "<sentence 3>", or "sentence 3".
        Returns None if it cannot be parsed.
        """
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            match = re.fullmatch(r"\s*<?\s*(?:(?:paragraph|sentence|line|word)\s*)?(\d+)\s*>?\s*", value)
            if match:
                return int(match.group(1))
        return None
