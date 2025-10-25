from typing import List, Dict, Union
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