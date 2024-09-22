import json
import yaml
from collections import defaultdict
from typing import List

def result_tree():
    return defaultdict(result_tree)

def readlines_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def save_txt(data: List, file: str) -> None:
    with open(file, 'w') as f:
        f.writelines(data)
    print(f'Saved to {file}.')
    return

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f'Saved to {path}.')
    return

def load_yaml(file: str):
    with open(file) as reader:
        return yaml.safe_load(reader)

def truncate_response(response: str, truncate_list: List[str], start_truncation_len: int=0) -> str:
    """
    response: the raw response requires truncating.
    truncate_list: a list of truncation keywords.
    start_truncation_len: the minimum length of truncation
    
    return: response after truncation.
    """
    for keyword in truncate_list:
        if len(response) <= start_truncation_len:
            response = response
        else:
            response = response[:start_truncation_len] + response[start_truncation_len:].split(keyword)[0]
    return response

def apply_template(template, data):

    # Source: https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba/gemba_mqm_utils.py

    if isinstance(template, str):
        return template.format(**data)
    elif isinstance(template, list):
        prompt = []
        for conversation_turn in template:
            p = conversation_turn.copy()
            p['content'] = p['content'].format(**data)
            prompt.append(p)
        return prompt
    else:
        raise ValueError(f"Unknown template type {type(template)}")