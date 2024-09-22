import os.path as osp
import re
from typing import List, Dict, Tuple

from basemodule import BaseModule
from inference import Inference
from prompts.prompts import TEMPLATE_GEMBA_MQM_FEWSHOT
from utils import (
    apply_template,
    save_json,
    save_txt, 
    truncate_response, 
    load_yaml,
    readlines_txt
)

class Error_Analysis_Evaluator(BaseModule):
    def __init__(self, 
                 inference: Inference,
                 max_tokens: int=512,
                 temperature: float=0):
        self.inference = inference
        self.max_tokens = max_tokens
        self.temperature = temperature


    def pipeline(self, 
                srcs: List[str], 
                tgts: List[str],
                src_lang: str,
                tgt_lang: str) -> Tuple[List[Dict[str, str]],
                                        List[Dict[str, str]], 
                                        List[Dict[str, Dict[str, str]]], 
                                        List[str]]:
        """pipeline of evaluator"""
        # identify errors
        inputs = self.preprocess(srcs, tgts, src_lang, tgt_lang)
        outputs = self.query(inputs)
        errors, messages = self.postprocess(outputs)

        return inputs, outputs, errors, messages


    def preprocess(self, 
                   srcs: List[str], 
                   tgts: List[str], 
                   src_lang: str, 
                   tgt_lang: str) -> List[Dict[str, str]]:
        
        """return inputs dict"""
        inputs = []
        for src, tgt in zip(srcs, tgts):
            inputs.append({
                'source_lang': src_lang,
                'source_seg': src.strip(),
                'target_lang': tgt_lang,
                'target_seg': tgt.strip()
            })

        return inputs


    def query(self, 
              inputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        
        """
        inputs: [{'source_lang', 'source_seg', 'target_lang', 'target_seg'}, {...}, ...]
        return: [{'prompt', 'generated_text'}, {...}, ...]
        """

        # generate query
        query_list = [apply_template(TEMPLATE_GEMBA_MQM_FEWSHOT, _input) for _input in inputs]

        outputs = self.inference.inference(query_list, 
                                           self.temperature,
                                           self.max_tokens)
        
        return outputs


    def postprocess(self, 
                    outputs: List[Dict[str, str]]) -> Tuple[List[Dict[str, Dict[str, str]]], List[str]]:
        
        """
        extract error spans from generated text.
        inputs: outputs list
        return: A List of error annotations, and messages about omitted lines.
        """

        # truncate error spans
        error_texts = [truncate_response(_output['generated_text'], ['<|eot_id|>', ]) for _output in outputs]

        # extract error category and span
        messages = []        
        error_list = [self.error_text2dict(_text, messages) for _text in error_texts]

        return error_list, messages


    def error_text2dict(self, 
                        error_text: str, 
                        message: List[str]=[]) -> Dict[str, Dict[str, str]]:
        
        """
        extract errors from responses and record omit lines in message.

        input: error_text after truncation.
        output: {
            'critical': [{'category': ..., 'span': ...}, {'category': ..., 'span': ...}, ...]
            'major': [{'category': ..., 'span': ...}, {'category': ..., 'span': ...}, ...]
            'minor': [{'category': ..., 'span': ...}, {'category': ..., 'span': ...}, ...]
        }
        
        Note: The number of errors from each category can be zero.
        """

        errors_dict = {
            'critical': [],
            'major': [],
            'minor': []
        }

        error_level = 'minor'
        for line in error_text.split('\n'):
            if 'critical:' in line.lower() or 'critical error' in line.lower():
                error_level = 'critical'
                continue
            if 'major:' in line.lower() or 'major error' in line.lower():
                error_level = 'major'
                continue
            if 'minor:' in line.lower() or 'minor error' in line.lower():
                error_level = 'minor'
                continue

            if 'no-error' in line.lower() or "no error" in line.lower():
                continue

            if " - " not in line:
                # print(f"This line will omit: {line}")
                message.append(f"This line will omit: {line}\n")
                continue

            category = line.split(' - ')[0] # match categories
            match = re.search(r'["”](.*?)["”]', line) # match errorspan within "" or “”

            if match:
                error_span = match.group(1)
            else:
                message.append(f"This line will omit: {line}\n") # record omit lines
                continue

            errors_dict[error_level].append({
                'category': category,
                'span': error_span
            })
        
        return errors_dict


if __name__ == "__main__":

    # current dir settings
    current_dir = osp.dirname(osp.abspath(__file__))

    # read files
    configs = load_yaml(osp.join(current_dir, "configs/llmconfig.yaml"))
    srcs = readlines_txt(osp.join(current_dir, "test/srcs_zh.txt"))
    tgts = readlines_txt(osp.join(current_dir, "test/tgts_en.txt"))
    src_lang, tgt_lang = 'zh', 'en'

    # init Inference
    inference = Inference(**configs['inference'])
    evaluator_module = Error_Analysis_Evaluator(inference=inference, **configs['evaluator'])

    # evaluate samples
    inputs, outputs, errors, messages = evaluator_module.pipeline(srcs, tgts, src_lang='Chinese', tgt_lang='English')

    input_with_errors = [{**_input, 'error_dict': _error} for _input, _error in zip(inputs, errors)]

    # save
    save_txt(messages, osp.join(current_dir, "test/lines_omitted.log"))
    save_json(input_with_errors, osp.join(current_dir, "test/errorspans.json"))