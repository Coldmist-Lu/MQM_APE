import os.path as osp
from typing import Any, List, Dict, Tuple

from basemodule import BaseModule
from inference import Inference
from prompts.prompts import TEMPLATE_VERIFIER
from utils import (
    apply_template, 
    truncate_response,
    read_json, 
    save_json,
    load_yaml,
)

class Pairwise_Quality_Verifier(BaseModule):
    def __init__(self, 
                 inference: Inference,
                 use_twice_verify: bool=True, # verify twice to avoid positional bias
                 max_tokens: int=512,
                 temperature: float=0):
        
        self.use_twice_verify = use_twice_verify
        self.inference = inference
        self.max_tokens = max_tokens
        self.temperature = temperature

    
    def pipeline(self,
                 sample_inputs: List[Dict[str, str]],
                 errors_ape: List[Dict[str, Dict[str, str]]]
                 ) -> Tuple[List[Dict[str, str]], List[Dict[str, Dict[str, Any]]]]:

        # evaluate samples
        inputs = self.preprocess(sample_inputs=sample_inputs, errors_ape=errors_ape)
        outputs = self.query(inputs)
        errors = self.postprocess(errors_ape=errors_ape, outputs=outputs)

        return outputs, errors


    def preprocess(self, 
                   sample_inputs: List[Dict[str, str]],
                   errors_ape: List[Dict[str, Dict[str, str]]]) -> List[Dict[str, str]]:
        
        """return inputs dict with comparative translations"""

        inputs = []
        for sample_input, error_dict in zip(sample_inputs, errors_ape):
            for _error in error_dict['critical'] + error_dict['major'] + error_dict['minor']:
                inputs.append({
                    'source_lang': sample_input['source_lang'],
                    'source_seg': sample_input['source_seg'],
                    'target_lang': sample_input['target_lang'],
                    'transA_seg': sample_input['target_seg'],
                    'transB_seg': _error['post_edit'],
                })
                if self.use_twice_verify is True:
                    inputs.append({ # swap transA and transB
                        'source_lang': sample_input['source_lang'],
                        'source_seg': sample_input['source_seg'],
                        'target_lang': sample_input['target_lang'],
                        'transA_seg': _error['post_edit'], 
                        'transB_seg': sample_input['target_seg'], 
                    })

        return inputs


    def query(self, 
              inputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        
        """
        inputs: [{'source_lang', 'source_seg', 'target_lang', 'target_seg'}, {...}, ...]
        return: [{'prompt', 'generated_text'}, {...}, ...] or [{'transA': score, 'transB': score}]
        """

        # generate query
        query_list = [apply_template(TEMPLATE_VERIFIER, _input) for _input in inputs]

        outputs = self.inference.inference(query_list, 
                                           self.temperature,
                                           self.max_tokens)
        
        return outputs


    def postprocess(self, 
                    errors_ape: List[Dict[str, Dict[str, str]]], 
                    outputs: List[Dict[str, Any]]) -> List[Dict[str, Dict[str, Any]]]:
        
        """
        extract error spans from generated text.
        
        errors: A List of error annotations from Evaluator.
        outputs: pairwise outputs list
        return: A List of error annotations, with APE translation, pairwise score for each error annotation.
        """

        def list_generator(my_list):
            for item in my_list:
                yield item

        # truncate error spans
        verifier_texts = [truncate_response(_output['generated_text'], ['<|eot_id|>', ]) for _output in outputs]

        # extract verifier results from error_list
        verifier_results_generator = list_generator(self.verifier_pairwise(_text) for _text in verifier_texts)

        # parse verifier judgments to error annotation
        for error_dict in errors_ape:
            for severity in 'critical', 'major', 'minor':
                if len(error_dict[severity]) == 0:
                    continue

                for _error in error_dict[severity]:

                    if self.use_twice_verify is True:
                        res1 = next(verifier_results_generator) # tgt vs. ape
                        res2 = next(verifier_results_generator) # ape vs. tgt

                        if res1 == 'A' and res2 == 'B': # target best
                            _error['pe_valid_score'] = 0
                        elif res1 == 'B' and res2 == 'A': # pe best
                            _error['pe_valid_score'] = 1
                        else: # tie
                            _error['pe_valid_score'] = 0.5
                    
                    else:
                        res = next(verifier_results_generator) # tgt vs. ape
                        if res == 'A':
                            _error['pe_valid_score'] = 0
                        elif res == 'B':
                            _error['pe_valid_score'] = 1
                        else:
                            _error['pe_valid_score'] = 0.5

        return errors_ape


    def verifier_pairwise(self, response: str) -> str: 

        """
        return 'A' or 'B'
        """

        def paircheck(text: str, a_key: str, b_key: str):
            num_a = text.lower().find(a_key)
            num_b = text.lower().find(b_key)
            if num_a != -1 and num_b == -1:
                return True, 'A'
            
            if num_a == -1 and num_b != -1:
                return True, 'B'
            
            if num_a != -1 and num_b != -1 and num_a < num_b:
                return True, 'A'
                
            if num_a != -1 and num_b != -1 and num_a > num_b:
                return True, 'B'
            
            return False, ''

        response = response.split('<|eot_id|>')[0]
        response_lines = [r.strip() for r in response.split('\n') if r.strip() != ""]
        text = response_lines[0]
        # for llama
        text = '\n'.join(response_lines)

        if text in ['A', 'B']: # direct A or B
            return text
        
        for A_indicator, B_indicator in ('translation a', 'translation b'), ('translations a', 'translations b'), ('a:', 'b:'), (':a', ':b'), ('"a"', '"b"'), ('option a', 'option b'):

            valid, res = paircheck(text, A_indicator, B_indicator)
            if valid is True:
                return res
        
        A_cnt = text.count('A ') + text.count(' A')
        B_cnt = text.count('B ') + text.count(' B')

        if A_cnt > B_cnt:
            return 'A'
        
        if B_cnt > A_cnt:
            return 'B'
        
        # other situations
        num_a = text.lower().find('A')
        num_b = text.lower().find('B')
        if num_a < num_b:
            return 'A'
        else:
            return 'B'
        
if __name__ == "__main__":

    # current dir settings
    current_dir = osp.dirname(osp.abspath(__file__))

    # read files
    configs = load_yaml(osp.join(current_dir, "configs/llmconfig.yaml"))
    inputs = read_json(osp.join(current_dir, "test/errorspans_ape.json"))
    errors = [_input['error_dict'] for _input in inputs]

    # init Inference
    inference = Inference(**configs['inference'])
    verifier_module = Pairwise_Quality_Verifier(inference=inference, **configs['verifier'])

    # evaluate samples
    outputs, errors = verifier_module.pipeline(inputs, errors)
   
    input_with_errors = [{**_input, 'error_dict': _error} for _input, _error in zip(inputs, errors)]

    # save
    save_json(input_with_errors, osp.join(current_dir, "test/errorspans_ape_verifier.json"))