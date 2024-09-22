import os.path as osp
from typing import List, Dict, Tuple

from basemodule import BaseModule
from inference import Inference
from prompts.prompts import TEMPLATE_POSTEDIT
from utils import (
    apply_template, 
    truncate_response, 
    load_yaml,
    read_json,
    save_json
)


class Automatic_Post_Editor(BaseModule):

    def __init__(self, 
                 inference: Inference,
                 max_tokens: int=512,
                 temperature: float=0):
        
        self.inference = inference
        self.max_tokens = max_tokens
        self.temperature = temperature


    def pipeline(self,
                 sample_inputs: List[Dict[str, str]],
                 errors: List[Dict[str, Dict[str, str]]],
                 ) -> Tuple[List[Dict[str, str]], List[Dict[str, Dict[str, str]]]]:
        
        """pipeline of ape."""
        inputs_ape = self.preprocess(sample_inputs=sample_inputs, errors=errors)
        outputs_ape = self.query(inputs_ape)
        errors_ape = self.postprocess(errors=errors, outputs=outputs_ape)

        return outputs_ape, errors_ape


    def preprocess(self, 
                   sample_inputs: List[Dict[str, str]],
                   errors: List[Dict[str, Dict[str, str]]]) -> List[Dict[str, str]]:
        
        """return inputs dict with errors"""

        inputs = []
        for sample_input, error_dict in zip(sample_inputs, errors):
            for _error in error_dict['critical'] + error_dict['major'] + error_dict['minor']:
                
                inputs.append({
                    'source_lang': sample_input['source_lang'],
                    'source_seg': sample_input['source_seg'],
                    'target_lang': sample_input['target_lang'],
                    'target_seg': sample_input['target_seg'],
                    'error_category': _error['category'],
                    'error_content': _error['span']
                })

        return inputs


    def query(self, 
              inputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        
        """
        inputs: [{'source_lang', 'source_seg', 'target_lang', 'target_seg'}, {...}, ...]
        return: [{'prompt', 'generated_text'}, {...}, ...]
        """

        # generate query
        query_list = [apply_template(TEMPLATE_POSTEDIT, _input) for _input in inputs]

        outputs = self.inference.inference(query_list, 
                                           self.temperature,
                                           self.max_tokens)
        
        return outputs


    def postprocess(self, 
                    errors: List[Dict[str, Dict[str, str]]], 
                    outputs: List[Dict[str, str]]) -> List[Dict[str, Dict[str, str]]]:
        """
        extract error spans from generated text.

        errors: A List of error annotations from Evaluator.
        outputs: APE outputs list
        return: A List of error annotations, with APE translation for each error annotation.
        """
        def list_generator(my_list):
            for item in my_list:
                yield item

        # truncate error spans
        ape_texts = [truncate_response(_output['generated_text'], ['<|eot_id|>', ]) for _output in outputs]

        # extract ape from error_list
        apes_generator = list_generator(self.response2ape_translation(_text) for _text in ape_texts)

        # parse post-edited translation to error annotation
        for error_dict in errors:
            for severity in 'critical', 'major', 'minor':
                if len(error_dict[severity]) == 0:
                        continue                
                for _error in error_dict[severity]:
                    _error['post_edit'] = next(apes_generator)

        return errors


    def response2ape_translation(self, 
                                 response: str) -> str:
        
        """
        Convert response text to translation.
        """

        position = response.lower().find("corrected translation")

        # Check if "Corrected Translation" was found
        if position != -1:
            # Extract the content after "Corrected Translation"
            result = response[position + len("corrected translation:"):].strip()

            if result == "": # For tower situation: "APE text" -- corrected translation
                result = response[:position].split('--')[0].strip()

            result = result.split('<|eot_id|>')[0]
            result = result.split('\n')[0]
            result = result.strip('\"')
            return result

        else: # only provide pe response
            result = response.split('<|eot_id|>')[0].strip()
            result = result.split('\n')[0]
            result = result.strip('\"')
            return result
        
        
if __name__ == "__main__":

    # current dir settings
    current_dir = osp.dirname(osp.abspath(__file__))

    # read files
    configs = load_yaml(osp.join(current_dir, "configs/llmconfig.yaml"))
    inputs = read_json(osp.join(current_dir, "test/errorspans.json"))
    errors = [_input['error_dict'] for _input in inputs]

    # init Inference
    inference = Inference(**configs['inference'])
    ape_module = Automatic_Post_Editor(inference=inference, **configs['evaluator'])

    # evaluate samples
    outputs, errors = ape_module.pipeline(inputs, errors)

    # errors
    input_with_errors = [{**_input, 'error_dict': _error} for _input, _error in zip(inputs, errors)]

    # save
    save_json(input_with_errors, osp.join(current_dir, "test/errorspans_ape.json"))