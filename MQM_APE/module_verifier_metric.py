import os.path as osp
from typing import Any, List, Dict, Tuple

from basemodule import BaseModule
from cometkiwi import COMETKiwi
from utils import (
    read_json, 
    save_json,
    load_yaml,
)

class Pairwise_Quality_Verifier_Metric(BaseModule):
    def __init__(self, # use metrics such as COMET-Kiwi to replace LLM verifier
                 metric_path: str,
                 metric_threshold: float=0.03):
        
        self.metric_threshold = metric_threshold
        self.metric_scorer = COMETKiwi(metric_path)
        

    def pipeline(self,
                sample_inputs: List[Dict[str, str]],
                errors_ape: List[Dict[str, Dict[str, str]]],
                ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Dict[str, Any]]]]:
        
        """pipeline of verifier"""
        
        tgt_inputs, ape_inputs = self.preprocess(sample_inputs=sample_inputs, errors_ape=errors_ape)
        tgt_scores = self.query(tgt_inputs)
        ape_scores = self.query(ape_inputs)
        samples_inputs_scores, errors_ape_scores = self.postprocess(sample_inputs=sample_inputs, 
                                                                    errors_ape=errors_ape, 
                                                                    tgt_scores=tgt_scores, 
                                                                    ape_scores=ape_scores)

        return samples_inputs_scores, errors_ape_scores
    

    def preprocess(self, 
                   sample_inputs: List[Dict[str, str]],
                   errors_ape: List[Dict[str, Dict[str, str]]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        
        """return inputs dict with comparative translations"""

        tgt_inputs, ape_inputs = [], []
        for sample_input, error_dict in zip(sample_inputs, errors_ape):

            tgt_inputs.append({
                'source_seg': sample_input['source_seg'],
                'target_seg': sample_input['target_seg'],
            })

            for _error in error_dict['critical'] + error_dict['major'] + error_dict['minor']:
                ape_inputs.append({
                    'source_seg': sample_input['source_seg'],
                    'target_seg': _error['post_edit'],
                })

        return tgt_inputs, ape_inputs
    

    def query(self, 
              inputs: List[Dict[str, str]]) -> List[float]:
        
        """
        inputs: [{'source_lang', 'source_seg', 'target_lang', 'target_seg'}, {...}, ...]
        return: [{'transA': score, 'transB': score}]
        """

        # cometeval
        srcs = [_input['source_seg'] for _input in inputs]
        tgts = [_input['target_seg'] for _input in inputs]

        scores, _ = self.metric_scorer.cometkiwi_eval(srcs=srcs, hyps=tgts)

        return scores
    

    def postprocess(self, 
                    sample_inputs: List[Dict[str, str]],
                    errors_ape: List[Dict[str, Dict[str, str]]], 
                    tgt_scores: List[float],
                    ape_scores: List[float]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Dict[str, Any]]]]:
        
        """
        extract error spans from generated text.
        
        errors: A List of error annotations from Evaluator.
        outputs: pairwise outputs list
        return: A List of error annotations, with APE translation, pairwise score for each error annotation.
        """

        def list_generator(my_list):
            for item in my_list:
                yield item

        # extract verifier results from error_list
        verifier_results_generator = list_generator(ape_scores)

        # parse verifier judgments to error annotation
        for _input, error_dict, tgt_score in zip(sample_inputs, errors_ape, tgt_scores):

            _input['cometkiwi_score'] = tgt_score

            for severity in 'critical', 'major', 'minor':
                if len(error_dict[severity]) == 0:
                    continue

                for _error in error_dict[severity]:
                    ape_score = next(verifier_results_generator) # tgt vs. ape

                    _error['postedit_cometkiwi_score'] = ape_score

                    if ape_score - tgt_score > self.metric_threshold: # ape better
                        _error['pe_valid_score'] = 1

                    elif tgt_score - ape_score > self.metric_threshold: # tgt better
                        _error['pe_valid_score'] = 0

                    else: # tie
                        _error['pe_valid_score'] = 0.5

        return sample_inputs, errors_ape

        
if __name__ == "__main__":

    # current dir settings
    current_dir = osp.dirname(osp.abspath(__file__))

    # read files
    configs = load_yaml(osp.join(current_dir, "configs/llmconfig_metric.yaml"))
    inputs = read_json(osp.join(current_dir, "test/errorspans_ape.json"))
    errors = [_input['error_dict'] for _input in inputs]

    # init Inference
    verifier_module = Pairwise_Quality_Verifier_Metric(**configs['verifier'])

    # evaluate samples
    inputs, errors = verifier_module.pipeline(sample_inputs=inputs, 
                                              errors_ape=errors)

    input_with_errors = [{**_input, 'error_dict': _error} for _input, _error in zip(inputs, errors)]

    # save
    save_json(input_with_errors, osp.join(current_dir, "test/errorspans_ape_verifier_metrics.json"))