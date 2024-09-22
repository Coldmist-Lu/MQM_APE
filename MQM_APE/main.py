import argparse
import os
import os.path as osp
from typing import Dict, Any, Literal, List, Tuple

from inference import Inference
from module_evaluator import Error_Analysis_Evaluator
from module_ape import Automatic_Post_Editor
from scorer import Scorer
from utils import (
    save_json, 
    load_yaml, 
    readlines_txt, 
    save_txt
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of configuration yaml.")
    parser.add_argument("--src", type=str, required=True, help="Path of src.")
    parser.add_argument("--tgt", type=str, required=True, help="Path of tgt.")
    parser.add_argument("--srclang", type=str, required=True, help="Source Language, such as English, German, Chinese.")
    parser.add_argument("--tgtlang", type=str, required=True, help="Target Language, such as English, German, Chinese.")
    parser.add_argument("--out", type=str, required=True, help="Save directory.")
    
    parser.add_argument("--metric_verifier", action="store_true", default=False, help="Whether to replace verifier with cometkiwi.")
    parser.add_argument("--save_llm_response", action="store_true", default=False, help="Whether to save response of llm.")

    args = parser.parse_args()
    return args


class MQM_APE():
    
    def __init__(self,
                 configs: Dict[Any, Any],
                 verifier_type: Literal['metric', 'llm']='llm',
                 ):
        
        self.verifer_type = verifier_type
        self.inference = Inference(**configs['inference'])
        self.evaluator_module = Error_Analysis_Evaluator(self.inference, **configs['evaluator'])
        self.ape_module = Automatic_Post_Editor(self.inference, **configs['ape'])
        self.scorer = Scorer(scorer_type='MQM-APE')

        if verifier_type == 'llm': # init different verifier for llm or metrics
            from module_verifier import Pairwise_Quality_Verifier
            self.verifier_module = Pairwise_Quality_Verifier(self.inference, **configs['verifier'])
        else:
            from module_verifier_metric import Pairwise_Quality_Verifier_Metric
            self.verifier_module = Pairwise_Quality_Verifier_Metric(**configs['verifier'])


    def eval(self, 
             srcs: List[str], 
             tgts: List[str], 
             src_lang: str, 
             tgt_lang: str,
             save_llm_response_dir: str=None
             ) -> Tuple[Dict[str, Any], List[float]]:
        
        # identify errors
        inputs, outputs, errors, messages = self.evaluator_module.pipeline(srcs, tgts, src_lang, tgt_lang)

        if save_llm_response_dir is not None:
            save_json(outputs, osp.join(save_llm_response_dir, "llm_responses_evaluator.json"))
            save_txt(messages, osp.join(save_llm_response_dir, "llm_evaluator_omitted_messages.json"))

        # post-edit
        outputs_ape, errors_ape = self.ape_module.pipeline(inputs, errors)

        if save_llm_response_dir is not None:
            save_json(outputs_ape, osp.join(save_llm_response_dir, "llm_responses_ape.json"))

        # verifier
        if self.verifer_type == 'llm':
            outputs_verifier, errors_w_scores = self.verifier_module.pipeline(sample_inputs=inputs, errors_ape=errors_ape)
            
            if save_llm_response_dir is not None:
                save_json(outputs_verifier, osp.join(save_llm_response_dir, "llm_responses_verifier.json"))
                
        else:
            inputs, errors_w_scores = self.verifier_module.pipeline(sample_inputs=inputs, 
                                                                    errors_ape=errors_ape)

        # concatenate results
        results = [{**_input, 'error_dict': _error} for _input, _error in zip(inputs, errors_w_scores)]

        # score
        scores = self.scorer.score_list(errors_w_scores)
        
        results = [{**_res, 'MQM_APE_score': _score} for _res, _score in zip(results, scores)]
        scores = [str(_score)+'\n' for _score in scores]
        
        return results, scores
    
    
if __name__ == "__main__":
    args = parse_args()
    
    configs = load_yaml(args.config)
    srcs = readlines_txt(args.src)
    tgts = readlines_txt(args.tgt)
    
    # check dir exist
    if osp.exists(args.out) is False:
        os.makedirs(args.out)

    mqm_ape = MQM_APE(configs, 'llm' if args.metric_verifier is False else 'metric')
    
    save_llm_response_dir = args.out if args.save_llm_response is True else None
    
    results, scores = mqm_ape.eval(srcs, tgts, args.srclang, args.tgtlang, save_llm_response_dir)
    
    save_json(results, osp.join(args.out, "results.json"))
    save_txt(scores, osp.join(args.out, "scores.txt"))
