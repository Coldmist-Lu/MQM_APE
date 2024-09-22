import os
import os.path as osp
from typing import List, Dict, Literal
from utils import read_json, save_txt

class Scorer():

    def __init__(self,
                 scorer_type: Literal['MQM', 'MQM-APE']):

        self.scorer_type = scorer_type

    def score_list(self, errors: List[Dict[str, Dict[str, str]]]) -> List[float]:
        return [self.score(error_dict) for error_dict in errors]

    def score(self, error_dict: List[Dict[str, Dict[str, str]]]) -> List[float]:
        
        if self.scorer_type == 'MQM':
            num_of_critical = len(error_dict['critical'])
            num_of_major = len(error_dict['major'])
            num_of_minor = len(error_dict['minor'])

            final_score = -25 * num_of_critical - 5 * num_of_major - 1 * num_of_minor

            return -25 if final_score < -25 else final_score
        
        elif self.scorer_type == 'MQM-APE':
            num_of_critical = sum([error['pe_valid_score'] for error in error_dict['critical']])
            num_of_major = sum([error['pe_valid_score'] for error in error_dict['major']])
            num_of_minor = sum([error['pe_valid_score'] for error in error_dict['minor']])

            final_score = -25 * num_of_critical -5 * num_of_major - 1 * num_of_minor

            return -25 if final_score < -25 else final_score
        

if __name__ == "__main__":

    # current dir settings
    current_dir = osp.dirname(osp.abspath(__file__))

    # read files
    inputs = read_json(osp.join(current_dir, "test/errorspans_ape_verifier.json"))
    errors = [_input['error_dict'] for _input in inputs]

    # mqm scorer
    mqm_scorer = Scorer(scorer_type='MQM')
    scores = mqm_scorer.score_list(errors)
    save_txt([str(score)+"\n" for score in scores], osp.join(current_dir, "test/mqm_scores.txt"))

    # mqm-ape scorer
    mqm_scorer = Scorer(scorer_type='MQM-APE')
    scores = mqm_scorer.score_list(errors)
    save_txt([str(score)+"\n" for score in scores], osp.join(current_dir, "test/mqm_ape_scores.txt"))
