"""
Evaluate MT sentence-pairs using WMT22-cometkiwi-da.
"""

from comet import load_from_checkpoint
from typing import List, Union


class COMETKiwi():
    
    def __init__(self,
                 model_path: str
                 ) -> None:
        # load comet model
        self.model_path = model_path
        self.model = load_from_checkpoint(self.model_path, reload_hparams=True)


    def cometkiwi_eval(self, 
                       srcs: List[str], 
                       hyps: List[str], 
                       batch_size: int=8,
                       gpus: int=1
                       ) -> Union[List[float], float]:
                
        assert len(srcs) == len(hyps), "length of srcs and hyps should be the same!"

        data = [{"src": src, "mt": mt} for src, mt in zip(srcs, hyps)]

        output = self.model.predict(data, batch_size=batch_size, gpus=gpus)

        return output[0], output[1]

if __name__ == "__main__":

    # an example of using cometkiwi model
    srcs = [
        "The output signal provides constant sync so the display never glitches.",
        "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
        "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
    ] 
    tgts = [
        "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört.",
        "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років.",
        "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。"
    ]

    scorer = COMETKiwi(model_path="/path/to/model.ckpt")

    segment_scores, system_score = scorer.cometkiwi_eval(srcs, tgts)

    print(f"{segment_scores=}")
    print(f"{system_score=}")