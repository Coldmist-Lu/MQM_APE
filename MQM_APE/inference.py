"""
LLM Inference using VLLM.
"""

import time
from typing import List, Dict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class Inference():

    def __init__(self, 
                 model_path: str,
                 tp: int=1,
                 ) -> None:
        
        # load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.8,
            trust_remote_code=True
        )
        
    def input2prompt(self, sample):
        """
        convert input into prompt str.

        input format 1: Prompt string. e.g. "Input String" -> "Input String"
        input format 2: Dict with system prompt.
        e.g. [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Post edit the translation."}
        ]
        input format 3: Dict without system prompt.
        e.g. [
            {"role": "user", "content": "Post edit the translation."}
        ]
        """
        if isinstance(sample, str): # input format 1
            return sample
        
        try: # input format 2
            return self.tokenizer.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=True
            )
        except: # input format 3
            return 'system:' + sample[0]['content'] + self.tokenizer.apply_chat_template(
                sample[1:],
                tokenize=False,
                add_generation_prompt=True
            )
    
    def inference(self, 
                  inputs,
                  temperature: float=0,
                  max_tokens: int=256) -> List[Dict[str, str]]:
        """
        inputs: List of input with prompt formats.
        return: [{'prompt': ..., 'generated_text': ...}, {'prompt': ..., 'generated_text': ...}, ...]
        """
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )

        step_timer = time.time()

        inference_inputs_str = [self.input2prompt(_input) for _input in inputs]

        # VLLM generate
        outputs = self.model.generate(
            prompts=inference_inputs_str,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        # parse response
        responses = []
        for prompt, output in zip(inference_inputs_str, outputs):
            if prompt == output.prompt:
                responses.append(
                    {
                        'prompt': prompt,
                        'generated_text': output.outputs[0].text
                    }
                )
            else:
                raise ValueError("Can't align input prompt")
    
        print(f"[INFO] Generating {len(inference_inputs_str)} samples finished. Time passed {(time.time() - step_timer)/60} mins.")

        return responses

