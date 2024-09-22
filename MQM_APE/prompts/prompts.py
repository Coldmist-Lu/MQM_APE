from .prompt_contexts import (
    GEMBA_MQM_INPUT_PROMPT, 
    GEMBA_MQM_SYSTEM_PROMPT, 
    GEMBA_MQM_FEW_SHOTS_PROMPT,
    POST_EDIT_INPUT_PROMPT,
    VERIFIER_PAIRWISE_SELECT_PROMPT
)

def simple_query(prompt: str):
    return [
        {
            "role": "user",
            "content": prompt
        }
    ]

# Source: https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba/gemba_mqm_utils.py
def gemba_mqm_fewshot(few_shots):
    prompts = [
        {
            "role": "system",
            "content": GEMBA_MQM_SYSTEM_PROMPT
        }
    ]

    template = GEMBA_MQM_INPUT_PROMPT

    for shot in few_shots:
        prompts.append({
            "role": "user",
            "content": template.format(**shot)
        })
        answer = shot['answer']

        prompts.append({
            "role": "assistant",
            "content": answer
        })

    prompts.append({
            "role": "user",
            "content": template
        })

    return prompts

TEMPLATE_GEMBA_MQM_FEWSHOT = gemba_mqm_fewshot([GEMBA_MQM_FEW_SHOTS_PROMPT['ende'], 
                                                GEMBA_MQM_FEW_SHOTS_PROMPT['encs'], 
                                                GEMBA_MQM_FEW_SHOTS_PROMPT['zhen']])

# postedit
TEMPLATE_POSTEDIT = simple_query(POST_EDIT_INPUT_PROMPT)
TEMPLATE_VERIFIER = simple_query(VERIFIER_PAIRWISE_SELECT_PROMPT)