# GEMBA_MQM prompt source: https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba/gemba_mqm_utils.py https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba/gemba_mqm_utils.py
GEMBA_MQM_SYSTEM_PROMPT = "You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation."
GEMBA_MQM_INPUT_PROMPT = """{source_lang} source:
```{source_seg}```
{target_lang} translation:
```{target_seg}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."""
GEMBA_MQM_FEW_SHOTS_PROMPT = {
    "ende": {
            "source_lang": "English",
            "source_seg": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
            "target_lang": "German",
            "target_seg": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
            "answer": """Critical:
no-error
Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
""",
        },
    "encs": {
            "source_lang": "English",
            "source_seg": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",
            "target_lang": "Czech",
            "target_seg": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",
            "answer": """Critical:
no-error
Major:
accuracy/addition - "ve Vídni"
accuracy/omission - "the stop-start"
Minor:
terminology/inappropriate for context - "partaje"
""",
        },
    "zhen": {
            "source_lang": "Chinese",
            "source_seg": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
            "target_lang": "English",
            "target_seg": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",
            "answer": """Critical:
accuracy/addition - "of high-speed rail"
Major:
accuracy/mistranslation - "go to the reviews"
Minor:
style/awkward - "etc.,"
""",
        },
}

# PostEdit Prompt
POST_EDIT_INPUT_PROMPT = '{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nPlease post-edit the translation to address the identified error: "{error_category} - {error_content}". Provide only the corrected {target_lang} translation after "Corrected Translation:" without adding any additional explanations or translation information.'

# Pairwise Verifier Prompt
VERIFIER_PAIRWISE_SELECT_PROMPT = '{source_lang} source: "{source_seg}"\nEvaluate the following translations:\n{target_lang} translation A: "{transA_seg}"\n{target_lang} translation B: "{transB_seg}"\nWhich translation is better? Please output either "A" or "B" only, without any additional explanation.\n\nAnswer:'




