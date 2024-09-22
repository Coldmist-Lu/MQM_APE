# MQM-APE

<b>MQM-APE: Toward High-Quality Error Annotation Predictors with Automatic Post-Editing in LLM Translation Evaluators</b>. [Full paper](Published on arXiv soon)

This repository presents MQM-APE, an enhanced framework for leveraging LLMs in translation evaluation. We also provide the test performance results of MQM-APE for the replication of the study.

## Abstract

Large Language Models (LLMs) have shown significant potential as judges for Machine Translation (MT) quality assessment, providing both scores and fine-grained feedback. Although approaches such as [GEMBA-MQM](https://aclanthology.org/2023.wmt-1.64.pdf) has shown SOTA performance on reference-free evaluation, the predicted errors do not align well with those annotated by human, limiting their interpretability as feedback signals. To *enhance the quality of error annotations* predicted by LLM evaluators, we introduce a **universal and training-free** framework, **MQM-APE**, based on the idea of **filtering out non-impactful errors by Automatically Post-Editing (APE) the original translation based on each error, leaving only those errors that contribute to quality improvement.** Specifically, we prompt the LLM to act as ① *evaluator* to provide error annotations, ② *post-editor* to determine whether errors impact quality improvement and ③ *pairwise quality verifier* as the error filter. Experiments show that our approach consistently improves both the reliability and quality of error spans against GEMBA-MQM, across eight LLMs in both high- and low-resource languages. Orthogonal to trained approaches, MQM-APE complements translation-specific evaluators such as [Tower](https://arxiv.org/pdf/2402.17733), highlighting its broad applicability. Further analysis confirm the effectiveness of each module and offer valuable insights into evaluator design and LLMs selection. 

## Overview

<div align="center">
    <img width="80%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/main.jpg">
</div>

## How to Use MQM-APE?

Please refer to [./MQM_APE/run.sh](./MQM_APE/run.sh) for an example of using MQM-APE.

```bash
cd ./MQM_APE

python3 main.py [-h] --config CONFIG --src SRC --tgt TGT --srclang SRCLANG --tgtlang TGTLANG --out OUT [--metric_verifier] [--save_llm_response]
```

MQM-APE can be performed in two ways, differing in the verifier module, which can use either an LLM or a metric ([COMETKiwi](https://aclanthology.org/2022.wmt-1.60.pdf) in our experiments). Here is the introduction of the parameters:

* **config**: The path of configuration, containing the LLM for inference, the inference hyper-parameters for each module. Please related to [./MQM_APE/configs/](./MQM_APE/configs/) for more details.

* **src**: The path of source segments.

* **tgt**: The path of target segments.

* **srclang**: Source language, such as "English", "Chinese".

* **tgtlang**: Target language, such as "English", "Chinese".

* **out**: The path of output annotations, scores and other informations.

* **metric_verifier**: A bool value controlling whether COMETKiwi is used to replace LLM verifier.

* **save_llm_response**: A bool value controlling whether to save the responses of LLM in each module.


## Comparison with Other MT Evaluation Strategies

MQM-APE is a **training-free** approach that improves upon GEMBA-MQM and complements training-dependent approaches such as Tower. It offers high-quality error annotations and post-edited translations.

| **Error-based MT Evaluation** | **Fine-grained Feedback** | **Error Span Enhancement** | **Post-Edited Translation** |
|-------------------------------|---------------------------|----------------------------|-----------------------------|
| **_Training-Dependent Approaches_** ||||  
| InstructScore ([Xu et al., 2023](https://aclanthology.org/2023.emnlp-main.365.pdf)) | ✔️ | ✔️ | ❌ |
| xCOMET ([Guerreiro et al., 2023](https://arxiv.org/pdf/2310.10482)) | ✔️ | ✔️ | ❌ |
| LLMRefine ([Xu et al., 2024](https://aclanthology.org/2024.findings-naacl.92.pdf)) | ✔️ | ❌ | ✔️ |
| Tower ([Alves et al., 2024](https://arxiv.org/pdf/2402.17733)) | ✔️ | ❌ | ✔️ |
| **_Training-Free Approaches_** ||||  
| GEMBA ([Kocmi & Federmann, 2023](https://aclanthology.org/2023.eamt-1.19.pdf)) | ❌ | ❌ | ❌ |
| EAPrompt ([Lu et al., 2024](https://aclanthology.org/2024.findings-acl.520.pdf)) | ✔️ | ❌ | ❌ |
| AutoMQM ([Fernandes et al., 2023](https://aclanthology.org/2023.wmt-1.100.pdf)) | ✔️ | ❌ | ❌ |
| GEMBA-MQM ([Kocmi & Federmann, 2023](https://aclanthology.org/2023.wmt-1.64.pdf)) | ✔️ | ❌ | ❌ |
| **MQM-APE (This work)** | ✔️ | ✔️ | ✔️ |

## Performance Comparison Between GEMBA-MQM and MQM-APE on Different LLMs

<div align="center">
    <img width="80%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/image.png">
</div>

>Comparison of performance between GEMBA-MQM ("MQM") and MQM-APE on WMT22 with human-labeled MQM, evaluated using pairwise accuracy (\%) at the system level, pairwise accuracy with tie calibration (\%) at the segment level, and error span precision of errors (SP) and major errors (MP), respectively.

>Building upon GEMBA-MQM, our purposed MQM-APE has the following advantages:

1. **Better Reliability**: MQM-APE consistently enhances GEMBA-MQM at both system and segment levels.

2. **Better Interpretability**: MQM-APE obtains higher error annotation quality compared with GEMBA-MQM.

3. **Evaluator Applicability**: MQM-APE complements MQM-based evaluators specifically trained for translation-related tasks.

4. **Language Generalizability**: MQM-APE obtains consistent improvements for almost all tested LLM on both high- and low-resource langauges.

## How to select LLM for Translation Evaluation?

Based on our analysis, we provide a guide on how to select LLMs as translation evaluators. For instance, Mixtral-8x22b-inst is the optimal choice for evaluator reliability when adopting large-scale LLMs for quality assessment. Users can download the model off-the-shelf and perform MQM-APE evaluation directly.

| **Aspect**         | **Model Scale** | **LLM Selection**      |
|--------------------|-----------------|------------------------|
| **Reliability**     | ○ Small         | [Tower-13b-inst](https://huggingface.co/Unbabel/TowerInstruct-13B-v0.1)         |
|                    | ● Large         | [Mixtral-8x22b-inst](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)     |
| **Interpretability**| ○ Small         | [Tower-13b-inst](https://huggingface.co/Unbabel/TowerInstruct-13B-v0.1)         |
|                    | ● Large         | [Llama3-70b-inst](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)        |
| **Inference Cost**  | ○ Small         | [Qwen1.5-14b-chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat)       |
|                    | ● Large         | [Qwen1.5-72b-chat](https://huggingface.co/Qwen/Qwen1.5-72B-Chat)       |

## Other Results and Findings of the Paper

> 1. APE translations exhibit superior overall quality compared to the original translations.

> 2. Quality Verifier aligns with modern metrics like [COMETKiwi](https://aclanthology.org/2022.wmt-1.60.pdf), which can be replaced by these metrics with comparable effects.

> 3. MQM-APE introduces acceptable costs against GEMBA-MQM, and preserves error distribution across severities and categories.

Please refer to our arXiv preprint for more details.

