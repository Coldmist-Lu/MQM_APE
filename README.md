# MQM-APE

<b>MQM-APE: Toward High-Quality Error Annotation Predictors with Automatic Post-Editing in LLM Translation Evaluators</b>. [Full paper](https://arxiv.org/pdf/2409.14335)

This repository presents MQM-APE, an enhanced framework for leveraging LLMs in translation evaluation. We also provide the performance of MQM-APE for the replication of the study.

## Abstract

Large Language Models (LLMs) have shown significant potential as judges for Machine Translation (MT) quality assessment, providing both scores and fine-grained feedback. Although approaches such as [GEMBA-MQM](https://aclanthology.org/2023.wmt-1.64.pdf) has shown SOTA performance on reference-free evaluation, the predicted errors do not align well with those annotated by human, limiting their interpretability as feedback signals. To *enhance the quality of error annotations* predicted by LLM evaluators, we introduce a **universal and training-free** framework, **MQM-APE**, based on the idea of **filtering out non-impactful errors by Automatically Post-Editing (APE) the original translation based on each error, leaving only those errors that contribute to quality improvement.** Specifically, we prompt the LLM to act as ① *evaluator* to provide error annotations, ② *post-editor* to determine whether errors impact quality improvement and ③ *pairwise quality verifier* as the error filter. Experiments show that our approach consistently improves both the reliability and quality of error spans against GEMBA-MQM, across eight LLMs in both high- and low-resource languages. Orthogonal to trained approaches, MQM-APE complements translation-specific evaluators such as [Tower](https://arxiv.org/pdf/2402.17733), highlighting its broad applicability. Further analysis confirm the effectiveness of each module and offer valuable insights into evaluator design and LLMs selection. 

## Overview

<div align="center">
    <img width="80%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/main.jpg">
</div>

We employ MQM-APE by prompting the same LLM to perform multiple roles without fine-tuning for each task. MQM-APE evaluates a given translation $y$ of source $x$ through three sequential modules:

1. **Error Analysis Evaluator** identifies errors in $y$, providing error demonstrations $\mathcal{E}$ with error span, category and severity;

2. **Automatic Post-Editor** post-edits $y$ based on each identified error $e_i \in \mathcal{E}$, producing a set of corrected translations $\mathcal{Y}_{pe}$;

3. **Pairwise Quality Verifier** checks whether the post-edited translations improve upon the original translation $y$. 

> Errors for which the APE translation fails to improve on the original are discarded, leaving a refined set of errors $\mathcal{E}^* \in \mathcal{E}$ that contribute to quality improvement. The translation is finally scored based on $\mathcal{E}^*$ following the MQM weighting scheme.

## How to Use MQM-APE?

Please refer to [./MQM_APE/run.sh](./MQM_APE/run.sh) for an example of using MQM-APE.

```bash
cd ./MQM_APE

python3 main.py \
  --config ./configs/llmconfig.yaml \
  --src ./test/srcs_zh.txt \
  --tgt ./test/tgts_en.txt \
  --srclang Chinese \
  --tgtlang English \
  --out ./test/outs/llm_verifier \
  [--metric_verifier] [--save_llm_response]
```

MQM-APE can be performed in two ways, differing in the verifier module, which can use either an LLM or a metric ([COMETKiwi](https://aclanthology.org/2022.wmt-1.60.pdf) in our experiments). Here is the introduction of the parameters:

* **config**: The path of configuration, containing the LLM for inference, the inference hyper-parameters for each module. Please related to [./MQM_APE/configs/](./MQM_APE/configs/) for more details.

* **src**: The path of source segments.

* **tgt**: The path of target segments.

* **srclang**: Source language, such as "English", "Chinese".

* **tgtlang**: Target language, such as "English", "Chinese".

* **out**: The path of output annotations, scores and other informations. A example output can be found in [./MQM_APE/test/outs/](./MQM_APE/test/outs/).

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
    <img width="95%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/main_res.png">
</div>

>Table: Comparison of performance between GEMBA-MQM ("MQM") and MQM-APE on WMT22 with human-labeled MQM, evaluated using pairwise accuracy (\%) at the system level, pairwise accuracy with tie calibration (\%) at the segment level, and error span precision of errors (SP) and major errors (MP), respectively.

Building upon GEMBA-MQM, our purposed MQM-APE has the following advantages:

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

### 1. APE translations exhibit superior overall quality compared to the original translations.

<div align="center">
    <img width="80%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/ape.png">
</div>

> Table: **Performance of Automatic Post Editor** measured with $CometKiwi_{22}^{QE}$ and $BLEURT_{20}$. "†" indicates that the metrics difference ($\Delta$) has >95\% estimated accuracy with humans ([kocmi et al., 2024](https://aclanthology.org/2024.acl-long.110.pdf)). For segment comparison, we define *Win* as cases where both $CometKiwi_{22}^{QE}$ and $BLEURT_{20}$ rate APE higher than TGT, *Lose* where they rate APE lower, and *Tie* when their evaluations conflict.

### 2. Quality Verifier aligns with modern metrics like [COMETKiwi](https://aclanthology.org/2022.wmt-1.60.pdf) and [BLEURT20](https://aclanthology.org/2020.acl-main.704).

<div align="center">
    <img width="40%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/verifier.png">
</div>

> Table: **Comparison of the pairwise quality verifier's consistency** with $CometKiwi_{22}^{QE}$ and $BLEURT_{20}$, which serve as ground truth.

### 3. MQM-APE exhibits superior performance compared to random error filter.

<div align="center">
    <img width="40%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/randomcompare_seg.png">
</div>

> Figure: **Comparison between MQM-APE, random error filter ("Random")** and GEMBA-MQM ("MQM") on segment-level performance.

### 4. MQM-APE introduces an acceptable inference cost compared to GEMBA-MQM. 

<div align="center">
    <img width="80%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/inference.png">
</div>

> Table: **Analysis of inference cost** averaged for each segment across different LLMs for each module, presenting input and generated tokens seperately.

### 5. A cost-reducing alternative of implementing MQM-APE is to replace the verifier with metrics for comparable performance.

<div align="center">
    <img width="80%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/metriccompare.png">
</div>

> Figure: **Comparison between MQM-APE with an LLM verifier and viwh $COMETKiwi_{22}^{QE}$** as a replacement on segment-level performance.

### 6. MQM-APE preserves error distribution across severities and categories.

<div align="center">
    <img width="80%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/errordist.png">
</div>
<div align="center">
    <img width="80%" alt="image" src="https://github.com/Coldmist-Lu/MQM_APE/blob/main/sources/category_pie.png">
</div>

> Figure: (Upper) Average Number of errors retained or discarded for each severity level with MQM-APE. (Lower) Distribution of error categories generated from GEMBA-MQM ("MQM") evaluator, MQM-APE, discarded errors, and human-annotated MQM, respectively.

Please refer to our [arXiv preprint](https://arxiv.org/pdf/2409.14335) for more details.

## Citation
If you find this work helpful, please consider citing as follows:  

```ruby
@article{Lu2024MQMAPE,
  title={MQM-APE: Toward High-Quality Error Annotation Predictors with Automatic Post-Editing in LLM Translation Evaluators},
  author={Lu, Qingyu and Ding, Liang and Zhang, Kanjian and Zhang, Jinxia and Tao, Dacheng},
  journal={arXiv preprint},
  url={https://arxiv.org/pdf/2409.14335},
  year={2024}
}
```

