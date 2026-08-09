# Efficient Table QA via TableGrid Navigation and Progressive Inference Prompting
 
**Amritansh Maurya**¹ · **Navjot Singh**¹ · **Mohammed Javed**² · **Omar Moured**³
 
¹ Vision Intelligence Lab, IIIT Allahabad, Prayagraj, India — `{rsi2024503, navjot}@iiita.ac.in`
² iMeDIA Lab, IIIT Allahabad, Prayagraj, India — `javed@iiita.ac.in`
³ CV:HCI Lab, Karlsruhe Institute of Technology, Karlsruhe, Germany — `omar.moured@kit.edu`
 
---
 
## Abstract
 
Large Language Models (LLMs) have shown promising results on NLP tasks, however, their performance on tabular data still needs research attention, because Table Question-Answering (TQA) requires precise cell retrieval and multi-step structured reasoning. Existing work improves TQA either by fine-tuning or training LLMs on task-specific tabular data, but often lacks verifiable control over how the model navigates tables and derives answers. In this work, we propose a training-free TQA approach with two structured prompting frameworks: **TableGrid Navigation (TGN)**, which iteratively navigates rows and columns via a three-module loop to locate evidence and refine answers, and **Progressive Inference Prompting (PIP)**, which enforces columns identification for explicit progressive row selection constraint according to the query. We evaluate **17 LLMs** against **6 baselines** on TableBench and FeTaQA dataset. On TableBench, TGN improves over the strongest baseline by **3.8 points**, and on FeTaQA, PIP achieves **SOTA performance** over ReAct and Chain-of-Thought. Beyond inference-time gains, PIP and TGN also narrows the performance gap to larger architectures in resource-constrained settings, offering cost-efficient solution for TQA.
 
**Keywords:** LLM · Table · Question-Answering · Prompt Engineering
 
---
 
## 1. Introduction
 
Table serves as backbone for representing structured data in professional documents, scientific reports, and financial records. Unlike unstructured text, tabular data encodes complex relationship through spatial arrangements, hierarchical headers, and heterogeneous formats. Consequently, table reasoning has emerged as a critical task in document analysis [[Fang et al., 2024](#references); [Zhang et al., 2025](#references)] and information retrieval, where the task of extracting, processing, and inferring information from a table is required to answer a specific query.
 
The recent evolution of language models such as GPT, Llama [[Grattafiori et al., 2024](#references)], Qwen [[Yang et al., 2024, 2025](#references)] has transformed natural language tasks, showcasing an extraordinary capacity to handle complex linguistic tasks, generating immense interest in a range of domains. However, from processing linear text to reasoning over table remains a notable challenge. Recent surveys [[Zhang et al., 2025](#references); [Deng et al., 2024](#references); [Fang et al., 2024](#references)] show that even state-of-the-art language models struggle with tabular data, because TQA is more than just semantic understanding; it requires structured reasoning, precise cell level extraction, and the ability to perform stepwise inference across table schemas.
 
**Figure 1** — Comparison of 4 prompting baselines, (a) Direct Prompting (DP), (b) Tree-of-Thought (ToT), (c) Chain-of-Thought (CoT), and (d) Reason+Act (ReAct), solving a TableBench question:
 
![Comparison of prompting baselines vs PIP](assets/pip_comparison.png)
 
Despite adaption of various prompting techniques aimed at improving LLM performance, it lacks systematic evaluation of these heuristics. Current research often relies on ad-hoc prompting strategies without a rigorous comparison of their efficiency, reliability, or generalizability across different model architectures. This gap raises a vital question: *Do general-purpose LLMs possess robust tabular reasoning skills in relation to real-world features of tabular data, or are they merely limited by the way we interface with them?*
 
Typically, these models take the user's input (prompt) for which they generate an output in response. This adaptability is different from traditional paradigms, where model re-training or extensive fine-tuning is often required for task-specific performance. The best way to check the potential of large language models is to evaluate how often they generate the correct answer for factoid-like questions. Existing prompting strategies such as Direct Prompting (DP) and Chain-of-Thought (CoT) reasoning improve performance but often suffers from redundancy, hallucinations, or a lack of grounding from table.
 
In this paper, we address these challenges by introducing two novel structured prompting framework: **TableGrid Navigation (TGN)** and **Progressive Inference Prompting (PIP)**, designed to enhance the reasoning capabilities of language models by prompt-based tuning for TQA task. Where PIP focuses on breaking down complex queries into a chain of logical sub-tasks, TGN treats the table as a spatial coordinate system, enabling the model to "navigate" the grid with higher precision. Our main contributions are threefold:
 
1. We introduce two structured prompting frameworks **TGN** and **PIP**, a novel LLM instruction-tuning strategy that significantly improves structured information extraction and complex reasoning over tabular data.
2. We conduct extensive experiments demonstrating that our frameworks significantly improves reasoning capabilities of language models for TQA task and help reduce hallucinations when dealing with tabular data.
3. We demonstrate that optimized prompting framework allows smaller models to bridge the performance gap with their larger counterparts.
---
 
## 2. Related Work
 
LLMs have demonstrated strong capabilities in understanding natural language and solving complex tasks via text generation, by which LLMs have achieved remarkable progress in many NLP tasks. Instead of text generation, language models are also being used for reasoning over several downstream tasks, performing logical and systematic problem solving. These advances in downstream tasks are largely enabled by in-context learning, a training-free paradigm in which models acquire task-specific behavior directly from instructions or demonstrations provided within the prompt. Prompting, therefore, serves as a mechanism for steering LLMs to achieve a specific task without updating model parameters.
 
The most known work of using LLMs for reasoning is Chain-of-Thought (CoT) [[Wei et al., 2022](#references)], which shows the LLMs ability to use their own thinking procedure for problem solving and showed promising results in complex reasoning benchmarks, following this several more works have been performed, including least-to-most prompting for solving complicated tasks [[Zhou et al., 2022](#references)], zero-shot CoT [[Kojima et al., 2022](#references)], and reasoning with self-consistency [[Wang et al., 2022](#references)]. While CoT generates a single linear reasoning chain before providing the final answer, it still fails when task requires strict constraint, symbolic precision or execution grounding. To counter these limitations, recent studies have also explored more sophisticated reasoning architecture like Symbolic Chain-of-Thought (SCoT) [[Xu et al., 2024](#references)] — combining reasoning with symbolic representations, Tree-of-Thought (ToT) [[Yao et al., 2023](#references)] — exploring multiple reasoning branches instead of one reasoning path and ReAct [[Yao et al., 2023](#references)] — combining action with reasoning to solve the problem. While these techniques have proven effective but domain-specific surveys for table reasoning highlight challenges like structured data navigation, multi-step reasoning, and precise cell retrieval required for TQA task. In contrast to these challenges, current literature do not explicitly consider the reasoning procedure required for table specific tasks and also lacks grounding of reasoning claims from tabular data.
 
---
 
## 3. Proposed Prompting Strategies
 
We propose two novel prompting strategies intended to enhance the complex reasoning capabilities of large language models over tabular question answering. By guiding language models toward more organized, iterative, and verifiable reasoning processes when working with tabular data, these strategies help the model to reduce more likely common issues like hallucinations, forgetting important data points, and inefficient computation (see Figure 1).
 
**Figure 2** — Framework of PIP, TGN and flow diagram of using prompting strategies for inference:
 
![Framework of PIP and TGN](assets/flow.png)
 
### 3.1 TableGrid Navigation (TGN)
 
Unlike conventional prompting techniques that rely solely on linear reasoning or unverified execution, TGN is a novel prompting methodology for structured tabular reasoning which introduces an iterative *analyzing–execution–validation* loop explicitly designed to operate over tabular schemas. The TGN prompting strategy on comparison with CoT, imposes structured execution and validation steps, moving away from free-form reasoning to ensure precision and data fidelity. In contrast to other prompting techniques such as DP, SCoT, ReAct, TGN adopts a linear, cyclic loop that incorporates a validation phase, ensuring that performed actions are cross-checked against the table for error correction and also avoids unnecessary explorations like ToT.
 
We define TGN as an ordered sequence of operations over a tabular dataset where $`Q`$ denotes the input query, $`T`$ represents the tabular schema, and $`R`$ signifies the final answer. The TGN process is a stateful function that operates over a sequence of iterations, each comprising Analyze $`A`$, Execute $`E`$ and Validate $`V`$ operations over a state space $`S`$.
 
```math
TGN(Q, T, S_0) = R
```
 
*(Equation 1)*
 
where $`R = \lim_{n \to k} S_n`$ and $`k`$ is the number of iterations until $`R`$ is found for a given $`Q`$.
 
$`S_n`$ can be written as:
 
```math
S_n = \mathcal{T}_n(S_{n-1}, Q, T)
```
 
*(Equation 2)*
 
The state $`S_n \in \mathcal{S}`$ at iteration $`n`$, initialized as $`S_0 = \emptyset`$, representing the initial state with no prior computations and the state transition function $`\mathcal{T}_n : \mathcal{S} \times Q \times T \to \mathcal{S}`$ at iteration $`n`$, can be defined as:
 
```math
\mathcal{T}_n(S_{n-1}, Q, T) = V_n(E_n(A_n(Q, T, S_{n-1}), T), T)
```
 
*(Equation 3)*
 
Where, the analysis function $`A_n(Q, T, S_{n-1}) : Q \times T \times \mathcal{S} \to P_n`$ generates a reasoning about how to interpret the data grid based on $`Q`$, $`T`$, and the previous state $`S_{n-1}`$. It maps to a plan space $`P_n \subseteq \mathcal{P}`$, then the execution function $`E_n(P_n, T) : \mathcal{P} \times T \to I_n`$ applies operations specified in $`P_n`$ on $`T`$, producing an intermediate result $`I_n \in \mathcal{I}`$, followed by the validation function $`V_n(I_n, T) : \mathcal{I} \times T \to \mathcal{S}`$ which verifies $`I_n`$ against $`T`$ and the cycle is repeated until convergence of $`R`$.
 
The framework overview of TGN prompting is that it decomposes the process into three distinct stages:
 
- **Analyze** — This stage involves interpreting of tabular schema and the query requirements by schema traversal and aggregating the targets or data points that can reach to the target present within the data grid.
- **Execute** — After aggregating the data points from the table grid, the model is directed to perform the execution of specific operations aligned with the analysis, such as value lookup, arithmetic computation, logical computation or relational computation.
- **Validate** — This step ensures the correctness of the execution by validating the result with aggregated data points, thereby reducing hallucinations.
This cycle of different stages can be repeated multiple times by the model if it detects any inconsistencies or ambiguities during the inference of suitable answer.
 
### 3.2 Progressive Inference Prompting (PIP)
 
In contrast to TGN's operation-centric loop, we developed another novel strategy PIP, which employs a linear, pattern-based progression to support the language model's reasoning, focused on gradual analysis. PIP decomposes the task into discrete, non-overlapping steps that ensure comprehensive coverage of the table while avoiding redundant processing. The distinction of PIP from existing prompt approaches can be viewed, as it doesn't work unconstrained like CoT and also identifies the columns for explicit progressive row selection constraint according to the query which is not present in other prompting approaches such as ToT, SCoT, and ReAct. Each intermediate step in PIP is explicitly tied to a selected row, which reduces variability in reasoning style and mitigates hallucination.
 
The PIP process is modeled as a composite function (see Figure 2) that applies five sequential steps, each producing an intermediate output that feeds into the next, with a minimal state to track progress. To understand the structure of PIP, let $`Q`$ denote the input query, $`T`$ represent the tabular schema, and $`R`$ represent the answer to the query; then the prompt can be written as $`PIP(Q, T) = R`$, where:
 
```math
R = F_5(F_4(F_3(F_2(F_1(Q, T), Q), C, T), Q', C))
```
 
*(Equation 4)*
 
Where:
 
- $`F_1(Q, T) : T \to C`$ — Identifies the columns of $`T`$, producing $`C = \lbrace c_i \mid c_i \in \text{columns}(T) \rbrace`$, where each $`c_i`$ is annotated with its semantic meaning.
- $`F_2(Q, C) : Q \to Q'`$ — Restates the query $`Q`$ into a clarified version $`Q'`$, aligning it with the column meanings in $`C`$.
- $`F_3(Q', C, T) : Q' \times C \times T \to R_s`$ — Extracts relevant rows $`R_s \subseteq \text{rows}(T)`$, selected based on a relevance measure of how well $`r_j`$ satisfies $`Q'`$.
- $`F_4(Q', R_s, C) : Q' \times R_s \times C \to I_j`$ — Performs intermediate analysis on $`R_s`$, processing each row once to produce intermediate results $`I_j = \lbrace i_j \mid i_j = \psi(r_j, C) \rbrace`$, where $`\psi`$ is an operation based on $`C`$.
- $`F_5(I_j) : I \to R`$ — Synthesizes the final answer $`R`$ by aggregating $`I_j`$s.
This constrained sequential steps of PIP directs the model only to reason over the required rows and columns according to the query, also helps in reducing hallucination by stoping the model from unnecessary processing of whole table. In empirical experiments, we will observe the utility of TGN and PIP by evaluating their performance on two tabular datasets.
 
---
 
## 4. Experiment
 
### 4.1 Benchmark Datasets
 
For experiment this study utilizes two tabular datasets: **TableBench** [[Wu et al., 2025](#references)] and **FeTaQA** [[Nan et al., 2022](#references)], which serves as an established benchmark in the domain of TableQA. TableBench consists large portion of tables from financial reports and data from competitive events, which are moderately sized, featuring an average of 6.68 columns and 16.71 rows. Notably, 65.74% of all table cells contain numerical values and the average reasoning steps required per question is 6.26, highlighting the multi-hop and compositional nature of the quantitative reasoning required. Whereas in FeTaQA, most of the instances are related to biography, sports, geographical regions, media, politics, and government, which necessitates the synthesis of free-form, natural language responses based on Wikipedia tables, addressing the generative challenges of TableQA. This requires the model to not only identify relevant data points across discontinuous cells but also to aggregate them into coherent, 'faithful' explanations.
 
### 4.2 Experimental Setup & Baselines
 
In this study, we conducted all our experiments on a machine with dual 24 GB NVIDIA GeForce RTX 4090 GPUs using the vLLM framework for transformer-based model architectures where the maximum sequence length for each model was set to 8000 tokens, and the decoding configuration used a temperature of 0.7 with probabilistic sampling to prioritize grounded and contextually reliable outputs, and Flash Attention was enabled to accelerate attention mechanisms.
 
We compare our methods against six baselines: **DP**, **CoT**, **SCoT**, **ToT**, **ReAct** and a hybrid prompt **ToT with SelfAsk**, combining the features of SelfAsk with ToT. For inference, we utilize **17 SOTA models** with sizes ranging from 0.6B to 8B parameters, including general open-source LLMs and table-based fine-tuned models. For open-source LLMs, we evaluate on DeepSeek-R1-Distill-Llama-8B, Llama3.1-8B-Instruct, Llama3.2-3B-Instruct, Llama3-8B-Instruct, Qwen2-7B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B-Instruct, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B. For fine-tuned models, we evaluate on TableGPT2-7B and TableLLM by finetuning all parameters of baseline LLMs CodeQwen-7B, DeepseekCoder-7B, Llama3-8B, Llama3.1-8B and Qwen2-7B to learn from the TableInstruct set.
 
### 4.3 Evaluation Metrics
 
To maintain consistency with prior work and ensure comparability, we adopt evaluation protocols from the original publications. For TableBench, we employ Exact Match for categories like Fact Checking, `EM_with_error_10` used for Correlation Analysis, Trend Forecasting, and Statistical Analysis, and for other data analysis tasks Rouge-L is used to assess the quality of the generated answers. For FeTaQA, we report sacreBLEU, ROUGE-{1, 2, L} and METEOR which evaluates the n-gram match between generated and reference answers. As these measures lack semantic meanings of sentences, we also report BERTScore and BLEURT scores, that incorporate semantics using contextual embeddings.
 
### 4.4 Quantitative Results
 
The quantitative results of our framework against baselines are presented in four parts. First we benchmark TGN and PIP on TableBench: **Table 1** presents accuracy of all methods across sub-tasks — *Fact Checking, Numerical Reasoning* and *Data Analysis* — where TGN scored SOTA accuracy of **85.42%**, **64.48%** and **26.63%** respectively; in contrast PIP also demonstrates the in-line performance with second best consistent performer across number of models.
 
**Table 1** — Quantitative analysis of methods on Fact Checking, Numerical Reasoning and Data Analysis task present in TableBench dataset. Row best value is shown in **bold** and second best in *italics*.
 
***Fact Checking***
 
| Models | DP | CoT | SCoT | ReAct | ToT | ToT-SelfAsk | PIP (Ours) | TGN (Ours) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1-Distill-Llama-8B | 43.75 | 51.04 | 59.38 | **62.50** | 54.17 | 46.88 | *59.38* | 44.79 |
| Llama-3.1-8B-Instruct | 51.04 | 41.67 | **55.21** | 45.83 | 16.67 | 41.67 | 47.92 | 42.71 |
| Llama-3.2-3B-Instruct | **25.00** | 5.21 | 1.04 | 11.46 | 3.12 | 5.21 | *12.50* | 7.29 |
| Meta-Llama-3-8B-Instruct | 59.38 | 33.33 | 56.25 | 55.21 | 52.08 | 56.25 | **62.50** | 50.00 |
| Qwen2-7B-Instruct | 46.88 | 42.71 | 61.46 | 63.54 | 43.75 | 48.96 | **69.79** | 60.42 |
| Qwen2.5-7B-Instruct | 55.21 | 56.25 | 50.00 | 32.29 | 38.54 | 52.08 | 51.04 | **58.33** |
| Qwen2.5-Coder-7B-Instruct | 54.17 | **73.96** | 63.54 | 65.62 | 67.71 | 59.38 | 67.71 | *70.83* |
| Qwen3-0.6B | 18.75 | 18.75 | 3.12 | 14.58 | 5.21 | 9.38 | **28.12** | 17.71 |
| Qwen3-1.7B | 59.38 | 57.29 | 55.21 | **61.46** | 3.12 | 3.12 | 51.04 | 45.83 |
| Qwen3-4B | 81.25 | 81.25 | **85.42** | 79.17 | 76.04 | 77.08 | 76.04 | *82.29* |
| Qwen3-8B | 79.17 | 78.12 | 82.29 | 84.38 | 75.00 | 82.29 | 82.29 | **85.42** |
| TableGPT2-7B | 36.46 | 48.96 | 31.25 | 25.00 | 42.71 | **54.17** | *53.12* | 43.75 |
| TableLLM-CodeQwen-7B | 64.58 | 57.29 | 59.38 | 42.71 | 66.67 | 67.71 | **68.75** | *67.71* |
| TableLLM-DeepseekCoder-7B | 59.38 | **66.67** | 64.58 | 60.42 | 63.54 | 60.42 | 61.46 | *65.96* |
| TableLLM-Llama3-8B | 59.38 | 61.46 | 63.54 | 62.50 | 62.50 | **63.54** | 61.46 | 60.42 |
| TableLLM-Llama3.1-8B | 64.58 | 61.46 | 55.21 | 63.54 | 65.62 | **67.71** | *65.62* | 64.58 |
| TableLLM-Qwen2-7B | 62.50 | **67.71** | 59.38 | 64.58 | 61.46 | 45.83 | 63.54 | 61.46 |
 
***Numerical Reasoning***
 
| Models | DP | CoT | SCoT | ReAct | ToT | ToT-SelfAsk | PIP (Ours) | TGN (Ours) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1-Distill-Llama-8B | 34.51 | 44.58 | 32.24 | 44.84 | 32.24 | 32.49 | **50.13** | 30.98 |
| Llama-3.1-8B-Instruct | 7.81 | 12.59 | 13.35 | **13.85** | 3.53 | 12.09 | 11.34 | 10.08 |
| Llama-3.2-3B-Instruct | **4.03** | 3.53 | 1.01 | 3.02 | 0.00 | 0.25 | 2.52 | *3.78* |
| Meta-Llama-3-8B-Instruct | 10.08 | 15.87 | 15.87 | **21.91** | 18.14 | 16.37 | *21.66* | 12.09 |
| Qwen2-7B-Instruct | 11.59 | 11.59 | 16.37 | 23.68 | 4.53 | 12.59 | **24.69** | 11.84 |
| Qwen2.5-7B-Instruct | 9.82 | **25.69** | 16.88 | 14.11 | 9.82 | 12.59 | *24.94* | 23.68 |
| Qwen2.5-Coder-7B-Instruct | 13.35 | **37.03** | 18.89 | 35.26 | 12.34 | 20.40 | 31.23 | 27.96 |
| Qwen3-0.6B | 14.11 | 14.61 | 4.03 | **18.39** | 12.59 | 9.82 | *15.37* | 13.35 |
| Qwen3-1.7B | 26.70 | 30.98 | 38.04 | 43.58 | 0.76 | 1.51 | **43.83** | 38.29 |
| Qwen3-4B | 48.11 | 44.33 | **61.46** | 53.65 | 56.42 | 51.64 | 42.07 | *58.69* |
| Qwen3-8B | 55.92 | 54.16 | 62.97 | 62.47 | 57.18 | 60.71 | 49.37 | **64.48** |
| TableGPT2-7B | 15.87 | 29.22 | 12.85 | 16.37 | 18.64 | 19.40 | **31.23** | 22.67 |
| TableLLM-CodeQwen-7B | 10.58 | **23.93** | 8.82 | 14.86 | 10.33 | 12.09 | 10.58 | 10.33 |
| TableLLM-DeepseekCoder-7B | 14.86 | **31.74** | 15.37 | 24.94 | 18.64 | 14.61 | 13.60 | 19.44 |
| TableLLM-Llama3-8B | 12.09 | **28.97** | 13.85 | 13.35 | 12.34 | 12.85 | 11.59 | 12.85 |
| TableLLM-Llama3.1-8B | 13.85 | **29.72** | 15.37 | 24.18 | 12.09 | 12.85 | 13.35 | 11.84 |
| TableLLM-Qwen2-7B | 14.86 | 35.26 | 19.65 | 31.74 | 33.75 | 30.98 | **38.54** | 31.99 |
 
***Data Analysis***
 
| Models | DP | CoT | SCoT | ReAct | ToT | ToT-SelfAsk | PIP (Ours) | TGN (Ours) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1-Distill-Llama-8B | 18.08 | **22.91** | 18.39 | 20.28 | 18.72 | 20.63 | *22.17* | 19.55 |
| Llama-3.1-8B-Instruct | 16.20 | 16.54 | 14.81 | 17.03 | 10.27 | 13.44 | 15.59 | **17.48** |
| Llama-3.2-3B-Instruct | **14.10** | 10.12 | 9.94 | 13.25 | 10.32 | 8.35 | 11.32 | 12.33 |
| Meta-Llama-3-8B-Instruct | 19.05 | 17.60 | 17.05 | 19.61 | 12.76 | 14.88 | 17.97 | **20.01** |
| Qwen2-7B-Instruct | 17.81 | 16.52 | 17.45 | **20.72** | 14.81 | 15.50 | 16.38 | 16.01 |
| Qwen2.5-7B-Instruct | 17.84 | 18.78 | 17.38 | 21.75 | 17.56 | 17.11 | **22.23** | 20.61 |
| Qwen2.5-Coder-7B-Instruct | 22.07 | 21.69 | 18.72 | 21.44 | 11.89 | 15.02 | **23.12** | 19.48 |
| Qwen3-0.6B | 13.36 | 13.66 | 11.64 | **13.84** | 11.72 | 12.05 | 11.18 | 12.59 |
| Qwen3-1.7B | 16.15 | 17.60 | 15.48 | **19.38** | 10.96 | 12.88 | 14.43 | 17.85 |
| Qwen3-4B | 22.23 | **22.41** | 20.43 | 21.01 | 17.71 | 19.38 | 20.71 | 20.44 |
| Qwen3-8B | 26.42 | 23.14 | 23.47 | 25.93 | 20.82 | 21.58 | 23.36 | **26.63** |
| TableGPT2-7B | 7.86 | 11.31 | 16.80 | 13.46 | 12.65 | 14.94 | **17.82** | 14.34 |
| TableLLM-CodeQwen-7B | **23.93** | 20.28 | 20.52 | 20.03 | 21.83 | 23.20 | *23.46* | 21.25 |
| TableLLM-DeepseekCoder-7B | 24.54 | 22.24 | 20.06 | 24.51 | **24.65** | 22.52 | 23.31 | 23.68 |
| TableLLM-Llama3-8B | 21.69 | 20.57 | 21.38 | 21.32 | 22.43 | 22.61 | **23.23** | 22.22 |
| TableLLM-Llama3.1-8B | 23.27 | 21.75 | 21.02 | 20.57 | 22.12 | **24.86** | 21.19 | *23.57* |
| TableLLM-Qwen2-7B | 23.99 | 22.39 | 23.67 | 24.71 | 22.57 | 21.10 | **26.51** | 23.77 |
 
After task-wise analysis, **Table 2** showcases the overall accuracy of methods on TableBench dataset, where TGN demonstrated clear superiority, outperforming baselines including ReAct and CoT and achieving SOTA performance of **48.46%**. For tabular fine-tuned LLMs, PIP scored highest with an accuracy of **34.41%** and achieving highest in 5 and second highest score in 3 out of 17 LLMs.
 
**Table 2** — Quantitative analysis of overall accuracy on TableBench dataset. Row best value is shown in **bold** and second best in *italics*.
 
***General LLMs***
 
| Models | DP | CoT | SCoT | ReAct | ToT | ToT-SelfAsk | PIP (Ours) | TGN (Ours) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1-Distill-Llama-8B | 27.20 | 34.38 | 28.00 | 34.71 | 27.56 | 27.63 | **37.48** | 26.31 |
| Llama-3.1-8B-Instruct | 15.31 | 16.56 | 17.70 | **17.77** | 7.36 | 15.13 | 16.31 | 15.91 |
| Llama-3.2-3B-Instruct | **9.97** | 6.06 | 4.41 | 7.72 | 4.34 | 3.91 | 6.87 | 7.26 |
| Meta-Llama-3-8B-Instruct | 18.33 | 17.53 | 19.81 | 23.40 | 18.71 | 19.19 | **23.44** | 18.58 |
| Qwen2-7B-Instruct | 17.17 | 16.21 | 20.75 | **25.51** | 12.51 | 16.95 | *24.96* | 18.05 |
| Qwen2.5-7B-Instruct | 17.29 | 24.88 | 19.71 | 18.24 | 15.37 | 17.91 | **25.31** | *24.91* |
| Qwen2.5-Coder-7B-Instruct | 20.39 | **33.00** | 22.60 | 31.21 | 17.47 | 21.39 | 30.29 | 27.74 |
| Qwen3-0.6B | 13.53 | 13.87 | 6.65 | **15.18** | 10.74 | 10.08 | *14.26* | 12.77 |
| Qwen3-1.7B | 24.65 | 26.90 | 29.02 | **33.68** | 4.92 | 6.00 | *30.76* | 29.03 |
| Qwen3-4B | 38.97 | 37.34 | **44.70** | 40.75 | 40.38 | 38.99 | 35.11 | *43.13* |
| Qwen3-8B | 43.86 | 41.69 | 46.22 | 47.17 | 41.81 | 44.47 | 40.08 | **48.46** |
 
***Fine-tuned LLMs***
 
| Models | DP | CoT | SCoT | ReAct | ToT | ToT-SelfAsk | PIP (Ours) | TGN (Ours) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TableGPT2-7B | 14.11 | 22.78 | 15.65 | 15.25 | 17.88 | 20.34 | **26.65** | 20.45 |
| TableLLM-CodeQwen-7B | 21.00 | **24.78** | 18.33 | 19.04 | 20.30 | 21.74 | 21.28 | 20.19 |
| TableLLM-DeepseekCoder-7B | 22.59 | **30.06** | 21.65 | 27.21 | 24.78 | 21.81 | 21.77 | 25.00 |
| TableLLM-Llama3-8B | 20.24 | **27.61** | 21.37 | 21.01 | 20.98 | 21.40 | 20.84 | 20.91 |
| TableLLM-Llama3.1-8B | 22.22 | **28.40** | 21.00 | 25.68 | 21.09 | 22.72 | 21.29 | 21.43 |
| TableLLM-Qwen2-7B | 22.72 | 31.81 | 24.40 | 30.78 | 30.52 | 27.02 | **34.41** | 30.20 |
 
Next, we evaluate on FeTaQA test set, by running inference using 17 LLMs for each baseline, resulting a total of 136 inferences. To keep the results insightful and compact, we select the best performing output of each baseline across the LLMs, as shown in **Table 3**. We observe that the methodology using PIP framework achieved SOTA accuracy on FeTaQA with TGN as second best scorer and outperforming other baselines.
 
**Table 3** — Quantitative Analysis of metrics on FeTaQA test set by selection of best performance of each baseline. Column best value is shown in **bold** and second best in *italics*.
 
| Methodology | sacreBLEU ↑ | R-1 ↑ | R-2 ↑ | R-L ↑ | METEOR ↑ | BERTScore ↑ | BLEURT ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Meta-Llama-3-8B-Instruct + DP | 17.01 | 0.52 | 0.31 | 0.43 | 0.42 | **0.67** | 0.54 |
| DeepSeek-R1-Distill-Llama-8B + TCoT | 13.91 | 0.47 | 0.27 | 0.38 | 0.38 | 0.44 | 0.51 |
| Llama-3.1-8B-Instruct + TCoT | 14.83 | 0.47 | 0.28 | 0.39 | 0.38 | 0.42 | 0.51 |
| TableGPT2-7B + SCoT | 15.37 | 0.48 | 0.29 | 0.39 | 0.40 | 0.44 | 0.58 |
| DeepSeek-R1-Distill-Llama-8B + ReAct | 17.64 | 0.54 | 0.32 | 0.44 | 0.45 | 0.52 | 0.58 |
| Llama-3.1-8B-Instruct + ToT | 17.28 | 0.54 | 0.32 | 0.44 | 0.44 | 0.51 | 0.58 |
| **Meta-Llama-3-8B-Instruct + PIP (Ours)** | **19.32** | **0.58** | **0.35** | **0.47** | **0.48** | *0.56* | **0.60** |
| **Meta-Llama-3-8B-Instruct + TGN (Ours)** | *17.96* | *0.54* | *0.33* | *0.45* | *0.45* | 0.52 | *0.58* |
 
In **Table 4**, we further demonstrate the capability of our proposed framework on TQA task by comparing the overall accuracy of LLMs greater than 8B parameters with smaller LLM. Here, Qwen3-8B with TGN, having 48.46% accuracy, surpasses many large models with baselines on the TableBench dataset.
 
**Table 4** — Overall accuracy comparison of language models with PIP and TGN framework against models ≥ 8B parameters using baselines on TableBench dataset.
 
| Methodology | # Params ↓ | Overall Acc. ↑ |
|---|:---:|:---:|
| GPT-3.5-Turbo + PoT | – | 37.15 |
| Llama3-70B-Chat + TCoT | 70B | 38.68 |
| Llama3.1-70B-Instruct + TCoT | 70B | 41.05 |
| QWQ-32B + DP | 32B | 43.87 |
| Qwen2.5-Coder-32B-Instruct + TCoT | 32B | 45.51 |
| Llama-4-Scout-17B-16E-Instruct + TCoT | 17B | 46.53 |
| Qwen2.5-72B-Instruct + TCoT | 72B | 48.79 |
| Llama-3.1-405B-Instruct + TCoT | 405B | 48.87 |
| **DeepSeek-R1-Distill-Llama-8B + PIP (ours)** | **8B** | **37.48** |
| **Qwen3-8B + TGN (ours)** | **8B** | **48.46** |
 
These findings challenge the prevailing 'bigger is better' paradigm in LLM performance, showing that sophisticated framework enables compact models to transcend the accuracy benchmarks set by significantly larger models using conventional methods. For more detailed analysis, we share the accuracy of our proposed frameworks on different question categories present in TableBench dataset in [Appendix B](#appendix-b-detailed-analysis).
 
For cost-effectiveness analysis of TGN & PIP we analyzed the inference-time token consumption across all 17 models with each framework as shown in **Table 5**, where TGN required **680.22 tokens/query** on average, outperforming competitive baselines such as ReAct (788.09 tokens) and SCoT (1105.58 tokens).
 
**Table 5** — Average token consumption of language models during inference with PIP and TGN framework against baselines on TableBench dataset.
 
| Framework | Avg. Token Count / Example ↓ |
|---|:---:|
| SCoT | 1105.58 |
| ToT | 795.67 |
| ReAct | 788.09 |
| CoT | 778.83 |
| ToT-SelfAsk | 713.23 |
| **PIP** | **794.09** |
| **TGN** | **680.22** |
 
### 4.5 Error Analysis
 
To analyze systematic limitations, we conducted an analysis of failures across both datasets. In TableBench, while the model achieves high proficiency in Fact Checking and Numerical Reasoning it primarily fails in Data Analysis, where its subtypes require pattern interpretation over tables. Causal Analysis (38.13%), Anomaly Detection (32.32%), and Trend Forecasting (18.0%) — all hover near chance, as the model produces plausible but table-unsupported reasoning; Statistical Analysis (28.0%) and Descriptive Analysis (23.67%) similarly fails at open-ended summarization, while Impact Analysis (36.0%) improve as expected outputs become more constrained. Within Numerical Reasoning, vulnerabilities also emerge in Domain-Specific tasks and Counting tasks due to missing background knowledge and multi-condition filtering omissions.
 
In FeTaQA, model struggles most with questions that require multi-fact synthesis or inferential generation. Causal questions are the hardest, with approx. 30.8% answered correctly, as the model produces surface-level responses unsupported by table evidence. Questions which requires enumeration fails because the model retrieves only the most salient match rather than aggregating all qualifying rows values into a complete answer. Comparative/Ranking questions suffer from both incorrect extremum identification and incomplete contextualisation of the result. Answer which includes person or entity shows bimodal behavior, either correct retrieval or a hallucinated name with no partial credit. Ultimately, analysis of the poorest-performing FeTaQA generations isolates partial or incomplete answers (18.9% of samples) and partial hallucinations (12.6% of samples) as the dominant error modalities, underscoring a persistent systemic limitation.
 
### 4.6 Ablation Study
 
To evaluate the effectiveness, we conducted the ablation studies by selectively removing key modules and by measuring the sensitivity of frameworks as shown in **Table 6**. In TGN, removing the last two modules caused the model to reason over tables with incorrect actions (Stage 1), while removing only the validation module led to hallucinations due to poor grounding (Stage 2), resulting in a significant performance drop from its SOTA results. Similarly, ablation results on PIP demonstrated the critical role of structural decomposition: removing schema identification and intermediate reasoning (Case 1) disrupted logical consistency, whereas removing row extraction and analysis (Case 2) reduced accuracy by preventing the model from isolating relevant variables from tabular noise.
 
**Table 6** — Ablation study of TGN & PIP Framework respectively by performing inference on TableBench using top scoring LLMs from overall result.
 
| Framework | Model | Stage 1 | Stage 2 |
|---|---|:---:|:---:|
| **TGN** | Qwen3-8B | 41.6 | 47.17 |
| | Qwen3-4B | 37.34 | 40.75 |
| | Qwen2.5-7B-Instruct | 24.88 | 18.24 |
 
| Framework | Model | Case 1 | Case 2 |
|---|---|:---:|:---:|
| **PIP** | Qwen2.5-7B-Instruct | 21.39 | 19.14 |
| | DeepSeek-R1-Distill-Llama-8B | 35.83 | 35.13 |
| | TableLLM-Qwen2-7B | 31.79 | 33.29 |
 
For sensitivity, we evaluate TGN & PIP using heavily paraphrased templates without disturbing the logical structure of the framework. Applying these altered templates to Qwen3-8B on TableBench showed remarkable stability, for TGN the accuracy shifted from 48.46% to 45.45% & for PIP it remained identical from 40.08% to 40.52%. This minimal variance gives evidence that gains of TGN & PIP are driven by logic of the framework.
 
---
 
## 5. Conclusion
 
In this paper, we introduced **TGN** and **PIP**, the first prompting framework designed to enhance the complex reasoning capabilities of LLMs in TableQA. By formalizing TGN as an iterative, validation-driven cycle for answer grounding from table and PIP as a constraint enforcer of column and row selection related to the query, we addressed key limitations in existing methods, such as unstructured reasoning and inefficiency in tabular data handling. Through comprehensive experiments on seventeen state-of-the-art large language models across two different categories, we demonstrate that TGN and PIP consistently outperformed baselines like Chain-of-Thought, and ReAct.
 
---
 
## Acknowledgements
 
This research work was funded by Visvesvaraya PhD Scheme for Electronics & IT Phase-II under the Ministry of Electronics and Information Technology, Government of India.
 
**Disclosure of Interests.** The authors have no competing interests to declare that are relevant to the content of this article.
 
---
 
## Appendix A: Prompts
 
In this section, we present demonstration used across TableBench dataset. We select the same answer output format from TableBench across multiple sub-tasks to keep the evaluation fair and the metrics on same scale as original.
 
### A.1 Demonstration of Progressive Inference Prompting
 
Our primary goal is to help the model understand the process of progressive inference workflow through the zero-shot five steps instructions given to it. The `{table_str}` contains the full table string in JSON format after pre-processing and `{question}` contains the query for the given table, for which the LLM generates the answer in the given answer format.
 
**Progressive Inference Prompting on TableBench Dataset**
 
```text
You are a table reasoning assistant. You will be given a table and a question.
Your goal is to progressively reason over the table before giving the final answer.
 
[Guidelines]
You should act in following patterns step by step to analyze the table and then
give the final answer:
 
[Action Patterns]
Step 1: Identify the table columns and their meaning.
Step 2: Identify the question and restate the question in your own words.
Step 3: Extract relevant rows from the table in a sequence.
Step 4: Do intermediate analysis, calculations, or comparisons step by step for
        each relevant row only once.
Step 5: Provide the final answer.
 
The answer should follow the format below:
 
[Answer Format]
Final Answer: AnswerName1, AnswerName2...
 
Ensure the final answer format is the last output line and can only be in the
"Final Answer: AnswerName1, AnswerName2..." form, no other form. Ensure the
"AnswerName" is a number or entity name, as short as possible, without any
explanation.
 
[TABLE]
{table_str}
 
Let's get start!
Question: {question}
```
 
### A.2 Demonstration of TableGrid Navigation Prompting
 
The prompting technique of TGN uses three modules, where we tried to explain the model about how each module works and these modules can be repeated until the query is resolved. Here also the `{table_str}` contains the full table string in JSON format after pre-processing and `{question}` contains the query for the given table.
 
**TableGrid Navigation Prompting on TableBench Dataset**
 
```text
You are a table assistant who solves the query by analyzing the question and
executing operations.
 
[Guidelines]
Use the following format for processing tabular queries:
 
Analyze: [reasoning about how to interpret the data grid or query]
Execute: [specific operation to perform on the tabular schema, e.g., lookup,
         calculation, or aggregation]
Validate: [verification of the result against the data grid]
... (repeat Analyze - Execute - Validate as needed to resolve the query)
 
The answer should follow the format below:
 
[Answer Format]
Final Answer: AnswerName1, AnswerName2...
 
Ensure the final answer format is the last output line and can only be in the
"Final Answer: AnswerName1, AnswerName2..." form, no other form. Ensure the
"AnswerName" is a number or entity name, as short as possible, without any
explanation.
 
[TABLE]
{table_str}
 
Let's get start!
Question: {question}
```
 
### A.3 Prompt used on FeTaQA Dataset
 
Inference on the FeTaQA dataset was performed using the similar framework configurations specified above for PIP and TGN, respectively. As FeTaQA employs free-form natural language answers rather than structured outputs in its ground truth annotations, corresponding adjustments were made to the inference configuration. Specifically, we set the `[Answer Format]` to `AnswerSentence` to ensure format compatibility between model outputs and reference answers, thereby enabling fair and meaningful evaluation.
 
**Answer Format for FeTaQA Dataset**
 
```text
The answer should follow the format below:
 
[Answer Format]
Final Answer: AnswerSentence
 
Ensure the final answer format is the last output line and can only be in the
"Final Answer: AnswerSentence" form, no other form. Ensure the "AnswerSentence"
is a single sentence, as short as possible, without any explanation.
 
[TABLE]
{table_str}
 
Let's get start!
Question: {question}
```
 
---
 
## Appendix B: Detailed Analysis
 
In this section, we share the results of our proposed frameworks on different question categories present in TableBench dataset.
 
**Abbreviations:** NR = Numerical Reasoning; FC = Fact Checking; DA = Data Analysis; A = Aggregation; MB = Match-Based; I = Impact Analysis; C = Causal Analysis; MH = Multi-Hop; Cn = Counting; Ds = Descriptive Analysis; An = Anomaly Detection; Dm = Domain-Specific; T = Time-based Calculation; Ar = Arithmetic Calculation; Rk = Ranking; Tr = Trend Forecasting; St = Statistical Analysis; Cp = Comparison.
 
**Table 7** — Accuracy across different question categories present in TableBench dataset using **TGN** framework (Part 1).
 
| Model | NR-A | FC-MB | DA-I | DA-C | NR-MH | FC-MH | NR-Cn | DA-Ds |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1-Distill-Llama-8B | 24 | 52.17 | 22 | 36.36 | 21.57 | 38 | 32 | 24.94 |
| Llama-3.1-8B-Instruct | 0 | 52.17 | 10 | 39.48 | 1.96 | 34 | 6 | 25.87 |
| Llama-3.2-3B-Instruct | 0 | 6.52 | 2 | 43.71 | 0 | 8 | 4 | 18.55 |
| Meta-Llama-3-8B-Instruct | 2 | 67.39 | 12 | 49.34 | 3.92 | 34 | 8 | 28.31 |
| Qwen2-7B-Instruct | 2 | 67.39 | 14 | 37.76 | 1.96 | 54 | 18 | 23.62 |
| Qwen2.5-7B-Instruct | 10 | 73.91 | 24 | 37.31 | 17.65 | 44 | 12 | 28.33 |
| Qwen2.5-Coder-7B-Instruct | 6 | 82.61 | 26 | 36.34 | 17.65 | 60 | 18 | 21.57 |
| Qwen3-0.6B | 18 | 26.09 | 4 | 35.77 | 7.84 | 10 | 10 | 14.78 |
| Qwen3-1.7B | 46 | 54.35 | 8 | 40.17 | 33.33 | 38 | 58 | 17.32 |
| Qwen3-4B | 48 | 97.83 | 18 | 30.28 | 41.18 | 68 | 72 | 25.34 |
| Qwen3-8B | 56 | 97.83 | 36 | 38.13 | 50.98 | 74 | 76 | 23.67 |
| TableGPT2-7B | 12 | 67.39 | 24 | 25.65 | 13.73 | 22 | 30 | 15.85 |
| TableLLM-CodeQwen-7B | 4 | 82.61 | 20 | 44.34 | 1.96 | 54 | 10 | 23.61 |
| TableLLM-DeepseekCoder-7B | 8 | 77.78 | 30.61 | 45.71 | 17.65 | 55.1 | 18.37 | 25.73 |
| TableLLM-Llama3-8B | 4 | 73.91 | 32 | 45.92 | 0 | 48 | 20 | 25.44 |
| TableLLM-Llama3.1-8B | 6 | 84.78 | 32 | 47.08 | 1.96 | 46 | 14 | 26.29 |
| TableLLM-Qwen2-7B | 12 | 73.91 | 32 | 45.4 | 19.61 | 50 | 42 | 26.06 |
 
**Table 8** — Accuracy across different question categories present in TableBench dataset using **TGN** framework (Part 2).
 
| Model | DA-An | NR-Dm | NR-T | NR-Ar | NR-Rk | DA-Tr | DA-St | NR-Cp |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1-Distill-Llama-8B | 23.43 | 38.78 | 38.3 | 30 | 34 | 20 | 6 | 30 |
| Llama-3.1-8B-Instruct | 23.62 | 16.33 | 12.77 | 0 | 24 | 22 | 0 | 20 |
| Llama-3.2-3B-Instruct | 15.39 | 12.24 | 4.26 | 0 | 10 | 10 | 0 | 0 |
| Meta-Llama-3-8B-Instruct | 25.49 | 16.33 | 14.89 | 12 | 22 | 26 | 2 | 18 |
| Qwen2-7B-Instruct | 16.75 | 8.16 | 10.64 | 12 | 22 | 18 | 4 | 20 |
| Qwen2.5-7B-Instruct | 28.36 | 20.41 | 23.4 | 32 | 34 | 16 | 12 | 40 |
| Qwen2.5-Coder-7B-Instruct | 22.57 | 32.65 | 27.66 | 30 | 40 | 16 | 16 | 52 |
| Qwen3-0.6B | 22.46 | 2.04 | 19.15 | 28 | 4 | 6 | 8 | 18 |
| Qwen3-1.7B | 24.14 | 22.45 | 38.3 | 46 | 24 | 18 | 16 | 38 |
| Qwen3-4B | 28.99 | 55.1 | 51.06 | 56 | 82 | 16 | 18 | 64 |
| Qwen3-8B | 32.32 | 55.1 | 65.96 | 74 | 74 | 18 | 28 | 64 |
| TableGPT2-7B | 12.23 | 22.45 | 17.02 | 28 | 22 | 18 | 6 | 36 |
| TableLLM-CodeQwen-7B | 33.86 | 12.24 | 12.77 | 6 | 26 | 18 | 8 | 10 |
| TableLLM-DeepseekCoder-7B | 38.67 | 20.41 | 19.15 | 10 | 34 | 16 | 10 | 28 |
| TableLLM-Llama3-8B | 35.14 | 18.37 | 12.77 | 6 | 22 | 14 | 6 | 20 |
| TableLLM-Llama3.1-8B | 36.73 | 16.33 | 10.64 | 6 | 22 | 20 | 6 | 18 |
| TableLLM-Qwen2-7B | 33.19 | 38.78 | 23.40 | 22 | 48 | 16 | 16 | 50 |
 
**Table 9** — Accuracy across different question categories present in TableBench dataset using **PIP** framework (Part 1).
 
| Model | NR-A | FC-MB | DA-I | DA-C | NR-MH | FC-MH | NR-Cn | DA-Ds |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1-Distill-Llama-8B | 56 | 76.09 | 10 | 38.88 | 47.06 | 44 | 38 | 26.34 |
| Llama-3.1-8B-Instruct | 6 | 58.7 | 2 | 39.71 | 3.92 | 38 | 4 | 23.61 |
| Llama-3.2-3B-Instruct | 0 | 15.22 | 2 | 37.28 | 0 | 10 | 2 | 17.55 |
| Meta-Llama-3-8B-Instruct | 12 | 80.43 | 6 | 42.37 | 7.84 | 46 | 12 | 28.2 |
| Qwen2-7B-Instruct | 10 | 82.61 | 18 | 29.01 | 15.69 | 58 | 20 | 21.15 |
| Qwen2.5-7B-Instruct | 10 | 65.22 | 26 | 31.99 | 11.76 | 38 | 10 | 26.17 |
| Qwen2.5-Coder-7B-Instruct | 8 | 89.13 | 38 | 41.18 | 15.69 | 48 | 20 | 26.62 |
| Qwen3-0.6B | 20 | 34.78 | 2 | 35.42 | 5.88 | 22 | 16 | 5.26 |
| Qwen3-1.7B | 56 | 60.87 | 4 | 34.27 | 35.29 | 42 | 54 | 0.9 |
| Qwen3-4B | 36 | 93.48 | 18 | 31.61 | 17.65 | 60 | 32 | 26.3 |
| Qwen3-8B | 30 | 95.65 | 32 | 32.44 | 33.33 | 70 | 44 | 25.26 |
| TableGPT2-7B | 14 | 73.91 | 20 | 35.43 | 15.69 | 34 | 36 | 20.41 |
| TableLLM-CodeQwen-7B | 4 | 86.96 | 32 | 44.94 | 1.96 | 52 | 12 | 24.52 |
| TableLLM-DeepseekCoder-7B | 4 | 78.26 | 26 | 45.5 | 3.92 | 46 | 18 | 25.87 |
| TableLLM-Llama3-8B | 4 | 73.91 | 34 | 47.23 | 0 | 50 | 6 | 26.23 |
| TableLLM-Llama3.1-8B | 4 | 84.78 | 24 | 46.71 | 3.92 | 48 | 12 | 24.87 |
| TableLLM-Qwen2-7B | 14 | 84.78 | 30 | 46.69 | 21.57 | 44 | 52 | 27.78 |
 
**Table 10** — Accuracy across different question categories present in TableBench dataset using **PIP** framework (Part 2).
 
| Model | DA-An | NR-Dm | NR-T | NR-Ar | NR-Rk | DA-Tr | DA-St | NR-Cp |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1-Distill-Llama-8B | 21.53 | 48.98 | 53.19 | 54 | 54 | 26 | 24 | 50 |
| Llama-3.1-8B-Instruct | 20.78 | 16.33 | 8.51 | 2 | 34 | 18 | 6 | 16 |
| Llama-3.2-3B-Instruct | 14.88 | 0 | 8.51 | 0 | 10 | 8 | 0 | 0 |
| Meta-Llama-3-8B-Instruct | 23.78 | 20.41 | 23.4 | 28 | 34 | 14 | 14 | 36 |
| Qwen2-7B-Instruct | 22.1 | 30.61 | 23.4 | 34 | 32 | 12 | 12 | 32 |
| Qwen2.5-7B-Instruct | 28.6 | 32.65 | 36.17 | 26 | 44 | 26 | 18 | 30 |
| Qwen2.5-Coder-7B-Instruct | 23.79 | 38.78 | 25.53 | 38 | 54 | 16 | 18 | 50 |
| Qwen3-0.6B | 20.61 | 4.08 | 19.15 | 28 | 8 | 10 | 8 | 22 |
| Qwen3-1.7B | 24.42 | 20.41 | 57.45 | 52 | 38 | 18 | 16 | 38 |
| Qwen3-4B | 34.55 | 55.1 | 36.17 | 44 | 72 | 14 | 14 | 44 |
| Qwen3-8B | 28.86 | 55.1 | 53.19 | 48 | 76 | 14 | 14 | 56 |
| TableGPT2-7B | 24.73 | 32.65 | 23.4 | 44 | 36 | 6 | 20 | 48 |
| TableLLM-CodeQwen-7B | 33.53 | 12.24 | 17.02 | 4 | 16 | 20 | 10 | 18 |
| TableLLM-DeepseekCoder-7B | 39.04 | 16.33 | 12.77 | 10 | 28 | 20 | 10 | 16 |
| TableLLM-Llama3-8B | 31.59 | 22.45 | 14.89 | 6 | 24 | 18 | 8 | 16 |
| TableLLM-Llama3.1-8B | 33.7 | 20.41 | 10.64 | 10 | 24 | 20 | 2 | 22 |
| TableLLM-Qwen2-7B | 33.49 | 38.78 | 38.3 | 30 | 60 | 30 | 20 | 54 |
 
---
 
## References
 
- Wu et al. — *TableBench: A comprehensive and complex benchmark for table question answering.* AAAI 2025.
- Nan et al. — *FeTaQA: Free-form table question answering.* TACL 2022.
- Wei et al. — *Chain-of-thought prompting elicits reasoning in large language models.* NeurIPS 2022.
- Xu et al. — *Faithful logical reasoning via symbolic chain-of-thought.* 2024.
- Yao et al. — *ReAct: Synergizing reasoning and acting in language models.* ICLR 2023.
- Yao et al. — *Tree of Thoughts: Deliberate problem solving with large language models.* 2023.
- Zhou et al. — *Least-to-most prompting enables complex reasoning in large language models.* 2022.
- Kojima et al. — *Large language models are zero-shot reasoners.* 2022.
- Wang et al. — *Self-consistency improves chain of thought reasoning in language models.* 2022.
- Guo et al. — *DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning.* 2025.
- Grattafiori et al. — *The Llama 3 herd of models.* 2024.
- Yang et al. — *Qwen2 Technical Report.* 2024. / *Qwen2.5 Technical Report.* 2025. / *Qwen3 Technical Report.* 2025.
- Su et al. — *TableGPT2: A large multimodal model with tabular data integration.* 2024.
- Zhang et al. — *TableLLM: Enabling tabular data manipulation by LLMs in real office usage scenarios.* Findings of ACL 2025.
- Zhang et al. — *A survey of table reasoning with large language models.* 2025.
- Fang et al. — *Large Language Models (LLMs) on tabular data: Prediction, generation, and understanding — a survey.* 2024.
- Deng et al. — *Tables as texts or images: Evaluating the table reasoning ability of LLMs and MLLMs.* 2024.
- Chen — *Large language models are few(1)-shot table reasoners.* Findings of EACL 2023.
- Lin — *Looking for a few good metrics: ROUGE and its evaluation.* NTCIR 2004.
- Post — *A call for clarity in reporting BLEU scores.* WMT 2018.
- Banerjee & Lavie — *METEOR: An automatic metric for MT evaluation with improved correlation with human judgments.* ACL Workshop 2005.
- Zhang et al. — *BERTScore: Evaluating text generation with BERT.* 2019.
- Sellam et al. — *BLEURT: Learning robust metrics for text generation.* ACL 2020.
---
 
## Citation
 
If you find this work useful, please cite:
 
```bibtex
@misc{maurya2026efficienttableqatablegrid,
      title={Efficient Table QA via TableGrid Navigation and Progressive Inference Prompting}, 
      author={Amritansh Maurya and Navjot Singh and Mohammed Javed and Omar Moured},
      year={2026},
      eprint={2605.20254},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2605.20254}, 
}
```
 
