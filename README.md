# CAS NLP Module 4 - Finetuning LLMs for fairy telling

This project investigates the effect of fine-tunining large language models (LLMs) on a domain-specific corpus, such as fairytale text. The performance is assessed for the task of generating text that is perceived as fairy-tale–like.

## Project Overview

The project uses the Hugging Face Transformers library to:

- Load and fine-tune GPT-2 models on fairytale datasets
- Generate new fairytale text based on prompts
- Compare baseline GPT-2 performance with fine-tuned models

## Problem

The objective of this project is to study the effects of fine-tuning LLMs in order to understand how fine-tuning influences model performance on fairy-tale text generation. Models of different sizes were fine-tuned and their performance was evaluated using a combination of quantitative metrics and qualitative human assessment.

## Dataset

The project uses the [`vicclab/fairy_tales`](https://huggingface.co/datasets/vicclab/fairy_tales) dataset from Hugging Face.
The dataset contains a concatenated and edited collection of fairy tales taken from the Project Gutenberg. The collection is organized in ca. 100'000 rows containing sentences with a lenght between 0 and 150 characters (including white spaces).

## Transformers vs NNs

Fairy-tale text generation requires modeling long-range dependencies, narrative structure, and thematic consistency across extended sequences of text. Traditional neural network architectures such as recurrent neural networks (RNNs), while capable of sequence modeling, struggle to capture long-term dependencies effectively and are limited by their sequential processing nature. Transformer-based models address these limitations through self-attention mechanisms, which allow direct modeling of relationships between distant tokens in a sequence. This makes transformers particularly well suited for narrative generation tasks compared to earlier neural approaches.

## Implementation

GPT-2-small, medium, large and GPT-2-xl are finetuned using the `vicclab/fairy_tales` dataset (`training_gpt-2_small//medium/large/xl.ipynb`). Checkpoints are saved locally and can be provided upon request.
Trained models are used to generate text which is analyzed with different procedures (`story_telling_gpt-2_small/medium/large/xl.ipynb`). Results are finally visualized (`presentation_mh.ipynb`).

## Results and evaluations

Model performance was evaluated using a combination of quantitative diversity metrics and qualitative human assessment. Self-BLEU and lexical diversity (Distinct-n) analyses indicate that fine-tuning increased stylistic consistency in the generated fairy-tale texts, particularly for the smaller GPT-2 model, but at the cost of reduced diversity. Zero-shot classification using a BART-based NLI model further confirmed that fine-tuning successfully shifted the generated text toward a fairy-tale narrative style. However, qualitative evaluation revealed that fine-tuning also led to a reduction in global coherence, with weaker sentence-to-sentence continuity and occasional grammatical degradation, especially in shorter generations. These results highlight a trade-off between stylistic specialization and narrative coherence introduced by domain-specific fine-tuning.

## Discussion

The results show that fine-tuning pretrained transformer-based language models on a domain-specific corpus can successfully steer text generation toward a fairy-tale narrative style. However, this adaptation is accompanied by reduced global coherence and grammatical consistency, particularly in smaller models. This trade-off suggests that continued pretraining on a narrow dataset may cause the model to emphasize local stylistic patterns at the expense of the narrative structure. The observed divergence between automatic evaluation metrics and qualitative human judgments highlights the need for complementary evaluation strategies. Future work could explore parameter-efficient fine-tuning methods, such as Low-Rank Adaptation (LoRA), to better preserve general language modeling capabilities while enabling stylistic adaptation.

## Limitations of Approach

The size of the fine-tuning dataset (approximately one million tokens) may be insufficient to preserve global coherence when adapting large language models to a narrow domain. Comparisons with alternative neural architectures were beyond the scope of this project, but represent an interesting direction for future work. Furthermore, fine-tuning was conducted using a single training objective and dataset, without systematic exploration of alternative training strategies or regularization techniques. Finally, the evaluation relied in part on qualitative human judgment, which, while appropriate for assessing narrative quality, inherently introduces a degree of subjectivity.

## Authors

Ana & Michael - CAS NLP Module 4 Project
