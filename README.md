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

GPT-2-small and GPT-2-large are finetuned using the `vicclab/fairy_tales` dataset (`training_gpt-2_small/large.ipynb`). Checkpoints are saved locally and can be provided upon request.
Trained models are used to generate text which is analyzed with different procedures (`story_telling_gpt-2_small/large.ipynb`). Results are finally visualized (`data_visualization.ipynb`).

## Results and evaluations





## Authors

Ana & Michael - CAS NLP Module 4 Project
