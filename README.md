# CAS NLP Module 4 - Finetuning LLMs for fairy telling

This project investigates the effect of fine-tunining LLMs on fairytale text to generate new fairytale stories.

## Project Overview

The project uses the Hugging Face Transformers library to:

- Load and fine-tune GPT-2 models on fairytale datasets
- Generate new fairytale text based on prompts
- Compare baseline GPT-2 performance with fine-tuned models

## Dataset

The project uses the `vicclab/fairy_tales` dataset from Hugging Face.

## Implementation

GPT-2-small and GPT-2-large are finetuned using the `vicclab/fairy_tales` dataset (`training_gpt-2_small/large.ipynb`). Checkpoints are saved locally and can be provided upon request.
Trained models are used to generate text which is analyzed with different procedures (`story_telling_gpt-2_small/large.ipynb`). Results are finally visualized (`data_visualization.ipynb`).

## Authors

Ana & Michael - CAS NLP Module 4 Project
