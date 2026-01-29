"""
LoRA Fine-tuning Script for Apertus-8B on Fairy Tales
Using PEFT (Parameter-Efficient Fine-Tuning)

This script is identical to the full fine-tuning approach but uses LoRA adapters
for efficient training with reduced memory and trainable parameters.
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os


def main(args):
    print(f"=" * 80)
    print(f"Starting LoRA Training - {args.num_epochs} Epoch(s)")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resume from: {args.resume_from if args.resume_from else 'Base model'}")
    print(f"=" * 80)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Load model
    print(f"Loading model from: {args.resume_from if args.resume_from else args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.resume_from if args.resume_from else args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply LoRA configuration (only if starting from base model)
    if not args.resume_from:
        print("\nApplying LoRA configuration...")
        lora_config = LoraConfig(
            r=16,                          # LoRA rank
            lora_alpha=32,                 # LoRA alpha (scaling factor)
            target_modules=[               # Apply LoRA to attention and MLP layers
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Enable gradient checkpointing for LoRA
        model.enable_input_require_grads()
    else:
        print(f"\nResuming LoRA training from checkpoint: {args.resume_from}")
        # Model already has LoRA adapters from checkpoint
        model.enable_input_require_grads()
    
    # Load and prepare dataset
    print(f"\nLoading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    
    # Split into train and validation
    if "train" not in dataset:
        # If no predefined split, create one
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset["train"]
        # Create validation split if not exists
        if "validation" in dataset:
            eval_dataset = dataset["validation"]
        else:
            split = dataset["train"].train_test_split(test_size=0.2, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,  # Same as full fine-tuning
            padding=False,
        )
    
    print("\nTokenizing dataset...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments - OPTIMIZED for LoRA efficiency
    # LoRA trains only 0.49% of parameters, so we can use larger batch sizes
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,      
        per_device_train_batch_size=8,         # 8x larger than full fine-tuning
        per_device_eval_batch_size=8,          # Larger eval batch for speed
        gradient_accumulation_steps=1,         # Effective batch size = 8
        learning_rate=5e-5,                    
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",                 # Only evaluate at epoch end
        save_strategy="epoch",
        save_total_limit=None,                 # Keep all epoch checkpoints
        bf16=True,
        gradient_checkpointing=True,           # Still useful for LoRA
        dataloader_num_workers=4,              # More workers for faster data loading
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 80)
    print(f"Starting training for {args.num_epochs} epoch(s)")
    print("=" * 80)
    print()
    
    trainer.train()
    
    # Save final checkpoint
    print(f"\nSaving final LoRA checkpoint to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 80)
    print(f"Training completed! {args.num_epochs} epoch(s) finished")
    print(f"Final checkpoint saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tune Apertus-8B on Fairy Tales")
    parser.add_argument("--model_name", type=str, default="swiss-ai/Apertus-8B-2509",
                        help="Base model name or path")
    parser.add_argument("--dataset_name", type=str, default="vicclab/fairy_tales",
                        help="Dataset name from HuggingFace")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the LoRA checkpoint")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of epochs to train")
    
    args = parser.parse_args()
    main(args)
