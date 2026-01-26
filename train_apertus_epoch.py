"""
Training script for Apertus 8B on fairy tales dataset
Supports training from scratch or resuming from checkpoint
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
import torch
import os

def main(args):
    print(f"=" * 80)
    print(f"Starting Epoch {args.epoch_num}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir if args.checkpoint_dir else 'None (training from scratch)'}")
    print(f"=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Always load base model first - trainer.train() will handle checkpoint loading
    print(f"Loading base model: {args.model_name}")
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        print(f"(Will resume from checkpoint: {args.checkpoint_dir})")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Model loaded successfully. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    
    # Load and prepare dataset
    print("Loading dataset: vicclab/fairy_tales")
    dataset = load_dataset('vicclab/fairy_tales')
    train_val = dataset["train"].train_test_split(test_size=0.2, seed=42)
    dataset = DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"]
    })
    print(f"Train size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256
        )
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Filter empty sequences
    tokenized_datasets = tokenized_datasets.filter(
        lambda x: len(x["input_ids"]) > 0
    )
    print(f"After filtering: Train size: {len(tokenized_datasets['train'])}, Validation size: {len(tokenized_datasets['validation'])}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=None,  # Keep all checkpoints for evaluation
        report_to="none",
        bf16=True,  # Use bfloat16 for H100
        gradient_checkpointing=True,  # Save memory
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )
    
    # Train
    print(f"\nStarting training for epoch {args.epoch_num}...")
    trainer.train(resume_from_checkpoint=args.checkpoint_dir if args.checkpoint_dir and os.path.exists(args.checkpoint_dir) else None)
    
    # Save model with epoch-based naming
    epoch_checkpoint_dir = f"{args.output_dir}/checkpoint-epoch{args.epoch_num}"
    trainer.save_model(epoch_checkpoint_dir)
    trainer.state.save_to_json(os.path.join(epoch_checkpoint_dir, "trainer_state.json"))
    
    print(f"\n{'=' * 80}")
    print(f"Epoch {args.epoch_num} completed successfully!")
    print(f"Model saved to: {epoch_checkpoint_dir}")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Apertus 8B on fairy tales")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Checkpoint to resume from (optional)")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--epoch_num", type=int, required=True, help="Current epoch number (for logging)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    main(args)
