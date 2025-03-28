# fine_tuning/qlora_train.py
"""
QLoRA fine-tuning script for Mistral 7B on domain-specific credit assessment data.

Uses 4-bit quantization with rank-16 LoRA adapters. Designed for 2x A100 80GB —
if you're on a single A100, halve the batch size and double gradient accumulation.

Usage:
    python fine_tuning/qlora_train.py \
        --base-model mistralai/Mistral-7B-Instruct-v0.2 \
        --dataset training_data.jsonl \
        --output-dir ./models/mistral-7b-finetuned-credit \
        --lora-rank 16 \
        --lora-alpha 32 \
        --num-epochs 3 \
        --batch-size 4 \
        --gradient-accumulation 8
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

# Mistral instruct template — don't mess with this format or the model
# gets confused and starts generating garbage after the first turn
INSTRUCT_TEMPLATE = """<s>[INST] {instruction} [/INST] {response}</s>"""


def load_training_data(dataset_path: str) -> Dataset:
    """
    Load JSONL training data. Each line:
    {"instruction": "...", "response": "...", "context": "..."}
    """
    records = []
    with open(dataset_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            # format into instruct template
            text = INSTRUCT_TEMPLATE.format(
                instruction=rec['instruction'],
                response=rec['response'],
            )
            records.append({'text': text})

    logger.info(f'Loaded {len(records)} training examples from {dataset_path}')
    return Dataset.from_list(records)


def create_bnb_config() -> BitsAndBytesConfig:
    """4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # nested quantization — saves ~0.4GB
    )


def create_lora_config(rank: int, alpha: int) -> LoraConfig:
    """
    LoRA adapter config.

    Target modules are specific to Mistral architecture. If you're adapting
    this for Llama or another model, check the layer names first.
    """
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        # target all attention + MLP projection layers
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )


def train(args):
    logger.info(f'Starting QLoRA fine-tuning of {args.base_model}')
    logger.info(f'  LoRA rank={args.lora_rank}, alpha={args.lora_alpha}')
    logger.info(f'  Batch size={args.batch_size}, grad accum={args.gradient_accumulation}')
    logger.info(f'  Effective batch = {args.batch_size * args.gradient_accumulation}')

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # load model with 4-bit quantization
    bnb_config = create_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        # attn_implementation='flash_attention_2',  # uncomment if flash-attn installed
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # prep for kbit training + attach LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = create_lora_config(args.lora_rank, args.lora_alpha)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f'Trainable params: {trainable:,} / {total:,} '
        f'({100 * trainable / total:.2f}%)'
    )

    # load dataset
    dataset = load_training_data(args.dataset)

    # split 95/5 for eval — not a lot but we have RAGAS for the real eval
    split = dataset.train_test_split(test_size=0.05, seed=42)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        weight_decay=0.001,
        optim='paged_adamw_32bit',
        bf16=True,
        logging_steps=25,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        report_to='none',  # set to 'wandb' if you want experiment tracking
        max_grad_norm=0.3,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        args=training_args,
        max_seq_length=2048,  # credit docs are rarely longer than this
    )

    logger.info('Starting training...')
    trainer.train()

    # save the final adapter weights
    final_path = os.path.join(args.output_dir, 'final_adapter')
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f'Adapter saved to {final_path}')

    # also save merged model for vLLM serving
    # vLLM can load LoRA adapters directly, but a merged model is simpler
    logger.info('Merging adapter into base model...')
    merged = model.merge_and_unload()
    merged_path = os.path.join(args.output_dir, 'merged')
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    logger.info(f'Merged model saved to {merged_path}')
    logger.info('Done. Run evaluation before deploying to vLLM.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QLoRA fine-tuning for Mistral 7B')
    parser.add_argument('--base-model', default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--dataset', required=True, help='Path to JSONL training data')
    parser.add_argument('--output-dir', default='./models/mistral-7b-finetuned-credit')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    args = parser.parse_args()

    train(args)
