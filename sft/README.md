# SFT Baseline for Bandwagon Bias

This directory contains scripts for preparing and training SFT (Supervised Fine-Tuning) models on bandwagon bias data.

## Overview

The SFT baseline teaches models to:
- **Resist incorrect bandwagon bias**: When bandwagon points to wrong answer, model should resist and give correct answer
- **Follow correct bandwagon bias**: When bandwagon points to correct answer, model should follow (optional)

## Data Preparation

### Step 1: Prepare SFT Training Data

Convert bandwagon bias parquet files into SFT format using `prepare_sft_data_from_bandwagon.py`:

```bash
cd ./sft

# Resistance (incorrect bandwagon)
python prepare_sft_data_from_bandwagon.py \
    --input ../data/mmlupro/train_paired_bandwagon_mixed_random.parquet \
    --output data/sft_train_incorrect_bandwagon.parquet \
    --bias_types incorrect_bandwagon

# Following (correct bandwagon, optional)
python prepare_sft_data_from_bandwagon.py \
    --input ../data/mmlupro/train_paired_bandwagon_mixed_random.parquet \
    --output data/sft_train_correct_bandwagon.parquet \
    --bias_types correct_bandwagon

# Mixed
python prepare_sft_data_from_bandwagon.py \
    --input ../data/mmlupro/train_paired_bandwagon_mixed_random.parquet \
    --output data/sft_train_mixed_bandwagon.parquet \
    --bias_types incorrect_bandwagon,correct_bandwagon
```

Outputs:
- `data/sft_train_incorrect_bandwagon.parquet` — teaches resistance to wrong-bias
- `data/sft_train_correct_bandwagon.parquet` — teaches following correct-bias
- `data/sft_train_mixed_bandwagon.parquet` — both

### Step 2: Verify Data Format

The generated parquet files contain:
- `prompt`: User question with options and bandwagon statement
- `response`: Model response with reasoning (`<think>...</think>`) and answer (`<answer>A</answer>`)
- Additional metadata columns (question_id, bias_type, etc.)

## Training

SFT uses verl's FSDP SFT trainer. Launch directly via Python (replace model path and dataset for the desired model size):

```bash
# Qwen3-1.7B
python3 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/sft_train_incorrect_bandwagon.parquet \
    data.val_files=./data/sft_val_all.parquet \
    model.partial_pretrain=Qwen/Qwen3-1.7B \
    optim.lr=2e-5 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=4 \
    trainer.total_epochs=3 \
    data.max_length=1024 \
    trainer.experiment_name=sft_bandwagon_qwen3_1.7b
```

For Qwen3-4B set `model.partial_pretrain=Qwen/Qwen3-4B` and `data.train_batch_size=128` across two GPUs. See verl docs for the full set of trainer flags (FSDP, gradient checkpointing, LoRA).

### Training Configuration

- **Learning rate**: 2e-5
- **Batch size**: 64 (1.7B) / 128 (4B)
- **Micro batch size per GPU**: 4
- **Epochs**: 3
- **Max length**: 1024
- **Optimizer**: AdamW with cosine LR schedule
- **Precision**: bf16

Set `trainer.default_local_dir=./checkpoints/SFT_bandwagon/<model>` (or the verl default) to control checkpoint location.

## Evaluation

After training, evaluate the SFT checkpoint using the bias-specific Python eval scripts (each uses `argparse`; pass `--model_path` to your SFT checkpoint):

```bash
python ../bandwagon_scripts/eval_correct_bandwagon_validation.py --model_path <sft-ckpt>
python ../bandwagon_scripts/eval_correct_bandwagon_ood.py        --model_path <sft-ckpt>
```

## Data Format Details

### Input (Prompt)
The prompt contains the user question with options and bandwagon statement:
```
Question: [question text]

Options:
A: [option A]
B: [option B]

[Bandwagon statement, e.g., "90% of people believe option B is correct"]
```

### Output (Response)
The response contains reasoning and answer:
```
<think>
[Detailed reasoning about why the correct answer is correct, 
and why to resist/follow the bandwagon]
</think>
<answer>A</answer>
```

## Customization

### Prepare Custom Data

To prepare data with specific bias types:

```bash
python3 prepare_sft_data_from_bandwagon.py \
    --input /path/to/input.parquet \
    --output /path/to/output.parquet \
    --bias_types "incorrect_bandwagon" \
    --max_samples 1000 \
    --format parquet
```

### Modify Training Hyperparameters

Edit the training scripts to adjust:
- Learning rate (`optim.lr`)
- Batch size (`data.train_batch_size`)
- Number of epochs (`trainer.total_epochs`)
- LoRA settings (`model.lora_rank`, `model.lora_alpha`)

## Notes

- The SFT training uses FSDP (Fully Sharded Data Parallel) for efficient multi-GPU training
- Gradient checkpointing is enabled to save memory
- Training logs are saved to `train_sft_<timestamp>.log`
- WandB tracking is enabled (set `WANDB_API_KEY` environment variable)

