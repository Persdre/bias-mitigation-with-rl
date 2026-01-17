# SFT Baseline for Bandwagon Bias

This directory contains scripts for preparing and training SFT (Supervised Fine-Tuning) models on bandwagon bias data.

## Overview

The SFT baseline teaches models to:
- **Resist incorrect bandwagon bias**: When bandwagon points to wrong answer, model should resist and give correct answer
- **Follow correct bandwagon bias**: When bandwagon points to correct answer, model should follow (optional)

## Data Preparation

### Step 1: Prepare SFT Training Data

Run the data preparation script to convert bandwagon bias parquet files into SFT format:

```bash
cd /home/qian/Logic-RL-qwen3-local-env/sft
./prepare_all_sft_data.sh
```

This will generate:
- `data/sft_train_incorrect_bandwagon.parquet` - Training data with incorrect bandwagon (teaches resistance)
- `data/sft_train_correct_bandwagon.parquet` - Training data with correct bandwagon (teaches following)
- `data/sft_train_mixed_bandwagon.parquet` - Training data with both types
- `data/sft_val_all.parquet` - Validation data for evaluation

### Step 2: Verify Data Format

The generated parquet files contain:
- `prompt`: User question with options and bandwagon statement
- `response`: Model response with reasoning (`<think>...</think>`) and answer (`<answer>A</answer>`)
- Additional metadata columns (question_id, bias_type, etc.)

## Training

### Qwen3-1.7B (Single GPU)

```bash
cd /home/qian/Logic-RL-qwen3-local-env/sft
./train_sft_bandwagon_qwen3_1.7b.sh
```

### Qwen3-4B (Two GPUs)

```bash
cd /home/qian/Logic-RL-qwen3-local-env/sft
./train_sft_bandwagon_qwen3_4b.sh
```

### Training Configuration

Key hyperparameters:
- **Learning rate**: 2e-5
- **Batch size**: 64 (1.7B) / 128 (4B)
- **Micro batch size per GPU**: 4
- **Epochs**: 3
- **Max length**: 1024
- **Optimizer**: AdamW with cosine learning rate schedule
- **Precision**: bf16

Checkpoints are saved to:
- Qwen3-1.7B: `/ssd2/qian/models/SFT_bandwagon/Qwen3-1.7B/run_<timestamp>`
- Qwen3-4B: `/ssd2/qian/models/SFT_bandwagon/Qwen3-4B/run_<timestamp>`

## Evaluation

After training, evaluate the SFT model using the existing evaluation scripts:

```bash
# For bandwagon bias evaluation
cd /home/qian/Logic-RL-qwen3-local-env/bandwagon_scripts
# Update the model path in eval scripts to point to your SFT checkpoint
./eval_qwen3_1-7b_validation.sh
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

