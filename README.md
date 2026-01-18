# Bias Mitigation with Reinforcement Learning

This repository contains the open-source implementation for bias mitigation in large language models using reinforcement learning. 

## Overview

This project implements reinforcement learning-based training to mitigate cognitive biases in LLM reasoning. The framework supports training on multiple bias types and evaluation across in-domain and out-of-domain scenarios.

## Key Features

- **Multiple Bias Types**: Support for bandwagon, authority, position, and distraction biases
- **RL Training Framework**: Built on verl (Volcano Engine Reinforcement Learning)
- **Comprehensive Evaluation**: Scripts for validation and out-of-domain evaluation
- **Data Processing**: Tools for generating bias datasets and preprocessing

## Installation

```bash
conda create -n bias-mitigation python=3.9
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 ray
pip3 install flash-attn --no-build-isolation
pip install -e .  # For verl integration
pip install wandb IPython matplotlib
```

## Project Structure

```
bias-mitigation-with-rl/
├── verl/                    # Core RL framework (from verl)
│   ├── utils/
│   │   └── reward_score/   # Reward functions including bias mitigation rewards
│   ├── trainer/            # Training scripts
│   └── workers/            # Worker implementations
├── examples/               # Example scripts and data preprocessing
│   └── data_preprocess/   # Data generation and preprocessing for bias mitigation
├── bandwagon_scripts/      # Bandwagon bias evaluation scripts ⭐
├── authority_scripts/      # Authority bias evaluation scripts ⭐
├── position_scripts/       # Position bias evaluation scripts ⭐
├── distraction_scripts/    # Distraction bias evaluation scripts ⭐
├── sft/                    # Supervised fine-tuning scripts for bias mitigation ⭐
├── data/                   # Bias mitigation datasets (parquet format) ⭐
├── train_mmlu_bandwagon.sh # Bandwagon bias training script ⭐
├── mmlupro_bandwagon*.sh  # MMLU-Pro bandwagon training scripts ⭐
├── tests/                  # Test suite
└── docs/                   # Documentation

⭐ = Bias mitigation specific contributions (not from original Logic-RL)
```

## Data Preparation

The repository includes preprocessed datasets in the `data/` directory. For generating your own bias datasets:

### Bandwagon Bias
```bash
python ./examples/data_preprocess/mmlupro_pair_bandwagon_mixed_random.py
```

### Authority Bias
```bash
python ./examples/data_preprocess/anthority_bias/mmlupro_pair_authority_mixed_random.py
```

### Position Bias
```bash
python ./examples/data_preprocess/mmlupro.py --position_bias
```

### Distraction Bias
```bash
python ./examples/data_preprocess/distraction_bias/mmlupro_pair_distraction_mixed_random.py
```

## Training

### Bias Mitigation Training Scripts
- `train_mmlu_bandwagon.sh` - Bandwagon bias training on MMLU dataset
- `mmlupro_bandwagon.sh` - Bandwagon bias training on MMLU-Pro dataset
- `mmlupro_bandwagon_mixed.sh` - Mixed bandwagon bias training on MMLU-Pro

**Note**: The `main_grpo.sh` and `main_4grpo_7b_*.sh` scripts are from the original Logic-RL framework for math reasoning tasks (K&K puzzles), not part of the bias mitigation contribution.

### SFT Training
```bash
cd sft
bash train_sft_bandwagon_qwen3_1.7b.sh
```

## Evaluation

### Bandwagon Bias Evaluation
```bash
cd bandwagon_scripts
bash eval_qwen3_1.7b_correct_bandwagon_validation.sh
bash eval_qwen3_1.7b_correct_bandwagon_ood.sh
```

### Authority Bias Evaluation
```bash
cd authority_scripts
bash eval_qwen3_1.7b_correct_authority_validation.sh
bash eval_qwen3_1.7b_correct_authority_ood.sh
```

### Position Bias Evaluation
```bash
cd position_scripts
bash eval_qwen3_1.7b_position_validation.sh
bash eval_qwen3_1.7b_position_ood.sh
```

### Distraction Bias Evaluation
```bash
cd distraction_scripts
bash eval_qwen3_1.7b_distraction_validation.sh
bash eval_qwen3_1.7b_distraction_ood.sh
```

## Reward Functions

The reward functions for bias mitigation are located in:
- `verl/utils/reward_score/mmlu_bandwagon_bias.py` - Bandwagon bias reward
- `verl/utils/reward_score/gsm8k_authority_bias.py` - Authority bias reward
- `verl/utils/reward_score/mmlupro.py` - MMLU-Pro reward functions
- `verl/utils/reward_score/mmlupro_accuracy_independence.py` - Accuracy independence metrics

## Implementation Details

| Component | Location |
|-----------|----------|
| Reward Modeling | `verl/utils/reward_score/` |
| Data Preprocessing | `examples/data_preprocess/` |
| Training Scripts | `main_*.sh`, `train_*.sh` |
| Evaluation Scripts | `*_scripts/eval_*.py` |


## Acknowledgements

This work builds upon:
- [Verl](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning framework
- [Logic-RL](https://github.com/Unakar/Logic-RL) - Rule-based reinforcement learning for LLM reasoning

**Note**: This repository includes the verl framework and some Logic-RL training scripts for reference. The bias mitigation contributions include:
- Bias-specific reward functions
- Bias dataset generation and preprocessing scripts
- Bias evaluation scripts for all bias types
- Bias mitigation training scripts

## License

Apache License 2.0
