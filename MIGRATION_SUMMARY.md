# Migration Summary

This document summarizes the files migrated from `Logic-RL-qwen3-local-env` to `bias-mitigation-with-rl` for the open-source release.

## Files and Directories Migrated

### Core Framework
- ✅ `verl/` - Complete verl framework (excluding `verl_old/`)
  - All reward scoring functions including bias-specific rewards
  - Training infrastructure
  - Worker implementations
  - Model support

### Examples and Scripts
- ✅ `examples/` - All example scripts
  - Data preprocessing scripts for all bias types
  - Training examples (PPO, GRPO, SFT)
  - Generation scripts

### Evaluation Scripts
- ✅ `bandwagon_scripts/` - Bandwagon bias evaluation
- ✅ `authority_scripts/` - Authority bias evaluation
- ✅ `position_scripts/` - Position bias evaluation
- ✅ `distraction_scripts/` - Distraction bias evaluation
- ✅ `sft/` - Supervised fine-tuning scripts

### Configuration Files
- ✅ `setup.py`
- ✅ `requirements.txt`
- ✅ `pyproject.toml`
- ✅ `LICENSE`
- ✅ `Notice.txt`
- ✅ `.gitignore`
- ✅ `common.py`

### Training Scripts
- ✅ `main_*.sh` - Main training scripts
- ✅ `train_*.sh` - Training scripts
- ✅ `mmlupro_*.sh` - MMLU-Pro specific scripts

### Data
- ✅ `data/` - Preprocessed datasets (parquet format)
  - Excluded large JSON files (corpus.json, testset*.json, etc.)

### Documentation
- ✅ `docs/` - Complete documentation
- ✅ `README.md` - Updated for bias mitigation
- ✅ `pics/` - Images and figures

### Utilities
- ✅ `tests/` - Test suite
- ✅ `eval_kk/` - K&K evaluation scripts
- ✅ `math_eval/` - Math evaluation scripts
- ✅ `prompt_mitigation/` - Prompt-based mitigation scripts
- ✅ `patches/` - Code patches
- ✅ `docker/` - Docker configurations

## Files Excluded

The following files/directories were **NOT** migrated as they are not suitable for open-source release:

- ❌ `verl_old/` - Old version of verl (backup)
- ❌ `outputs/` - Training outputs and checkpoints
- ❌ `wandb/` - Weights & Biases logs
- ❌ `evaluation_results/` - Evaluation result files
- ❌ `baseline/` - Baseline result files
- ❌ `*.log` - All log files
- ❌ Large JSON files:
  - `corpus.json`
  - `testset*.json`
  - `MultiHopRAG.json`
  - `pandalm_typecase.json`
- ❌ `local_model_*.py` - Experiment-specific model files
- ❌ `__pycache__/` - Python cache directories
- ❌ `*.pyc` - Compiled Python files

## Statistics

- **Total files migrated**: ~650 files
- **Python files**: ~401 files
- **Total size**: ~452 MB
- **Directories**: 40+ directories

## Key Components for Bias Mitigation ⭐

**⭐ = Bias mitigation specific contributions (not from original Logic-RL)**

### Reward Functions (Bias Mitigation Contributions)
- `verl/utils/reward_score/mmlu_bandwagon_bias.py` - Bandwagon bias reward ⭐
- `verl/utils/reward_score/gsm8k_authority_bias.py` - Authority bias reward ⭐
- `verl/utils/reward_score/mmlupro.py` - MMLU-Pro rewards (may include bias-specific modifications) ⭐
- `verl/utils/reward_score/mmlupro_accuracy_independence.py` - Accuracy independence ⭐

### Data Preprocessing (Bias Mitigation Contributions)
- `examples/data_preprocess/mmlupro_pair_bandwagon*.py` - Bandwagon data ⭐
- `examples/data_preprocess/anthority_bias/` - Authority bias data ⭐
- `examples/data_preprocess/distraction_bias/` - Distraction bias data ⭐
- `examples/data_preprocess/mmlupro.py` - Position bias data (bias-specific modifications) ⭐

### Training Scripts
**Bias Mitigation Contributions:**
- `train_mmlu_bandwagon.sh` - Bandwagon bias training ⭐
- `mmlupro_bandwagon.sh` - MMLU-Pro bandwagon training ⭐
- `mmlupro_bandwagon_mixed.sh` - Mixed bandwagon training ⭐
- `sft/train_*.sh` - SFT scripts for bias mitigation ⭐

**From Original Logic-RL (for reference only):**
- `main_grpo.sh` - GRPO training for K&K puzzles (math reasoning)
- `main_4grpo_7b_*.sh` - GRPO configurations for math reasoning
- `examples/grpo_trainer/` - GRPO training examples for math
- `math_eval/` - Math evaluation scripts

### Evaluation Scripts (Bias Mitigation Contributions)
All evaluation scripts are organized by bias type in their respective directories:
- `bandwagon_scripts/eval_*.py` - Bandwagon bias evaluation ⭐
- `authority_scripts/eval_*.py` - Authority bias evaluation ⭐
- `position_scripts/eval_*.py` - Position bias evaluation ⭐
- `distraction_scripts/eval_*.py` - Distraction bias evaluation ⭐
- `*_scripts/eval_*.sh` - Evaluation shell scripts ⭐

## Attribution

### Bias Mitigation Contributions (This Work)
- All bias-specific reward functions
- All bias dataset generation scripts
- All bias evaluation scripts (`*_scripts/`)
- Bias mitigation training scripts (`train_mmlu_bandwagon.sh`, `mmlupro_bandwagon*.sh`)
- SFT scripts for bias mitigation (`sft/`)
- Bias-specific data preprocessing scripts

### From Logic-RL (Included for Reference)
- `main_grpo.sh` and `main_4grpo_7b_*.sh` - Original GRPO training scripts for math reasoning
- `examples/grpo_trainer/` - GRPO training examples
- `math_eval/` - Math evaluation scripts
- `eval_kk/` - K&K puzzle evaluation scripts
- Core verl framework structure

### From Verl Framework
- Core RL training infrastructure
- Worker implementations
- Model support

## Next Steps

1. Review the migrated code for any hardcoded paths or credentials
2. Update any model paths or data paths in scripts
3. Test the installation and basic functionality
4. Update documentation with specific examples
5. Add any missing dependencies to requirements.txt
6. Consider removing or clearly marking Logic-RL math/grpo scripts if not needed for bias mitigation

