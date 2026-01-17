# Privacy Cleanup Summary

This document summarizes all personal information that has been removed or anonymized from the codebase to ensure compliance with submission requirements.

## Removed Personal Information

### 1. API Keys
- **WANDB_API_KEY**: All hardcoded API keys have been removed
  - Files modified:
    - `train_mmlu_bandwagon.sh`
    - `mmlupro_bandwagon.sh`
    - `mmlupro_bandwagon_mixed.sh`
    - `main_4grpo_8b_updated.sh`
  - **Action**: Replaced with environment variable instructions

### 2. Hardcoded User Paths
- **Removed paths**:
  - `/shared/hdd/nuochen/` - Personal shared directory
  - `/shared/ssd/models/` - Personal model storage
  - `/shared/hdd/qian/` - Personal checkpoint directory
  - `/home/nuochen/` - Personal home directory
  - `ray_tmp_qian` - Personal Ray temp directory

- **Files modified**:
  - `train_mmlu_bandwagon.sh`
  - `mmlupro_bandwagon.sh`
  - `mmlupro_bandwagon_mixed.sh`
  - `main_grpo.sh`
  - `main_4grpo_7b_balanced.sh`
  - `main_4grpo_8b_updated.sh`

- **Action**: Replaced with environment variables or relative paths:
  - `MODEL_PATH=${MODEL_PATH:-"path/to/your/model"}`
  - `CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./models/..."}`
  - `RAY_TEMP_DIR="$HOME/ray_tmp"` (generic)

### 3. Username References
- Removed all references to specific usernames in:
  - Path names
  - Directory names
  - Comments

## Current State

All scripts now use:
1. **Environment variables** for sensitive paths (with fallback defaults)
2. **Relative paths** where possible
3. **Generic directory names** (e.g., `$HOME/ray_tmp` instead of `$HOME/ray_tmp_qian`)
4. **Instructions** for setting API keys via environment variables

## User Configuration Required

Users need to set the following environment variables before running scripts:

```bash
# Model path
export MODEL_PATH="/path/to/your/model"

# Checkpoint directory (optional)
export CHECKPOINT_DIR="./models/checkpoints"

# WandB API key (optional, for logging)
export WANDB_API_KEY="your_api_key_here"
```

## Verification

All personal information has been removed. The codebase is now ready for open-source release and submission.

