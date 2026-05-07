#!/usr/bin/env python3
"""
Script to update all SFT evaluation scripts:
1. Remove RL evaluation sections
2. Add OOD dataset evaluation
3. Update output directories
"""

import os
import re
from pathlib import Path

# Mapping of bias types to their datasets
BIAS_CONFIGS = {
    'wrong_bandwagon': {
        'validation': 'validation_paired_bandwagon_correct_random.parquet',
        'ood': 'test_paired_ood_mixed_bandwagon_random.parquet',
        'eval_script': 'eval_wrong_bandwagon_validation.py'
    },
    'correct_authority': {
        'validation': 'validation_paired_authority_correct_random.parquet',
        'ood': 'test_paired_ood_authority_correct_random.parquet',
        'eval_script': 'eval_correct_authority_validation.py'
    },
    'wrong_authority': {
        'validation': 'validation_paired_authority_incorrect_random.parquet',
        'ood': 'test_paired_ood_authority_incorrect_random.parquet',
        'eval_script': 'eval_wrong_authority_validation.py'
    },
    'correct_distraction': {
        'validation': 'validation_paired_distraction_correct_random.parquet',
        'ood': 'test_paired_ood_distraction_correct_random.parquet',
        'eval_script': 'eval_correct_distraction_validation.py'
    },
    'wrong_distraction': {
        'validation': 'validation_paired_distraction_incorrect_random.parquet',
        'ood': 'test_paired_ood_distraction_incorrect_random.parquet',
        'eval_script': 'eval_wrong_distraction_validation.py'
    },
    'wrong_verbosity': {
        'validation': 'validation_paired_verbosity_incorrect_random.parquet',
        'ood': 'test_paired_ood_verbosity_incorrect_random.parquet',
        'eval_script': 'eval_wrong_verbosity_validation.py'
    },
    'correct_verbosity': {
        'validation': 'validation_paired_verbosity_correct_random.parquet',
        'ood': 'test_paired_ood_verbosity_correct_random.parquet',
        'eval_script': 'eval_correct_verbosity_validation.py'
    }
}

def update_sft_script(filepath, bias_type, model_size):
    """Update a single SFT evaluation script"""
    
    if bias_type not in BIAS_CONFIGS:
        print(f"⚠️  Unknown bias type: {bias_type}")
        return False
    
    config = BIAS_CONFIGS[bias_type]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Update header comment
    content = re.sub(
        r'# Compares SFT models to RL models as baseline',
        '',
        content
    )
    
    # 2. Remove RL_RUN_DIR
    content = re.sub(
        r'# 2\. RL training run directory.*?\nRL_RUN_DIR=.*?\n\n',
        '',
        content,
        flags=re.DOTALL
    )
    
    # 3. Update dataset section
    validation_dataset = f"./data/data/mmlupro/{config['validation']}"
    ood_dataset = f"./data/data/mmlupro/{config['ood']}"
    
    # Replace single dataset with validation and OOD
    content = re.sub(
        r'# 4\. Dataset\nVALIDATION_DATASET="[^"]+"',
        f'# 3. Datasets\nVALIDATION_DATASET="{validation_dataset}"\nOOD_DATASET="{ood_dataset}"',
        content
    )
    
    # 4. Update output directories
    base_name = f"results_{bias_type}_validation_qwen3_{model_size}"
    ood_name = f"results_{bias_type}_ood_qwen3_{model_size}"
    
    # Replace output directories section
    content = re.sub(
        r'# 5\. Output directories\nBASELINE_OUTPUT_DIR="[^"]+"\nRL_OUTPUT_DIR="[^"]+"\nSFT_OUTPUT_DIR="[^"]+"',
        f'# 4. Output directories\nBASELINE_VAL_OUTPUT_DIR="./data/baseline/{base_name}_baseline"\nBASELINE_OOD_OUTPUT_DIR="./data/baseline/{ood_name}_baseline"\nSFT_VAL_OUTPUT_DIR="./data/baseline/{base_name}_sft"\nSFT_OOD_OUTPUT_DIR="./data/baseline/{ood_name}_sft"',
        content
    )
    
    # 5. Update step numbering (remove RL step, add OOD baseline step)
    content = re.sub(
        r'# ============================================================================\n# Step 2: Evaluate RL Models.*?# ============================================================================\n\n.*?# ============================================================================\n# Step 3: Evaluate SFT Models',
        '# ============================================================================\n# Step 2: Evaluate Baseline Model (OOD)\n# ============================================================================\n\nif [ -z "$SKIP_BASELINE" ]; then\n    cd "$SCRIPT_DIR"\n    \n    echo ""\n    echo "=========================================="\n    echo "Step 2: Evaluating Baseline Model (OOD - ' + bias_type.replace('_', ' ').title() + ')"\n    echo "=========================================="\n    echo "Model: $BASELINE_MODEL_PATH"\n    echo "Dataset: $OOD_DATASET"\n    echo ""\n    \n    mkdir -p "$BASELINE_OOD_OUTPUT_DIR"\n    \n    if [ -f "${BASELINE_OOD_OUTPUT_DIR}/summary.json" ] || [ -n "$(ls -A ${BASELINE_OOD_OUTPUT_DIR}/*.json 2>/dev/null)" ]; then\n        echo "✅ Baseline OOD evaluation results already exist. Skipping."\n    else\n        echo "Evaluating baseline model on OOD set..."\n        \n        CMD="python3 ' + config['eval_script'] + '"\n        CMD="$CMD --model_path $BASELINE_MODEL_PATH"\n        CMD="$CMD --dataset $OOD_DATASET"\n        CMD="$CMD --output_dir $BASELINE_OOD_OUTPUT_DIR"\n        CMD="$CMD --gpu_ids $GPU_IDS"\n        CMD="$CMD --tensor_parallel_size $TENSOR_PARALLEL_SIZE"\n        CMD="$CMD --temperature $TEMPERATURE"\n        CMD="$CMD --max_model_len $MAX_MODEL_LEN"\n        CMD="$CMD --gpu_memory_utilization $GPU_MEMORY_UTIL"\n        \n        echo "Command: $CMD"\n        eval $CMD\n        \n        if [ $? -ne 0 ]; then\n            echo "❌ Baseline OOD evaluation failed."\n        else\n            echo "✅ Baseline OOD evaluation completed!"\n        fi\n    fi\nfi\n\n# ============================================================================\n# Step 3: Evaluate SFT Models',
        content,
        flags=re.DOTALL
    )
    
    # 6. Update SFT evaluation to include both validation and OOD
    # This is complex, so we'll do it in parts
    
    # Update variable references in baseline evaluation
    content = re.sub(
        r'\$BASELINE_OUTPUT_DIR',
        '$BASELINE_VAL_OUTPUT_DIR',
        content
    )
    
    # Update SFT evaluation section to add OOD
    # Find the SFT evaluation loop and add OOD evaluation after validation
    sft_eval_pattern = r'(# Evaluate on validation set\n.*?STEP_OUTPUT_DIR="\$\{SFT_OUTPUT_DIR\}/.*?\n.*?done)'
    
    # Replace SFT_OUTPUT_DIR with SFT_VAL_OUTPUT_DIR in validation section
    content = re.sub(
        r'STEP_OUTPUT_DIR="\$\{SFT_OUTPUT_DIR\}',
        'VAL_STEP_OUTPUT_DIR="${SFT_VAL_OUTPUT_DIR}',
        content
    )
    
    # Add OOD evaluation after validation evaluation
    # This is tricky, let's do a more targeted replacement
    old_sft_section = r'(if \[ -f "\$\{STEP_OUTPUT_DIR\}/summary\.json" \] \|\| \[ -n "\$\(ls -A \$\{STEP_OUTPUT_DIR\}/\*\.json 2>/dev/null\)" \]; then.*?fi\n)done'
    
    # Actually, let's use a simpler approach - find the end of SFT validation and add OOD
    content = re.sub(
        r'(echo "✅ SFT evaluation completed for \$STEP_NAME"\n        fi\n    fi\ndone)',
        r'''echo "✅ SFT validation evaluation completed for $STEP_NAME"
        fi
    fi
    
    # Evaluate on OOD set
    OOD_STEP_OUTPUT_DIR="${SFT_OOD_OUTPUT_DIR}/${STEP_NAME}"
    mkdir -p "$OOD_STEP_OUTPUT_DIR"
    
    if [ -f "${OOD_STEP_OUTPUT_DIR}/summary.json" ] || [ -n "$(ls -A ${OOD_STEP_OUTPUT_DIR}/*.json 2>/dev/null)" ]; then
        echo "✅ SFT OOD results already exist for $STEP_NAME. Skipping."
    else
        echo "Evaluating SFT $STEP_NAME on OOD set..."
        
        cd "$SCRIPT_DIR"
        
        CMD="python3 ''' + config['eval_script'] + r'''"
        CMD="$CMD --model_path $HF_MODEL_DIR"
        CMD="$CMD --dataset $OOD_DATASET"
        CMD="$CMD --output_dir $OOD_STEP_OUTPUT_DIR"
        CMD="$CMD --gpu_ids $GPU_IDS"
        CMD="$CMD --tensor_parallel_size $TENSOR_PARALLEL_SIZE"
        CMD="$CMD --temperature $TEMPERATURE"
        CMD="$CMD --max_model_len $MAX_MODEL_LEN"
        CMD="$CMD --gpu_memory_utilization $GPU_MEMORY_UTIL"
        
        echo "Command: $CMD"
        eval $CMD
        
        if [ $? -ne 0 ]; then
            echo "❌ SFT OOD evaluation failed for $STEP_NAME."
        else
            echo "✅ SFT OOD evaluation completed for $STEP_NAME"
        fi
    fi
done''',
        content
    )
    
    # 7. Update final summary
    content = re.sub(
        r'echo "Results saved to:"\n.*?echo "  RL:.*?\n.*?echo "  SFT:.*?\n.*?echo "✅ All evaluations completed! Compare SFT vs RL results above\."',
        '''echo "Results saved to:"
echo "  Baseline Validation: $BASELINE_VAL_OUTPUT_DIR"
echo "  Baseline OOD: $BASELINE_OOD_OUTPUT_DIR"
echo "  SFT Validation: $SFT_VAL_OUTPUT_DIR"
echo "  SFT OOD: $SFT_OOD_OUTPUT_DIR"
echo ""
echo "✅ All evaluations completed!"''',
        content
    )
    
    # 8. Update eval script name in SFT validation section
    eval_script = config["eval_script"]
    content = re.sub(
        r'CMD="python3 eval_[^"]+_validation\.py"',
        f'CMD="python3 {eval_script}"',
        content,
        count=1  # Only replace the first occurrence (baseline)
    )
    
    # Write updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    
    # Find all SFT evaluation scripts
    scripts = list(script_dir.glob('eval_sft_*_qwen3_*.sh'))
    
    for script in scripts:
        # Parse bias type and model size from filename
        # Format: eval_sft_{bias_type}_qwen3_{size}.sh
        match = re.match(r'eval_sft_([^_]+(?:_[^_]+)*)_qwen3_([^\.]+)\.sh', script.name)
        if match:
            bias_type = match.group(1)
            model_size = match.group(2)
            
            print(f"Updating {script.name} (bias: {bias_type}, model: {model_size})...")
            if update_sft_script(script, bias_type, model_size):
                print(f"✅ Updated {script.name}")
            else:
                print(f"❌ Failed to update {script.name}")
        else:
            print(f"⚠️  Could not parse filename: {script.name}")

