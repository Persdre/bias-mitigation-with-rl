#!/usr/bin/env python3
"""Extract verbosity val results across 3 seeds (42, 0, 1) and report per-seed + mean."""
import json, glob, os, statistics

P = '/ssd3/qian/bias-mitigation-with-rl/baseline'

# (size_label, method_label, w_dir_template, c_dir_template, key_w, key_c, mit_filter)
# Templates use {seed_suffix} where seed=42 has empty suffix and seeds 0/1 have "_seedN_"
# But original dir names don't follow same template — seed=42 uses "results_verbosity_val_qwen3_4b_baseline"
# while new seeds use "results_verbosity_val_seed0_qwen3_4b_baseline" etc.
# Also EIT uses "_eit110" / "_eit70" for seed=42 but just "_eit" for seeds 0/1.

CONFIGS = [
    # (size, method, w_dir_seed42, c_dir_seed42, w_dir_seedN_template, c_dir_seedN_template, key_w, key_c, mit)
    ('4B',   'Baseline',
        f'{P}/results_verbosity_val_qwen3_4b_baseline',
        f'{P}/results_correct_verbosity_val_qwen3_4b_baseline',
        f'{P}/results_verbosity_val_seed{{seed}}_qwen3_4b_baseline',
        f'{P}/results_correct_verbosity_val_seed{{seed}}_qwen3_4b_baseline',
        'incorrect_verbosity', 'correct_verbosity', 'any'),
    ('4B',   'PromptMit',
        f'{P}/results_verbosity_val_qwen3_4b_promptmit',
        f'{P}/results_correct_verbosity_val_qwen3_4b_promptmit',
        f'{P}/results_verbosity_val_seed{{seed}}_qwen3_4b_promptmit',
        f'{P}/results_correct_verbosity_val_seed{{seed}}_qwen3_4b_promptmit',
        None, None, 'with'),
    ('4B',   'SFT',
        f'{P}/results_verbosity_val_qwen3_4b_sft',
        f'{P}/results_correct_verbosity_val_qwen3_4b_sft',
        f'{P}/results_verbosity_val_seed{{seed}}_qwen3_4b_sft',
        f'{P}/results_correct_verbosity_val_seed{{seed}}_qwen3_4b_sft',
        'incorrect_verbosity', 'correct_verbosity', 'any'),
    ('4B',   'EIT',
        f'{P}/results_verbosity_val_qwen3_4b_eit110',
        f'{P}/results_correct_verbosity_val_qwen3_4b_eit110',
        f'{P}/results_verbosity_val_seed{{seed}}_qwen3_4b_eit',
        f'{P}/results_correct_verbosity_val_seed{{seed}}_qwen3_4b_eit',
        'incorrect_verbosity', 'correct_verbosity', 'any'),
    ('1.7B', 'Baseline',
        f'{P}/results_verbosity_val_qwen3_1_7b_baseline',
        f'{P}/results_correct_verbosity_val_qwen3_1_7b_baseline',
        f'{P}/results_verbosity_val_seed{{seed}}_qwen3_1_7b_baseline',
        f'{P}/results_correct_verbosity_val_seed{{seed}}_qwen3_1_7b_baseline',
        'incorrect_verbosity', 'correct_verbosity', 'any'),
    ('1.7B', 'PromptMit',
        f'{P}/results_verbosity_val_qwen3_1_7b_promptmit',
        f'{P}/results_correct_verbosity_val_qwen3_1_7b_promptmit',
        f'{P}/results_verbosity_val_seed{{seed}}_qwen3_1_7b_promptmit',
        f'{P}/results_correct_verbosity_val_seed{{seed}}_qwen3_1_7b_promptmit',
        None, None, 'with'),
    ('1.7B', 'SFT',
        f'{P}/results_verbosity_val_qwen3_1_7b_sft',
        f'{P}/results_correct_verbosity_val_qwen3_1_7b_sft',
        f'{P}/results_verbosity_val_seed{{seed}}_qwen3_1_7b_sft',
        f'{P}/results_correct_verbosity_val_seed{{seed}}_qwen3_1_7b_sft',
        'incorrect_verbosity', 'correct_verbosity', 'any'),
    ('1.7B', 'EIT',
        f'{P}/results_verbosity_val_qwen3_1_7b_eit70',
        f'{P}/results_correct_verbosity_val_qwen3_1_7b_eit70',
        f'{P}/results_verbosity_val_seed{{seed}}_qwen3_1_7b_eit',
        f'{P}/results_correct_verbosity_val_seed{{seed}}_qwen3_1_7b_eit',
        'incorrect_verbosity', 'correct_verbosity', 'any'),
]

def get_summary(d, key, mit):
    files = sorted(glob.glob(os.path.join(d, '*.json')))
    if not files:
        return None
    if mit == 'with':
        files = [f for f in files if 'with_mitigation' in f]
        if not files:
            return None
    j = json.load(open(files[-1]))
    s = j.get('summary', j)
    if key:
        return {
            'reg_acc': s.get('regular_accuracy', None),
            'bias_acc': s.get(f'{key}_accuracy', None),
            'robust_rate': s.get(f'{key}_robust_rate', None),
        }
    else:
        return {
            'reg_acc': s.get('regular_accuracy', None),
            'bias_acc': s.get('bias_accuracy', None),
            'robust_rate': s.get('robustness_rate', None),
        }


def fmt(v):
    return f"{v:.3f}" if v is not None else "  N/A"


print(f"{'Model':<6} {'Method':<10} {'Direction':<4} {'seed42':>10} {'seed0':>10} {'seed1':>10} {'mean':>10} {'std':>8}")
print('-' * 80)

# Collect averaged val numbers for table fill
final = {}  # (size, method, direction, metric) -> mean

for size, method, w42, c42, wN, cN, key_w, key_c, mit in CONFIGS:
    for direction, dir42, dirN_template, key in [('W', w42, wN, key_w), ('C', c42, cN, key_c)]:
        # Get all 3 seeds
        s42 = get_summary(dir42, key, mit)
        s0 = get_summary(dirN_template.format(seed=0), key, mit)
        s1 = get_summary(dirN_template.format(seed=1), key, mit)

        for metric in ['bias_acc', 'robust_rate']:
            vals = []
            label = f"{direction}.{metric.split('_')[0]}_{direction}"
            v42 = s42[metric] if s42 else None
            v0  = s0[metric]  if s0  else None
            v1  = s1[metric]  if s1  else None
            for v in [v42, v0, v1]:
                if v is not None:
                    vals.append(v)
            mean = statistics.mean(vals) if vals else None
            std = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"{size:<6} {method:<10} {direction:<4} {fmt(v42):>10} {fmt(v0):>10} {fmt(v1):>10} {fmt(mean):>10} {std:>8.4f}    {metric}")
            final[(size, method, direction, metric)] = mean

# Pretty-print val table cells
print()
print('=' * 80)
print(' AVERAGED VAL VERBOSITY (3 seeds) — for LaTeX table cells ')
print('=' * 80)
for size in ['1.7B', '4B']:
    print(f'\n--- Qwen3-{size} val ---')
    print(f'{"Method":<12} {"Acc_C":>8} {"Acc_W":>8} {"RR_C":>8} {"RR_W":>8}')
    for method in ['Baseline', 'PromptMit', 'SFT', 'EIT']:
        ac = final.get((size, method, 'C', 'bias_acc'))
        aw = final.get((size, method, 'W', 'bias_acc'))
        rc = final.get((size, method, 'C', 'robust_rate'))
        rw = final.get((size, method, 'W', 'robust_rate'))
        print(f'{method:<12} {fmt(ac):>8} {fmt(aw):>8} {fmt(rc):>8} {fmt(rw):>8}')
