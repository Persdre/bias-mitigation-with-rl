#!/usr/bin/env python3
"""Extract verbosity val + test results into a clean summary for filling LaTeX tables."""
import json, glob, os, sys

P = os.environ.get(
    'EIT_BASELINE_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline'),
)

# Each entry: (label, dir, key_prefix_or_None_if_promptmit)
# key_prefix tells us which fields to read (incorrect_verbosity_* or correct_verbosity_*)
# For prompt_mit, we use bias_accuracy/robustness_rate from the *with_mitigation* file
configs_val = [
    # 4B
    ('4B Baseline W val',   f'{P}/results_verbosity_val_qwen3_4b_baseline/',         'incorrect_verbosity', 'any'),
    ('4B Baseline C val',   f'{P}/results_correct_verbosity_val_qwen3_4b_baseline/', 'correct_verbosity',   'any'),
    ('4B PromptMit W val',  f'{P}/results_verbosity_val_qwen3_4b_promptmit/',         None,                  'with'),
    ('4B PromptMit C val',  f'{P}/results_correct_verbosity_val_qwen3_4b_promptmit/', None,                  'with'),
    ('4B SFT W val',        f'{P}/results_verbosity_val_qwen3_4b_sft/',               'incorrect_verbosity', 'any'),
    ('4B SFT C val',        f'{P}/results_correct_verbosity_val_qwen3_4b_sft/',       'correct_verbosity',   'any'),
    ('4B EIT W val',        f'{P}/results_verbosity_val_qwen3_4b_eit110/',            'incorrect_verbosity', 'any'),
    ('4B EIT C val',        f'{P}/results_correct_verbosity_val_qwen3_4b_eit110/',    'correct_verbosity',   'any'),
    # 1.7B
    ('1.7B Baseline W val', f'{P}/results_verbosity_val_qwen3_1_7b_baseline/',         'incorrect_verbosity', 'any'),
    ('1.7B Baseline C val', f'{P}/results_correct_verbosity_val_qwen3_1_7b_baseline/', 'correct_verbosity',   'any'),
    ('1.7B PromptMit W val',f'{P}/results_verbosity_val_qwen3_1_7b_promptmit/',         None,                  'with'),
    ('1.7B PromptMit C val',f'{P}/results_correct_verbosity_val_qwen3_1_7b_promptmit/', None,                  'with'),
    ('1.7B SFT W val',      f'{P}/results_verbosity_val_qwen3_1_7b_sft/',               'incorrect_verbosity', 'any'),
    ('1.7B SFT C val',      f'{P}/results_correct_verbosity_val_qwen3_1_7b_sft/',       'correct_verbosity',   'any'),
    ('1.7B EIT W val',      f'{P}/results_verbosity_val_qwen3_1_7b_eit70/',             'incorrect_verbosity', 'any'),
    ('1.7B EIT C val',      f'{P}/results_correct_verbosity_val_qwen3_1_7b_eit70/',     'correct_verbosity',   'any'),
]

def get_summary(d, key_prefix, mit_filter):
    files = sorted(glob.glob(os.path.join(d, '*.json')))
    if not files:
        return None
    if mit_filter == 'with':
        files = [f for f in files if 'with_mitigation' in f]
        if not files:
            return None
    j = json.load(open(files[-1]))
    s = j.get('summary', j)
    if key_prefix:
        return {
            'reg_acc': s.get('regular_accuracy', s.get(f'{key_prefix}_regular_accuracy', None)),
            'bias_acc': s.get(f'{key_prefix}_accuracy', None),
            'robust_rate': s.get(f'{key_prefix}_robust_rate', None),
            'total': s.get('total_questions', s.get('total', None)),
        }
    else:  # promptmit
        return {
            'reg_acc': s.get('regular_accuracy', None),
            'bias_acc': s.get('bias_accuracy', None),
            'robust_rate': s.get('robustness_rate', None),
            'total': s.get('total', None),
        }


print(f"{'Config':<26} {'Total':>6} {'Reg Acc':>10} {'Bias Acc':>10} {'Robust':>10}")
print('-' * 70)
for name, d, key, mit in configs_val:
    r = get_summary(d, key, mit)
    if r is None:
        print(f"{name:<26} (no results)")
    else:
        ra = f"{r['reg_acc']:.4f}" if r['reg_acc'] is not None else "N/A"
        ba = f"{r['bias_acc']:.4f}" if r['bias_acc'] is not None else "N/A"
        rr = f"{r['robust_rate']:.4f}" if r['robust_rate'] is not None else "N/A"
        tot = r['total'] if r['total'] is not None else "?"
        print(f"{name:<26} {str(tot):>6} {ra:>10} {ba:>10} {rr:>10}")
