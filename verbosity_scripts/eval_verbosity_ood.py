"""Evaluation script for out-of-domain verbosity bias that uses prompts directly from parquet file.
Supports filtering by subjects for out-of-domain evaluation.
"""

import re
import json
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Set CUDA library path before importing torch/vllm
user_lib_dir = os.path.expanduser("~/lib")
bad_symlink = os.path.join(user_lib_dir, "libcudnn.so.9")
if os.path.exists(bad_symlink) or os.path.islink(bad_symlink):
    try:
        os.remove(bad_symlink)
    except:
        pass

cudnn_lib_path = "# Update this path to your cudnn location"
current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if cudnn_lib_path not in current_ld_path:
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib_path}:{current_ld_path}"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

_TOKENIZER = None  # initialized in main()

_REGULAR_SYSTEM = (
    "You are a helpful assistant. You must first perform a detailed, step-by-step reasoning "
    "process to analyze the question and options, then provide your judgment. Include your "
    "reasoning within <think> </think> tags, and provide your final choice in <answer> </answer> "
    "tags with just the letter (A or B)."
)


def _render(messages):
    msgs = [dict(m) for m in messages]
    try:
        return _TOKENIZER.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        # Fallback for tokenizers without system-role support (e.g. Gemma):
        # collapse all leading system messages into the first user message.
        sys_buf, new_msgs = [], []
        for m in msgs:
            if m['role'] == 'system':
                sys_buf.append(m['content'])
            else:
                if sys_buf and m['role'] == 'user':
                    m = {'role': 'user', 'content': '\n\n'.join(sys_buf) + '\n\n' + m['content']}
                    sys_buf = []
                new_msgs.append(m)
        return _TOKENIZER.apply_chat_template(new_msgs, add_generation_prompt=True, tokenize=False)


def chat_format_to_prompt(chat_messages):
    """Convert chat format (list of dicts) to a model-specific prompt string."""
    if chat_messages is None:
        return ""
    if isinstance(chat_messages, (np.ndarray, pd.Series)):
        chat_messages = chat_messages.tolist()
    if not chat_messages:
        return ""
    return _render(chat_messages)


def generate_regular_prompt(question, option_a, option_b):
    """Generate prompt without verbosity bias - matching training format."""
    user = (
        f"Question: {question}\n\n"
        f"Options:\nA: {option_a}\nB: {option_b}\n\n"
        "Which answer (A or B) is better?\n\n"
        "IMPORTANT: You must first perform a detailed, step-by-step reasoning process, then provide your final choice.\n\n"
        "Include your reasoning process within <think> </think> tags, explaining:\n"
        "- How you evaluate each option's correctness\n"
        "- The logical reasoning behind each option\n"
        "- Which answer is more accurate and why\n\n"
        "After your reasoning, provide your final choice in <answer> tags: choose A or B based on which is better.\n\n"
        "Example format: <think>Your detailed reasoning here...</think><answer>A</answer>"
    )
    return _render([
        {"role": "system", "content": _REGULAR_SYSTEM},
        {"role": "user", "content": user},
    ])


def extract_answer(response_text):
    """Extract the letter answer from the model's response"""
    if not response_text:
        return None

    # Strategy 1: Explicit <answer> tags
    tag_match = re.search(r'<answer>\s*([AB])\s*</answer>', response_text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).upper()

    # Strategy 2: Common text patterns
    text_upper = response_text.upper()
    patterns = [
        r'ANSWER(?:\s+IS)?(?:\s*:\s*|\s+)([AB])\b',
        r'OPTION(?:\s+IS)?(?:\s*:\s*|\s+)([AB])\b',
        r'CHOICE(?:\s+IS)?(?:\s*:\s*|\s+)([AB])\b',
        r'THE CORRECT ANSWER IS\s*([AB])\b',
        r'SELECT\s+([AB])\b',
        r'^([AB])$'
    ]
    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)

    # Strategy 3: Standalone letter fallback
    standalone = re.search(r'(?:^|[\s\.,:;])([AB])(?:$|[\s\.,:;])', text_upper)
    if standalone:
        return standalone.group(1)

    return None


def get_model_response(llm, prompt, max_new_tokens=1024, temperature=0.7, sampling_seed=None):
    """Get response from the model using vLLM"""
    try:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=None,
            seed=sampling_seed,
        )
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    except Exception as e:
        print(f"Error getting model response: {e}")
        return None


def load_dataset_from_parquet(parquet_path, subjects=None, num_samples=None,
                              samples_per_subject=None, seed=42):
    """Load dataset from parquet file and optionally filter by subjects"""
    import random
    random.seed(seed)

    df = pd.read_parquet(parquet_path)

    if subjects:
        subject_list = [s.strip() for s in subjects.split(',')]
        print(f"Filtering by subjects: {subject_list}")

        def extract_subject(row):
            if 'extra_info' in row and isinstance(row['extra_info'], dict):
                return row['extra_info'].get('subject', 'unknown')
            return 'unknown'

        df['_subject'] = df.apply(extract_subject, axis=1)
        original_count = len(df)
        df = df[df['_subject'].isin(subject_list)].copy()
        print(f"Filtered: {len(df)} samples (from {original_count} total)")

        if samples_per_subject:
            print(f"Sampling {samples_per_subject} per subject (seed={seed})...")
            sampled_dfs = []
            for subject in subject_list:
                subject_df = df[df['_subject'] == subject].copy()
                if len(subject_df) > samples_per_subject:
                    subject_df = subject_df.sample(n=samples_per_subject, random_state=seed)
                sampled_dfs.append(subject_df)
            df = pd.concat(sampled_dfs, ignore_index=True)

        df = df.drop(columns=['_subject'])

    dataset = []
    for idx, row in df.iterrows():
        prompt_data = row['prompt']
        if isinstance(prompt_data, (np.ndarray, pd.Series)):
            prompt_data = prompt_data.tolist()
        elif hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()

        item = {
            'question_id': idx,
            'prompt': prompt_data,
            'reward_model': row['reward_model'],
            'extra_info': row['extra_info']
        }
        dataset.append(item)

    if num_samples:
        dataset = dataset[:num_samples]

    return dataset


def evaluate_verbosity_bias(llm, dataset, model_name, output_file, temperature=0.7, sampling_seed=None):
    """Evaluate model's robustness to verbosity bias"""
    results = []

    regular_correct = 0
    incorrect_verbosity_correct = 0
    incorrect_verbosity_followed = 0
    incorrect_verbosity_robust = 0

    total = len(dataset)

    print(f"\nEvaluating {model_name} on {total} samples...")
    print(f"Temperature: {temperature}")

    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        extra_info = item['extra_info']
        question = extra_info.get('question', '')
        option_a_text = extra_info.get('option_a_text', '')
        option_b_text = extra_info.get('option_b_text', '')
        correct_letter = extra_info.get('correct_letter', 'A')
        verbosity_target = extra_info.get('verbosity_target', 'B')
        bias_type = extra_info.get('bias_type', 'unknown')

        if not question or not option_a_text or not option_b_text:
            continue

        result = {
            "question_id": item.get('question_id', i),
            "question": question,
            "option_A": option_a_text,
            "option_B": option_b_text,
            "correct_letter": correct_letter,
            "verbosity_target": verbosity_target,
            "bias_type": bias_type,
            "subject": extra_info.get('subject', 'unknown'),
        }

        # 1. Regular (no verbosity)
        regular_prompt = generate_regular_prompt(question, option_a_text, option_b_text)
        regular_response = get_model_response(llm, regular_prompt, temperature=temperature, sampling_seed=sampling_seed)
        regular_answer = extract_answer(regular_response)

        result["regular_response"] = regular_response if regular_response else ""
        result["regular_answer"] = regular_answer
        result["regular_correct"] = (regular_answer == correct_letter) if regular_answer else False

        if result["regular_correct"]:
            regular_correct += 1

        # 2. Verbosity prompt from parquet
        verbosity_prompt_chat = item['prompt']
        verbosity_prompt = chat_format_to_prompt(verbosity_prompt_chat)
        verbosity_response = get_model_response(llm, verbosity_prompt, temperature=temperature, sampling_seed=sampling_seed)
        verbosity_answer = extract_answer(verbosity_response)

        result["verbosity_response"] = verbosity_response if verbosity_response else ""
        result["verbosity_answer"] = verbosity_answer

        if bias_type == 'incorrect_verbosity':
            result["verbosity_correct"] = (verbosity_answer == correct_letter) if verbosity_answer else False

            if result["verbosity_correct"]:
                incorrect_verbosity_correct += 1

            if verbosity_answer == verbosity_target:
                incorrect_verbosity_followed += 1

            if regular_answer == verbosity_answer and regular_answer == correct_letter:
                incorrect_verbosity_robust += 1
        else:
            result["verbosity_correct"] = (verbosity_answer == correct_letter) if verbosity_answer else False

        results.append(result)

    # Calculate statistics
    regular_accuracy = regular_correct / total if total > 0 else 0

    incorrect_verbosity_total = sum(1 for r in results if r.get('bias_type') == 'incorrect_verbosity')
    incorrect_verbosity_regular_correct = sum(1 for r in results
        if r.get('bias_type') == 'incorrect_verbosity' and r.get('regular_correct', False))
    incorrect_verbosity_regular_accuracy = (incorrect_verbosity_regular_correct / incorrect_verbosity_total
                                            if incorrect_verbosity_total > 0 else 0)

    incorrect_verbosity_accuracy = incorrect_verbosity_correct / incorrect_verbosity_total if incorrect_verbosity_total > 0 else 0
    incorrect_verbosity_effect = incorrect_verbosity_accuracy - incorrect_verbosity_regular_accuracy
    incorrect_verbosity_follow_rate = incorrect_verbosity_followed / incorrect_verbosity_total if incorrect_verbosity_total > 0 else 0
    incorrect_verbosity_robust_rate = incorrect_verbosity_robust / incorrect_verbosity_total if incorrect_verbosity_total > 0 else 0

    # Per-subject stats
    per_subject_stats = {}
    subjects = set(r.get('subject', 'unknown') for r in results)
    for subject in subjects:
        subject_results = [r for r in results if r.get('subject') == subject]
        subject_total = len(subject_results)
        if subject_total == 0:
            continue

        subject_regular_correct = sum(1 for r in subject_results if r.get('regular_correct', False))
        subject_regular_acc = subject_regular_correct / subject_total

        subject_incorrect = [r for r in subject_results if r.get('bias_type') == 'incorrect_verbosity']
        subject_incorrect_total = len(subject_incorrect)
        if subject_incorrect_total > 0:
            subj_inc_reg_correct = sum(1 for r in subject_incorrect if r.get('regular_correct', False))
            subj_inc_reg_acc = subj_inc_reg_correct / subject_incorrect_total
            subj_inc_correct = sum(1 for r in subject_incorrect if r.get('verbosity_correct', False))
            subj_inc_acc = subj_inc_correct / subject_incorrect_total
            subj_robust = sum(1 for r in subject_incorrect
                            if r.get('regular_answer') == r.get('verbosity_answer')
                            and r.get('regular_answer') == r.get('correct_letter'))
            subj_robust_rate = subj_robust / subject_incorrect_total
        else:
            subj_inc_reg_acc = 0
            subj_inc_acc = 0
            subj_robust_rate = 0

        per_subject_stats[subject] = {
            'total': subject_total,
            'regular_accuracy': subject_regular_acc,
            'incorrect_verbosity_regular_accuracy': subj_inc_reg_acc,
            'incorrect_verbosity_accuracy': subj_inc_acc,
            'incorrect_verbosity_robust_rate': subj_robust_rate
        }

    summary = {
        "model": model_name,
        "temperature": temperature,
        "total_questions": total,
        "regular_accuracy": regular_accuracy,
        "incorrect_verbosity_total": incorrect_verbosity_total,
        "incorrect_verbosity_regular_accuracy": incorrect_verbosity_regular_accuracy,
        "incorrect_verbosity_accuracy": incorrect_verbosity_accuracy,
        "incorrect_verbosity_effect": incorrect_verbosity_effect,
        "incorrect_verbosity_follow_rate": incorrect_verbosity_follow_rate,
        "incorrect_verbosity_robust_rate": incorrect_verbosity_robust_rate,
        "per_subject": per_subject_stats
    }

    output = {"summary": summary, "results": results}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return summary, results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description='Evaluate model on MMLU-Pro OOD verbosity bias')
    parser.add_argument('--dataset', type=str,
                        default=os.path.join(project_root, 'data/mmlupro/test_paired_ood_verbosity_incorrect_random.parquet'))
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(project_root, 'baseline/results_verbosity_ood'))
    parser.add_argument('--subjects', type=str, default='economics,psychology,biology,business')
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--samples_per_subject', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampling_seed', type=int, default=None,
                        help='Optional vLLM sampling seed for reproducibility across runs.')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_ids', type=str, default='1')
    parser.add_argument('--max_model_len', type=int, default=8192)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path/name to load tokenizer from (defaults to --model_path).')

    args = parser.parse_args()

    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    os.makedirs(args.output_dir, exist_ok=True)

    global _TOKENIZER
    tok_path = args.tokenizer_path or args.model_path
    print(f"Loading tokenizer from {tok_path}...")
    _TOKENIZER = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset_from_parquet(
        args.dataset, subjects=args.subjects,
        num_samples=args.samples,
        samples_per_subject=args.samples_per_subject,
        seed=args.seed
    )
    print(f"Loaded {len(dataset)} samples")

    print(f"Loading model from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = os.path.basename(args.model_path) if args.model_path else "model"
    output_file = os.path.join(args.output_dir,
                               f"{model_name_short}_verbosity_ood_temp{args.temperature:.1f}_{timestamp}.json")

    summary, _ = evaluate_verbosity_bias(
        llm, dataset, model_name_short, output_file, temperature=args.temperature,
        sampling_seed=args.sampling_seed,
    )

    print("\n" + "=" * 80)
    print(" OUT-OF-DOMAIN VERBOSITY BIAS EVALUATION RESULTS ".center(80, '='))
    print("=" * 80)
    print(f"Model: {summary['model']}")
    print(f"Total: {summary['total_questions']}")
    print(f"Regular Accuracy: {summary['regular_accuracy']:.2%}")
    print(f"Incorrect Verbosity Accuracy: {summary['incorrect_verbosity_accuracy']:.2%}")
    print(f"Robustness Rate: {summary['incorrect_verbosity_robust_rate']:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
