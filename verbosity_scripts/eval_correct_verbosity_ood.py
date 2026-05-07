"""Evaluation script for out-of-domain CORRECT verbosity bias.
Uses prompts from parquet file where verbosity points to the CORRECT answer.
"""

import re
import json
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

user_lib_dir = os.path.expanduser("~/lib")
bad_symlink = os.path.join(user_lib_dir, "libcudnn.so.9")
if os.path.exists(bad_symlink) or os.path.islink(bad_symlink):
    try:
        os.remove(bad_symlink)
    except:
        pass

cudnn_lib_path = os.environ.get("CUDNN_LIB_PATH", "")
if cudnn_lib_path:
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
    if chat_messages is None:
        return ""
    if isinstance(chat_messages, (np.ndarray, pd.Series)):
        chat_messages = chat_messages.tolist()
    if not chat_messages:
        return ""
    return _render(chat_messages)


def generate_regular_prompt(question, option_a, option_b):
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
    if not response_text:
        return None
    tag_match = re.search(r'<answer>\s*([AB])\s*</answer>', response_text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).upper()
    text_upper = response_text.upper()
    for pattern in [
        r'ANSWER(?:\s+IS)?(?:\s*:\s*|\s+)([AB])\b',
        r'OPTION(?:\s+IS)?(?:\s*:\s*|\s+)([AB])\b',
        r'THE CORRECT ANSWER IS\s*([AB])\b',
        r'^([AB])$'
    ]:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)
    standalone = re.search(r'(?:^|[\s\.,:;])([AB])(?:$|[\s\.,:;])', text_upper)
    if standalone:
        return standalone.group(1)
    return None


def get_model_response(llm, prompt, max_new_tokens=1024, temperature=0.7, sampling_seed=None):
    try:
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, seed=sampling_seed)
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    except Exception as e:
        print(f"Error: {e}")
        return None


def load_dataset_from_parquet(parquet_path, subjects=None, num_samples=None,
                              samples_per_subject=None, seed=42):
    import random
    random.seed(seed)
    df = pd.read_parquet(parquet_path)
    if subjects:
        subject_list = [s.strip() for s in subjects.split(',')]
        df['_subject'] = df.apply(
            lambda row: row['extra_info'].get('subject', 'unknown')
            if isinstance(row.get('extra_info'), dict) else 'unknown', axis=1)
        df = df[df['_subject'].isin(subject_list)].copy()
        if samples_per_subject:
            sampled = []
            for s in subject_list:
                sdf = df[df['_subject'] == s].copy()
                if len(sdf) > samples_per_subject:
                    sdf = sdf.sample(n=samples_per_subject, random_state=seed)
                sampled.append(sdf)
            df = pd.concat(sampled, ignore_index=True)
        df = df.drop(columns=['_subject'])

    dataset = []
    for idx, row in df.iterrows():
        prompt_data = row['prompt']
        if isinstance(prompt_data, (np.ndarray, pd.Series)):
            prompt_data = prompt_data.tolist()
        elif hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()
        dataset.append({
            'question_id': idx,
            'prompt': prompt_data,
            'reward_model': row['reward_model'],
            'extra_info': row['extra_info']
        })
    if num_samples:
        dataset = dataset[:num_samples]
    return dataset


def evaluate_correct_verbosity(llm, dataset, model_name, output_file, temperature=0.7, sampling_seed=None):
    """Evaluate correct verbosity bias (verbosity points to correct answer)"""
    results = []
    regular_correct = 0
    correct_verbosity_correct = 0
    correct_verbosity_robust = 0
    total = len(dataset)

    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        extra_info = item['extra_info']
        question = extra_info.get('question', '')
        option_a = extra_info.get('option_a_text', '')
        option_b = extra_info.get('option_b_text', '')
        correct_letter = extra_info.get('correct_letter', 'A')
        verbosity_target = extra_info.get('verbosity_target', 'A')

        if not question or not option_a or not option_b:
            continue

        result = {
            "question_id": item.get('question_id', i),
            "correct_letter": correct_letter,
            "verbosity_target": verbosity_target,
            "subject": extra_info.get('subject', 'unknown'),
        }

        # Regular
        regular_prompt = generate_regular_prompt(question, option_a, option_b)
        regular_response = get_model_response(llm, regular_prompt, temperature=temperature, sampling_seed=sampling_seed)
        regular_answer = extract_answer(regular_response)
        result["regular_answer"] = regular_answer
        result["regular_correct"] = (regular_answer == correct_letter) if regular_answer else False
        if result["regular_correct"]:
            regular_correct += 1

        # Verbosity
        verbosity_prompt = chat_format_to_prompt(item['prompt'])
        verbosity_response = get_model_response(llm, verbosity_prompt, temperature=temperature, sampling_seed=sampling_seed)
        verbosity_answer = extract_answer(verbosity_response)
        result["verbosity_answer"] = verbosity_answer
        result["verbosity_correct"] = (verbosity_answer == correct_letter) if verbosity_answer else False
        if result["verbosity_correct"]:
            correct_verbosity_correct += 1
        if regular_answer == verbosity_answer:
            correct_verbosity_robust += 1

        results.append(result)

    regular_accuracy = regular_correct / total if total > 0 else 0
    correct_verbosity_accuracy = correct_verbosity_correct / total if total > 0 else 0
    correct_verbosity_robust_rate = correct_verbosity_robust / total if total > 0 else 0

    summary = {
        "model": model_name,
        "temperature": temperature,
        "total_questions": total,
        "regular_accuracy": regular_accuracy,
        "correct_verbosity_accuracy": correct_verbosity_accuracy,
        "correct_verbosity_robust_rate": correct_verbosity_robust_rate,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False, default=str)

    return summary, results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default=os.path.join(project_root, 'data/mmlupro/test_paired_ood_verbosity_correct_random.parquet'))
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(project_root, 'baseline/results_correct_verbosity_ood'))
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

    dataset = load_dataset_from_parquet(
        args.dataset, subjects=args.subjects,
        num_samples=args.samples,
        samples_per_subject=args.samples_per_subject, seed=args.seed)

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size,
              enforce_eager=True, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = os.path.basename(args.model_path) if args.model_path else "model"
    output_file = os.path.join(args.output_dir,
                               f"{model_name_short}_correct_verbosity_ood_{timestamp}.json")

    summary, _ = evaluate_correct_verbosity(llm, dataset, model_name_short, output_file,
                                            temperature=args.temperature,
                                            sampling_seed=args.sampling_seed)

    print("\n" + "=" * 80)
    print(" CORRECT VERBOSITY BIAS EVALUATION ".center(80, '='))
    print("=" * 80)
    print(f"Model: {summary['model']}")
    print(f"Regular Accuracy: {summary['regular_accuracy']:.2%}")
    print(f"Correct Verbosity Accuracy: {summary['correct_verbosity_accuracy']:.2%}")
    print(f"Robustness Rate: {summary['correct_verbosity_robust_rate']:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
