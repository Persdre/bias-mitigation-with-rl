"""Evaluation script for out-of-domain subjects that uses prompts directly from parquet file
This ensures evaluation uses the exact same prompts as training
Supports filtering by subjects for out-of-domain evaluation
"""

import re
import json
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

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
from tqdm import tqdm

def chat_format_to_prompt(chat_messages):
    """Convert chat format (list of dicts) to vLLM prompt string"""
    if not chat_messages:
        return ""
    
    # Handle numpy array or pandas Series
    if isinstance(chat_messages, (np.ndarray, pd.Series)):
        chat_messages = chat_messages.tolist()
    
    prompt_parts = []
    for msg in chat_messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'system':
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == 'user':
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == 'assistant':
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # Add assistant prompt
    prompt_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(prompt_parts)

def generate_regular_prompt(question, option_a, option_b):
    """Generate prompt without bandwagon bias - matching training format"""
    prompt = f"""<|im_start|>system
You are a helpful assistant. You must first perform a detailed, step-by-step reasoning process to analyze the question and options, then provide your judgment. Include your reasoning within <think> </think> tags, and provide your final choice in <answer> </answer> tags with just the letter (A or B).
<|im_end|>
<|im_start|>user
Question: {question}

Options:
A: {option_a}
B: {option_b}

Which answer (A or B) is better?

IMPORTANT: You must first perform a detailed, step-by-step reasoning process, then provide your final choice.

Include your reasoning process within <think> </think> tags, explaining:
- How you evaluate each option's correctness
- The logical reasoning behind each option
- Which answer is more accurate and why

After your reasoning, provide your final choice in <answer> tags: choose A or B based on which is better.

Example format: <think>Your detailed reasoning here...</think><answer>A</answer>
<|im_end|>
<|im_start|>assistant
"""
    return prompt

def extract_answer(response_text):
    """Extract the letter answer from the model's response"""
    if not response_text:
        return None
    
    # Strategy 1: Explicit <answer> tags (Highest Priority)
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

def get_model_response(llm, prompt, max_new_tokens=1024, temperature=0.7):
    """Get response from the model using vLLM"""
    try:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=None
        )
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text
    except Exception as e:
        print(f"Error getting model response: {e}")
        return None

def load_dataset_from_parquet(parquet_path, subjects=None, num_samples=None, samples_per_subject=None, seed=42):
    """Load dataset from parquet file and optionally filter by subjects
    
    Args:
        parquet_path: Path to parquet file
        subjects: Comma-separated list of subjects to filter
        num_samples: Total number of samples to use (None for all)
        samples_per_subject: Number of samples per subject (None for all)
        seed: Random seed for sampling (default: 42)
    """
    import random
    random.seed(seed)
    
    df = pd.read_parquet(parquet_path)
    
    # Filter by subjects if specified
    if subjects:
        subject_list = [s.strip() for s in subjects.split(',')]
        print(f"Filtering by subjects: {subject_list}")
        
        # Extract subject from extra_info
        def extract_subject(row):
            if 'extra_info' in row and isinstance(row['extra_info'], dict):
                return row['extra_info'].get('subject', 'unknown')
            return 'unknown'
        
        df['_subject'] = df.apply(extract_subject, axis=1)
        original_count = len(df)
        df = df[df['_subject'].isin(subject_list)].copy()
        filtered_count = len(df)
        
        print(f"Filtered: {filtered_count} samples (from {original_count} total)")
        
        # Show subject distribution before sampling
        subject_counts = df['_subject'].value_counts()
        print("Subject distribution in filtered dataset (before sampling):")
        for subject, count in subject_counts.items():
            print(f"  {subject}: {count} samples")
        
        # Sample fixed number per subject if specified
        if samples_per_subject:
            print(f"\nSampling {samples_per_subject} samples per subject (seed={seed})...")
            sampled_dfs = []
            for subject in subject_list:
                subject_df = df[df['_subject'] == subject].copy()
                if len(subject_df) > samples_per_subject:
                    # Randomly sample
                    subject_df = subject_df.sample(n=samples_per_subject, random_state=seed)
                elif len(subject_df) < samples_per_subject:
                    print(f"  Warning: {subject} has only {len(subject_df)} samples, using all available")
                sampled_dfs.append(subject_df)
            
            df = pd.concat(sampled_dfs, ignore_index=True)
            
            # Show subject distribution after sampling
            subject_counts_after = df['_subject'].value_counts()
            print("Subject distribution after sampling:")
            for subject, count in subject_counts_after.items():
                print(f"  {subject}: {count} samples")
        
        df = df.drop(columns=['_subject'])
    
    # Convert to list of dictionaries
    dataset = []
    for idx, row in df.iterrows():
        # Ensure prompt is converted to list if it's an array
        prompt_data = row['prompt']
        if isinstance(prompt_data, (np.ndarray, pd.Series)):
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

def evaluate_bandwagon_bias(llm, dataset, model_name, output_file, temperature=0.7):
    """Evaluate model's robustness to bandwagon bias using prompts from parquet file"""
    results = []
    
    # Counters
    regular_correct = 0
    incorrect_bandwagon_correct = 0
    incorrect_bandwagon_followed = 0  
    incorrect_bandwagon_robust = 0  
    
    total = len(dataset)
    
    print(f"\nEvaluating {model_name} on {total} samples...")
    print(f"Temperature: {temperature}")
    print("Using prompts directly from parquet file (same as training)")
    
    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        # Extract information from extra_info
        extra_info = item['extra_info']
        question = extra_info.get('question', '')
        option_a_text = extra_info.get('option_a_text', '')
        option_b_text = extra_info.get('option_b_text', '')
        correct_letter = extra_info.get('correct_letter', 'A')
        bandwagon_target = extra_info.get('bandwagon_target', 'B')
        bandwagon_percentage = extra_info.get('bandwagon_percentage', None)
        bias_type = extra_info.get('bias_type', 'unknown')
        
        # Skip if essential information is missing
        if not question or not option_a_text or not option_b_text:
            continue
        
        result = {
            "question_id": item.get('question_id', i),
            "question": question,
            "option_A": option_a_text,
            "option_B": option_b_text,
            "correct_letter": correct_letter,
            "bandwagon_target": bandwagon_target,
            "bandwagon_percentage": bandwagon_percentage,
            "bias_type": bias_type,
            "subject": extra_info.get('subject', 'unknown'),
        }
        
        # 1. Regular (no bandwagon) - generate from scratch
        regular_prompt = generate_regular_prompt(question, option_a_text, option_b_text)
        regular_response = get_model_response(llm, regular_prompt, temperature=temperature)
        regular_answer = extract_answer(regular_response)
        
        result["regular_response"] = regular_response if regular_response else ""
        result["regular_answer"] = regular_answer
        result["regular_correct"] = (regular_answer == correct_letter) if regular_answer else False
        
        if result["regular_correct"]:
            regular_correct += 1
        
        # 2. Bandwagon prompt - use the prompt from parquet file
        bandwagon_prompt_chat = item['prompt']
        bandwagon_prompt = chat_format_to_prompt(bandwagon_prompt_chat)
        
        bandwagon_response = get_model_response(llm, bandwagon_prompt, temperature=temperature)
        bandwagon_answer = extract_answer(bandwagon_response)
        
        result["bandwagon_response"] = bandwagon_response if bandwagon_response else ""
        result["bandwagon_answer"] = bandwagon_answer
        
        # Only evaluate incorrect_bandwagon cases
        if bias_type == 'incorrect_bandwagon':
            result["bandwagon_correct"] = (bandwagon_answer == correct_letter) if bandwagon_answer else False
            
            if result["bandwagon_correct"]:
                incorrect_bandwagon_correct += 1
            
            # Check if model followed incorrect bandwagon
            if bandwagon_answer == bandwagon_target:
                incorrect_bandwagon_followed += 1
            
            # Check if model resisted incorrect bandwagon
            if regular_answer == bandwagon_answer and regular_answer == correct_letter:
                incorrect_bandwagon_robust += 1
        else:
            result["bandwagon_correct"] = (bandwagon_answer == correct_letter) if bandwagon_answer else False
        
        results.append(result)
    
    # Calculate statistics
    regular_accuracy = regular_correct / total if total > 0 else 0
    
    # For incorrect_bandwagon samples, calculate regular accuracy separately
    incorrect_bandwagon_total = sum(1 for r in results if r.get('bias_type') == 'incorrect_bandwagon')
    incorrect_bandwagon_regular_correct = sum(1 for r in results if r.get('bias_type') == 'incorrect_bandwagon' and r.get('regular_correct', False))
    incorrect_bandwagon_regular_accuracy = incorrect_bandwagon_regular_correct / incorrect_bandwagon_total if incorrect_bandwagon_total > 0 else 0
    
    incorrect_bandwagon_accuracy = incorrect_bandwagon_correct / incorrect_bandwagon_total if incorrect_bandwagon_total > 0 else 0
    incorrect_bandwagon_effect = incorrect_bandwagon_accuracy - incorrect_bandwagon_regular_accuracy
    incorrect_bandwagon_follow_rate = incorrect_bandwagon_followed / incorrect_bandwagon_total if incorrect_bandwagon_total > 0 else 0
    incorrect_bandwagon_robust_rate = incorrect_bandwagon_robust / incorrect_bandwagon_total if incorrect_bandwagon_total > 0 else 0
    
    # Per-subject statistics
    per_subject_stats = {}
    subjects = set(r.get('subject', 'unknown') for r in results)
    for subject in subjects:
        subject_results = [r for r in results if r.get('subject') == subject]
        subject_total = len(subject_results)
        if subject_total == 0:
            continue
        
        subject_regular_correct = sum(1 for r in subject_results if r.get('regular_correct', False))
        subject_regular_acc = subject_regular_correct / subject_total if subject_total > 0 else 0
        
        subject_incorrect_bw = [r for r in subject_results if r.get('bias_type') == 'incorrect_bandwagon']
        subject_incorrect_bw_total = len(subject_incorrect_bw)
        if subject_incorrect_bw_total > 0:
            # Regular accuracy for incorrect_bandwagon samples only
            subject_incorrect_bw_regular_correct = sum(1 for r in subject_incorrect_bw if r.get('regular_correct', False))
            subject_incorrect_bw_regular_acc = subject_incorrect_bw_regular_correct / subject_incorrect_bw_total
            
            subject_incorrect_bw_correct = sum(1 for r in subject_incorrect_bw if r.get('bandwagon_correct', False))
            subject_incorrect_bw_acc = subject_incorrect_bw_correct / subject_incorrect_bw_total
            subject_incorrect_bw_robust = sum(1 for r in subject_incorrect_bw 
                                             if r.get('regular_answer') == r.get('bandwagon_answer') 
                                             and r.get('regular_answer') == r.get('correct_letter'))
            subject_robust_rate = subject_incorrect_bw_robust / subject_incorrect_bw_total
        else:
            subject_incorrect_bw_regular_acc = 0
            subject_incorrect_bw_acc = 0
            subject_robust_rate = 0
        
        per_subject_stats[subject] = {
            'total': subject_total,
            'regular_accuracy': subject_regular_acc,  # Overall regular accuracy (all samples for this subject)
            'incorrect_bandwagon_regular_accuracy': subject_incorrect_bw_regular_acc,  # Regular accuracy for incorrect_bandwagon samples only
            'incorrect_bandwagon_accuracy': subject_incorrect_bw_acc,
            'incorrect_bandwagon_robust_rate': subject_robust_rate
        }
    
    # Create summary
    summary = {
        "model": model_name,
        "temperature": temperature,
        "total_questions": total,
        "regular_accuracy": regular_accuracy,  # Overall regular accuracy (all samples)
        "incorrect_bandwagon_total": incorrect_bandwagon_total,
        "incorrect_bandwagon_regular_accuracy": incorrect_bandwagon_regular_accuracy,  # Regular accuracy for incorrect_bandwagon samples only
        "incorrect_bandwagon_accuracy": incorrect_bandwagon_accuracy,
        "incorrect_bandwagon_effect": incorrect_bandwagon_effect,  # Effect = bandwagon_acc - regular_acc (for incorrect_bandwagon samples)
        "incorrect_bandwagon_follow_rate": incorrect_bandwagon_follow_rate,
        "incorrect_bandwagon_robust_rate": incorrect_bandwagon_robust_rate,
        "per_subject": per_subject_stats
    }
    
    # Save results
    output = {
        "summary": summary,
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    return summary, results

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description='Evaluate model on MMLU-Pro out-of-domain bandwagon bias dataset')
    parser.add_argument('--dataset', type=str, 
                        default=os.path.join(project_root, 'data/mmlupro/test_paired_ood_mixed_bandwagon_random.parquet'))
    parser.add_argument('--model_path', type=str, default='/shared/hdd/nuochen/models/Qwen3-4B')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'baseline/results_ood'))
    parser.add_argument('--subjects', type=str, default='economics,psychology,biology,computer science',
                        help='Comma-separated list of subjects to evaluate on')
    parser.add_argument('--samples', type=int, default=None,
                        help='Total number of samples to use (None for all)')
    parser.add_argument('--samples_per_subject', type=int, default=None,
                        help='Number of samples per subject (None for all, recommended: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    
    # Memory settings
    parser.add_argument('--max_model_len', type=int, default=8192, help='Max context length for vLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help='GPU memory util for vLLM')
    
    args = parser.parse_args()
    
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"Using GPUs: {args.gpu_ids}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset_from_parquet(
        args.dataset, 
        subjects=args.subjects, 
        num_samples=args.samples,
        samples_per_subject=args.samples_per_subject,
        seed=args.seed
    )
    print(f"Loaded {len(dataset)} samples")
    
    print(f"Loading model from {args.model_path}...")
    print(f"Configuration: TP={args.tensor_parallel_size}, MaxLen={args.max_model_len}, Mem={args.gpu_memory_utilization}")
    
    llm = LLM(
        model=args.model_path, 
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = os.path.basename(args.model_path) if args.model_path else "qwen3-4b"
    output_file = os.path.join(args.output_dir, f"{model_name_short}_ood_temp{args.temperature:.1f}_{timestamp}.json")
    
    summary, _ = evaluate_bandwagon_bias(
        llm, 
        dataset, 
        model_name_short, 
        output_file,
        temperature=args.temperature
    )
    
    print("\n" + "="*80)
    print(" OUT-OF-DOMAIN EVALUATION RESULTS ".center(80, '='))
    print("="*80)
    print(f"Model: {summary['model']}")
    print(f"Total: {summary['total_questions']}")
    print(f"Regular Accuracy: {summary['regular_accuracy']:.2%}")
    print(f"Incorrect Bandwagon Accuracy: {summary['incorrect_bandwagon_accuracy']:.2%}")
    print(f"Robustness Rate: {summary['incorrect_bandwagon_robust_rate']:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()

