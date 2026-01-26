"""Evaluation script for position bias that uses prompts directly from parquet file
This ensures evaluation uses the exact same prompts as training
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

def chat_format_to_prompt(chat_messages):
    """Convert chat format (list of dicts) to vLLM prompt string"""
    if isinstance(chat_messages, (np.ndarray, pd.Series)):
        chat_messages = chat_messages.tolist()
    
    if not chat_messages:
        return ""
    
    prompt_parts = []
    for msg in chat_messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == 'user':
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == 'assistant':
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # Add assistant start tag for generation
    prompt_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(prompt_parts)

def generate_regular_prompt(question, option_a, option_b):
    """Generate prompt without position bias - matching training format"""
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
    
    # Try to find <answer> tags first (highest priority)
    tag_match = re.search(r'<answer>\s*([AB])\s*</answer>', response_text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).upper()
    
    # Try other patterns
    patterns = [
        r'(?i)answer(?:\s+is)?(?:\s*:\s*|\s+)([AB])',
        r'(?i)(?:option|letter)(?:\s+is)?(?:\s*:\s*|\s+)([AB])',
        r'(?i)I(?:\s+would)?(?:\s+)(?:choose|select|pick)(?:\s+)(?:option|letter)?(?:\s*)([AB])',
        r'(?i)The(?:\s+)(?:correct|right)(?:\s+)(?:answer|option|choice)(?:\s+is)?(?:\s*:\s*|\s+)([AB])',
        r'(?i)([AB])(?:\s+is)(?:\s+the)(?:\s+)(?:correct|right)(?:\s+)(?:answer|option|choice)',
        r'(?i)(?:select|choose|pick)(?:\s+)([AB])',
        r'^([AB])$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1).upper()
    
    # Fallback: look for standalone A or B
    standalone_pattern = r'(?:^|[\s\.\,\!\?])([AB])(?:$|[\s\.\,\!\?])'
    matches = re.findall(standalone_pattern, response_text.upper())
    if matches:
        from collections import Counter
        letter_counts = Counter(matches)
        most_common = letter_counts.most_common(1)
        if most_common:
            return most_common[0][0]
    
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

def load_dataset_from_parquet(parquet_path, num_samples=None):
    """Load dataset from parquet file"""
    df = pd.read_parquet(parquet_path)
    
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

def evaluate_position_bias(llm, dataset, model_name, output_file, temperature=0.7):
    """Evaluate model's robustness to position bias using prompts from parquet file
    Tests three scenarios:
    1. Random: correct answer appears in random position (baseline)
    2. Correct-in-A: correct answer always in position A
    3. Correct-in-B: correct answer always in position B
    """
    results = []
    
    # Counters for different position bias types
    random_correct = 0
    correct_in_a_correct = 0
    correct_in_b_correct = 0
    
    random_total = 0
    correct_in_a_total = 0
    correct_in_b_total = 0
    
    random_robust = 0
    correct_in_a_robust = 0
    correct_in_b_robust = 0
    
    total = len(dataset)
    
    print(f"\nEvaluating {model_name} on {total} samples...")
    print(f"Temperature: {temperature}")
    print("Testing three position bias scenarios: Random, Correct-in-A, Correct-in-B")
    
    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        # Extract information from extra_info
        extra_info = item['extra_info']
        question = extra_info.get('question', '')
        option_a_text = extra_info.get('option_a_text', '')
        option_b_text = extra_info.get('option_b_text', '')
        correct_letter = extra_info.get('correct_letter', 'A')
        bias_type = extra_info.get('bias_type', 'unknown')
        position_type = extra_info.get('position_type', 'random')  # 'random', 'correct_in_a', 'correct_in_b'
        
        # Skip if essential information is missing
        if not question or not option_a_text or not option_b_text:
            continue
        
        result = {
            "question_id": item.get('question_id', i),
            "question": question,
            "option_A": option_a_text,
            "option_B": option_b_text,
            "correct_letter": correct_letter,
            "position_type": position_type,
            "bias_type": bias_type,
            "subject": extra_info.get('subject', 'unknown'),
        }
        
        # 1. Regular (baseline) - always use original order (A: option_a, B: option_b)
        regular_prompt = generate_regular_prompt(question, option_a_text, option_b_text)
        regular_response = get_model_response(llm, regular_prompt, temperature=temperature)
        regular_answer = extract_answer(regular_response)
        
        result["regular_response"] = regular_response if regular_response else ""
        result["regular_answer"] = regular_answer
        result["regular_correct"] = (regular_answer == correct_letter) if regular_answer else False
        
        # 2. Position bias prompt - use the prompt from parquet file (may have reordered options)
        position_prompt_chat = item['prompt']
        position_prompt = chat_format_to_prompt(position_prompt_chat)
        
        position_response = get_model_response(llm, position_prompt, temperature=temperature)
        position_answer = extract_answer(position_response)
        
        result["position_response"] = position_response if position_response else ""
        result["position_answer"] = position_answer
        result["position_correct"] = (position_answer == correct_letter) if position_answer else False
        
        # Determine which position the correct answer is in the biased prompt
        # Check position_type from extra_info first (most reliable)
        if position_type == 'correct_in_a':
            correct_position = 'A'
        elif position_type == 'correct_in_b':
            correct_position = 'B'
        elif position_type == 'random':
            correct_position = 'random'
        else:
            # Fallback: try to extract from prompt text
            prompt_text = str(position_prompt_chat)
            # Check if we can determine from the prompt structure
            # This is a heuristic - ideally position_type should be set correctly in data
            if 'A:' in prompt_text and 'B:' in prompt_text:
                # Try to match option text to determine which is correct
                if option_a_text in prompt_text and option_b_text in prompt_text:
                    a_idx = prompt_text.find('A:')
                    b_idx = prompt_text.find('B:')
                    # Find which option text appears first after its label
                    a_text_start = prompt_text.find(option_a_text, a_idx)
                    b_text_start = prompt_text.find(option_b_text, b_idx)
                    # If correct_letter is A and A appears before B, or vice versa
                    if correct_letter == 'A' and a_text_start < b_text_start:
                        correct_position = 'A'
                    elif correct_letter == 'B' and b_text_start < a_text_start:
                        correct_position = 'B'
                    else:
                        correct_position = 'random'
                else:
                    correct_position = 'random'
            else:
                correct_position = 'random'
        
        result["correct_position"] = correct_position
        result["position_type"] = position_type
        
        # Categorize and count based on position type
        if position_type == 'random' or correct_position == 'random':
            random_total += 1
            if result["position_correct"]:
                random_correct += 1
            # Robustness: same answer as regular and correct
            if regular_answer == position_answer and regular_answer == correct_letter:
                random_robust += 1
        elif position_type == 'correct_in_a' or correct_position == 'A':
            correct_in_a_total += 1
            if result["position_correct"]:
                correct_in_a_correct += 1
            # Robustness: same answer as regular and correct
            if regular_answer == position_answer and regular_answer == correct_letter:
                correct_in_a_robust += 1
        elif position_type == 'correct_in_b' or correct_position == 'B':
            correct_in_b_total += 1
            if result["position_correct"]:
                correct_in_b_correct += 1
            # Robustness: same answer as regular and correct
            if regular_answer == position_answer and regular_answer == correct_letter:
                correct_in_b_robust += 1
        
        results.append(result)
    
    # Calculate statistics
    regular_correct_total = sum(1 for r in results if r.get('regular_correct', False))
    regular_accuracy = regular_correct_total / total if total > 0 else 0
    
    # Calculate metrics for each position type
    random_accuracy = random_correct / random_total if random_total > 0 else 0
    random_robust_rate = random_robust / random_total if random_total > 0 else 0
    
    correct_in_a_accuracy = correct_in_a_correct / correct_in_a_total if correct_in_a_total > 0 else 0
    correct_in_a_robust_rate = correct_in_a_robust / correct_in_a_total if correct_in_a_total > 0 else 0
    
    correct_in_b_accuracy = correct_in_b_correct / correct_in_b_total if correct_in_b_total > 0 else 0
    correct_in_b_robust_rate = correct_in_b_robust / correct_in_b_total if correct_in_b_total > 0 else 0
    
    # Overall position bias metrics (combining all types)
    position_bias_total = random_total + correct_in_a_total + correct_in_b_total
    position_bias_correct = random_correct + correct_in_a_correct + correct_in_b_correct
    position_bias_accuracy = position_bias_correct / position_bias_total if position_bias_total > 0 else 0
    position_bias_robust = random_robust + correct_in_a_robust + correct_in_b_robust
    position_bias_robust_rate = position_bias_robust / position_bias_total if position_bias_total > 0 else 0
    
    # Create summary
    summary = {
        "model": model_name,
        "temperature": temperature,
        "total_questions": total,
        "regular_accuracy": regular_accuracy,
        "position_bias_total": position_bias_total,
        "position_bias_accuracy": position_bias_accuracy,
        "position_bias_robust_rate": position_bias_robust_rate,
        # Detailed breakdown by position type
        "random": {
            "total": random_total,
            "accuracy": random_accuracy,
            "robust_rate": random_robust_rate
        },
        "correct_in_a": {
            "total": correct_in_a_total,
            "accuracy": correct_in_a_accuracy,
            "robust_rate": correct_in_a_robust_rate
        },
        "correct_in_b": {
            "total": correct_in_b_total,
            "accuracy": correct_in_b_accuracy,
            "robust_rate": correct_in_b_robust_rate
        }
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
    
    parser = argparse.ArgumentParser(description='Evaluate model on MMLU-Pro position bias dataset')
    parser.add_argument('--dataset', type=str, 
                        default=os.path.join(project_root, 'data/mmlupro/validation_paired_position_bias_random.parquet'))
    parser.add_argument('--model_path', type=str, default='/shared/hdd/nuochen/models/Qwen3-4B')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'baseline/results_position_validation'))
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    
    parser.add_argument('--max_model_len', type=int, default=8192, help='Max context length for vLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help='GPU memory util for vLLM')
    
    args = parser.parse_args()
    
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"Using GPUs: {args.gpu_ids}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset_from_parquet(args.dataset, args.samples)
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
    output_file = os.path.join(args.output_dir, f"{model_name_short}_position_validation_temp{args.temperature:.1f}_{timestamp}.json")
    
    summary, _ = evaluate_position_bias(
        llm, 
        dataset, 
        model_name_short, 
        output_file,
        temperature=args.temperature
    )
    
    print("\n" + "="*80)
    print(" POSITION BIAS EVALUATION RESULTS ".center(80, '='))
    print("="*80)
    print(f"Model: {summary['model']}")
    print(f"Total: {summary['total_questions']}")
    print(f"Regular Accuracy: {summary['regular_accuracy']:.2%}")
    print(f"\nOverall Position Bias:")
    print(f"  Accuracy: {summary['position_bias_accuracy']:.2%}")
    print(f"  Robustness Rate: {summary['position_bias_robust_rate']:.2%}")
    print(f"\nBreakdown by Position Type:")
    print(f"  Random (n={summary['random']['total']}):")
    print(f"    Accuracy: {summary['random']['accuracy']:.2%}, Robustness: {summary['random']['robust_rate']:.2%}")
    print(f"  Correct-in-A (n={summary['correct_in_a']['total']}):")
    print(f"    Accuracy: {summary['correct_in_a']['accuracy']:.2%}, Robustness: {summary['correct_in_a']['robust_rate']:.2%}")
    print(f"  Correct-in-B (n={summary['correct_in_b']['total']}):")
    print(f"    Accuracy: {summary['correct_in_b']['accuracy']:.2%}, Robustness: {summary['correct_in_b']['robust_rate']:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()

