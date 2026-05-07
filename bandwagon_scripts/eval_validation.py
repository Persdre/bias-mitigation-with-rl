"""Evaluation script that uses prompts directly from parquet file
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

cudnn_lib_path = os.environ.get("CUDNN_LIB_PATH", "")
if cudnn_lib_path:
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if cudnn_lib_path not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib_path}:{current_ld_path}"

from vllm import LLM, SamplingParams

def chat_format_to_prompt(chat_messages):
    """Convert chat format (list of dicts) to vLLM prompt string"""
    # FIX: Convert numpy array or pandas Series to list FIRST before checking truth value
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
    
    parser = argparse.ArgumentParser(description='Evaluate model on MMLU-Pro bandwagon bias dataset')
    parser.add_argument('--dataset', type=str, 
                        default=os.path.join(project_root, 'data/mmlupro/validation_paired_mixed_bandwagon_random.parquet'))
    parser.add_argument('--model_path', type=str, default='./models/Qwen3-4B')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'baseline/results_validation'))
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    
    # Added new arguments for Memory Fix
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
        # Use arguments instead of hardcoded values
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = os.path.basename(args.model_path) if args.model_path else "qwen3-4b"
    output_file = os.path.join(args.output_dir, f"{model_name_short}_validation_temp{args.temperature:.1f}_{timestamp}.json")
    
    summary, _ = evaluate_bandwagon_bias(
        llm, 
        dataset, 
        model_name_short, 
        output_file,
        temperature=args.temperature
    )
    
    print("\n" + "="*80)
    print(" EVALUATION RESULTS ".center(80, '='))
    print("="*80)
    print(f"Model: {summary['model']}")
    print(f"Total: {summary['total_questions']}")
    print(f"Robustness Rate: {summary['incorrect_bandwagon_robust_rate']:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()