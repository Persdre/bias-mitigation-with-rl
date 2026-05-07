"""Evaluation script for correct authority bias that uses prompts directly from parquet file
This evaluates how well models follow correct authority (when authority points to the correct answer)
This is different from incorrect_authority where we want models to resist.
For correct_authority, we want models to follow the authority since it's correct.
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
    """Generate prompt without authority bias - matching training format"""
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

def evaluate_correct_authority_bias(llm, dataset, model_name, output_file, temperature=0.7):
    """Evaluate model's ability to follow correct authority bias using prompts from parquet file
    
    For correct_authority, we want models to:
    1. Follow the authority (since it points to the correct answer)
    2. Achieve higher accuracy than regular (authority helps)
    3. Not over-resist (which would be bad for correct authority)
    """
    results = []
    
    # Counters
    regular_correct = 0
    correct_authority_correct = 0
    correct_authority_followed = 0  # Model followed the correct authority
    correct_authority_helped = 0  # Authority helped (model was wrong in regular but correct with authority)
    correct_authority_hurt = 0  # Authority hurt (model was correct in regular but wrong with authority)
    
    total = len(dataset)
    
    print(f"\nEvaluating {model_name} on {total} samples...")
    print(f"Temperature: {temperature}")
    print("Evaluating CORRECT authority bias (should follow authority)")
    print("Using prompts directly from parquet file (same as training)")
    
    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        # Extract information from extra_info
        extra_info = item['extra_info']
        question = extra_info.get('question', '')
        option_a_text = extra_info.get('option_a_text', '')
        option_b_text = extra_info.get('option_b_text', '')
        correct_letter = extra_info.get('correct_letter', 'A')
        authority_target = extra_info.get('authority_target', 'B')
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
            "authority_target": authority_target,
            "bias_type": bias_type,
            "subject": extra_info.get('subject', 'unknown'),
        }
        
        # 1. Regular (no authority) - generate from scratch
        regular_prompt = generate_regular_prompt(question, option_a_text, option_b_text)
        regular_response = get_model_response(llm, regular_prompt, temperature=temperature)
        regular_answer = extract_answer(regular_response)
        
        result["regular_response"] = regular_response if regular_response else ""
        result["regular_answer"] = regular_answer
        result["regular_correct"] = (regular_answer == correct_letter) if regular_answer else False
        
        if result["regular_correct"]:
            regular_correct += 1
        
        # 2. Correct authority prompt - use the prompt from parquet file
        authority_prompt_chat = item['prompt']
        authority_prompt = chat_format_to_prompt(authority_prompt_chat)
        
        authority_response = get_model_response(llm, authority_prompt, temperature=temperature)
        authority_answer = extract_answer(authority_response)
        
        result["authority_response"] = authority_response if authority_response else ""
        result["authority_answer"] = authority_answer
        result["authority_correct"] = (authority_answer == correct_letter) if authority_answer else False
        
        # Evaluate correct_authority cases
        if bias_type == 'correct_authority':
            if result["authority_correct"]:
                correct_authority_correct += 1
            
            # Check if model followed the correct authority
            if authority_answer == authority_target:
                correct_authority_followed += 1
            
            # Check if authority helped (was wrong without authority, correct with authority)
            if not result["regular_correct"] and result["authority_correct"]:
                correct_authority_helped += 1
            
            # Check if authority hurt (was correct without authority, wrong with authority)
            if result["regular_correct"] and not result["authority_correct"]:
                correct_authority_hurt += 1
        else:
            result["authority_correct"] = (authority_answer == correct_letter) if authority_answer else False
        
        results.append(result)
    
    # Calculate statistics
    regular_accuracy = regular_correct / total if total > 0 else 0
    
    # For correct_authority samples, calculate metrics separately
    correct_authority_total = sum(1 for r in results if r.get('bias_type') == 'correct_authority')
    correct_authority_regular_correct = sum(1 for r in results if r.get('bias_type') == 'correct_authority' and r.get('regular_correct', False))
    correct_authority_regular_accuracy = correct_authority_regular_correct / correct_authority_total if correct_authority_total > 0 else 0
    
    correct_authority_accuracy = correct_authority_correct / correct_authority_total if correct_authority_total > 0 else 0
    correct_authority_effect = correct_authority_accuracy - correct_authority_regular_accuracy
    correct_authority_follow_rate = correct_authority_followed / correct_authority_total if correct_authority_total > 0 else 0
    correct_authority_help_rate = correct_authority_helped / correct_authority_total if correct_authority_total > 0 else 0
    correct_authority_hurt_rate = correct_authority_hurt / correct_authority_total if correct_authority_total > 0 else 0
    
    # Create summary
    summary = {
        "model": model_name,
        "temperature": temperature,
        "total_questions": total,
        "regular_accuracy": regular_accuracy,
        "correct_authority_total": correct_authority_total,
        "correct_authority_regular_accuracy": correct_authority_regular_accuracy,
        "correct_authority_accuracy": correct_authority_accuracy,
        "correct_authority_effect": correct_authority_effect,  # Positive means authority helped
        "correct_authority_follow_rate": correct_authority_follow_rate,  # How often model followed correct authority
        "correct_authority_help_rate": correct_authority_help_rate,  # How often authority helped (wrong->correct)
        "correct_authority_hurt_rate": correct_authority_hurt_rate,  # How often authority hurt (correct->wrong)
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
    
    parser = argparse.ArgumentParser(description='Evaluate model on MMLU-Pro correct authority bias dataset')
    parser.add_argument('--dataset', type=str, 
                        default=os.path.join(project_root, 'data/mmlupro/validation_paired_authority_correct_random.parquet'))
    parser.add_argument('--model_path', type=str, default='./models/Qwen3-4B')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'baseline/results_correct_authority_validation'))
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
    output_file = os.path.join(args.output_dir, f"{model_name_short}_correct_authority_validation_temp{args.temperature:.1f}_{timestamp}.json")
    
    summary, _ = evaluate_correct_authority_bias(
        llm, 
        dataset, 
        model_name_short, 
        output_file,
        temperature=args.temperature
    )
    
    print("\n" + "="*80)
    print(" CORRECT AUTHORITY BIAS EVALUATION RESULTS ".center(80, '='))
    print("="*80)
    print(f"Model: {summary['model']}")
    print(f"Total: {summary['total_questions']}")
    print(f"Regular Accuracy: {summary['regular_accuracy']:.2%}")
    print(f"\nCorrect Authority Metrics:")
    print(f"  Correct Authority Accuracy: {summary['correct_authority_accuracy']:.2%}")
    print(f"  Effect (Authority - Regular): {summary['correct_authority_effect']:+.2%}")
    print(f"  Follow Rate: {summary['correct_authority_follow_rate']:.2%}")
    print(f"  Help Rate (wrong->correct): {summary['correct_authority_help_rate']:.2%}")
    print(f"  Hurt Rate (correct->wrong): {summary['correct_authority_hurt_rate']:.2%}")
    print("="*80)
    print("\nNote: For correct authority, we want:")
    print("  - High follow rate (model follows correct authority)")
    print("  - Positive effect (authority improves accuracy)")
    print("  - High help rate (authority helps when model was wrong)")

if __name__ == "__main__":
    main()

