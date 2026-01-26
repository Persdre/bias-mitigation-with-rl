"""Evaluation script for correct distraction bias that uses prompts directly from parquet file
This evaluates how well models follow correct distraction (when distraction points to the correct answer)
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
    """Generate prompt without distraction bias - matching training format"""
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
        r'I(?:\s+WOULD)?(?:\s+)(?:CHOOSE|SELECT|PICK)(?:\s+)(?:OPTION|LETTER)?(?:\s*)([AB])\b',
        r'THE(?:\s+)(?:CORRECT|RIGHT)(?:\s+)(?:ANSWER|OPTION|CHOICE)(?:\s+IS)?(?:\s*:\s*|\s+)([AB])\b',
        r'([AB])(?:\s+IS)(?:\s+THE)(?:\s+)(?:CORRECT|RIGHT)(?:\s+)(?:ANSWER|OPTION|CHOICE)',
        r'(?:SELECT|CHOOSE|PICK)(?:\s+)([AB])\b',
        r'^([AB])$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1).upper()
    
    # Strategy 3: Standalone A or B
    standalone_pattern = r'(?:^|[\s\.\,\!\?])([AB])(?:$|[\s\.\,\!\?])'
    matches = re.findall(standalone_pattern, text_upper)
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

def evaluate_correct_distraction_bias(llm, dataset, model_name, output_file, temperature=0.7):
    """Evaluate model's ability to follow correct distraction bias using prompts from parquet file"""
    results = []
    
    # Counters
    regular_correct = 0
    correct_distraction_correct = 0
    correct_distraction_followed = 0  
    correct_distraction_helped = 0  
    correct_distraction_hurt = 0  
    
    total = len(dataset)
    
    print(f"\nEvaluating {model_name} on {total} samples...")
    print(f"Temperature: {temperature}")
    print("Evaluating CORRECT distraction bias (should follow correct distraction)")
    print("Using prompts directly from parquet file (same as training)")
    
    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        # Extract information from extra_info
        extra_info = item['extra_info']
        question = extra_info.get('question', '')
        option_a_text = extra_info.get('option_a_text', '')
        option_b_text = extra_info.get('option_b_text', '')
        correct_letter = extra_info.get('correct_letter', 'A')
        distraction_target = extra_info.get('distraction_target', 'B')
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
            "distraction_target": distraction_target,
            "bias_type": bias_type,
            "subject": extra_info.get('subject', 'unknown'),
        }
        
        # 1. Regular (no distraction) - generate from scratch
        regular_prompt = generate_regular_prompt(question, option_a_text, option_b_text)
        regular_response = get_model_response(llm, regular_prompt, temperature=temperature)
        regular_answer = extract_answer(regular_response)
        
        result["regular_response"] = regular_response if regular_response else ""
        result["regular_answer"] = regular_answer
        result["regular_correct"] = (regular_answer == correct_letter) if regular_answer else False
        
        if result["regular_correct"]:
            regular_correct += 1
        
        # 2. Correct distraction prompt - use the prompt from parquet file
        distraction_prompt_chat = item['prompt']
        distraction_prompt = chat_format_to_prompt(distraction_prompt_chat)
        
        distraction_response = get_model_response(llm, distraction_prompt, temperature=temperature)
        distraction_answer = extract_answer(distraction_response)
        
        result["distraction_response"] = distraction_response if distraction_response else ""
        result["distraction_answer"] = distraction_answer
        result["distraction_correct"] = (distraction_answer == correct_letter) if distraction_answer else False
        
        # Evaluate correct_distraction cases
        if bias_type == 'correct_distraction':
            if result["distraction_correct"]:
                correct_distraction_correct += 1
            
            # Check if model followed the correct distraction
            if distraction_answer == distraction_target:
                correct_distraction_followed += 1
            
            # Check if distraction helped (was wrong without bias, correct with bias)
            if not result["regular_correct"] and result["distraction_correct"]:
                correct_distraction_helped += 1
            
            # Check if distraction hurt (was correct without bias, wrong with bias)
            if result["regular_correct"] and not result["distraction_correct"]:
                correct_distraction_hurt += 1
        
        results.append(result)
    
    # Calculate statistics
    regular_accuracy = regular_correct / total if total > 0 else 0
    
    # For correct_distraction samples, calculate metrics separately
    correct_distraction_total = sum(1 for r in results if r.get('bias_type') == 'correct_distraction')
    correct_distraction_regular_correct = sum(1 for r in results if r.get('bias_type') == 'correct_distraction' and r.get('regular_correct', False))
    correct_distraction_regular_accuracy = correct_distraction_regular_correct / correct_distraction_total if correct_distraction_total > 0 else 0
    
    correct_distraction_accuracy = correct_distraction_correct / correct_distraction_total if correct_distraction_total > 0 else 0
    correct_distraction_effect = correct_distraction_accuracy - correct_distraction_regular_accuracy
    correct_distraction_follow_rate = correct_distraction_followed / correct_distraction_total if correct_distraction_total > 0 else 0
    correct_distraction_help_rate = correct_distraction_helped / correct_distraction_total if correct_distraction_total > 0 else 0
    correct_distraction_hurt_rate = correct_distraction_hurt / correct_distraction_total if correct_distraction_total > 0 else 0
    
    # Create summary
    summary = {
        "model": model_name,
        "temperature": temperature,
        "total_questions": total,
        "regular_accuracy": regular_accuracy,
        "correct_distraction_total": correct_distraction_total,
        "correct_distraction_regular_accuracy": correct_distraction_regular_accuracy,
        "correct_distraction_accuracy": correct_distraction_accuracy,
        "correct_distraction_effect": correct_distraction_effect,  # Positive means bias helped
        "correct_distraction_follow_rate": correct_distraction_follow_rate,
        "correct_distraction_help_rate": correct_distraction_help_rate,
        "correct_distraction_hurt_rate": correct_distraction_hurt_rate,
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
    
    parser = argparse.ArgumentParser(description='Evaluate model on MMLU-Pro correct distraction bias dataset')
    parser.add_argument('--dataset', type=str, 
                        default=os.path.join(project_root, 'data/mmlupro/validation_paired_distraction_correct_random.parquet'))
    parser.add_argument('--model_path', type=str, default='/shared/hdd/nuochen/models/Qwen3-4B')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'baseline/results_correct_distraction_validation'))
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
    output_file = os.path.join(args.output_dir, f"{model_name_short}_correct_distraction_validation_temp{args.temperature:.1f}_{timestamp}.json")
    
    summary, _ = evaluate_correct_distraction_bias(
        llm, 
        dataset, 
        model_name_short, 
        output_file,
        temperature=args.temperature
    )
    
    print("\n" + "="*80)
    print(" CORRECT DISTRACTION BIAS EVALUATION RESULTS ".center(80, '='))
    print("="*80)
    print(f"Model: {summary['model']}")
    print(f"Total: {summary['total_questions']}")
    print(f"Regular Accuracy: {summary['regular_accuracy']:.2%}")
    print(f"\nCorrect Distraction Metrics:")
    print(f"  Correct Distraction Accuracy: {summary['correct_distraction_accuracy']:.2%}")
    print(f"  Effect (Distraction - Regular): {summary['correct_distraction_effect']:+.2%}")
    print(f"  Follow Rate: {summary['correct_distraction_follow_rate']:.2%}")
    print(f"  Help Rate (wrong->correct): {summary['correct_distraction_help_rate']:.2%}")
    print(f"  Hurt Rate (correct->wrong): {summary['correct_distraction_hurt_rate']:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()

