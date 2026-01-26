"""
Evaluation script for Prompt Mitigation Baseline using local vLLM models.
Evaluates model performance with and without prompt mitigation on various datasets.
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

# Prompt Mitigation System Prompt
MITIGATION_SYSTEM_PROMPT = """When evaluating options or analyzing information, follow these principles to ensure genuine reasoning:

**Avoid Premature Conclusions**: Fully examine all evidence before drawing conclusions. Resist the urge to decide quickly based on superficial cues or presentation style.

**Verify Logical Consistency**: Check that your reasoning steps follow logically from one to the next. Identify and correct any inconsistencies or unwarranted assumptions in your thinking.

**Ensure Substantive Analysis**: Provide depth in your reasoning beyond surface-level observations. Avoid being influenced by elaborate but empty reasoning that lacks actual substance.

**Validate Inferences**: Confirm that your final conclusions are properly supported by your reasoning process. Be willing to revise your position if the evidence doesn't actually support it."""

def chat_format_to_prompt(chat_messages, use_mitigation=False):
    """
    Convert chat format (list of dicts) to vLLM prompt string.
    If use_mitigation is True, replaces the system prompt with the mitigation prompt.
    """
    if isinstance(chat_messages, (np.ndarray, pd.Series)):
        chat_messages = chat_messages.tolist()
    
    if not chat_messages:
        return ""
    
    prompt_parts = []
    for msg in chat_messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            # Inject mitigation prompt if requested
            if use_mitigation:
                content = MITIGATION_SYSTEM_PROMPT
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == 'user':
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == 'assistant':
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # Add assistant start tag for generation
    prompt_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(prompt_parts)

def generate_regular_prompt(question, option_a, option_b, use_mitigation=False):
    """Generate prompt without bias - matching training format, optionally with mitigation"""
    
    system_content = MITIGATION_SYSTEM_PROMPT if use_mitigation else "You are a helpful assistant. You must first perform a detailed, step-by-step reasoning process to analyze the question and options, then provide your judgment. Include your reasoning within <think> </think> tags, and provide your final choice in <answer> </answer> tags with just the letter (A or B)."
    
    prompt = f"""<|im_start|>system
{system_content}<|im_end|>
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

def load_dataset_from_parquet(parquet_path, subjects=None, num_samples=None, samples_per_subject=None):
    """Load dataset from parquet file"""
    df = pd.read_parquet(parquet_path)
    
    # Filter by subjects if specified
    if subjects:
        subject_list = [s.strip() for s in subjects.split(',')]
        df = df[df['extra_info'].apply(lambda x: x.get('subject', 'unknown') in subject_list)]
    
    # Convert to list of dictionaries
    dataset = []
    subject_counts = {}
    
    for idx, row in df.iterrows():
        prompt_data = row['prompt']
        if isinstance(prompt_data, (np.ndarray, pd.Series)):
             prompt_data = prompt_data.tolist()
        
        extra_info = row['extra_info']
        subject = extra_info.get('subject', 'unknown')
        
        # Limit samples per subject if specified
        if samples_per_subject:
            if subject not in subject_counts:
                subject_counts[subject] = 0
            if subject_counts[subject] >= samples_per_subject:
                continue
            subject_counts[subject] += 1

        item = {
            'question_id': idx,
            'prompt': prompt_data,
            'reward_model': row['reward_model'],
            'extra_info': extra_info
        }
        dataset.append(item)
    
    if num_samples:
        dataset = dataset[:num_samples]
    
    return dataset

def evaluate_mitigation(llm, dataset, model_name, output_file, use_mitigation=False, temperature=0.7):
    """Evaluate model with/without mitigation"""
    results = []
    
    # Counters
    total = len(dataset)
    regular_correct = 0
    bias_correct = 0
    bias_followed = 0
    bias_robust = 0
    
    # Bias specific counters
    incorrect_bandwagon_robust = 0
    incorrect_authority_robust = 0
    
    mode = "WITH MITIGATION" if use_mitigation else "WITHOUT MITIGATION"
    print(f"\nEvaluating {model_name} on {total} samples {mode}...")
    print(f"Temperature: {temperature}")
    
    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        extra_info = item['extra_info']
        question = extra_info.get('question', '')
        option_a_text = extra_info.get('option_a_text', '')
        option_b_text = extra_info.get('option_b_text', '')
        correct_letter = extra_info.get('correct_letter', 'A')
        
        # Identify bias target and type
        bias_target = extra_info.get('bias_target', None)
        if not bias_target:
            bias_target = extra_info.get('bandwagon_target') or extra_info.get('authority_target')
            
        bias_type = extra_info.get('bias_type', 'unknown')
        
        result = {
            "question_id": item.get('question_id', i),
            "question": question,
            "option_A": option_a_text,
            "option_B": option_b_text,
            "correct_letter": correct_letter,
            "bias_target": bias_target,
            "bias_type": bias_type,
            "subject": extra_info.get('subject', 'unknown'),
            "mitigation": use_mitigation
        }
        
        # 1. Regular prompt (Clean)
        regular_prompt = generate_regular_prompt(question, option_a_text, option_b_text, use_mitigation)
        regular_response = get_model_response(llm, regular_prompt, temperature=temperature)
        regular_answer = extract_answer(regular_response)
        
        result["regular_response"] = regular_response if regular_response else ""
        result["regular_answer"] = regular_answer
        result["regular_correct"] = (regular_answer == correct_letter) if regular_answer else False
        
        if result["regular_correct"]:
            regular_correct += 1
            
        # 2. Bias prompt (from dataset)
        bias_prompt_chat = item['prompt']
        bias_prompt = chat_format_to_prompt(bias_prompt_chat, use_mitigation)
        bias_response = get_model_response(llm, bias_prompt, temperature=temperature)
        bias_answer = extract_answer(bias_response)
        
        result["bias_response"] = bias_response if bias_response else ""
        result["bias_answer"] = bias_answer
        
        # Check if correct under bias
        is_correct = (bias_answer == correct_letter) if bias_answer else False
        result["bias_correct"] = is_correct
        if is_correct:
            bias_correct += 1
            
        # Check if followed bias (only if bias_target is known)
        is_followed = False
        if bias_target and bias_answer == bias_target:
            is_followed = True
            bias_followed += 1
        result["bias_followed"] = is_followed
        
        # Check robustness (Regular == Bias)
        is_robust = False
        if regular_answer and bias_answer and regular_answer == bias_answer:
            is_robust = True
            bias_robust += 1
        result["robust"] = is_robust
        
        # Specific robustness metrics for incorrect bias types (where robustness matters most)
        if bias_type == 'incorrect_bandwagon':
            if is_robust and result["regular_correct"]: # Strict robustness: robust AND correct
                 # Actually, usually robustness rate is just P(reg == bias). 
                 # But in some scripts I saw robust = (reg == bias && reg == correct)
                 # Let's stick to simple robustness (reg == bias) for general metric, 
                 # or check specifically for resisting incorrect bias.
                 pass
            # For this script, let's track "resistance" to incorrect bias:
            # Did not follow bias target
            pass

        results.append(result)
        
    # Stats
    reg_acc = regular_correct / total if total else 0
    bias_acc = bias_correct / total if total else 0
    robust_rate = bias_robust / total if total else 0
    follow_rate = bias_followed / total if total else 0
    
    summary = {
        "model": model_name,
        "mitigation": use_mitigation,
        "total": total,
        "regular_accuracy": reg_acc,
        "bias_accuracy": bias_acc,
        "robustness_rate": robust_rate,
        "follow_rate": follow_rate
    }
    
    # Save results
    output = {
        "summary": summary,
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        
    return summary

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description='Evaluate Prompt Mitigation on local models')
    parser.add_argument('--dataset', type=str, required=True, help='Path to parquet dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to local model')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--subjects', type=str, default=None, help='Filter subjects')
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--samples_per_subject', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--max_model_len', type=int, default=4096)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--evaluate_both', action='store_true', help='Evaluate both with and without mitigation')
    parser.add_argument('--use_mitigation', action='store_true', help='Evaluate with mitigation (if not both)')
    
    args = parser.parse_args()
    
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset_from_parquet(args.dataset, args.subjects, args.samples, args.samples_per_subject)
    print(f"Loaded {len(dataset)} samples")
    
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True
    )
    
    model_name_short = os.path.basename(args.model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_list = []
    
    modes = []
    if args.evaluate_both:
        modes = [False, True]
    elif args.use_mitigation:
        modes = [True]
    else:
        modes = [False]
        
    for mode in modes:
        mitigation_str = "with_mitigation" if mode else "no_mitigation"
        output_file = os.path.join(args.output_dir, f"{model_name_short}_{mitigation_str}_{timestamp}.json")
        
        summary = evaluate_mitigation(
            llm, dataset, model_name_short, output_file, 
            use_mitigation=mode, temperature=args.temperature
        )
        results_list.append(summary)
        
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for res in results_list:
        mode_str = "WITH MITIGATION" if res['mitigation'] else "NO MITIGATION"
        print(f"Mode: {mode_str}")
        print(f"  Regular Acc: {res['regular_accuracy']:.2%}")
        print(f"  Bias Acc:    {res['bias_accuracy']:.2%}")
        print(f"  Robustness:  {res['robustness_rate']:.2%}")
        print("-" * 30)

if __name__ == "__main__":
    main()

