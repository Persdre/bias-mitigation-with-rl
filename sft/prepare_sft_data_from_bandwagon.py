"""Prepare SFT training data from bandwagon bias parquet files

This script converts bandwagon bias data (with prompts) into SFT format:
- Input: prompt (question + options + bandwagon statement)
- Output: response with reasoning and correct answer

For incorrect_bandwagon: model should resist and give correct answer
For correct_bandwagon: model should follow and give correct answer
"""

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os

def chat_format_to_prompt(chat_messages):
    """Convert chat format (list of dicts) to prompt string"""
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
    
    return "\n".join(prompt_parts)

def generate_sft_response(question, option_a, option_b, correct_letter, bias_type, bandwagon_target=None, bandwagon_percentage=None):
    """Generate the desired SFT response based on bias type
    
    For incorrect_bandwagon: Model should resist and give correct answer
    For correct_bandwagon: Model should follow and give correct answer
    """
    correct_option = option_a if correct_letter == 'A' else option_b
    incorrect_option = option_b if correct_letter == 'A' else option_a
    
    if bias_type == 'incorrect_bandwagon':
        # Model should resist incorrect bandwagon and give correct answer
        reasoning = f"""<think>
Let me carefully analyze this question and the options provided.

Question: {question}

Option A: {option_a}
Option B: {option_b}

I notice that there's a statement suggesting that {bandwagon_percentage or 'many'} people believe {bandwagon_target} is correct. However, I should not let popular opinion influence my judgment. I need to evaluate the options based on their actual correctness, not on what others think.

Let me analyze each option:
- Option A: {option_a}
- Option B: {option_b}

After careful consideration, I find that {correct_option} (Option {correct_letter}) is the correct answer based on the facts and logical reasoning, regardless of what others might believe.
</think>"""
    elif bias_type == 'correct_bandwagon':
        # Model should follow correct bandwagon (since it's pointing to the right answer)
        reasoning = f"""<think>
Let me carefully analyze this question and the options provided.

Question: {question}

Option A: {option_a}
Option B: {option_b}

I notice that there's a statement suggesting that {bandwagon_percentage or 'many'} people believe {bandwagon_target} is correct. In this case, the consensus aligns with my own analysis.

Let me evaluate each option:
- Option A: {option_a}
- Option B: {option_b}

After careful consideration, I find that {correct_option} (Option {correct_letter}) is indeed the correct answer. The consensus view is consistent with my independent analysis.
</think>"""
    else:
        # Regular case (no bias)
        reasoning = f"""<think>
Let me carefully analyze this question and the options provided.

Question: {question}

Option A: {option_a}
Option B: {option_b}

Let me evaluate each option:
- Option A: {option_a}
- Option B: {option_b}

After careful consideration, I find that {correct_option} (Option {correct_letter}) is the correct answer based on the facts and logical reasoning.
</think>"""
    
    # Combine reasoning with answer
    response = f"{reasoning}<answer>{correct_letter}</answer>"
    return response

def prepare_sft_data(parquet_path, output_path, bias_types=None, max_samples=None, output_format='parquet'):
    """Convert bandwagon bias parquet data to SFT format
    
    Args:
        parquet_path: Path to input parquet file
        output_path: Path to output file
        bias_types: List of bias types to include (None = all)
        max_samples: Maximum number of samples to process
        output_format: 'parquet' or 'json'
    """
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples")
    
    sft_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Extract information
        extra_info = row['extra_info']
        question = extra_info.get('question', '')
        option_a_text = extra_info.get('option_a_text', '')
        option_b_text = extra_info.get('option_b_text', '')
        correct_letter = extra_info.get('correct_letter', 'A')
        bias_type = extra_info.get('bias_type', 'unknown')
        bandwagon_target = extra_info.get('bandwagon_target', None)
        bandwagon_percentage = extra_info.get('bandwagon_percentage', None)
        
        # Filter by bias type if specified
        if bias_types and bias_type not in bias_types:
            continue
        
        # Skip if essential information is missing
        if not question or not option_a_text or not option_b_text:
            continue
        
        # Get prompt from parquet (already in chat format)
        prompt_chat = row['prompt']
        if isinstance(prompt_chat, (np.ndarray, pd.Series)):
            prompt_chat = prompt_chat.tolist()
        
        # Extract user message content (the actual prompt text)
        # The prompt in parquet is in chat format, we need to extract the user content
        user_content = ""
        for msg in prompt_chat:
            if msg.get('role') == 'user':
                user_content = msg.get('content', '')
                break
        
        if not user_content:
            # Fallback: construct prompt from question and options
            user_content = f"Question: {question}\n\nOptions:\nA: {option_a_text}\nB: {option_b_text}\n\nWhich answer (A or B) is better?"
        
        # Generate desired response
        response = generate_sft_response(
            question, option_a_text, option_b_text, correct_letter,
            bias_type, bandwagon_target, bandwagon_percentage
        )
        
        # Create SFT data entry (for parquet format, verl expects 'prompt' and 'response' columns)
        sft_entry = {
            "prompt": user_content,  # Just the user content, not full chat format
            "response": response,
            "question_id": idx,
            "question": question,
            "option_a": option_a_text,
            "option_b": option_b_text,
            "correct_letter": correct_letter,
            "bias_type": bias_type,
            "bandwagon_target": bandwagon_target,
            "bandwagon_percentage": bandwagon_percentage,
            "subject": extra_info.get('subject', 'unknown')
        }
        
        sft_data.append(sft_entry)
        
        if max_samples and len(sft_data) >= max_samples:
            break
    
    # Save to file
    print(f"\nSaving {len(sft_data)} SFT samples to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_format == 'parquet':
        sft_df = pd.DataFrame(sft_data)
        sft_df.to_parquet(output_path, index=False)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "="*60)
    print("SFT Data Preparation Summary")
    print("="*60)
    print(f"Total samples: {len(sft_data)}")
    
    # Count by bias type
    bias_counts = {}
    for entry in sft_data:
        bias_type = entry['bias_type']
        bias_counts[bias_type] = bias_counts.get(bias_type, 0) + 1
    
    print("\nSamples by bias type:")
    for bias_type, count in sorted(bias_counts.items()):
        print(f"  {bias_type}: {count}")
    
    print(f"\nOutput saved to: {output_path}")
    print("="*60)
    
    return sft_data

def main():
    parser = argparse.ArgumentParser(description='Prepare SFT data from bandwagon bias parquet files')
    parser.add_argument('--input', type=str, required=True,
                        help='Input parquet file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path (parquet or json)')
    parser.add_argument('--bias_types', type=str, default=None,
                        help='Comma-separated list of bias types to include (e.g., "incorrect_bandwagon,correct_bandwagon")')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--format', type=str, default='parquet', choices=['parquet', 'json'],
                        help='Output format: parquet (for verl) or json')
    
    args = parser.parse_args()
    
    # Parse bias types
    bias_types = None
    if args.bias_types:
        bias_types = [bt.strip() for bt in args.bias_types.split(',')]
    
    prepare_sft_data(
        args.input,
        args.output,
        bias_types=bias_types,
        max_samples=args.max_samples,
        output_format=args.format
    )

if __name__ == "__main__":
    main()