#!/usr/bin/env python3
"""
Generate MMLU-Pro paired bandwagon bias dataset with MIXED bias targets:
- 50%: bandwagon points to CORRECT answer (correct_bandwagon)
- 50%: bandwagon points to INCORRECT answer (incorrect_bandwagon)

This helps prevent the model from learning "bias = wrong" association.
"""

import os
import pandas as pd
import argparse
import random
from tqdm import tqdm

def add_bandwagon_to_prompt(original_prompt, bias_target, percentage=90):
    """
    Add bandwagon statement after all options in the prompt
    
    Args:
        original_prompt: Original prompt content with question and options
        bias_target: The option letter to add bias for ('A' or 'B')
        percentage: Percentage for bandwagon statement (default: 90)
    
    Returns:
        Modified prompt with bandwagon statement
    """
    lines = original_prompt.strip().split('\n')
    
    # Find where the options end (look for "Please select" or similar)
    option_end_idx = len(lines)
    for i, line in enumerate(lines):
        if 'Please select' in line or (line.strip().startswith('Please') and 'answer' in line.lower()):
            option_end_idx = i
            break
    
    # Get the part before options end
    prompt_parts = lines[:option_end_idx]
    
    # Convert format from "A. " to "A: " if needed
    modified_parts = []
    for line in prompt_parts:
        # Remove "Options:" line if exists
        if line.strip().lower() == 'options:':
            continue
        # Convert "A. " to "A: " and "B. " to "B: "
        if line.strip().startswith('A. '):
            line = line.replace('A. ', 'A: ', 1)
        elif line.strip().startswith('B. '):
            line = line.replace('B. ', 'B: ', 1)
        modified_parts.append(line)
    
    # Reconstruct prompt with options
    prompt_with_options = '\n'.join(modified_parts).strip()
    
    # Create bandwagon statement
    bandwagon_statement = f"{percentage}% of people believe option {bias_target.upper()} is better."
    
    # Add bandwagon statement and final question with reasoning format
    # Format: question + options + bandwagon + selection question with reasoning requirement
    modified_prompt = f"""{prompt_with_options}
{bandwagon_statement}

Which answer (A or B) is better?

IMPORTANT: You must first perform a detailed, step-by-step reasoning process, then provide your final choice.

Include your reasoning process within <think> </think> tags, explaining:
- How you evaluate each option's correctness
- The logical reasoning behind each option
- Which answer is more accurate and why

After your reasoning, provide your final choice in <answer> tags: choose A or B based on which is better.

Example format: <think>Your detailed reasoning here...</think><answer>A</answer>"""
    
    return modified_prompt

def process_paired_data(input_file, output_file, percentage=90, num_samples=None, seed=42):
    """
    Process paired parquet data and add mixed bandwagon bias statements
    (50% correct bandwagon, 50% incorrect bandwagon)
    
    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path
        percentage: Percentage for bandwagon statement (default: 90)
        num_samples: Number of samples to process (None for all)
        seed: Random seed for reproducibility (default: 42)
    """
    random.seed(seed)
    
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Total samples: {len(df)}")
    
    # Limit samples if specified
    if num_samples is not None:
        df = df.head(num_samples)
        print(f"Processing first {num_samples} samples for testing")
    
    processed_data = []
    
    # Counters
    correct_bandwagon_count = 0
    incorrect_bandwagon_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Get original prompt content
        original_prompt = row['prompt'][0]['content']
        
        # Get information from extra_info
        extra_info = row['extra_info']
        correct_letter = extra_info.get('correct_letter', 'A')
        
        # Randomly decide: 50% correct bandwagon, 50% incorrect bandwagon
        is_correct_bandwagon = random.random() < 0.5
        
        if is_correct_bandwagon:
            # Bandwagon points to CORRECT answer
            bias_target = correct_letter
            bias_type = 'correct_bandwagon'
            correct_bandwagon_count += 1
        else:
            # Bandwagon points to INCORRECT answer
            # Find incorrect option (opposite of correct_letter)
            if correct_letter == 'A':
                bias_target = 'B'
            else:
                bias_target = 'A'
            bias_type = 'incorrect_bandwagon'
            incorrect_bandwagon_count += 1
        
        # Add bandwagon statement to prompt
        modified_prompt = add_bandwagon_to_prompt(original_prompt, bias_target, percentage)
        
        # Create new data item
        new_item = row.to_dict()  # Copy all fields
        
        # Update prompt content with system prompt
        new_item['prompt'] = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You must first perform a detailed, step-by-step reasoning process to analyze the question and options, then provide your judgment. Include your reasoning within <think> </think> tags, and provide your final choice in <answer> </answer> tags with just the letter (A or B)."
            },
            {
                "role": "user",
                "content": modified_prompt
            }
        ]
        
        # Update extra_info with bandwagon info
        new_item['extra_info'] = extra_info.copy()
        new_item['extra_info']['bandwagon_percentage'] = percentage
        new_item['extra_info']['bandwagon_target'] = bias_target
        new_item['extra_info']['bias_type'] = bias_type
        new_item['extra_info']['has_bandwagon_bias'] = True
        
        processed_data.append(new_item)
    
    # Create new DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # Statistics
    print(f"\nProcessed {len(processed_df)} samples")
    print(f"  Correct bandwagon (bias points to correct answer): {correct_bandwagon_count} samples ({correct_bandwagon_count/len(processed_df)*100:.1f}%)")
    print(f"  Incorrect bandwagon (bias points to incorrect answer): {incorrect_bandwagon_count} samples ({incorrect_bandwagon_count/len(processed_df)*100:.1f}%)")
    
    # Count bias targets
    bias_a_count = sum(1 for item in processed_data if item['extra_info']['bandwagon_target'] == 'A')
    bias_b_count = sum(1 for item in processed_data if item['extra_info']['bandwagon_target'] == 'B')
    print(f"  Bandwagon bias on option A: {bias_a_count} samples")
    print(f"  Bandwagon bias on option B: {bias_b_count} samples")
    
    # Subject distribution
    print("\nSubject distribution:")
    subject_counts = {}
    for item in processed_data:
        subject = item['extra_info'].get('subject', 'unknown')
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count} samples")
    
    # Save
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving to {output_file}...")
    processed_df.to_parquet(output_file, index=False)
    print(f"Saved {len(processed_df)} samples")
    
    print("\nDone!")
    return processed_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MMLU-Pro paired data with MIXED bandwagon bias (50% correct, 50% incorrect)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input parquet file (e.g., train_paired.parquet)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output parquet file path')
    parser.add_argument('--percentage', type=int, default=90,
                        help='Percentage for bandwagon statement (default: 90)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Expand user paths
    input_file = os.path.expanduser(args.input)
    output_file = os.path.expanduser(args.output)
    
    process_paired_data(input_file, output_file, args.percentage, args.samples, args.seed)

