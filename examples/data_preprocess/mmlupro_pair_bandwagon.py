"""
Process MMLU-Pro paired training data with bandwagon bias injection
Adds bandwagon statement after all options are presented
"""

import os
import pandas as pd
import argparse
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

def process_paired_data(input_file, output_file, percentage=90, num_samples=None):
    """
    Process paired parquet data and add bandwagon bias statements
    
    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path
        percentage: Percentage for bandwagon statement (default: 90)
        num_samples: Number of samples to process (None for all)
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Total samples: {len(df)}")
    
    # Limit samples if specified
    if num_samples is not None:
        df = df.head(num_samples)
        print(f"Processing first {num_samples} samples for testing")
    
    processed_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Get original prompt content
        original_prompt = row['prompt'][0]['content']
        
        # Get bias target from extra_info
        extra_info = row['extra_info']
        bias_target = extra_info.get('bias_target', 'B')  # Default to 'B' if not found
        
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
        new_item['extra_info']['has_bandwagon_bias'] = True
        
        processed_data.append(new_item)
    
    # Create new DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # Statistics
    print(f"\nProcessed {len(processed_df)} samples")
    
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
    parser = argparse.ArgumentParser(description='Process MMLU-Pro paired data with bandwagon bias')
    parser.add_argument('--input', type=str, required=True,
                        help='Input parquet file (e.g., train_paired.parquet)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output parquet file path')
    parser.add_argument('--percentage', type=int, default=90,
                        help='Percentage for bandwagon statement (default: 90)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    
    args = parser.parse_args()
    
    # Expand user paths
    input_file = os.path.expanduser(args.input)
    output_file = os.path.expanduser(args.output)
    
    process_paired_data(input_file, output_file, args.percentage, args.samples)

