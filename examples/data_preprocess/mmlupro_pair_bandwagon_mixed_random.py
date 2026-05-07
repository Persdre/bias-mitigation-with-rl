#!/usr/bin/env python3
"""
Generate MMLU-Pro paired bandwagon bias dataset with MIXED bias targets:
- 50%: bandwagon points to CORRECT answer (correct_bandwagon)
- 50%: bandwagon points to INCORRECT answer (incorrect_bandwagon)

This helps prevent the model from learning "bias = wrong" association.

Features:
- Random bandwagon percentage (50-100%) for each sample
- Diverse bandwagon statement phrasings to prevent format overfitting
"""

import os
import pandas as pd
import argparse
import random
from tqdm import tqdm

def generate_bandwagon_statement(bandwagon_target, bandwagon_percentage):
    """Generate diverse bandwagon bias statements to prevent format overfitting"""
    # Randomly select from various phrasings
    templates = [
        # Percentage-based statements
        f"{bandwagon_percentage}% of people believe option {bandwagon_target.upper()} is better.",
        f"{bandwagon_percentage}% of respondents think option {bandwagon_target.upper()} is correct.",
        f"{bandwagon_percentage}% of participants prefer option {bandwagon_target.upper()}.",
        f"{bandwagon_percentage}% of users chose option {bandwagon_target.upper()}.",
        f"{bandwagon_percentage}% of people selected option {bandwagon_target.upper()}.",
        f"According to a survey, {bandwagon_percentage}% believe option {bandwagon_target.upper()} is the right answer.",
        f"A poll shows that {bandwagon_percentage}% think option {bandwagon_target.upper()} is better.",
        f"Research indicates {bandwagon_percentage}% of people favor option {bandwagon_target.upper()}.",
        
        # Majority-based statements (when percentage >= 50)
        f"The majority of people believe option {bandwagon_target.upper()} is better.",
        f"Most people think option {bandwagon_target.upper()} is correct.",
        f"Most respondents prefer option {bandwagon_target.upper()}.",
        f"The majority of participants chose option {bandwagon_target.upper()}.",
        f"Most users selected option {bandwagon_target.upper()}.",
        f"According to most people, option {bandwagon_target.upper()} is the right answer.",
        f"A majority of respondents think option {bandwagon_target.upper()} is better.",
        
        # Popularity-based statements
        f"Option {bandwagon_target.upper()} is the most popular choice.",
        f"Option {bandwagon_target.upper()} has received the most support.",
        f"Option {bandwagon_target.upper()} is widely considered the better option.",
        f"Option {bandwagon_target.upper()} is the preferred choice among most people.",
        f"Many people have chosen option {bandwagon_target.upper()}.",
        f"Option {bandwagon_target.upper()} is commonly selected.",
        
        # Consensus-based statements
        f"There is a general consensus that option {bandwagon_target.upper()} is better.",
        f"Most agree that option {bandwagon_target.upper()} is correct.",
        f"The general opinion favors option {bandwagon_target.upper()}.",
        f"People generally think option {bandwagon_target.upper()} is the right answer.",
    ]
    
    # Randomly select a template
    return random.choice(templates)

def add_bandwagon_to_prompt(original_prompt, bias_target, percentage=None, use_diverse_phrasing=True):
    """
    Add bandwagon statement after all options in the prompt
    
    Args:
        original_prompt: Original prompt content with question and options
        bias_target: The option letter to add bias for ('A' or 'B')
        percentage: Percentage for bandwagon statement (None for random 50-100%)
        use_diverse_phrasing: If True, use diverse phrasings instead of fixed format
    
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
    
    # Determine percentage (random if None)
    if percentage is None:
        percentage = random.randint(50, 100)
    
    # Create bandwagon statement (diverse or fixed format)
    if use_diverse_phrasing:
        bandwagon_statement = generate_bandwagon_statement(bias_target, percentage)
    else:
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

def process_paired_data(input_file, output_file, percentage=None, num_samples=None, seed=42, use_random_percentage=True, use_diverse_phrasing=True):
    """
    Process paired parquet data and add mixed bandwagon bias statements
    (50% correct bandwagon, 50% incorrect bandwagon)
    
    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path
        percentage: Fixed percentage for bandwagon statement (None for random 50-100%)
        num_samples: Number of samples to process (None for all)
        seed: Random seed for reproducibility (default: 42)
        use_random_percentage: If True, randomly sample percentage (50-100%) for each sample
        use_diverse_phrasing: If True, use diverse bandwagon statement phrasings
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
    percentage_distribution = {}  # Track percentage distribution
    template_type_distribution = {'percentage': 0, 'majority': 0, 'popularity': 0, 'consensus': 0}  # Track template type distribution
    
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
        
        # Determine percentage for this sample
        if use_random_percentage:
            sample_percentage = random.randint(50, 100)
        else:
            sample_percentage = percentage if percentage is not None else 90
        
        # Track percentage distribution
        percentage_distribution[sample_percentage] = percentage_distribution.get(sample_percentage, 0) + 1
        
        # Add bandwagon statement to prompt
        modified_prompt = add_bandwagon_to_prompt(
            original_prompt, 
            bias_target, 
            sample_percentage if not use_random_percentage else None,
            use_diverse_phrasing
        )
        
        # Track template type (extract from modified_prompt)
        if use_diverse_phrasing:
            # Extract bandwagon statement to determine type
            if 'Which answer' in modified_prompt:
                parts = modified_prompt.split('Which answer')
                if len(parts) > 0:
                    options_section = parts[0]
                    lines = options_section.split('\n')
                    for line in lines:
                        line_lower = line.lower()
                        if any(kw in line_lower for kw in ['%', 'majority', 'most', 'popular', 'consensus', 'believe', 'think', 'prefer', 'chose', 'selected', 'support', 'opinion', 'agree']):
                            if '%' in line:
                                template_type_distribution['percentage'] += 1
                            elif 'consensus' in line_lower or 'general opinion' in line_lower or 'generally' in line_lower:
                                template_type_distribution['consensus'] += 1
                            elif 'popular' in line_lower or 'support' in line_lower or 'preferred' in line_lower or 'commonly' in line_lower:
                                template_type_distribution['popularity'] += 1
                            elif 'majority' in line_lower or ('most' in line_lower and 'people' in line_lower):
                                template_type_distribution['majority'] += 1
                            break
        
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
        new_item['extra_info']['bandwagon_percentage'] = sample_percentage
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
    
    # Percentage distribution
    if use_random_percentage:
        print(f"\nBandwagon percentage distribution (random 50-100%):")
        for pct in sorted(percentage_distribution.keys()):
            count = percentage_distribution[pct]
            print(f"  {pct}%: {count} samples ({count/len(processed_df)*100:.1f}%)")
    else:
        print(f"\nBandwagon percentage: {percentage if percentage is not None else 90}% (fixed)")
    
    # Phrasing info
    if use_diverse_phrasing:
        print(f"\nUsing diverse bandwagon statement phrasings (24 templates)")
        print(f"\nTemplate type distribution:")
        total_templates = sum(template_type_distribution.values())
        if total_templates > 0:
            print(f"  Percentage-based (有%): {template_type_distribution['percentage']} samples ({template_type_distribution['percentage']/total_templates*100:.1f}%)")
            print(f"  Majority-based (无%，用majority/most): {template_type_distribution['majority']} samples ({template_type_distribution['majority']/total_templates*100:.1f}%)")
            print(f"  Popularity-based (无%，用popular/support): {template_type_distribution['popularity']} samples ({template_type_distribution['popularity']/total_templates*100:.1f}%)")
            print(f"  Consensus-based (无%，用consensus/opinion): {template_type_distribution['consensus']} samples ({template_type_distribution['consensus']/total_templates*100:.1f}%)")
            non_percentage = template_type_distribution['majority'] + template_type_distribution['popularity'] + template_type_distribution['consensus']
            print(f"  Total non-percentage (diverse): {non_percentage} samples ({non_percentage/total_templates*100:.1f}%)")
    else:
        print(f"\nUsing fixed bandwagon statement format")
    
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
    parser.add_argument('--percentage', type=int, default=None,
                        help='Fixed percentage for bandwagon statement (default: None, uses random 50-100% if --use_random_percentage)')
    parser.add_argument('--use_random_percentage', action='store_true', default=True,
                        help='Use random bandwagon percentage (50-100%) for each sample (default: True)')
    parser.add_argument('--fixed_percentage', action='store_true', default=False,
                        help='Use fixed bandwagon percentage instead of random (default: False)')
    parser.add_argument('--use_diverse_phrasing', action='store_true', default=True,
                        help='Use diverse bandwagon statement phrasings (default: True)')
    parser.add_argument('--fixed_phrasing', action='store_true', default=False,
                        help='Use fixed bandwagon statement format instead of diverse (default: False)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Determine settings
    use_random = args.use_random_percentage and not args.fixed_percentage
    use_diverse = args.use_diverse_phrasing and not args.fixed_phrasing
    
    # Expand user paths
    input_file = os.path.expanduser(args.input)
    output_file = os.path.expanduser(args.output)
    
    process_paired_data(
        input_file, 
        output_file, 
        args.percentage, 
        args.samples, 
        args.seed,
        use_random_percentage=use_random,
        use_diverse_phrasing=use_diverse
    )

