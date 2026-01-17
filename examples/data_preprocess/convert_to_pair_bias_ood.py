#!/usr/bin/env python3
"""
Convert multi-choice questions to A/B pairs (one correct, one incorrect)
for OUT-OF-DOMAIN evaluation subjects from MMLU-Pro dataset

This script converts original test.parquet (with 10 options) to paired format (2 options)
for subjects that were NOT used in training, enabling out-of-domain evaluation.
"""

import pandas as pd
import argparse
import os
import random
from datetime import datetime

def extract_answer_index(answer, choices):
    """Extract answer index from answer field (could be letter or index)"""
    if isinstance(answer, int):
        return answer if 0 <= answer < len(choices) else None
    elif isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
        idx = ord(answer.upper()) - ord('A')
        return idx if 0 <= idx < len(choices) else None
    return None

def format_pair_question(question_text, option_a_text, option_b_text):
    """Format question with two options (A and B)"""
    formatted_options = f"A. {option_a_text}\nB. {option_b_text}\n"
    
    formatted_prompt = (
        f"Question: {question_text}\n\n"
        f"Options:\n{formatted_options}\n"
        f"Please select the correct answer by responding with only the letter of the correct option (A or B)."
    )
    
    return formatted_prompt

def extract_subject_from_row(row):
    """Extract subject from a row, handling different data formats"""
    # Try category field (for original test.parquet format)
    if 'category' in row:
        return row['category']
    
    # Try extra_info
    if 'extra_info' in row and isinstance(row['extra_info'], dict):
        subject = row['extra_info'].get('subject', None)
        if subject:
            return subject
    
    # Try subject field directly
    if 'subject' in row:
        return row['subject']
    
    return 'unknown'

def convert_to_pairs_ood(input_file, output_file, subjects=None, seed=42):
    """
    Convert multi-choice questions to A/B pairs for OUT-OF-DOMAIN subjects
    
    Args:
        input_file: input parquet file (e.g., test.parquet with original format)
        output_file: output parquet file path
        subjects: list of subjects to convert (if None, converts all subjects)
        seed: random seed for reproducibility
    """
    random.seed(seed)
    
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Total samples: {len(df)}")
    
    # Get all available subjects
    all_subjects = df.apply(extract_subject_from_row, axis=1).value_counts()
    print(f"\nAvailable subjects in dataset ({len(all_subjects)} total):")
    for subject, count in sorted(all_subjects.items()):
        marker = " ← SELECTED" if (subjects is None or subject in subjects) else ""
        print(f"  {subject}: {count} samples{marker}")
    
    # Filter by subjects if specified
    if subjects is not None:
        if isinstance(subjects, str):
            subject_list = [s.strip() for s in subjects.split(',')]
        else:
            subject_list = subjects
        
        print(f"\nFiltering by subjects: {subject_list}")
        mask = df.apply(lambda row: extract_subject_from_row(row) in subject_list, axis=1)
        df = df[mask].reset_index(drop=True)
        print(f"Filtered to {len(df)} samples")
    
    converted_data = []
    
    for idx, row in df.iterrows():
        # Extract subject
        subject = extract_subject_from_row(row)
        
        # Extract question, choices, and answer from different possible formats
        # Format 1: Original test.parquet format (has question, options, answer directly)
        if 'question' in row and 'options' in row:
            question_text = row['question']
            choices = row['options']
            answer = row.get('answer', row.get('answer_index', None))
        # Format 2: Has extra_info with question, choices, answer
        elif 'extra_info' in row and isinstance(row['extra_info'], dict):
            extra_info = row['extra_info']
            question_text = extra_info.get('question', '')
            choices = extra_info.get('choices', [])
            answer = extra_info.get('answer', None)
        else:
            print(f"Warning: Skipping sample {idx} - cannot find question/choices/answer")
            continue
        
        # Convert choices to list if it's numpy array or other format
        if hasattr(choices, 'tolist'):
            choices = choices.tolist()
        elif not isinstance(choices, list):
            choices = list(choices)
        
        if len(choices) < 2:
            print(f"Warning: Skipping sample {idx} - less than 2 choices")
            continue
        
        # Extract correct answer index
        correct_idx = extract_answer_index(answer, choices)
        if correct_idx is None or correct_idx >= len(choices):
            print(f"Warning: Skipping sample {idx} - invalid answer index")
            continue
        
        correct_answer_text = choices[correct_idx]
        
        # Select a random incorrect answer
        incorrect_indices = [i for i in range(len(choices)) if i != correct_idx]
        if not incorrect_indices:
            print(f"Warning: Skipping sample {idx} - no incorrect options")
            continue
        
        incorrect_idx = random.choice(incorrect_indices)
        incorrect_answer_text = choices[incorrect_idx]
        
        # Randomly shuffle the order to avoid position bias
        options = [correct_answer_text, incorrect_answer_text]
        correct_is_a = random.choice([True, False])
        
        if correct_is_a:
            option_a = correct_answer_text
            option_b = incorrect_answer_text
            ground_truth_letter = 'A'
        else:
            option_a = incorrect_answer_text
            option_b = correct_answer_text
            ground_truth_letter = 'B'
        
        # Format the prompt
        prompt_content = format_pair_question(question_text, option_a, option_b)
        
        # Get question_id
        question_id = row.get('question_id', idx)
        
        # Build new data structure (matching format from convert_to_pair_bias.py)
        new_data = {
            "data_source": row.get('data_source', 'mmlupro_paired_ood'),
            "prompt": [{
                "role": "user",
                "content": prompt_content
            }],
            "ability": row.get('ability', 'reasoning'),
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth_letter  # A or B
            },
            "extra_info": {
                'split': 'test_ood',
                'index': idx,
                'original_index': question_id,
                'question': question_text,
                'original_choices': choices,
                'original_answer': answer,
                'original_correct_idx': correct_idx,
                'original_correct_text': correct_answer_text,
                'selected_incorrect_idx': incorrect_idx,
                'selected_incorrect_text': incorrect_answer_text,
                'subject': subject,
                'option_a_text': option_a,
                'option_b_text': option_b,
                'correct_letter': ground_truth_letter,
                'correct_is_a': correct_is_a,
                'bias_target': 'B' if correct_is_a else 'A'  # The incorrect option for bias injection
            }
        }
        
        # Copy other fields from original if they exist
        if 'extra_info' in row and isinstance(row['extra_info'], dict):
            extra_info = row['extra_info']
            for key in ['category', 'src', 'cot_content']:
                if key in extra_info:
                    new_data['extra_info'][key] = extra_info[key]
        else:
            # Copy directly from row if exists
            for key in ['category', 'src', 'cot_content']:
                if key in row:
                    new_data['extra_info'][key] = row[key]
        
        converted_data.append(new_data)
    
    # Create new DataFrame
    converted_df = pd.DataFrame(converted_data)
    
    print(f"\nConverted {len(converted_df)} samples to A/B pairs")
    
    # Statistics
    if len(converted_data) > 0:
        correct_as_a = sum(1 for item in converted_data if item['extra_info']['correct_is_a'])
        correct_as_b = len(converted_data) - correct_as_a
        print(f"  Correct answer as A: {correct_as_a} ({correct_as_a/len(converted_data)*100:.1f}%)")
        print(f"  Correct answer as B: {correct_as_b} ({correct_as_b/len(converted_data)*100:.1f}%)")
        
        # Subject distribution
        print("\nSubject distribution:")
        subject_counts = {}
        for item in converted_data:
            subj = item['extra_info']['subject']
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
        for subj, count in sorted(subject_counts.items()):
            print(f"  {subj}: {count} samples")
    
    # Save
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving to {output_file}...")
    converted_df.to_parquet(output_file, index=False)
    print(f"Saved {len(converted_df)} samples")
    
    print("\nDone!")
    return converted_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert multi-choice questions to A/B pairs for OUT-OF-DOMAIN evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all subjects from test.parquet
  python3 convert_to_pair_bias_ood.py --input test.parquet --output test_paired_ood.parquet

  # Convert specific subjects (out-of-domain)
  python3 convert_to_pair_bias_ood.py --input test.parquet --output biology_paired.parquet --subjects biology,psychology

  # Convert all subjects except training subjects (math, physics, chemistry, law)
  python3 convert_to_pair_bias_ood.py --input test.parquet --output ood_paired.parquet \\
      --subjects biology,business,computer science,economics,engineering,health,history,other,philosophy,psychology
        """
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input parquet file (e.g., test.parquet with original format)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output parquet file path')
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma-separated list of subjects to convert (default: all subjects)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Expand user paths
    input_file = os.path.expanduser(args.input)
    output_file = os.path.expanduser(args.output)
    
    convert_to_pairs_ood(input_file, output_file, args.subjects, args.seed)

