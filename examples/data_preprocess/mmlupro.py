# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MMLU-Pro dataset to parquet format
"""

import os
import datasets
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse


def format_question_with_choices(question, choices):
    """Format question with multiple choice options"""
    formatted_choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    formatted_question = f"{question}\n\n{formatted_choices}\n\nAnswer:"
    return formatted_question


def extract_answer(answer_idx, choices):
    """Extract the answer text from answer index"""
    if isinstance(answer_idx, int):
        if 0 <= answer_idx < len(choices):
            return choices[answer_idx]
    elif isinstance(answer_idx, str):
        # Try to convert letter to index (A=0, B=1, etc.)
        if len(answer_idx) == 1 and answer_idx.isalpha():
            idx = ord(answer_idx.upper()) - ord('A')
            if 0 <= idx < len(choices):
                return choices[idx]
        # If it's already the answer text, return as is
        if answer_idx in choices:
            return answer_idx
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/mmlupro',
                        help='Local directory to save processed parquet files')
    parser.add_argument('--hdfs_dir', default=None,
                        help='HDFS directory to upload processed files (optional)')
    parser.add_argument('--cache_dir', default=None,
                        help='Cache directory for HuggingFace datasets (default: ~/.cache/huggingface/datasets)')
    parser.add_argument('--instruction', type=str, default="Let's think step by step.", 
                        help='Instruction to append to questions')
    parser.add_argument('--preview', action='store_true', 
                        help='Preview mode: only show samples without saving')
    parser.add_argument('--preview_subjects', type=str, nargs='+', default=None,
                        help='List of subjects to preview (e.g., --preview_subjects math physics)')
    parser.add_argument('--preview_num', type=int, default=3,
                        help='Number of samples to preview per subject (default: 3)')

    args = parser.parse_args()

    data_source = 'TIGER-Lab/MMLU-Pro'

    # Load dataset - MMLU-Pro may have different splits
    # Note: Dataset will be downloaded from HuggingFace and cached locally
    # Default cache location: ~/.cache/huggingface/datasets
    # You can set cache_dir to specify a custom location
    print(f"Loading dataset: {data_source}...")
    print(f"Note: Dataset will be downloaded from HuggingFace and cached locally.")
    if args.cache_dir:
        cache_dir = os.path.expanduser(args.cache_dir)
        print(f"Using custom cache directory: {cache_dir}")
        dataset = datasets.load_dataset(data_source, cache_dir=cache_dir)
    else:
        print(f"Using default cache directory: ~/.cache/huggingface/datasets")
        dataset = datasets.load_dataset(data_source)
    
    # Check available splits
    print(f"Available splits: {list(dataset.keys())}")
    
    # Use 'test' split if available, otherwise use the first available split
    if 'test' in dataset:
        test_dataset = dataset['test']
    elif 'validation' in dataset:
        test_dataset = dataset['validation']
    else:
        # Use the first available split
        first_split = list(dataset.keys())[0]
        test_dataset = dataset[first_split]
        print(f"Using split '{first_split}' as test set")

    # Check if train split exists
    if 'train' in dataset:
        train_dataset = dataset['train']
    else:
        train_dataset = None
        print("No train split found, only processing test/validation split")

    # Check dataset structure by looking at first example
    all_subjects = set()
    if len(test_dataset) > 0:
        example = test_dataset[0]
        print(f"\nDataset structure (first example keys): {list(example.keys())}")
        
        # Get all unique subjects
        for item in test_dataset:
            subject = item.get('subject', item.get('Subject', item.get('category', 'general')))
            all_subjects.add(subject)
        print(f"\nAvailable subjects ({len(all_subjects)}): {sorted(all_subjects)}")
    else:
        print("\nWarning: Test dataset is empty!")
    
    instruction_following = args.instruction
    
    # Preview mode: show samples from specified subjects
    if args.preview:
        if len(test_dataset) == 0:
            print("\nError: Cannot preview - dataset is empty!")
            exit(1)
            
        print("\n" + "="*80)
        print("PREVIEW MODE - Showing sample questions")
        print("="*80)
        
        subjects_to_preview = args.preview_subjects if args.preview_subjects else sorted(all_subjects)
        preview_num = args.preview_num
        
        for subject in subjects_to_preview:
            print(f"\n{'='*80}")
            print(f"Subject: {subject}")
            print(f"{'='*80}")
            
            count = 0
            for idx, item in enumerate(test_dataset):
                item_subject = item.get('subject', item.get('Subject', item.get('category', 'general')))
                if item_subject == subject:
                    count += 1
                    if count > preview_num:
                        break
                    
                    question_raw = item.get('question', item.get('Question', ''))
                    choices = item.get('choices', item.get('Choices', item.get('options', [])))
                    answer = item.get('answer', item.get('Answer', item.get('answer_idx', None)))
                    
                    print(f"\n[Sample {count}] (Index: {idx})")
                    # Handle multi-line questions better
                    if '\n' in question_raw:
                        print(f"Question:\n{question_raw}")
                    else:
                        print(f"Question: {question_raw}")
                    print(f"Choices:")
                    for i, choice in enumerate(choices):
                        marker = " ✓" if (isinstance(answer, int) and answer == i) or \
                                      (isinstance(answer, str) and len(answer) == 1 and 
                                       ord(answer.upper()) - ord('A') == i) else ""
                        print(f"  {chr(65+i)}. {choice}{marker}")
                    
                    # Show answer with more detail
                    if isinstance(answer, int):
                        answer_text = choices[answer] if 0 <= answer < len(choices) else f"Index {answer}"
                        answer_letter = chr(65 + answer) if 0 <= answer < 26 else "N/A"
                        print(f"Answer: {answer_letter} (index {answer}) - {answer_text}")
                    elif isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                        answer_idx = ord(answer.upper()) - ord('A')
                        answer_text = choices[answer_idx] if 0 <= answer_idx < len(choices) else "N/A"
                        print(f"Answer: {answer.upper()} (index {answer_idx}) - {answer_text}")
                    else:
                        print(f"Answer: {answer}")
                    print("-" * 80)
            
            if count == 0:
                print(f"No samples found for subject: {subject}")
            else:
                print(f"\nTotal samples shown: {min(count, preview_num)}")
        
        print("\n" + "="*80)
        print("Preview completed. Use without --preview flag to process and save data.")
        print("="*80)
        exit(0)

    def make_map_fn(split):
        def process_fn(example, idx):
            # MMLU-Pro typically has: question, choices, answer (or answer_idx)
            # Handle different possible field names
            question_raw = example.get('question', example.get('Question', ''))
            choices = example.get('choices', example.get('Choices', example.get('options', [])))
            answer = example.get('answer', example.get('Answer', example.get('answer_idx', None)))
            
            # Format question with choices
            question = format_question_with_choices(question_raw, choices)
            
            # Add instruction
            if instruction_following:
                question = question + ' ' + instruction_following
            
            # Extract ground truth answer
            if isinstance(answer, int):
                # Answer is an index
                ground_truth = choices[answer] if 0 <= answer < len(choices) else str(answer)
            elif isinstance(answer, str):
                # Answer might be a letter (A, B, C, D) or the answer text itself
                if len(answer) == 1 and answer.isalpha():
                    idx = ord(answer.upper()) - ord('A')
                    ground_truth = choices[idx] if 0 <= idx < len(choices) else answer
                else:
                    ground_truth = answer
            else:
                # Fallback: use the first choice or empty string
                ground_truth = choices[0] if choices else ""
            
            # Get subject/category if available
            subject = example.get('subject', example.get('Subject', example.get('category', 'general')))
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "reasoning",  # MMLU-Pro focuses on reasoning
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question_raw,
                    'choices': choices,
                    'answer': answer,
                    'subject': subject,
                }
            }
            return data

        return process_fn

    # Process datasets
    if train_dataset is not None:
        print("\nProcessing train dataset...")
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        print(f"Train samples: {len(train_dataset)}")
    
    print("\nProcessing test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    print(f"Test samples: {len(test_dataset)}")
    
    # Show subject distribution
    if len(test_dataset) > 0:
        subject_counts = {}
        for item in test_dataset:
            subject = item['extra_info']['subject']
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        print(f"\nSubject distribution:")
        for subject, count in sorted(subject_counts.items()):
            print(f"  {subject}: {count} samples")

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet
    print(f"\nSaving to {local_dir}...")
    if train_dataset is not None:
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
        print(f"Saved train.parquet: {len(train_dataset)} samples")
    
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Saved test.parquet: {len(test_dataset)} samples")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")

    print("\nPreprocessing completed!")
