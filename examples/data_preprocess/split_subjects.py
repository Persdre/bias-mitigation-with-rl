#!/usr/bin/env python3
"""
Split specific subjects from test.parquet into train/validation sets
95% training, 5% validation per subject
"""

import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import random

def split_subjects(input_file, output_dir, subjects, train_ratio=0.95, random_seed=42):
    """
    Split specified subjects into train/validation sets
    
    Args:
        input_file: path to input parquet file (test.parquet)
        output_dir: directory to save train.parquet and validation.parquet
        subjects: list of subject names to include
        train_ratio: ratio for training set (default 0.95)
        random_seed: random seed for reproducibility
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Total samples: {len(df)}")
    
    # Filter by subjects
    print(f"\nFiltering subjects: {subjects}")
    filtered_df = df[df['extra_info'].apply(lambda x: x.get('subject') in subjects)]
    print(f"Filtered samples: {len(filtered_df)}")
    
    # Show distribution by subject
    print("\nSubject distribution:")
    for subject in subjects:
        count = len(filtered_df[filtered_df['extra_info'].apply(lambda x: x.get('subject') == subject)])
        print(f"  {subject}: {count} samples")
    
    # Split each subject separately, then combine
    train_dfs = []
    val_dfs = []
    
    random.seed(random_seed)
    
    for subject in subjects:
        subject_df = filtered_df[filtered_df['extra_info'].apply(lambda x: x.get('subject') == subject)].copy()
        
        if len(subject_df) == 0:
            print(f"Warning: No samples found for subject '{subject}'")
            continue
        
        # Shuffle the subject data
        subject_df = subject_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Split
        n_train = int(len(subject_df) * train_ratio)
        subject_train = subject_df.iloc[:n_train].copy()
        subject_val = subject_df.iloc[n_train:].copy()
        
        print(f"\n{subject}:")
        print(f"  Training: {len(subject_train)} samples ({len(subject_train)/len(subject_df)*100:.1f}%)")
        print(f"  Validation: {len(subject_val)} samples ({len(subject_val)/len(subject_df)*100:.1f}%)")
        
        train_dfs.append(subject_train)
        val_dfs.append(subject_val)
    
    # Combine all subjects
    if not train_dfs:
        print("\nError: No data to save!")
        return
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    
    # Shuffle combined datasets
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"Final split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Total: {len(train_df) + len(val_df)} samples")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    train_file = os.path.join(output_dir, 'train.parquet')
    val_file = os.path.join(output_dir, 'validation.parquet')
    
    print(f"\nSaving training set to {train_file}...")
    train_df.to_parquet(train_file, index=False)
    print(f"Saved {len(train_df)} samples")
    
    print(f"\nSaving validation set to {val_file}...")
    val_df.to_parquet(val_file, index=False)
    print(f"Saved {len(val_df)} samples")
    
    print("\nDone!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split specific subjects into train/validation sets')
    parser.add_argument('--input', type=str, default='~/data/mmlupro/test.parquet',
                        help='Input parquet file (default: ~/data/mmlupro/test.parquet)')
    parser.add_argument('--output_dir', type=str, default='~/data/mmlupro',
                        help='Output directory for train.parquet and validation.parquet')
    parser.add_argument('--subjects', type=str, nargs='+', 
                        default=['math', 'physics', 'law', 'chemistry'],
                        help='Subjects to include (default: math physics law chemistry)')
    parser.add_argument('--train_ratio', type=float, default=0.95,
                        help='Ratio for training set (default: 0.95)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Expand user path
    input_file = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    
    split_subjects(input_file, output_dir, args.subjects, args.train_ratio, args.seed)

