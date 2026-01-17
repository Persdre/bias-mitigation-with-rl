#!/usr/bin/env python3
"""
Inspect parquet file format and structure
"""

import pandas as pd
import json
import argparse
from pprint import pprint

def inspect_parquet(file_path, num_samples=3):
    """Inspect parquet file structure and show sample data"""
    
    print("="*80)
    print(f"Inspecting: {file_path}")
    print("="*80)
    
    # Load parquet file
    df = pd.read_parquet(file_path)
    
    print(f"\n📊 Basic Information:")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Column names: {list(df.columns)}")
    
    print(f"\n📋 Column Data Types:")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"  {col}: {dtype}")
    
    print(f"\n🔍 Sample Data (first {num_samples} rows):")
    print("="*80)
    
    for idx in range(min(num_samples, len(df))):
        print(f"\n--- Row {idx} ---")
        row = df.iloc[idx]
        
        for col in df.columns:
            print(f"\n[{col}]")
            value = row[col]
            
            # Handle different data types
            if isinstance(value, (dict, list)):
                # Pretty print JSON-like structures
                print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
            elif isinstance(value, str):
                # Truncate long strings
                if len(value) > 500:
                    print(f"{value[:500]}... (truncated, total length: {len(value)})")
                else:
                    print(value)
            else:
                print(value)
    
    print(f"\n📝 Detailed Column Analysis:")
    print("="*80)
    
    for col in df.columns:
        print(f"\nColumn: {col}")
        print(f"  Type: {df[col].dtype}")
        print(f"  Non-null count: {df[col].notna().sum()}/{len(df)}")
        print(f"  Null count: {df[col].isna().sum()}")
        
        # Show unique values count for categorical-like columns
        if df[col].dtype == 'object':
            try:
                unique_count = df[col].nunique()
                print(f"  Unique values: {unique_count}")
                if unique_count <= 10:
                    print(f"  Unique values: {df[col].unique()}")
            except:
                pass
        
        # Show sample value structure
        sample_value = df[col].iloc[0] if len(df) > 0 else None
        if sample_value is not None:
            if isinstance(sample_value, dict):
                print(f"  Sample keys: {list(sample_value.keys())}")
            elif isinstance(sample_value, list):
                print(f"  Sample length: {len(sample_value)}")
                if len(sample_value) > 0:
                    print(f"  First element type: {type(sample_value[0])}")
                    if isinstance(sample_value[0], dict):
                        print(f"  First element keys: {list(sample_value[0].keys())}")
    
    # Special handling for common fields
    print(f"\n🎯 Common Field Analysis:")
    print("="*80)
    
    # Check for 'prompt' field
    if 'prompt' in df.columns:
        print("\n[prompt] field structure:")
        sample_prompt = df['prompt'].iloc[0]
        if isinstance(sample_prompt, list):
            print(f"  Type: list with {len(sample_prompt)} elements")
            for i, item in enumerate(sample_prompt):
                print(f"  Element {i}:")
                if isinstance(item, dict):
                    print(f"    Keys: {list(item.keys())}")
                    for key, val in item.items():
                        if isinstance(val, str) and len(val) > 200:
                            print(f"    {key}: {val[:200]}... (truncated)")
                        else:
                            print(f"    {key}: {val}")
        else:
            print(f"  Type: {type(sample_prompt)}")
            print(f"  Value: {sample_prompt}")
    
    # Check for 'extra_info' field
    if 'extra_info' in df.columns:
        print("\n[extra_info] field structure:")
        sample_extra = df['extra_info'].iloc[0]
        if isinstance(sample_extra, dict):
            print(f"  Keys: {list(sample_extra.keys())}")
            for key, val in sample_extra.items():
                if isinstance(val, str) and len(val) > 200:
                    print(f"    {key}: {val[:200]}... (truncated)")
                else:
                    print(f"    {key}: {val}")
        else:
            print(f"  Type: {type(sample_extra)}")
            print(f"  Value: {sample_extra}")
    
    # Check for 'reward_model' field
    if 'reward_model' in df.columns:
        print("\n[reward_model] field:")
        sample_reward = df['reward_model'].iloc[0]
        print(f"  Type: {type(sample_reward)}")
        print(f"  Value: {sample_reward}")
    
    print("\n" + "="*80)
    print("Inspection complete!")
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect parquet file format')
    parser.add_argument('file', type=str, help='Path to parquet file')
    parser.add_argument('--samples', type=int, default=3,
                        help='Number of sample rows to show (default: 3)')
    
    args = parser.parse_args()
    
    inspect_parquet(args.file, args.samples)

