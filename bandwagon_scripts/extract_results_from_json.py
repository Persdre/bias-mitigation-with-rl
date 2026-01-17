#!/usr/bin/env python3
"""
Extract and recalculate metrics from existing JSON result files
This script reads existing evaluation results and recalculates the corrected metrics
(incorrect_bandwagon_regular_accuracy) from the results array
"""

import json
import os
import glob
import re
import argparse
from pathlib import Path

def extract_step_number(path):
    """Extract step number from path like global_step_10"""
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None

def recalculate_metrics_from_results(json_file):
    """Recalculate metrics from results array in JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    results = data.get('results', [])
    
    # Calculate incorrect_bandwagon_regular_accuracy from results
    incorrect_bandwagon_results = [r for r in results if r.get('bias_type') == 'incorrect_bandwagon']
    incorrect_bandwagon_total = len(incorrect_bandwagon_results)
    
    if incorrect_bandwagon_total > 0:
        incorrect_bandwagon_regular_correct = sum(
            1 for r in incorrect_bandwagon_results 
            if r.get('regular_correct', False)
        )
        incorrect_bandwagon_regular_accuracy = incorrect_bandwagon_regular_correct / incorrect_bandwagon_total
        
        # Recalculate effect
        incorrect_bandwagon_accuracy = summary.get('incorrect_bandwagon_accuracy', 0)
        incorrect_bandwagon_effect = incorrect_bandwagon_accuracy - incorrect_bandwagon_regular_accuracy
    else:
        incorrect_bandwagon_regular_accuracy = 0
        incorrect_bandwagon_effect = 0
    
    # Update summary
    summary['incorrect_bandwagon_regular_accuracy'] = incorrect_bandwagon_regular_accuracy
    summary['incorrect_bandwagon_effect'] = incorrect_bandwagon_effect
    
    # Update per-subject stats
    if 'per_subject' in summary:
        for subject, subject_stats in summary['per_subject'].items():
            subject_incorrect_bw = [
                r for r in results 
                if r.get('subject') == subject and r.get('bias_type') == 'incorrect_bandwagon'
            ]
            subject_incorrect_bw_total = len(subject_incorrect_bw)
            
            if subject_incorrect_bw_total > 0:
                subject_incorrect_bw_regular_correct = sum(
                    1 for r in subject_incorrect_bw 
                    if r.get('regular_correct', False)
                )
                subject_incorrect_bw_regular_acc = subject_incorrect_bw_regular_correct / subject_incorrect_bw_total
                subject_stats['incorrect_bandwagon_regular_accuracy'] = subject_incorrect_bw_regular_acc
            else:
                subject_stats['incorrect_bandwagon_regular_accuracy'] = 0
    
    # Save updated JSON
    output_data = {
        'summary': summary,
        'results': results
    }
    
    return output_data, summary

def process_trajectory_results(base_dir, output_csv=None):
    """Process all checkpoint results in a trajectory directory"""
    base_path = Path(base_dir)
    
    # Find all global_step directories
    step_dirs = sorted(base_path.glob('global_step_*'), key=lambda x: extract_step_number(str(x)))
    
    if not step_dirs:
        print(f"No global_step directories found in {base_dir}")
        return
    
    print(f"Found {len(step_dirs)} checkpoints to process")
    print()
    
    # Collect all results
    all_results = []
    
    for step_dir in step_dirs:
        step_num = extract_step_number(str(step_dir))
        if step_num is None:
            continue
        
        # Find JSON files in this directory
        json_files = list(step_dir.glob('*.json'))
        if not json_files:
            print(f"⚠️  No JSON files found in {step_dir}")
            continue
        
        # Use the most recent JSON file
        json_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        print(f"Processing {step_dir.name}...")
        print(f"  JSON file: {json_file.name}")
        
        try:
            output_data, summary = recalculate_metrics_from_results(json_file)
            
            # Save updated JSON (backup original first)
            backup_file = json_file.with_suffix('.json.backup')
            if not backup_file.exists():
                import shutil
                shutil.copy2(json_file, backup_file)
                print(f"  ✅ Backed up original to {backup_file.name}")
            
            # Save updated JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"  ✅ Updated JSON with corrected metrics")
            
            # Extract metrics for CSV
            all_results.append({
                'step': step_num,
                'regular_acc': summary.get('regular_accuracy', 0),
                'incorrect_bw_regular_acc': summary.get('incorrect_bandwagon_regular_accuracy', 0),
                'incorrect_bw_acc': summary.get('incorrect_bandwagon_accuracy', 0),
                'incorrect_bw_effect': summary.get('incorrect_bandwagon_effect', 0),
                'robustness_rate': summary.get('incorrect_bandwagon_robust_rate', 0),
            })
            
            print(f"  Regular Acc: {summary.get('regular_accuracy', 0):.4f}")
            print(f"  Incorrect BW Regular Acc: {summary.get('incorrect_bandwagon_regular_accuracy', 0):.4f}")
            print(f"  Incorrect BW Acc: {summary.get('incorrect_bandwagon_accuracy', 0):.4f}")
            print(f"  Incorrect BW Effect: {summary.get('incorrect_bandwagon_effect', 0):.4f}")
            print(f"  Robustness Rate: {summary.get('incorrect_bandwagon_robust_rate', 0):.4f}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing {json_file}: {e}")
            print()
            continue
    
    # Write CSV summary
    if output_csv and all_results:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w') as f:
            f.write("Step,Regular_Acc,Incorrect_BW_Regular_Acc,Incorrect_BW_Acc,Incorrect_BW_Effect,Robustness_Rate\n")
            for r in sorted(all_results, key=lambda x: x['step']):
                f.write(f"{r['step']},{r['regular_acc']:.4f},{r['incorrect_bw_regular_acc']:.4f},"
                       f"{r['incorrect_bw_acc']:.4f},{r['incorrect_bw_effect']:.4f},{r['robustness_rate']:.4f}\n")
        
        print(f"✅ Summary CSV saved to: {csv_path}")
        print()
        print("CSV Summary:")
        print("Step | Regular Acc | Incorrect BW Regular Acc | Incorrect BW Acc | Effect | Robustness")
        print("-" * 80)
        for r in sorted(all_results, key=lambda x: x['step']):
            print(f"{r['step']:4d} | {r['regular_acc']:11.4f} | {r['incorrect_bw_regular_acc']:26.4f} | "
                 f"{r['incorrect_bw_acc']:17.4f} | {r['incorrect_bw_effect']:6.4f} | {r['robustness_rate']:10.4f}")

def process_single_result(json_file, update_file=True):
    """Process a single JSON result file"""
    json_path = Path(json_file)
    
    if not json_path.exists():
        print(f"❌ File not found: {json_file}")
        return None
    
    print(f"Processing: {json_path.name}")
    
    try:
        output_data, summary = recalculate_metrics_from_results(json_path)
        
        if update_file:
            # Backup original
            backup_file = json_path.with_suffix('.json.backup')
            if not backup_file.exists():
                import shutil
                shutil.copy2(json_path, backup_file)
                print(f"✅ Backed up original to {backup_file.name}")
            
            # Save updated JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ Updated JSON with corrected metrics")
        
        print()
        print("Summary:")
        print(f"  Regular Accuracy: {summary.get('regular_accuracy', 0):.4f}")
        print(f"  Incorrect BW Regular Accuracy: {summary.get('incorrect_bandwagon_regular_accuracy', 0):.4f}")
        print(f"  Incorrect BW Accuracy: {summary.get('incorrect_bandwagon_accuracy', 0):.4f}")
        print(f"  Incorrect BW Effect: {summary.get('incorrect_bandwagon_effect', 0):.4f}")
        print(f"  Robustness Rate: {summary.get('incorrect_bandwagon_robust_rate', 0):.4f}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and recalculate metrics from existing JSON results')
    parser.add_argument('--trajectory_dir', type=str, 
                       help='Directory containing global_step_* subdirectories (e.g., results_trajectory_ood)')
    parser.add_argument('--json_file', type=str,
                       help='Single JSON file to process')
    parser.add_argument('--output_csv', type=str,
                       help='Output CSV file path (only used with --trajectory_dir)')
    parser.add_argument('--no_update', action='store_true',
                       help='Do not update JSON files, only print results')
    
    args = parser.parse_args()
    
    if args.trajectory_dir:
        process_trajectory_results(args.trajectory_dir, args.output_csv)
    elif args.json_file:
        process_single_result(args.json_file, update_file=not args.no_update)
    else:
        print("Please specify either --trajectory_dir or --json_file")
        parser.print_help()

