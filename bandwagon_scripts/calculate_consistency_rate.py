#!/usr/bin/env python3
"""
Calculate Consistency Rate between baseline and trained models.

Consistency rate measures:
1. Within-model consistency: regular_answer vs bandwagon_answer for each model
2. Cross-model consistency: same question answers between baseline and trained models
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_consistency_rate(baseline_file, trained_file):
    """Calculate consistency rates and robustness rates between two models."""
    
    # Load data
    baseline_data = load_json(baseline_file)
    trained_data = load_json(trained_file)
    
    baseline_results = baseline_data['results']
    trained_results = trained_data['results']
    
    # Ensure same number of questions
    if len(baseline_results) != len(trained_results):
        print(f"Warning: Different number of questions!")
        print(f"  Baseline: {len(baseline_results)}")
        print(f"  Trained: {len(trained_results)}")
        min_len = min(len(baseline_results), len(trained_results))
        baseline_results = baseline_results[:min_len]
        trained_results = trained_results[:min_len]
    
    # Statistics
    stats = {
        'total_questions': len(baseline_results),
        # Internal consistency (regular == bandwagon)
        'baseline_internal_consistent': 0,
        'trained_internal_consistent': 0,
        # Robustness: regular == bandwagon AND regular == correct
        'baseline_robust': 0,
        'trained_robust': 0,
        # Cross-model consistency: regular answers
        'baseline_trained_regular_consistent': 0,
        # Cross-model consistency: bandwagon answers
        'baseline_trained_bandwagon_consistent': 0,
        # Cross-model robustness: both models robust on same question
        'both_robust': 0,
    }
    
    # Per-subject statistics
    per_subject_stats = defaultdict(lambda: {
        'total': 0,
        'baseline_internal_consistent': 0,
        'trained_internal_consistent': 0,
        'baseline_robust': 0,
        'trained_robust': 0,
        'baseline_trained_regular_consistent': 0,
        'baseline_trained_bandwagon_consistent': 0,
        'both_robust': 0,
    })
    
    # Calculate consistency and robustness
    for baseline_item, trained_item in zip(baseline_results, trained_results):
        # Ensure same question_id
        if baseline_item['question_id'] != trained_item['question_id']:
            print(f"Warning: Question ID mismatch: {baseline_item['question_id']} vs {trained_item['question_id']}")
            continue
        
        subject = baseline_item.get('subject', 'unknown')
        per_subject_stats[subject]['total'] += 1
        
        baseline_regular = baseline_item.get('regular_answer', '')
        baseline_bandwagon = baseline_item.get('bandwagon_answer', '')
        trained_regular = trained_item.get('regular_answer', '')
        trained_bandwagon = trained_item.get('bandwagon_answer', '')
        correct_letter = baseline_item.get('correct_letter', '')
        
        # Baseline internal consistency (regular == bandwagon)
        if baseline_regular == baseline_bandwagon:
            stats['baseline_internal_consistent'] += 1
            per_subject_stats[subject]['baseline_internal_consistent'] += 1
        
        # Trained internal consistency (regular == bandwagon)
        if trained_regular == trained_bandwagon:
            stats['trained_internal_consistent'] += 1
            per_subject_stats[subject]['trained_internal_consistent'] += 1
        
        # Baseline robustness: regular == bandwagon (without bias)
        if baseline_regular == baseline_bandwagon:
            stats['baseline_robust'] += 1
            per_subject_stats[subject]['baseline_robust'] += 1
        
        # Trained robustness: regular == bandwagon (without bias)
        if trained_regular == trained_bandwagon:
            stats['trained_robust'] += 1
            per_subject_stats[subject]['trained_robust'] += 1
        
        # Cross-model consistency: regular answers
        if baseline_regular == trained_regular:
            stats['baseline_trained_regular_consistent'] += 1
            per_subject_stats[subject]['baseline_trained_regular_consistent'] += 1
        
        # Cross-model consistency: bandwagon answers
        if baseline_bandwagon == trained_bandwagon:
            stats['baseline_trained_bandwagon_consistent'] += 1
            per_subject_stats[subject]['baseline_trained_bandwagon_consistent'] += 1
        
        # Both models robust on same question
        baseline_is_robust = (baseline_regular == baseline_bandwagon)
        trained_is_robust = (trained_regular == trained_bandwagon)
        if baseline_is_robust and trained_is_robust:
            stats['both_robust'] += 1
            per_subject_stats[subject]['both_robust'] += 1
    
    # Calculate rates
    total = stats['total_questions']
    
    results = {
        'total_questions': total,
        # Internal consistency rates (regular == bandwagon)
        'baseline_internal_consistency_rate': stats['baseline_internal_consistent'] / total if total > 0 else 0,
        'trained_internal_consistency_rate': stats['trained_internal_consistent'] / total if total > 0 else 0,
        # Robustness rates (regular == bandwagon, without bias)
        'baseline_robustness_rate': stats['baseline_robust'] / total if total > 0 else 0,
        'trained_robustness_rate': stats['trained_robust'] / total if total > 0 else 0,
        # Cross-model consistency rates
        'cross_model_regular_consistency_rate': stats['baseline_trained_regular_consistent'] / total if total > 0 else 0,
        'cross_model_bandwagon_consistency_rate': stats['baseline_trained_bandwagon_consistent'] / total if total > 0 else 0,
        'both_robust_rate': stats['both_robust'] / total if total > 0 else 0,
        'per_subject': {}
    }
    
    # Per-subject rates
    for subject, subject_stats in per_subject_stats.items():
        subject_total = subject_stats['total']
        if subject_total > 0:
            results['per_subject'][subject] = {
                'total': subject_total,
                'baseline_internal_consistency_rate': subject_stats['baseline_internal_consistent'] / subject_total,
                'trained_internal_consistency_rate': subject_stats['trained_internal_consistent'] / subject_total,
                'baseline_robustness_rate': subject_stats['baseline_robust'] / subject_total,
                'trained_robustness_rate': subject_stats['trained_robust'] / subject_total,
                'cross_model_regular_consistency_rate': subject_stats['baseline_trained_regular_consistent'] / subject_total,
                'cross_model_bandwagon_consistency_rate': subject_stats['baseline_trained_bandwagon_consistent'] / subject_total,
                'both_robust_rate': subject_stats['both_robust'] / subject_total,
            }
    
    return results


def print_results(results, baseline_file, trained_file):
    """Print results in a readable format."""
    print("=" * 80)
    print("CONSISTENCY RATE REPORT")
    print("=" * 80)
    print()
    print(f"Baseline File: {Path(baseline_file).name}")
    print(f"Trained File:  {Path(trained_file).name}")
    print()
    print(f"Total Questions: {results['total_questions']}")
    print()
    
    print("=" * 80)
    print("WITHIN-MODEL CONSISTENCY & ROBUSTNESS RATES")
    print("=" * 80)
    print()
    
    print("1. Baseline Model:")
    print(f"   Internal Consistency (Regular == Bandwagon): {results['baseline_internal_consistency_rate']:.4f} ({results['baseline_internal_consistency_rate']*100:.2f}%)")
    print(f"   Robustness Rate (Regular == Bandwagon):     {results['baseline_robustness_rate']:.4f} ({results['baseline_robustness_rate']*100:.2f}%)")
    print()
    
    print("2. Trained Model:")
    print(f"   Internal Consistency (Regular == Bandwagon): {results['trained_internal_consistency_rate']:.4f} ({results['trained_internal_consistency_rate']*100:.2f}%)")
    print(f"   Robustness Rate (Regular == Bandwagon):     {results['trained_robustness_rate']:.4f} ({results['trained_robustness_rate']*100:.2f}%)")
    print()
    
    print("=" * 80)
    print("CROSS-MODEL CONSISTENCY RATES")
    print("(Baseline vs Trained)")
    print("=" * 80)
    print()
    
    print("3. WITHOUT BIAS Consistency (Baseline Regular vs Trained Regular):")
    print(f"   {results['cross_model_regular_consistency_rate']:.4f} ({results['cross_model_regular_consistency_rate']*100:.2f}%)")
    print()
    
    print("4. WITH BIAS Consistency (Baseline Bandwagon vs Trained Bandwagon):")
    print(f"   {results['cross_model_bandwagon_consistency_rate']:.4f} ({results['cross_model_bandwagon_consistency_rate']*100:.2f}%)")
    print()
    
    print("5. Both Models Robust on Same Questions:")
    print(f"   {results['both_robust_rate']:.4f} ({results['both_robust_rate']*100:.2f}%)")
    print()
    
    if results['per_subject']:
        print("=" * 80)
        print("PER-SUBJECT CONSISTENCY RATES")
        print("=" * 80)
        print()
        
        for subject in sorted(results['per_subject'].keys()):
            subject_data = results['per_subject'][subject]
            print(f"Subject: {subject.upper()} (n={subject_data['total']})")
            print(f"  Baseline Internal Consistency: {subject_data['baseline_internal_consistency_rate']:.4f} ({subject_data['baseline_internal_consistency_rate']*100:.2f}%)")
            print(f"  Baseline Robustness Rate:      {subject_data['baseline_robustness_rate']:.4f} ({subject_data['baseline_robustness_rate']*100:.2f}%)")
            print(f"  Trained Internal Consistency:  {subject_data['trained_internal_consistency_rate']:.4f} ({subject_data['trained_internal_consistency_rate']*100:.2f}%)")
            print(f"  Trained Robustness Rate:       {subject_data['trained_robustness_rate']:.4f} ({subject_data['trained_robustness_rate']*100:.2f}%)")
            print(f"  Cross-Model WITHOUT BIAS (Regular):     {subject_data['cross_model_regular_consistency_rate']:.4f} ({subject_data['cross_model_regular_consistency_rate']*100:.2f}%)")
            print(f"  Cross-Model WITH BIAS (Bandwagon):      {subject_data['cross_model_bandwagon_consistency_rate']:.4f} ({subject_data['cross_model_bandwagon_consistency_rate']*100:.2f}%)")
            print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Baseline Model:")
    print(f"  - Internal Consistency (Regular==Bandwagon): {results['baseline_internal_consistency_rate']:.4f} ({results['baseline_internal_consistency_rate']*100:.2f}%)")
    print(f"  - Robustness Rate (Regular==Bandwagon):       {results['baseline_robustness_rate']:.4f} ({results['baseline_robustness_rate']*100:.2f}%)")
    print()
    print(f"Trained Model:")
    print(f"  - Internal Consistency (Regular==Bandwagon): {results['trained_internal_consistency_rate']:.4f} ({results['trained_internal_consistency_rate']*100:.2f}%)")
    print(f"  - Robustness Rate (Regular==Bandwagon):       {results['trained_robustness_rate']:.4f} ({results['trained_robustness_rate']*100:.2f}%)")
    print()
    print(f"Cross-Model Comparison:")
    print(f"  - WITHOUT BIAS (Regular): {results['cross_model_regular_consistency_rate']:.4f} ({results['cross_model_regular_consistency_rate']*100:.2f}%)")
    print(f"  - WITH BIAS (Bandwagon):  {results['cross_model_bandwagon_consistency_rate']:.4f} ({results['cross_model_bandwagon_consistency_rate']*100:.2f}%)")
    print(f"  - Both Robust:             {results['both_robust_rate']:.4f} ({results['both_robust_rate']*100:.2f}%)")
    print()
    print("=" * 80)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 calculate_consistency_rate.py <baseline_json> <trained_json>")
        print()
        print("Example:")
        print("  python3 calculate_consistency_rate.py \\")
        print("    baseline/results_correct_bandwagon_ood_qwen3_1.7b_baseline/Qwen3-1.7B_*.json \\")
        print("    baseline/results_correct_bandwagon_ood_qwen3_1.7b_trained/global_step_70/hf_merged_model_*.json")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    trained_file = sys.argv[2]
    
    # Check if files exist
    if not Path(baseline_file).exists():
        print(f"Error: Baseline file not found: {baseline_file}")
        sys.exit(1)
    
    if not Path(trained_file).exists():
        print(f"Error: Trained file not found: {trained_file}")
        sys.exit(1)
    
    # Calculate consistency
    results = calculate_consistency_rate(baseline_file, trained_file)
    
    # Print results
    print_results(results, baseline_file, trained_file)
    
    # Save results to JSON
    output_file = Path(baseline_file).parent / "consistency_rate_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

