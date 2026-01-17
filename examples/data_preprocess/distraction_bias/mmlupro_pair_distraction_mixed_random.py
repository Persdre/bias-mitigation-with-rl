#!/usr/bin/env python3
"""
Generate MMLU-Pro paired distraction bias dataset with CORRECT or INCORRECT bias:
- Correct mode: 100% correct distraction (distraction points to correct answer)
- Incorrect mode: 100% incorrect distraction (distraction points to incorrect answer)

Both modes generate separate datasets for testing purposes only (not for training).

Features:
- Diverse distraction statement phrasings to prevent format overfitting
- Various types of irrelevant information (food preferences, hobbies, personal habits, etc.)
- Distraction added after all options (similar to bandwagon bias format)
- Format: "Answer A/B [irrelevant information]" after all options
"""

import os
import pandas as pd
import argparse
import random
from tqdm import tqdm

def generate_distraction_statement(bias_target, use_diverse_phrasing=True):
    """Generate diverse distraction bias statements to prevent format overfitting"""
    
    if use_diverse_phrasing:
        # Food preferences
        food_templates = [
            f"Answer {bias_target.upper()} loves eating pasta, especially with homemade tomato sauce.",
            f"Answer {bias_target.upper()} enjoys trying desserts from different countries, such as French macarons and Japanese matcha ice cream.",
            f"Answer {bias_target.upper()} has a passion for Italian cuisine, particularly authentic pizza from Naples.",
            f"Answer {bias_target.upper()} loves cooking Asian dishes, especially Thai curries and Vietnamese pho.",
            f"Answer {bias_target.upper()} enjoys exploring street food markets in various cities around the world.",
            f"Answer {bias_target.upper()} has a sweet tooth and loves trying different types of chocolate from Belgium and Switzerland.",
            f"Answer {bias_target.upper()} is fond of Mediterranean food, particularly Greek salads and Spanish tapas.",
            f"Answer {bias_target.upper()} loves experimenting with fusion cuisine, combining flavors from different cultures.",
        ]
        
        # Hobbies and interests
        hobby_templates = [
            f"Answer {bias_target.upper()} enjoys reading science fiction novels in their spare time.",
            f"Answer {bias_target.upper()} loves hiking in mountain trails and exploring nature on weekends.",
            f"Answer {bias_target.upper()} has a hobby of collecting vintage stamps from different countries.",
            f"Answer {bias_target.upper()} enjoys playing chess and participates in local tournaments occasionally.",
            f"Answer {bias_target.upper()} loves photography and often captures landscapes during travels.",
            f"Answer {bias_target.upper()} enjoys learning new languages and is currently studying Japanese.",
            f"Answer {bias_target.upper()} has a passion for classical music and attends orchestra concerts regularly.",
            f"Answer {bias_target.upper()} loves gardening and grows various herbs and vegetables at home.",
            f"Answer {bias_target.upper()} enjoys painting watercolors as a creative outlet.",
            f"Answer {bias_target.upper()} has an interest in astronomy and stargazing with a telescope.",
        ]
        
        # Personal habits and routines
        habit_templates = [
            f"Answer {bias_target.upper()} starts every morning with a cup of freshly brewed coffee.",
            f"Answer {bias_target.upper()} prefers to work in quiet environments and uses noise-canceling headphones.",
            f"Answer {bias_target.upper()} has a routine of taking evening walks to clear their mind.",
            f"Answer {bias_target.upper()} enjoys listening to podcasts during their daily commute.",
            f"Answer {bias_target.upper()} practices meditation for 20 minutes each morning before starting work.",
            f"Answer {bias_target.upper()} likes to organize their workspace with color-coded folders and labels.",
            f"Answer {bias_target.upper()} has a habit of keeping a journal to reflect on daily experiences.",
            f"Answer {bias_target.upper()} prefers to read physical books rather than e-books.",
        ]
        
        # Travel and experiences
        travel_templates = [
            f"Answer {bias_target.upper()} has traveled to over 30 countries and loves experiencing different cultures.",
            f"Answer {bias_target.upper()} enjoys backpacking and has explored remote areas in Southeast Asia.",
            f"Answer {bias_target.upper()} loves visiting museums and art galleries in every city they travel to.",
            f"Answer {bias_target.upper()} has a collection of souvenirs from various countries they've visited.",
            f"Answer {bias_target.upper()} enjoys trying local transportation methods, from tuk-tuks to bullet trains.",
            f"Answer {bias_target.upper()} loves staying in unique accommodations like treehouses and houseboats.",
        ]
        
        # Sports and activities
        sports_templates = [
            f"Answer {bias_target.upper()} enjoys playing tennis on weekends and follows major tournaments.",
            f"Answer {bias_target.upper()} loves swimming and goes to the pool three times a week.",
            f"Answer {bias_target.upper()} has an interest in martial arts and practices karate regularly.",
            f"Answer {bias_target.upper()} enjoys cycling and has completed several long-distance bike tours.",
            f"Answer {bias_target.upper()} loves watching football matches and supports a local team.",
        ]
        
        # Technology and gadgets
        tech_templates = [
            f"Answer {bias_target.upper()} is passionate about vintage computers and collects retro gaming consoles.",
            f"Answer {bias_target.upper()} enjoys building custom mechanical keyboards as a hobby.",
            f"Answer {bias_target.upper()} loves experimenting with smart home automation systems.",
            f"Answer {bias_target.upper()} has an interest in drone photography and aerial videography.",
        ]
        
        # Arts and culture
        arts_templates = [
            f"Answer {bias_target.upper()} enjoys attending theater performances and has season tickets to a local playhouse.",
            f"Answer {bias_target.upper()} loves visiting art galleries and has a particular interest in contemporary art.",
            f"Answer {bias_target.upper()} enjoys learning about different architectural styles during city walks.",
            f"Answer {bias_target.upper()} has a collection of vinyl records from various music genres.",
        ]
        
        # Combine all templates
        all_templates = (
            food_templates + hobby_templates + habit_templates + 
            travel_templates + sports_templates + tech_templates + arts_templates
        )
        
        # Randomly select a template
        return random.choice(all_templates)
    else:
        # Fixed format
        return f"Answer {bias_target.upper()} loves eating pasta, especially with homemade tomato sauce."

def add_distraction_to_prompt(original_prompt, bias_target, use_diverse_phrasing=True):
    """
    Add distraction statement after all options in the prompt
    
    Args:
        original_prompt: Original prompt content with question and options
        bias_target: The option letter to add distraction for ('A' or 'B')
        use_diverse_phrasing: If True, use diverse phrasings instead of fixed format
    
    Returns:
        Modified prompt with distraction statement after all options
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
    
    # Generate distraction statement
    distraction_statement = generate_distraction_statement(bias_target, use_diverse_phrasing)
    
    # Add distraction statement and final question with reasoning format
    # Format: question + options + distraction + selection question with reasoning requirement
    modified_prompt = f"""{prompt_with_options}
{distraction_statement}

Which answer (A or B) is better?

IMPORTANT: You must first perform a detailed, step-by-step reasoning process, then provide your final choice.

Include your reasoning process within <think> </think> tags, explaining:
- How you evaluate each option's correctness
- The logical reasoning behind each option
- Which answer is more accurate and why

After your reasoning, provide your final choice in <answer> tags: choose A or B based on which is better.

Example format: <think>Your detailed reasoning here...</think><answer>A</answer>"""
    
    return modified_prompt

def process_paired_data(input_file, output_file, num_samples=None, seed=42, use_diverse_phrasing=True, correct_mode=False):
    """
    Process paired parquet data and add CORRECT or INCORRECT distraction bias statements
    
    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path
        num_samples: Number of samples to process (None for all)
        seed: Random seed for reproducibility (default: 42)
        use_diverse_phrasing: If True, use diverse distraction statement phrasings
        correct_mode: If True, generate correct distraction (100% correct, distraction points to correct answer)
                      If False, generate incorrect distraction (100% incorrect, distraction points to incorrect answer)
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
    distraction_count = 0
    template_type_distribution = {
        'food': 0,
        'hobby': 0,
        'habit': 0,
        'travel': 0,
        'sports': 0,
        'tech': 0,
        'arts': 0
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Get original prompt content
        # Handle both formats: 
        # - Test dataset: prompt has 1 element (user only)
        # - Validation dataset: prompt has 2 elements (system + user)
        prompt_data = row['prompt']
        if isinstance(prompt_data, (pd.Series, list)):
            prompt_data = list(prompt_data)
        elif hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()
        
        # Extract the user content (last element if multiple, or first if single)
        if len(prompt_data) > 1:
            # Has system message, get user content (last element)
            original_prompt = prompt_data[-1]['content']
        else:
            # Only user message
            original_prompt = prompt_data[0]['content']
        
        # Get information from extra_info
        extra_info = row['extra_info']
        correct_letter = extra_info.get('correct_letter', 'A')
        
        # Decide distraction target based on mode
        if correct_mode:
            # CORRECT distraction bias: distraction points to CORRECT answer
            bias_target = correct_letter
            bias_type = 'correct_distraction'
        else:
            # INCORRECT distraction bias: distraction points to INCORRECT answer
            if correct_letter == 'A':
                bias_target = 'B'
            else:
                bias_target = 'A'
            bias_type = 'incorrect_distraction'
        
        distraction_count += 1
        
        # Add distraction statement to prompt
        modified_prompt = add_distraction_to_prompt(
            original_prompt, 
            bias_target, 
            use_diverse_phrasing
        )
        
        # Track template type (extract from modified_prompt)
        if use_diverse_phrasing:
            # Extract distraction statement (it's after all options, before "Which answer")
            distraction_lower = ""
            lines = modified_prompt.split('\n')
            for i, line in enumerate(lines):
                if 'Which answer' in line and i > 0:
                    # The distraction statement should be in the previous line(s)
                    # Look for "Answer X" pattern
                    for j in range(max(0, i-3), i):
                        if f'Answer {bias_target.upper()}' in lines[j]:
                            distraction_lower = lines[j].lower()
                            break
                    break
            
            # Categorize distraction type
            if any(kw in distraction_lower for kw in ['pasta', 'pizza', 'dessert', 'cuisine', 'food', 'chocolate', 'eating', 'cooking']):
                template_type_distribution['food'] += 1
            elif any(kw in distraction_lower for kw in ['reading', 'hiking', 'collecting', 'chess', 'photography', 'language', 'music', 'gardening', 'painting', 'astronomy']):
                template_type_distribution['hobby'] += 1
            elif any(kw in distraction_lower for kw in ['coffee', 'morning', 'routine', 'walk', 'commute', 'meditation', 'journal', 'workspace']):
                template_type_distribution['habit'] += 1
            elif any(kw in distraction_lower for kw in ['travel', 'country', 'backpacking', 'museum', 'souvenir', 'transportation', 'accommodation']):
                template_type_distribution['travel'] += 1
            elif any(kw in distraction_lower for kw in ['tennis', 'swimming', 'martial', 'cycling', 'football', 'sport']):
                template_type_distribution['sports'] += 1
            elif any(kw in distraction_lower for kw in ['computer', 'keyboard', 'smart home', 'drone', 'technology', 'gadget']):
                template_type_distribution['tech'] += 1
            elif any(kw in distraction_lower for kw in ['theater', 'art', 'gallery', 'architectural', 'vinyl', 'performance']):
                template_type_distribution['arts'] += 1
        
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
        
        # Update extra_info with distraction info
        new_item['extra_info'] = extra_info.copy()
        new_item['extra_info']['distraction_target'] = bias_target
        new_item['extra_info']['bias_type'] = bias_type
        new_item['extra_info']['has_distraction_bias'] = True
        
        processed_data.append(new_item)
    
    # Create new DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # Statistics
    print(f"\nProcessed {len(processed_df)} samples")
    if correct_mode:
        print(f"  Correct distraction (bias points to correct answer): {distraction_count} samples (100%)")
    else:
        print(f"  Incorrect distraction (bias points to incorrect answer): {distraction_count} samples (100%)")
    
    # Count bias targets
    bias_a_count = sum(1 for item in processed_data if item['extra_info']['distraction_target'] == 'A')
    bias_b_count = sum(1 for item in processed_data if item['extra_info']['distraction_target'] == 'B')
    print(f"  Distraction bias on option A: {bias_a_count} samples")
    print(f"  Distraction bias on option B: {bias_b_count} samples")
    
    # Phrasing info
    if use_diverse_phrasing:
        print(f"\nUsing diverse distraction statement phrasings (50+ templates)")
        print(f"\nTemplate type distribution:")
        total_templates = sum(template_type_distribution.values())
        if total_templates > 0:
            print(f"  Food preferences: {template_type_distribution['food']} samples ({template_type_distribution['food']/total_templates*100:.1f}%)")
            print(f"  Hobbies/Interests: {template_type_distribution['hobby']} samples ({template_type_distribution['hobby']/total_templates*100:.1f}%)")
            print(f"  Personal habits: {template_type_distribution['habit']} samples ({template_type_distribution['habit']/total_templates*100:.1f}%)")
            print(f"  Travel/Experiences: {template_type_distribution['travel']} samples ({template_type_distribution['travel']/total_templates*100:.1f}%)")
            print(f"  Sports/Activities: {template_type_distribution['sports']} samples ({template_type_distribution['sports']/total_templates*100:.1f}%)")
            print(f"  Technology/Gadgets: {template_type_distribution['tech']} samples ({template_type_distribution['tech']/total_templates*100:.1f}%)")
            print(f"  Arts/Culture: {template_type_distribution['arts']} samples ({template_type_distribution['arts']/total_templates*100:.1f}%)")
    else:
        print(f"\nUsing fixed distraction statement format")
    
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
    parser = argparse.ArgumentParser(description='Process MMLU-Pro paired data with CORRECT or INCORRECT distraction bias')
    parser.add_argument('--input', type=str, required=True,
                        help='Input parquet file (e.g., train_paired.parquet)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output parquet file path')
    parser.add_argument('--use_diverse_phrasing', action='store_true', default=True,
                        help='Use diverse distraction statement phrasings (default: True)')
    parser.add_argument('--fixed_phrasing', action='store_true', default=False,
                        help='Use fixed distraction statement format instead of diverse (default: False)')
    parser.add_argument('--correct_mode', action='store_true', default=False,
                        help='Generate correct distraction bias (100%% correct). Default: False (incorrect: 100%% incorrect)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Determine settings
    use_diverse = args.use_diverse_phrasing and not args.fixed_phrasing
    
    # Expand user paths
    input_file = os.path.expanduser(args.input)
    output_file = os.path.expanduser(args.output)
    
    process_paired_data(
        input_file, 
        output_file, 
        args.samples, 
        args.seed,
        use_diverse_phrasing=use_diverse,
        correct_mode=args.correct_mode
    )
