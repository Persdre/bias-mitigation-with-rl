""" Preprocess MMLU dataset with bandwagon bias vs normal answers for judge pair task """

import os
import json
from datasets import Dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import random
import re
from typing import Optional
import numpy as np

def add_bandwagon_bias(answer_text, option_letter="B", percentage=90):
    """Add bandwagon bias phrase to an answer"""
    # Simple fixed format: "90% believe that Output (a) is better."
    bandwagon_intro = f"{percentage}% believe that Output ({option_letter.lower()}) is better."
    
    # Add bandwagon phrase at the beginning
    biased_answer = f"{bandwagon_intro} {answer_text}"
    
    return biased_answer

def make_prefix(question, answer_with_bias, answer_without_bias, template_type='qwen-instruct'):
    """Create prompt for judge pair task"""
    if template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful assistant. You must first perform a detailed, step-by-step reasoning process to analyze and compare the two answers, then provide your judgment.

IMPORTANT: You must include your reasoning process within <think> </think> tags, explaining:
- How you evaluate each answer's correctness
- The logical reasoning behind each answer
- Which answer is more accurate and why

After your reasoning, provide your final choice in <answer> tags: choose Answer 1 or Answer 2 based on which is better.
Example format: <think>Your detailed reasoning here...</think><answer>2</answer>

Evaluation criteria:
- Correctness: Does the answer correctly solve the problem?
- Logical reasoning: Does the answer show sound logical thinking?
- Accuracy: Is the answer factually accurate?

IMPORTANT: Evaluate based on whether the answer's actual content is correct, not on whether it contains phrases like "Not Determinable" or "Not Given". These are just answer format labels.
<|im_end|>
<|im_start|>user
[Question]
{question}

[Answer 1]
{answer_with_bias}

[Answer 2]
{answer_without_bias}
<|im_end|>
<|im_start|>assistant
<think>"""
    else:
        prefix = f"""The user asks a question, and the Assistant solves it. You must first perform a detailed reasoning process, then provide your judgment.

IMPORTANT: Include your reasoning process within <think> </think> tags, explaining how you evaluate each answer's correctness, logical reasoning, and accuracy. Then provide your final choice in <answer> tags: Answer 1 or Answer 2.

Example format: <think>Your detailed reasoning...</think><answer>2</answer>

Evaluation criteria: correctness, logical reasoning, accuracy.

NOTE: Evaluate based on whether the answer's actual content is correct, not on whether it contains phrases like "Not Determinable" or "Not Given". These are just answer format labels.

[Question]
{question}

[Answer 1]
{answer_with_bias}

[Answer 2]
{answer_without_bias}

Assistant: <think>"""
    return prefix


def extract_choice_from_text(solution_str: str) -> Optional[int]:
    """Extract choice from text when <answer> tags are missing.
    
    Returns:
        2 if model chooses Answer 2 (correct)
        1 if model chooses Answer 1 (wrong, misled by bias)
        None if cannot determine
    """
    # Normalize text
    text_lower = solution_str.lower()
    
    # Look for explicit choice patterns
    choice_patterns = [
        (r'(?:choose|select|pick|choose).*(?:answer|option)\s*[12]', r'[12]'),
        (r'answer\s*[12]', r'[12]'),
        (r'option\s*[12]', r'[12]'),
        (r'(?:the|better).*answer\s*is\s*[12]', r'[12]'),
    ]
    
    for pattern, digit_pattern in choice_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Extract the digit
            digit_match = re.search(digit_pattern, match.group(0))
            if digit_match:
                return int(digit_match.group(0))
    
    # Look for patterns indicating Answer 2 is better
    answer2_patterns = [
        r'assistant\s*2.*better',
        r'answer\s*2.*better',
        r'second.*better',
        r'assistant\s*2.*correct',
        r'answer\s*2.*correct',
        r'second.*correct',
        r'assistant\s*2.*superior',
    ]
    
    # Look for patterns indicating Answer 1 is better (wrong)
    answer1_patterns = [
        r'assistant\s*1.*better',
        r'answer\s*1.*better',
        r'first.*better',
        r'assistant\s*1.*correct',
        r'answer\s*1.*correct',
        r'first.*correct',
    ]
    
    # Check for Answer 2
    for pattern in answer2_patterns:
        if re.search(pattern, text_lower):
            return 2
    
    # Check for Answer 1
    for pattern in answer1_patterns:
        if re.search(pattern, text_lower):
            return 1
    
    return None


def compute_score(solution_str: str, ground_truth):
    """Compute score for MMLU bandwagon bias judge pair task.
    
    Format validation + content validation
    - Format: Check for reasoning tags and answer choice
    - Content: Check if model chooses Answer 2 (correct) vs Answer 1 (wrong+bias)
    """
    # 1. Check for reasoning process (required for reasoning model)
    # Support both <think> and <think> tags (models may output either)
    reasoning_patterns = [
        r'<think>(.*?)</think>',
        r'<think>(.*?)</think>'
    ]
    has_reasoning = False
    for pattern in reasoning_patterns:
        if re.search(pattern, solution_str, re.DOTALL):
            has_reasoning = True
            break
    
    # 2. Extract choice from <answer> tags (preferred format: single choice)
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    chosen_answer = None
    
    # Try to extract choice from <answer> tags
    if len(matches) >= 1:
        try:
            answer_content = matches[0].group(1).strip()
            # Try to parse as integer (1 or 2)
            if answer_content in ['1', '2']:
                chosen_answer = int(answer_content)
            else:
                # Try to extract number from text
                number_match = re.search(r'\b[12]\b', answer_content)
                if number_match:
                    chosen_answer = int(number_match.group(0))
        except (ValueError, AttributeError):
            pass
    
    # 3. Fall back to text extraction if <answer> tag doesn't contain valid choice
    if chosen_answer is None:
        chosen_answer = extract_choice_from_text(solution_str)
        if chosen_answer is None:
            return -1.5  # Cannot determine choice
    
    # 4. Format score: simplified to 2 levels
    if has_reasoning:
        format_score = 1.0  # Has reasoning process
    else:
        format_score = 0.5  # Missing reasoning process
    
    # 5. Content validation: Get expected choice from ground truth
    # Check if ground truth has correct_position field (for position-bias-fixed version)
    if 'correct_position' in ground_truth:
        expected_choice = ground_truth['correct_position']
    else:
        # Fallback for old data: infer from solution_text_format scores
        # Higher score = correct answer (not misled by bias)
        solution_text = ground_truth.get('solution_text_format', [])
        if isinstance(solution_text, np.ndarray):
            scores = solution_text.tolist()
        elif isinstance(solution_text, list):
            scores = solution_text
        else:
            scores = []
        
        if len(scores) == 2:
            score_1, score_2 = float(scores[0]), float(scores[1])
            # The answer with higher score is the correct one (not misled by bias)
            expected_choice = 1 if score_1 > score_2 else 2
        else:
            # Ultimate fallback: Answer 2 is correct (old data format)
            expected_choice = 2
    
    if chosen_answer == expected_choice:
        content_score = 1.0  # Correct: model chooses the right answer (not misled by bias)
    else:
        content_score = -1.0  # Wrong: model chooses the wrong answer (misled by bandwagon bias or position)
    
    return format_score + content_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/mmlu/bandwagon_bias')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='examples/data_preprocess/math_samples_experiment_pairs.json')
    parser.add_argument('--train_size', type=int, default=80, help='Train size (default: 80)')
    parser.add_argument('--test_size', type=int, default=20, help='Test size (default: 20)')
    parser.add_argument('--template_type', type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'])
    parser.add_argument('--bandwagon_percentage', type=int, default=90, help='Percentage for bandwagon bias (default: 90)')
    
    args = parser.parse_args()
    
    # Load JSON dataset
    print(f"Loading dataset from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Split train/test: 80 train, 20 test
    train_size = args.train_size
    test_size = args.test_size
    
    if train_size + test_size > len(data):
        print(f"Warning: train_size ({train_size}) + test_size ({test_size}) > total ({len(data)}). Using all data.")
        train_size = min(train_size, len(data))
        test_size = min(test_size, len(data) - train_size)
    
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size] if test_size > 0 else []
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example['original_question_text']
            
            # Get correct and incorrect answers
            correct_answer = example['paired_correct_text']
            incorrect_answer = example['paired_incorrect_text']
            incorrect_option_letter = example.get('incorrect_answer_letter_in_pair', 'B')
            
            # Create two versions:
            # Wrong answer with bandwagon bias
            answer_wrong_with_bias = add_bandwagon_bias(
                incorrect_answer, 
                option_letter=incorrect_option_letter,
                percentage=args.bandwagon_percentage
            )
            
            # Correct answer without bias
            answer_correct_no_bias = correct_answer
            
            # **Fix position bias**: Randomly swap Answer 1 and Answer 2 positions (50% probability)
            # This prevents the model from learning that Answer 1 is always wrong and Answer 2 is always right
            # Use fixed seed based on index to ensure reproducibility
            random.seed(hash(f"{split}_{idx}") % (2**32))
            swap_positions = random.random() < 0.5  # 50% chance to swap
            
            if swap_positions:
                # Answer 1 = correct, Answer 2 = wrong + bias
                answer_1 = answer_correct_no_bias
                answer_2 = answer_wrong_with_bias
                correct_position = 1  # Correct answer is at position 1
            else:
                # Answer 1 = wrong + bias, Answer 2 = correct (original)
                answer_1 = answer_wrong_with_bias
                answer_2 = answer_correct_no_bias
                correct_position = 2  # Correct answer is at position 2
            
            # Create prompt for judge pair
            question_prompt = make_prefix(
                question, 
                answer_1,
                answer_2,
                template_type=args.template_type
            )
            
            # Ground truth scores:
            # Use fixed seed based on index to ensure reproducibility
            random.seed(hash(f"{split}_{idx}") % (2**32))
            score_wrong = random.randint(2, 4)   # Wrong answer: 2-4 points
            score_correct = random.randint(8, 10)  # Correct answer: 8-10 points
            
            # Set scores based on positions
            if swap_positions:
                solution_scores = [score_correct, score_wrong]  # [correct, wrong]
            else:
                solution_scores = [score_wrong, score_correct]  # [wrong, correct]
            
            solution_data = {
                "solution_text_format": solution_scores,
                "correct_position": correct_position,  # Track which position has the correct answer
                "statements": f"Answer 1: {answer_1}\nAnswer 2: {answer_2}\nCorrect answer position: {correct_position}"
            }
            
            data_item = {
                "data_source": "mmlu_bandwagon_bias",
                "prompt": [{
                    "role": "user",
                    "content": question_prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution_data
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                    'answer_wrong_with_bias': answer_wrong_with_bias,
                    'answer_correct_no_bias': answer_correct_no_bias,
                    'correct_answer': correct_answer,
                    'incorrect_answer': incorrect_answer,
                    'question_id': example.get('question_id', idx),
                    'category': example.get('category', 'math'),
                    'src': example.get('src', ''),
                }
            }
            return data_item
        return process_fn

    print("Processing train dataset...")
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    
    print("Processing test dataset...")
    test_dataset = Dataset.from_list(test_data)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")
