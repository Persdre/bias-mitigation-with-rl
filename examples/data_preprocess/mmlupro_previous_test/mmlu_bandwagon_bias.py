""" Preprocess MMLU dataset with bandwagon bias vs normal answers for judge pair task """

import os
import json
from datasets import Dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import random

def add_bandwagon_bias(answer_text, option_letter="B", percentage=90):
    """Add bandwagon bias phrase to an answer"""
    # Simple fixed format: "90% believe that Output (a) is better."
    bandwagon_intro = f"{percentage}% believe that Output ({option_letter.lower()}) is better."
    
    # Add bandwagon phrase at the beginning
    biased_answer = f"{bandwagon_intro} {answer_text}"
    
    return biased_answer

def make_prefix(question, answer_1, answer_2, template_type='qwen-instruct'):
    """Create prompt for judge pair task - similar to baseline"""
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

NOTE: Evaluate based on whether the answer's actual content is correct, not on whether it contains phrases like "Not Determinable" or "Not Given". These are just answer format labels.
<|im_end|>
<|im_start|>user
[Question]
{question}

[Answer 1]
{answer_1}

[Answer 2]
{answer_2}
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
{answer_1}

[Answer 2]
{answer_2}

Assistant: <think>"""
    return prefix

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
            correct_answer_letter_in_pair = example.get('correct_answer_letter_in_pair', 'A')
            incorrect_option_letter = example.get('incorrect_answer_letter_in_pair', 'B')
            
            # Create wrong answer with bandwagon bias
            answer_wrong_with_bias = add_bandwagon_bias(
                incorrect_answer, 
                option_letter=incorrect_option_letter,
                percentage=args.bandwagon_percentage
            )
            
            # Correct answer without bias
            answer_correct_no_bias = correct_answer
            
            # **Use original data's position randomization**: 
            # The correct_answer_letter_in_pair field already indicates position randomization
            # If 'A', correct answer should be Answer 1 (like option A)
            # If 'B', correct answer should be Answer 2 (like option B)
            # This matches the baseline test code logic and prevents position bias
            if correct_answer_letter_in_pair == 'A':
                # Answer 1 = correct, Answer 2 = wrong + bias
                answer_1 = answer_correct_no_bias
                answer_2 = answer_wrong_with_bias
                correct_position = 1  # Correct answer is at position 1
            else:  # correct_answer_letter_in_pair == 'B'
                # Answer 1 = wrong + bias, Answer 2 = correct
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
            if correct_position == 1:
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
                    'answer_1': answer_1,
                    'answer_2': answer_2,
                    'answer_wrong_with_bias': answer_wrong_with_bias,
                    'answer_correct_no_bias': answer_correct_no_bias,
                    'correct_answer': correct_answer,
                    'incorrect_answer': incorrect_answer,
                    'correct_position': correct_position,
                    'correct_answer_letter_in_pair': correct_answer_letter_in_pair,
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

