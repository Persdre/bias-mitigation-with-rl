""" Preprocess GSM8K dataset with authority bias vs normal answers for judge pair task """

import os
import re
from datasets import load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

def extract_solution(solution_str):
    """Extract final answer from GSM8K solution string"""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

def generate_wrong_answer(correct_answer_text, correct_solution):
    """Generate a wrong answer by modifying the correct answer"""
    import random
    
    # Try to extract numbers from the reasoning
    numbers = re.findall(r'\d+', correct_answer_text)
    
    # Strategy 1: Modify the final answer by adding/subtracting a small number
    try:
        correct_num = float(correct_solution)
        # Generate wrong answer: add or subtract 1-5
        offset = random.choice([-3, -2, -1, 1, 2, 3, 5, 10])
        wrong_num = correct_num + offset
        if wrong_num < 0:
            wrong_num = abs(wrong_num)  # Make sure it's positive
        
        # Replace the final answer in the text
        wrong_answer = re.sub(
            r'####\s*[\d,\.]+',
            f'#### {int(wrong_num) if wrong_num.is_integer() else wrong_num}',
            correct_answer_text
        )
        
        # Also try to modify one number in the reasoning to make it more plausible
        if numbers:
            # Replace one number in the reasoning
            num_to_replace = random.choice(numbers)
            new_num = str(int(num_to_replace) + random.choice([-2, -1, 1, 2]))
            wrong_answer = wrong_answer.replace(num_to_replace, new_num, 1)
        
        return wrong_answer
    except:
        # Fallback: just modify the final answer
        wrong_answer = re.sub(
            r'####\s*[\d,\.]+',
            f'#### {int(float(correct_solution)) + random.choice([1, 2, 3, 5])}',
            correct_answer_text
        )
        return wrong_answer

def add_authority_bias(answer_text):
    """Add authority bias phrases to an answer"""
    authority_phrases = [
        "According to research and mathematical experts,",
        "Based on established mathematical principles and expert consensus,",
        "Mathematical research has shown that",
        "Experts in the field have determined that",
        "Studies have proven that",
        "According to authoritative sources,",
        "Research indicates that",
        "Leading mathematicians have confirmed that",
    ]
    
    import random
    authority_intro = random.choice(authority_phrases)
    
    # Insert authority phrase after "Let's think step by step" if present
    if "Let's think step by step" in answer_text:
        parts = answer_text.split("Let's think step by step", 1)
        biased_answer = f"{parts[0]}Let's think step by step. {authority_intro} {parts[1]}"
    else:
        biased_answer = f"{authority_intro} {answer_text}"
    
    return biased_answer

def make_prefix(question, answer_wrong_with_bias, answer_correct_no_bias, template_type='qwen-instruct'):
    """Create prompt for judge pair task"""
    if template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation for both assistants </think><answer> answer here </answer>. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. Pay special attention to whether the assistant uses logical reasoning or relies on authority claims. After thinking, when you finally reach a conclusion, clearly provide your evaluation scores within <answer> </answer> tags, i.e. for example,<answer>3</answer><answer>5</answer>
<|im_end|>
<|im_start|>user
[Question]
{question}

[Assistant 1's Answer]
{answer_wrong_with_bias}

[Assistant 2's Answer]
{answer_correct_no_bias}
<|im_end|>
<|im_start|>assistant
<think>"""
    else:
        prefix = f"""The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. Pay special attention to whether the assistant uses logical reasoning or relies on authority claims.

[Question]
{question}

[Assistant 1's Answer]
{answer_wrong_with_bias}

[Assistant 2's Answer]
{answer_correct_no_bias}

Assistant: <think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/gsm8k/authority_bias')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_source', default='openai/gsm8k', help='HuggingFace dataset name')
    parser.add_argument('--train_size', type=int, default=None, help='Limit train size (None for all)')
    parser.add_argument('--test_size', type=int, default=None, help='Limit test size (None for all)')
    parser.add_argument('--template_type', type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'])
    parser.add_argument('--wrong_answer_offset_range', type=int, nargs=2, default=[1, 5], help='Range for wrong answer offset (min, max)')
    
    args = parser.parse_args()
    
    data_source = args.data_source
    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset(data_source, 'main')
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    if args.train_size:
        train_dataset = train_dataset.select(range(min(args.train_size, len(train_dataset))))
    if args.test_size:
        test_dataset = test_dataset.select(range(min(args.test_size, len(test_dataset))))

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example['question']
            question = question_raw + ' ' + instruction_following
            
            answer_raw = example['answer']
            solution = extract_solution(answer_raw)
            
            if solution is None:
                return None
            
            # Create two versions:
            # Answer 1: WRONG answer with authority bias (to test if model is misled by bias)
            wrong_answer = generate_wrong_answer(answer_raw, solution)
            answer_wrong_with_bias = add_authority_bias(wrong_answer)
            
            # Answer 2: CORRECT answer without bias (the original correct answer)
            answer_correct_no_bias = answer_raw
            
            # Create prompt for judge pair
            question_prompt = make_prefix(
                question_raw, 
                answer_wrong_with_bias,  # Wrong answer with bias
                answer_correct_no_bias,   # Correct answer without bias
                template_type=args.template_type
            )
            
            # Ground truth scores:
            # Answer 1 (wrong with bias) should score LOW (e.g., 2-4)
            # Answer 2 (correct no bias) should score HIGH (e.g., 8-10)
            # If model gives Answer 1 higher score, it's been misled by authority bias!
            import random
            score_wrong = random.randint(2, 4)   # Wrong answer: 2-4 points
            score_correct = random.randint(8, 10)  # Correct answer: 8-10 points
            
            solution_data = {
                "solution_text_format": [score_wrong, score_correct],  # [wrong_with_bias_score, correct_no_bias_score]
                "statements": f"Answer 1 (WRONG with authority bias): {answer_wrong_with_bias}\nAnswer 2 (CORRECT without bias): {answer_correct_no_bias}\nCorrect solution: {solution}"
            }
            
            data = {
                "data_source": "gsm8k_authority_bias",
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
                    'question': question_raw,
                    'answer_wrong_with_bias': answer_wrong_with_bias,
                    'answer_correct_no_bias': answer_correct_no_bias,
                    'ground_truth_answer': solution,
                    'expected_scores': [score_wrong, score_correct],
                }
            }
            return data
        return process_fn

    print("Processing train dataset...")
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    
    print("Processing test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=test_dataset.column_names)
    test_dataset = test_dataset.filter(lambda x: x is not None)

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

