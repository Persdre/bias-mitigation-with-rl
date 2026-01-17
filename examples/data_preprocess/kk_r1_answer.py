""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def make_prefix(dp, template_type):
    question = dp['question_body']
    answer_1 = dp['answer1_body']
    answer_2 = dp['answer2_body']
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        # prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation for both assistants </think><answer> answer here </answer>. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. After thinking, when you finally reach a conclusion, clearly  provide your evaluation scores within <answer> </answer> tags, i.e. for example,<answer>3</answer><answer>5</answer>\n<|im_end|>\n<|im_start|>user\n[Question]\n{question}\n\n[Assistant 1's Answer]\n{answer_1}\n\n[Assistant 2's Answer]\n{answer_2}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

def load_jsonl(file_path):
    """读取 JSONL 文件"""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def query_deepseek_r1_think(instruction):
    client = OpenAI(
        base_url="https://api.chatfire.cn/v1",
        api_key="sk-mYDMUkBi0LXmnCqjxF81CARnouWkHzFv3tNDFZaEjVigjQrZ"
    )
    response = client.chat.completions.create(
        model="deepseek-r1-think",
        messages=[{"role": "user", "content": instruction}],
        stream=True
    )
    output = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            output += chunk.choices[0].delta.content
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/shared/hdd/nuochen/Logic-RL/data/kk/instruct/jppl')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_train_system_100k_no_metadata_cleaned.jsonl')
    parser.add_argument('--val_data_path', default='/shared/hdd/nuochen/datasets/JudgeLM-100K/judgelm_val_5k_gpt4_updated_no_metadata.jsonl')
    parser.add_argument('--train_size', type=int, default=900)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    parser.add_argument('--max_samples', type=int, default=15)  # 自定义条数
    
    args = parser.parse_args()
    
    data_source = "kk"
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # 读取 JSONL 文件
    test_dataset = load_jsonl(args.val_data_path)
    if args.max_samples:  # 如果指定了最大条数，则截取部分数据
        test_dataset = test_dataset[:args.max_samples]

    # Process testset with deepseek-r1-think using thread pool
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for idx, example in enumerate(test_dataset):
            instruction = make_prefix(example, template_type=args.template_type)
            futures.append((idx, executor.submit(query_deepseek_r1_think, instruction)))
        
        for idx, future in futures:
            try:
                output = future.result()
                test_dataset[idx]['output'] = output  # 直接添加 output 字段
                print(f"Processed {idx + 1}/{len(test_dataset)} examples")
            except Exception as e:
                print(f"Error processing example {idx + 1}: {str(e)}")

    # Save the results to a JSON file
    output_file = "testset_with_responses.json"
    with open(output_file, 'w') as f:
        json.dump(test_dataset, f, indent=4)

    print(f"Testset with responses saved to {output_file}")