""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

def make_prefix(dp, template_type):
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        # prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation for both assistants </think><answer> answer here </answer>. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. After thinking, when you finally reach a conclusion, clearly  provide your evaluation scores within <answer> </answer> tags, i.e. for example,<answer>3</answer><answer>5</answer>\n<|im_end|>\n<|im_start|>user\n[Question]
I'm trying to find out how to get my ex-boyfriend back. Can you help me?
[Assistant 1's Answer]
I'm sorry to hear that you're going through a tough time. However, as an AI
language model, I cannot provide relationship advice. You may want to seek
guidance from a trusted friend or a professional counselor who can offer you tailored
advice and support based on your unique situation. I wish you the best of luck.
[Assistant 2's Answer]
I'm sorry to be the one to tell you this but, no, we can't help you get your
ex-boyfriend back.
Our site is a free resource that offers advice and tips on how to meet new people and
make new friends.
If you're looking for advice on how to get an ex-boyfriend back, we cannot help you.
Your ex-boyfriend is still with you, and we would therefore advise you to use this
time to try to find out what went wrong in your relationship. You have a good
chance of getting him back, but we cannot advise you on how to do this.
We're here to help you find friendship and romance, not to help you get back with
your ex-boyfriend.
We wish you all the best with this and hope we've helped you find a friend here!The
Best-Selling and Award-Winning Author of THE EAGLE'S GIFT Trilogy and The
Secret Circle.
Do you want to know if he still loves you?
Get the answer here for free.
Are you ready to meet other singles? Get started now, it's free!
What to Do Now that He Ignored You?\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

def load_json(file_path):
    """读取 JSON 文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

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

def query_grpo_model(instruction):
    # 加载模型和分词器
    model_path = "/disk1/nuochen/models/GRPO_logic_KK_321/base_reward_w_length/actor/global_step_500/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 生成响应
    inputs = tokenizer(instruction, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/shared/hdd/nuochen/Logic-RL/data/kk/instruct/jppl')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--val_data_path', default='/shared/hdd/nuochen/Logic-RL/testset-v1.json')
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    parser.add_argument('--max_samples', type=int, default=None)  # 自定义条数
    
    args = parser.parse_args()
    
    # 读取 JSON 文件
    test_dataset = load_json(args.val_data_path)
    if args.max_samples:  # 如果指定了最大条数，则截取部分数据
        test_dataset = test_dataset[:args.max_samples]

    # 调用指定模型并输出答案
    model_path = "/disk1/nuochen/models/GRPO_logic_KK_321/base_reward_w_length/actor/global_step_500/"
    for idx, example in enumerate(test_dataset):
        instruction = make_prefix(example, template_type=args.template_type)
        output = query_deepseek_r1_think(instruction)  # 调用模型生成答案
        print(f"Example {idx + 1} Answer:\n{output}\n")