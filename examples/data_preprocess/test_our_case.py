from transformers import pipeline
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time


# Example JSON input
json_input = {"review_id": "abtw7no6nDD2dxjG9yAkBo", "question_id": 99999, "answer1_id": "", "answer2_id": "", "reviewer_id": 1, "text": "1 10\nAssistant 1's response is completely irrelevant to the question asked. The user asked for the categorization of a specific artwork, but the assistant provided a programming exercise which is not related to the question at all. Therefore, Assistant 1 gets a score of 1. On the other hand, Assistant 2's response is accurate and directly answers the user's question. The artwork by Mark Rothko is indeed categorized as Abstract. Therefore, Assistant 2 gets a score of 10.", "score": [1.0, 10.0], "question_body": "Categorize the following artwork: Abstraction of Reality, by Mark Rothko.\n", "answer1_body": "![Image](assets/abstraction2.png)\n---  \n  \n## [Exercise](https://repl.it/@trashman/artistic-conflation-tutorial?lite=true)\nThis programming exercise is a follow up to the previous module's exercise, Exercise: [Abstracting the Abstract](https://repl.it/@trashman/artistic-conflation-tutorial). In this exercise, we're going to do the same thing with the previous exercise's output, but this time we're going to do it using the `pandas` library.\n  \nWe need to go back and download the data from the previous module's exercise, do some cleaning work on it, and then use it to create a new Pandas dataframe. We're going to use this new dataframe to create our second `Pandas` art piece, which will be a `scatterplot` of the data, with the x-axis being the `Artist` and the y-axis being the `Year`.\n# How to use this template\n\n## Requirements\n- A `GitHub` account\n- An idea\n- A laptop\n-", "answer2_body": "Abstract", "answer1_model_id": "mpt-7b", "answer2_model_id": "flan-t5-xxl", "answer1_metadata": {"decoding_method": "top_p_sampling"}, "answer2_metadata": {"decoding_method": "top_p_sampling"}, "reference": {"decoding_method": "None", "model": "itwgpt4/40500", "text": "The artwork \"Abstraction of Reality\" by Mark Rothko would be categorized as Abstract Expressionism."}, "score_w_reference": [1.0, 5.0], "text_w_reference": "1 5\n\nAssistant 1's response is completely irrelevant to the question asked. The user asked for the categorization of an artwork, but the assistant started discussing a programming exercise, which is not related to the question at all. Therefore, it gets a score of 1.\n\nAssistant 2's response is somewhat relevant as it identifies the artwork as \"Abstract\". However, it lacks detail and accuracy. The correct categorization should be \"Abstract Expressionism\", not just \"Abstract\". Therefore, it gets a score of 5. It provided a relevant but incomplete answer.", "system": "You are a helpful assistant tasked with evaluating the performance of two AI assistants in response to a user\u2019s question. Your role is to assess the helpfulness, relevance, accuracy, and level of detail in each assistant\u2019s response.\n\nFor each assistant, provide an overall score between 1 and 10, where a higher score indicates better performance.\n\nYour output must follow this format:\n\nFirst, a single line with two scores separated by a space: the first score for Assistant 1 and the second for Assistant 2.\nNext, a comprehensive explanation of your evaluation, ensuring no bias and that the order of responses does not influence your judgment.\nBase your evaluation on how well each assistant addressed the user\u2019s question, the correctness of the information, and the clarity and thoroughness of the response.\n<|im_end|>\n<|im_start|>user\n[Question]\n{question}\n\n[Assistant 1\u2019s Answer]\n{answer_1}\n\n[Assistant 2\u2019s Answer]\n{answer_2}\n<|im_end|>\n<|im_start|>assistant", "instruction": "[Question]\nCategorize the following artwork: Abstraction of Reality, by Mark Rothko.\n\n[Assistant 1\u2019s Answer]\n![Image](assets/abstraction2.png)\n---  \n  \n## [Exercise](https://repl.it/@trashman/artistic-conflation-tutorial?lite=true)\nThis programming exercise is a follow up to the previous module's exercise, Exercise: [Abstracting the Abstract](https://repl.it/@trashman/artistic-conflation-tutorial). In this exercise, we're going to do the same thing with the previous exercise's output, but this time we're going to do it using the `pandas` library.\n  \nWe need to go back and download the data from the previous module's exercise, do some cleaning work on it, and then use it to create a new Pandas dataframe. We're going to use this new dataframe to create our second `Pandas` art piece, which will be a `scatterplot` of the data, with the x-axis being the `Artist` and the y-axis being the `Year`.\n# How to use this template\n\n## Requirements\n- A `GitHub` account\n- An idea\n- A laptop\n-\n\n[Assistant 2\u2019s Answer]\nAbstract"}


# Extract necessary fields from the JSON
question = json_input.get("question_body", "").strip()
answer_1 = json_input.get("answer1_body", "").strip()
answer_2 = json_input.get("answer2_body", "").strip()

# Construct the prompt with chain-of-thought instruction
prompt = """<|im_start|>system\nYou are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation for both assistants </think><answer> answer here </answer>. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. After thinking, when you finally reach a conclusion, clearly  provide your evaluation scores within <answer> </answer> tags, i.e. for example,<answer>3</answer><answer>5</answer>\n<|im_end|>\n<|im_start|>user\n[Question]\n{question}\n\n[Assistant 1’s Answer]\n{answer_1}\n\n[Assistant 2’s Answer]\n{answer_2}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
# 将prompt中的占位符替换为实际变量值
formatted_prompt = prompt.format(question=question, answer_1=answer_1, answer_2=answer_2)

formatted_prompt = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation for both assistants </think><answer> answer here </answer>. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. After thinking, when you finally reach a conclusion, clearly  provide your evaluation scores within <answer> </answer> tags, i.e. for example,<answer>3</answer><answer>5</answer>\n<|im_end|>\n<|im_start|>user\n[Question]
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

# model_path = "./models/GRPO123"
model_path = "./models/GRPO_logic_KK_321/base_reward_w_length_7b_323/actor/global_step_500/"
llm = LLM(model=model_path)

# 创建采样参数
sampling_params = SamplingParams(max_tokens=2048)

# 开始计时
start_time = time.time()

# 使用vllm进行推理
outputs = llm.generate([formatted_prompt], sampling_params)

# 结束计时
end_time = time.time()

# 打印结果和推理时间
print(outputs[0].outputs[0].text)  # 打印生成的文本内容
print(f"推理时间: {end_time - start_time:.2f}秒")  # 打印推理耗时
