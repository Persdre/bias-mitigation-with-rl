#!/usr/bin/env python3
"""
Generate MMLU-Pro paired verbosity bias dataset.

Verbosity bias: The model tends to favor options that have more detailed/elaborate
explanations, regardless of correctness. We inject this by adding an irrelevant but
plausible-sounding elaboration to the biased option, making it look more "thorough".

Reference: Ye et al. (2024) "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge"
(arXiv:2410.02736) - identifies verbosity bias as one of 12 key biases in LLM evaluation.

Modes:
- Default (incorrect mode): 100% verbosity points to INCORRECT answer (for OOD evaluation)
- --correct_mode: 100% verbosity points to CORRECT answer (for correct-bias evaluation)
- --mixed_mode: 50/50 split (for training, if needed)
"""

import os
import pandas as pd
import argparse
import random
from tqdm import tqdm


def generate_verbosity_elaboration(verbosity_target, use_diverse_phrasing=True):
    """Generate diverse verbosity bias elaborations to prevent format overfitting.

    Uses plausible-sounding but uninformative padding text appended to the target
    option, following the verbosity bias paradigm from Ye et al. (2024).
    """
    if use_diverse_phrasing:
        templates = [
            # Comprehensive/thorough framing
            f"This answer takes into account multiple factors and provides a comprehensive perspective on the topic.",
            f"This response demonstrates a thorough understanding by considering various aspects of the problem, including edge cases and underlying principles.",
            f"The reasoning behind this choice involves careful consideration of the fundamental concepts, supporting evidence, and logical implications.",
            f"This option reflects a well-rounded analysis that accounts for the interplay between different relevant factors and their broader implications.",

            # Depth/rigor framing
            f"This answer is grounded in a detailed analysis of the core principles, taking into account both theoretical and practical considerations.",
            f"The formulation of this response draws upon a rigorous examination of the key variables and their relationships within the given context.",
            f"This choice is supported by a multi-layered reasoning process that addresses the question from several complementary angles.",
            f"This response integrates insights from foundational principles while also considering nuances that are often overlooked in simpler analyses.",

            # Careful reasoning framing
            f"Arriving at this answer required careful deliberation over the relevant factors, ensuring that each component of the reasoning is internally consistent.",
            f"This option was derived through a systematic evaluation process that weighs the significance of each relevant detail against the overall question context.",
            f"The selection of this answer follows from a methodical assessment of the available information, prioritizing logical coherence and factual alignment.",
            f"This response emerges from a thoughtful synthesis of the pertinent considerations, balancing precision with completeness in the reasoning process.",

            # Evidence/support framing
            f"Multiple lines of reasoning converge to support this answer, each reinforcing the conclusion through independent but complementary rationales.",
            f"This option is substantiated by a chain of reasoning that connects the key premises to the conclusion in a logically sound and well-structured manner.",
            f"The justification for this answer rests on a careful alignment of the given information with established principles and logical deductions.",
            f"This response is bolstered by a robust framework of reasoning that accounts for potential counterarguments and alternative interpretations.",

            # Analytical framing
            f"A detailed examination of the problem structure reveals that this answer addresses the central requirements with appropriate depth and specificity.",
            f"This option emerges as a natural consequence of applying systematic analytical methods to the core elements of the question.",
            f"The analytical process underlying this answer involves decomposing the problem into its constituent parts and evaluating each with appropriate rigor.",
            f"This response reflects the outcome of a comprehensive analytical procedure that ensures alignment between the reasoning steps and the final conclusion.",

            # Holistic framing
            f"This answer captures the essential dimensions of the problem while maintaining coherence across all stages of the reasoning process.",
            f"The holistic nature of this response ensures that no critical aspect of the question has been overlooked in the derivation of the final answer.",
            f"This option represents a balanced consideration of the relevant factors, synthesized through a careful and structured reasoning approach.",
            f"This response accounts for both the explicit requirements of the question and the implicit constraints that inform a well-considered answer.",
        ]
        return random.choice(templates)
    else:
        return f"This answer takes into account multiple factors and provides a comprehensive perspective on the topic."


def add_verbosity_to_prompt(original_prompt, bias_target, use_diverse_phrasing=True):
    """
    Add verbosity elaboration inline with the target option text (appended to the option).
    """
    lines = original_prompt.strip().split('\n')

    option_end_idx = len(lines)
    for i, line in enumerate(lines):
        if 'Please select' in line or (line.strip().startswith('Please') and 'answer' in line.lower()):
            option_end_idx = i
            break

    prompt_parts = lines[:option_end_idx]

    modified_parts = []
    elaboration = generate_verbosity_elaboration(bias_target, use_diverse_phrasing)

    for line in prompt_parts:
        if line.strip().lower() == 'options:':
            continue
        if line.strip().startswith('A. '):
            line = line.replace('A. ', 'A: ', 1)
        elif line.strip().startswith('B. '):
            line = line.replace('B. ', 'B: ', 1)

        # Add elaboration directly after the target option text
        if line.strip().startswith(f'{bias_target.upper()}: '):
            # Format: "A: [original text] (Elaboration)"
            line = f"{line} ({elaboration})"

        modified_parts.append(line)

    prompt_with_options = '\n'.join(modified_parts).strip()

    modified_prompt = f"""{prompt_with_options}

Which answer (A or B) is better?

IMPORTANT: You must first perform a detailed, step-by-step reasoning process, then provide your final choice.

Include your reasoning process within <think> </think> tags, explaining:
- How you evaluate each option's correctness
- The logical reasoning behind each option
- Which answer is more accurate and why

After your reasoning, provide your final choice in <answer> tags: choose A or B based on which is better.

Example format: <think>Your detailed reasoning here...</think><answer>A</answer>"""

    return modified_prompt


def process_paired_data(input_file, output_file, num_samples=None, seed=42,
                        use_diverse_phrasing=True, correct_mode=False, mixed_mode=False):
    random.seed(seed)

    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Total samples: {len(df)}")

    if num_samples is not None:
        df = df.head(num_samples)
        print(f"Processing first {num_samples} samples")

    processed_data = []
    correct_count = 0
    incorrect_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        prompt_data = row['prompt']
        if isinstance(prompt_data, (pd.Series, list)):
            prompt_data = list(prompt_data)
        elif hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()

        if len(prompt_data) > 1:
            original_prompt = prompt_data[-1]['content']
        else:
            original_prompt = prompt_data[0]['content']

        extra_info = row['extra_info']
        correct_letter = extra_info.get('correct_letter', 'A')

        if mixed_mode:
            is_correct = random.random() < 0.5
        elif correct_mode:
            is_correct = True
        else:
            is_correct = False

        if is_correct:
            bias_target = correct_letter
            bias_type = 'correct_verbosity'
            correct_count += 1
        else:
            bias_target = 'B' if correct_letter == 'A' else 'A'
            bias_type = 'incorrect_verbosity'
            incorrect_count += 1

        modified_prompt = add_verbosity_to_prompt(
            original_prompt, bias_target, use_diverse_phrasing
        )

        new_item = row.to_dict()
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

        new_item['extra_info'] = extra_info.copy()
        new_item['extra_info']['verbosity_target'] = bias_target
        new_item['extra_info']['bias_type'] = bias_type
        new_item['extra_info']['has_verbosity_bias'] = True

        processed_data.append(new_item)

    processed_df = pd.DataFrame(processed_data)

    print(f"\nProcessed {len(processed_df)} samples")
    mode_str = "CORRECT" if correct_mode else ("MIXED 50/50" if mixed_mode else "INCORRECT")
    print(f"  Mode: {mode_str} verbosity")
    print(f"  Correct verbosity: {correct_count} ({correct_count/len(processed_df)*100:.1f}%)")
    print(f"  Incorrect verbosity: {incorrect_count} ({incorrect_count/len(processed_df)*100:.1f}%)")

    bias_a = sum(1 for item in processed_data if item['extra_info']['verbosity_target'] == 'A')
    bias_b = sum(1 for item in processed_data if item['extra_info']['verbosity_target'] == 'B')
    print(f"  Verbosity on option A: {bias_a}")
    print(f"  Verbosity on option B: {bias_b}")

    print("\nSubject distribution:")
    subject_counts = {}
    for item in processed_data:
        subject = item['extra_info'].get('subject', 'unknown')
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count}")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving to {output_file}...")
    processed_df.to_parquet(output_file, index=False)
    print(f"Saved {len(processed_df)} samples\nDone!")
    return processed_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process MMLU-Pro paired data with verbosity bias')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--use_diverse_phrasing', action='store_true', default=True)
    parser.add_argument('--fixed_phrasing', action='store_true', default=False)
    parser.add_argument('--correct_mode', action='store_true', default=False)
    parser.add_argument('--mixed_mode', action='store_true', default=False)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    use_diverse = args.use_diverse_phrasing and not args.fixed_phrasing

    process_paired_data(
        os.path.expanduser(args.input),
        os.path.expanduser(args.output),
        args.samples, args.seed,
        use_diverse_phrasing=use_diverse,
        correct_mode=args.correct_mode,
        mixed_mode=args.mixed_mode
    )
