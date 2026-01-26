"""
Create position-bias evaluation datasets.

Outputs:
1) validation_paired_position_bias_random.parquet  (from validation_paired.parquet)
2) test_paired_ood_position_bias_random.parquet    (from test_paired_ood.parquet)

For each source QA pair, we emit three rows with identical question content:
  - position_type = "random"        (original ordering)
  - position_type = "correct_in_a"  (correct option forced to A)
  - position_type = "correct_in_b"  (correct option forced to B)

Prompts keep the same format as other MMLU-Pro paired datasets and do not add
any extra bias text besides swapping option order.
"""

import os
import numpy as np
import pandas as pd


def build_prompt(question: str, opt_a: str, opt_b: str) -> np.ndarray:
    """Return a numpy array of chat messages in the expected format."""
    content = (
        f"Question: {question}\n\n\n"
        f"Options:\n"
        f"A. {opt_a}\n"
        f"B. {opt_b}\n\n"
        f"Please select the correct answer by responding with only the letter of the correct option (A or B)."
    )
    return np.array([{"role": "user", "content": content}], dtype=object)


def make_rows(df: pd.DataFrame, split_name: str):
    """Generate rows for a given split (validation or train)."""
    rows = []
    for _, row in df.iterrows():
        ei = row["extra_info"]
        question = ei.get("question", "")
        opt_correct = ei.get("original_correct_text", "")
        opt_incorrect = ei.get("selected_incorrect_text", "")
        correct_letter = ei.get("correct_letter", "A")
        subject = ei.get("subject", "unknown")

        # Skip if essential fields are missing
        if not question or not opt_correct or not opt_incorrect:
            continue

        # Define three variants: random (use original ordering), correct_in_a, correct_in_b
        variants = [
            ("random", opt_correct if correct_letter == "A" else opt_incorrect,
             opt_correct if correct_letter == "B" else opt_incorrect,
             correct_letter),
            ("correct_in_a", opt_correct, opt_incorrect, "A"),
            ("correct_in_b", opt_incorrect, opt_correct, "B"),
        ]

        for position_type, opt_a, opt_b, corr_letter in variants:
            extra_info = {
                "question": question,
                "option_a_text": opt_a,
                "option_b_text": opt_b,
                "correct_letter": corr_letter,
                "correct_is_a": corr_letter == "A",
                "subject": subject,
                "position_type": position_type,
                "bias_type": "position",
            }

            rows.append(
                {
                    "data_source": row.get("data_source", "mmlupro"),
                    "prompt": build_prompt(question, opt_a, opt_b),
                    "ability": row.get("ability", None),
                    "reward_model": row.get("reward_model", ""),
                    "extra_info": extra_info,
                }
            )

    return pd.DataFrame(rows)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    val_path = os.path.join(base_dir, "validation_paired.parquet")
    test_path = os.path.join(base_dir, "test_paired_ood.parquet")

    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Missing source file: {val_path}")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Missing source file: {test_path}")

    # print("Loading validation source ...")
    # val_df = pd.read_parquet(val_path)
    # print("Building validation position-bias dataset ...")
    # val_out = make_rows(val_df, split_name="validation")
    # val_out_path = os.path.join(base_dir, "validation_paired_position_bias_random.parquet")
    # val_out.to_parquet(val_out_path, index=False)
    # print(f"Saved {len(val_out)} rows to {val_out_path}")

    print("Loading test OOD source ...")
    test_df = pd.read_parquet(test_path)
    print("Building OOD position-bias dataset ...")
    test_out = make_rows(test_df, split_name="test_ood")
    test_out_path = os.path.join(base_dir, "test_paired_ood_position_bias_random.parquet")
    test_out.to_parquet(test_out_path, index=False)
    print(f"Saved {len(test_out)} rows to {test_out_path}")


if __name__ == "__main__":
    main()

