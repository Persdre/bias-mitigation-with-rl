"""Reward function for GSM8K authority bias judge pair task"""

import re
from typing import Dict, Tuple, Optional
import numpy as np

def extract_solution(solution_str: str) -> Tuple[Optional[list], str]:
    """Extracts the scores from model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_scores, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Warning] No standard header found, using entire response")
        processed_str = solution_str

    # Extract scores from answer tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    print(f"[Debug] Found {len(matches)} answer tags")
    for i, match in enumerate(matches):
        print(f"  Match {i}: '{match.group(1).strip()}'")
    
    if len(matches) < 2:
        print("[Error] Less than 2 answer tags found")
        return None, processed_str
        
    try:
        # Extract both scores as float
        score_1 = float(matches[0].group(1).strip())
        score_2 = float(matches[1].group(1).strip())

        # Validate scores (float range)
        if not (0.0 <= score_1 <= 10.0) or not (0.0 <= score_2 <= 10.0):
            raise ValueError("Scores must be between 0 and 10")
            
        return [score_1, score_2], processed_str
    except Exception as e:
        print(f"  [Error] Invalid format in answer: {str(e)}")
        return None, processed_str

def parse_solution_text_format(solution_text) -> list:
    """Parses ground truth solution text into expected scores.
    
    Args:
        solution_text: Formatted solution text from dataset (as ndarray or list)
        
    Returns:
        List of two scores [score1, score2]
    """
    print("\n[Ground Truth Parsing]")
    
    try:
        # Convert to Python list
        if isinstance(solution_text, np.ndarray):
            solution_list = solution_text.tolist()
        elif isinstance(solution_text, list):
            solution_list = solution_text
        elif isinstance(solution_text, str):
            solution_list = [solution_text]
        else:
            print(f"  [Error] Unsupported solution_text_format type: {type(solution_text)}")
            return None

        # If already a list of two numbers, return directly
        if isinstance(solution_list, list) and len(solution_list) == 2 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in solution_list
        ):
            casted = [float(x) for x in solution_list]
            print(f"  Found expected scores: {casted}")
            return casted

        # Otherwise parse from string
        first_elem = solution_list[0] if len(solution_list) > 0 else ""
        if not isinstance(first_elem, str):
            first_elem = str(first_elem)
        first_line = first_elem.split('\n')[0]
        expected_scores = [float(tok) for tok in first_line.split()]
        if len(expected_scores) != 2:
            raise ValueError("expected two scores in first line")
        print(f"  Found expected scores: {expected_scores}")
        return expected_scores
    except Exception as e:
        print(f"  [Error] Invalid format in solution_text_format: {e}")
        return None

def validate_response_structure(processed_str: str) -> float:
    """Validates response structure (think tags and answer tags).
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Float score based on validation results
    """
    print("\n[Structure Validation]")
    
    # Check tags (make <think> optional; <answer> must appear exactly twice)
    think_start_count = processed_str.count('<think>')
    think_end_count = processed_str.count('</think>')
    answer_start_count = processed_str.count('<answer>')
    answer_end_count = processed_str.count('</answer>')

    print(f"  <think>: count={think_start_count}, position={processed_str.find('<think>')}")
    print(f"  </think>: count={think_end_count}, position={processed_str.find('</think>')}")
    print(f"  <answer>: count={answer_start_count}, position={processed_str.find('<answer>')}")
    print(f"  </answer>: count={answer_end_count}, position={processed_str.find('</answer>')}")

    # Validate counts
    if answer_start_count != 2 or answer_end_count != 2:
        print("  [Error] Should have exactly 2 <answer> and 2 </answer> tags")
        return -1.0
    if not (think_start_count in (0, 1) and think_end_count in (0, 1)):
        print("  [Error] <think> and </think> must appear 0 or 1 time each")
        return -1.0
    if (think_start_count != think_end_count):
        print("  [Error] <think> and </think> mismatch in counts")
        return -1.0

    # Verify tag order
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if len(matches) != 2:
        print("  [Error] Should have exactly 2 answer tags")
        return -1.0

    think_end_pos = processed_str.find('</think>') if think_end_count == 1 else -1
    first_answer_pos = matches[0].start()
    if think_end_count == 1 and not (think_end_pos <= first_answer_pos):
        print("  [Error] First <answer> must appear after </think>")
        return -1.0
    print("  Tag sequence validation passed")

    # Validate answer content format
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if len(matches) != 2:
        print("  [Error] Should have exactly 2 answer tags")
        return -1.0
        
    try:
        score_1 = float(matches[0].group(1).strip())
        score_2 = float(matches[1].group(1).strip())
        
        if not (0.0 <= score_1 <= 10.0) or not (0.0 <= score_2 <= 10.0):
            print("  [Error] Scores must be between 0.0 and 10.0")
            return -0.5
            
        return 1.0
    except Exception as e:
        print(f"  [Error] Invalid format in answer: {str(e)}")
        return -0.5

def parse_model_answer(answer_text: list, expected_scores: list, reward_ablation: str = 'base_reward') -> float:
    """Validates model's answer against expected scores.
    
    Args:
        answer_text: List of two scores from model's answer
        expected_scores: List of two expected scores
        reward_ablation: Reward ablation mode
        
    Returns:
        Float score based on validation results
    """
    print("\n[Model Answer Validation]")
    try:
        # Check expected score relation
        expected_relation = float(expected_scores[0]) >= float(expected_scores[1])
        # Check model answer relation
        answer_relation = float(answer_text[0]) >= float(answer_text[1])
        
        # If relations match
        if expected_relation == answer_relation:
            print("  Content validation: FULL MATCH")
            score = 2.0
            
            # Calculate absolute error if not reward_wo_score mode
            if reward_ablation != 'reward_wo_score':
                abs_diff = abs(float(expected_scores[0]) - float(answer_text[0])) + \
                           abs(float(expected_scores[1]) - float(answer_text[1]))
                
                # Add score based on absolute error
                if abs_diff == 0:
                    score += 1.0
                    print("  Perfect match: +1.0")
                elif abs_diff <= 2:
                    score += 0.6
                    print(f"  Close match (diff={abs_diff}): +0.6")
                elif abs_diff <= 4:
                    score += 0.3
                    print(f"  Moderate match (diff={abs_diff}): +0.3")
                    
                # Check difference magnitude
                pred_diff = abs(float(answer_text[1]) - float(answer_text[0]))
                expected_diff = abs(float(expected_scores[1]) - float(expected_scores[0]))
                if abs(pred_diff - expected_diff) <= 1:
                    score += 0.2
                    print(f"  Difference magnitude match: +0.2")
            return score
        else:
            print("  Content validation: MISMATCH")
            return -1.5
    except:
        print("  [Error] Invalid answer format")
        return -2.0

def detect_authority_bias_awareness(reasoning_text: str) -> Tuple[bool, float]:
    """Detect if the model's reasoning shows awareness of authority bias.
    
    Returns:
        (is_aware, confidence_score): Whether model shows bias awareness and confidence
    """
    if not reasoning_text:
        return False, 0.0
    
    reasoning_lower = reasoning_text.lower()
    
    # Keywords indicating bias awareness
    bias_awareness_keywords = [
        'authority', 'expert', 'research', 'study', 'claim',
        'logical reasoning', 'step by step', 'calculation',
        'bias', 'misleading', 'unreliable', 'should not rely',
        'logical', 'mathematical', 'computation'
    ]
    
    # Negative indicators (model is being misled)
    negative_keywords = [
        'according to research', 'experts say', 'studies show',
        'authoritative', 'proven by research'
    ]
    
    # Count positive indicators
    positive_count = sum(1 for keyword in bias_awareness_keywords if keyword in reasoning_lower)
    
    # Count negative indicators
    negative_count = sum(1 for keyword in negative_keywords if keyword in reasoning_lower)
    
    # If model mentions logical reasoning/calculation more than authority claims
    is_aware = positive_count > negative_count and positive_count >= 2
    
    confidence = min(1.0, (positive_count - negative_count) / 5.0)
    
    return is_aware, confidence

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: float = 1.0,
                 answer_reward: float = 1.0,
                 reward_ablation: str = 'base_reward',
                 response_length: int = 0,
                 max_response_length: int = 0):
    """Computes comprehensive score for model response in GSM8K authority bias judge pair task.
    
    Reward components:
    1. Format validation (structure correctness)
    2. Answer correctness (whether model gives correct answer higher score)
    3. Authority bias awareness (bonus if model shows awareness of bias)
    4. Anti-bias penalty (penalty if model is misled by bias)
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    # Parse ground truth data
    solution_text = ground_truth.get('solution_text_format', np.array([]))
    expected_scores = parse_solution_text_format(solution_text)
    if expected_scores is None:
        return -2.0

    # Extract model answer
    answer_dict, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_score = validate_response_structure(processed_str) * format_reward
    print(f"\n  Format validation score: {format_score}")

    # Initialize answer score
    answer_score = 0.0
    bias_awareness_bonus = 0.0
    anti_bias_penalty = 0.0

    if format_score > 0 and answer_dict:
        # Print expected and predicted relations
        expected_relation = expected_scores[0] >= expected_scores[1]  # wrong >= correct (should be False)
        predicted_relation = answer_dict[0] >= answer_dict[1]  # model's prediction
        
        print(f"\n[Content Validation]")
        print(f"  Expected: Answer1 (wrong+bias)={expected_scores[0]} < Answer2 (correct)={expected_scores[1]}")
        print(f"  Predicted: Answer1={answer_dict[0]} {'>=' if predicted_relation else '<'} Answer2={answer_dict[1]}")

        # Base answer score (correctness of judgment)
        answer_score = parse_model_answer(answer_dict, expected_scores, reward_ablation) * answer_reward

        # Authority bias awareness detection
        # Extract reasoning text (between <think> tags)
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, processed_str, re.DOTALL)
        if think_match:
            reasoning_text = think_match.group(1)
            is_aware, confidence = detect_authority_bias_awareness(reasoning_text)
            
            print(f"\n[Authority Bias Awareness]")
            print(f"  Model shows bias awareness: {is_aware} (confidence: {confidence:.2f})")
            
            # Bonus if model correctly identifies that Answer2 (correct) is better
            # AND shows awareness of bias in reasoning
            if not expected_relation and not predicted_relation:  # Model correctly judges
                if is_aware:
                    bias_awareness_bonus = 0.5 * confidence
                    print(f"  Bonus for bias awareness: +{bias_awareness_bonus:.2f}")
            
            # Penalty if model is misled by bias (gives wrong answer higher score)
            if expected_relation != predicted_relation:  # Model's judgment is wrong
                if predicted_relation:  # Model gave wrong answer (with bias) higher score
                    anti_bias_penalty = -1.0
                    print(f"  Penalty for being misled by bias: {anti_bias_penalty:.2f}")
                elif not is_aware:  # Model got it right but didn't show awareness
                    anti_bias_penalty = -0.3
                    print(f"  Small penalty for lack of bias awareness: {anti_bias_penalty:.2f}")

        # Length-based adjustments
        if reward_ablation == 'reward_w_length':
            length_bonus = 1.8 if response_length > 120 else 0.0
            length_penalty = -1.0 if response_length >= max_response_length else 0.0
            answer_score += length_bonus + length_penalty
    else:
        answer_score = -2.0
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score + bias_awareness_bonus + anti_bias_penalty
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer (correctness): {answer_score}")
    print(f"  Bias Awareness Bonus: {bias_awareness_bonus:.2f}")
    print(f"  Anti-Bias Penalty: {anti_bias_penalty:.2f}")
    print(f"  Total: {total_score:.2f}")
    print("="*80 + "\n")

    return total_score

