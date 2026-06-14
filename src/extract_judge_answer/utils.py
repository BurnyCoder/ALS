# Local: file src/extract_judge_answer/utils.py provides first-party ALS source context. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
import json  # Local: imports json for this module. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
import re  # Local: imports re for this module. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
# Local: imports selected helpers from math_verify. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
from math_verify import parse, verify
# Local: imports selected helpers from .grader. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
from .grader import math_equal_process
# Local: imports selected helpers from .math_equivalent_MATH. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
from .math_equivalent_MATH import is_equiv
# Local: imports selected helpers from .parse_utils_qwen. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
from .parse_utils_qwen import extract_answer as extract_fn
# Local: defines the extract_true_answer function. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
def extract_true_answer(text, name="gsm8k"):
    # Local: starts a multi-line text literal that Python treats as one value. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    '''
    Extract answer from text

    Args:
        text: input text
        name: name of the dataset

    Returns:
        answer: extracted answer
    '''
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if "gsm8k" in name:
        # Local: sets label for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        label = text.split("#### ")[1]
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return label
    # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    elif "MATH-500" in name:
        return text  # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    elif "AIME_2024" in name:
        return text  # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    else:
        # Local: stops execution with an explicit error for invalid state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        raise ValueError(f"Unknown dataset name: {name}")


# Local: defines the judge_answer function. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
def judge_answer(input, label, data_name="gsm8k", extract=True, prompt_idx=0):
    # Local: starts a multi-line text literal that Python treats as one value. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    """Score.

    Judge whether the answer is correct or not.
    Only exact match is considered correct.

    Args:
        input (str): model response
        label (str): ground truth
        data_name (str): name of the dataset, ["gsm8k", "MATH-500"]
        extract (bool): whether to extract answer from model response
        prompt_idx (int): index of the solver prompt (different format) 

    Returns:
        bool: True if the answer is correct, False otherwise
    """
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if "gsm8k" in data_name:
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if extract:
            # Local: sets input for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            input = extract_answer(input, data_name="gsm8k", prompt_idx=prompt_idx)
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return (input == label)
    # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    elif "MATH-500" in data_name:
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if extract:
            # Local: sets input for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            input = extract_answer(input, data_name="MATH-500", prompt_idx=prompt_idx)

        # huggingface math_verify
        # Local: sets hf_input for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        hf_input = parse(input)
        # Local: sets hf_verifier_judge for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        hf_verifier_judge = verify(label, hf_input)
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if hf_verifier_judge:
            # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            return True

        # qwen2.5-math 
        # Local: sets qwen_verifier_judge for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        qwen_verifier_judge = math_equal_process((label, input))
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if qwen_verifier_judge:
            # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            return True

        # exact match
        # Local: executes this statement in the current code path. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        exact_judge = (str(input) == str(label))
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if exact_judge:
            # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            return True

        # MATH-500
        # Local: sets MATH_500_judge for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        MATH_500_judge = is_equiv(str(label), str(input))
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if MATH_500_judge:
            # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            return True
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return False

    # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    elif "AIME_2024" in data_name:
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if extract:
            # Local: sets input for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            input = extract_answer(input, data_name="AIME_2024", prompt_idx=prompt_idx)
            # Local: sets input for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            input = str(input)
            # Local: sets label for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            label = str(label)
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return (input == label)

    # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    else:
        # Local: stops execution with an explicit error for invalid state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        raise ValueError(f"Unknown dataset name: {data_name} for judge answer")
    
    
# Local: defines the extract_answer function. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
def extract_answer(text, data_name="gsm8k", prompt_idx=0, model_name="Qwen2.5-7B-Instruct"):
    # Local: starts a multi-line text literal that Python treats as one value. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    '''
    Extract answer from model response

    Args:
        text: Raw response string from the language model
        data_name: name of the dataset, ["gsm8k", "MATH-500"]
        prompt_idx: index of the solver prompt (different format)

    Returns:
        answer: extracted answer(pure numbers)
    '''
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if "gsm8k" in data_name:
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if prompt_idx == 0:
            # 0: boxed
            # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            if "qwen2.5-1.5b-instruct" in model_name.lower():
                # well, well, well
                # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                temp = _extract_qwen25_1_5B_answer(text)
            # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            else:
                # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                temp = _extract_answer(text)
            # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            return temp

        # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        elif prompt_idx == 1:
            # 1: json
            # Local: starts a protected operation that may fail on external or parsed input. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            try:
                # Local: parses generated text as JSON when the prompt requires structure. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                answer = json.loads(text.strip('` \n'))
                # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                final_answer = answer.get('final answer', '')
                # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                if not isinstance(final_answer, str):
                    # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    final_answer = str(final_answer)
                # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                temp = _extract_answer(final_answer)
                # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                return temp

            # Local: handles a recoverable failure from the protected operation. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            except json.JSONDecodeError:
                # Local: sets pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)[}<]'

                # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL) 
                
                # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                if match:
                    # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    temp = _extract_answer(match.group(1))
                    # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    return temp
                # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                else:
                    # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    temp = _extract_answer(text)
                    # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    return temp


        # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        else:
            # Local: stops execution with an explicit error for invalid state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")

    # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    elif "MATH-500" in data_name:
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if prompt_idx == 0:
            # 0: boxed
            # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            temp = extract_fn(text, data_name='math')
            # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            return temp

        # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        elif prompt_idx == 1:
            # json
            # Local: starts a protected operation that may fail on external or parsed input. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            try:
                # Local: parses generated text as JSON when the prompt requires structure. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                answer = json.loads(text.strip('` \n'))
                # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                final_answer = answer.get('final answer', '')
                # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                if not isinstance(final_answer, str):
                    # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    final_answer = str(final_answer)
                # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                final_answer = final_answer.replace("\n", "")
                # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                final_answer = final_answer.replace("\"", "")
                # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                final_answer = final_answer.replace("\'", "")
                # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                return final_answer

            # Local: handles a recoverable failure from the protected operation. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            except json.JSONDecodeError:
                # Local: sets text for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                text = text.replace("\n", "")
                # Local: sets pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)(}<|<\|)'


                # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL) 
                
                # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                if match:
                    # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    temp = match.group(1)
                    # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    temp = temp.replace("\n", "")
                    # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    temp = temp.replace("\"", "")
                    # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    temp = temp.replace("\'", "")
                    # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    return temp
                # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                else:
                    # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    return None

    # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    elif "AIME_2024" in data_name:
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if prompt_idx == 0:
            # 0: boxed
            # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            temp = _extract_answer(text)
            # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            return temp

        # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        elif prompt_idx == 1:
            # 1: json, {"final answer": ...}
            # Local: starts a protected operation that may fail on external or parsed input. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            try:
                # Local: parses generated text as JSON when the prompt requires structure. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                answer = json.loads(text.strip('` \n'))
                # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                final_answer = answer.get('final answer', '')
                # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                if not isinstance(final_answer, str):
                    # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    final_answer = str(final_answer)
                # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                temp = _extract_answer(final_answer)
                # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                return temp

            # Local: handles a recoverable failure from the protected operation. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            except json.JSONDecodeError:
                # Local: sets pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)[}<]'

                # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL) 
                
                # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                if match:
                    # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    temp = _extract_answer(match.group(1))
                    # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    return temp
                # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                else:
                    # Local: sets temp for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    temp = _extract_answer(text)
                    # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                    return temp


        # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        else:
            # Local: stops execution with an explicit error for invalid state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")
    # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    else:
        # Local: stops execution with an explicit error for invalid state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        raise ValueError(f"Unknown dataset name: {data_name} for extract answer")



######################
#       MATH         #
######################

# Local: defines the extract_MATH_solution function. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
def extract_MATH_solution(solution_str: str):
    # Local: starts a multi-line text literal that Python treats as one value. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        extracted final answer
    """""
    # Split response to isolate assistant output
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if "Assistant:" in solution_str:
        # Local: sets processed_str for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        processed_str = solution_str.split("Assistant:", 1)[1]
    # Local: checks the next mutually exclusive condition. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    elif "<|im_start|>assistant" in solution_str:
        # Local: sets processed_str for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # Local: handles the remaining branch after earlier checks fail. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    else:
        # Local: sets processed_str for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        processed_str = solution_str

    # Extract final answer using XML-style tags
    # Local: sets answer_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    answer_pattern = r'<answer>.*?(\\boxed{.*}).*?</answer>'
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if not matches:
        # Local: sets answer_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        answer_pattern = r'\\boxed{(.*)}'
        # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if not matches:
        # Local: reports progress or diagnostics to the run log. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        print("[Error] No valid answer tags found")
        return None  # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    # Local: sets final_answer for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    final_answer = matches[-1].group(1).strip()
    # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    return final_answer


# Local: defines the _extract_answer function. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
def _extract_answer(text):
    # Local: starts a multi-line text literal that Python treats as one value. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    """
    Extract numerical answer from generated text.
    handling various edge cases.
    
    Args:
        text (str): Generated text to extract answer from.
    
    Returns:
        str or None: Extracted numerical answer, or None if not found.
    """
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if text is None:
        return None  # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    
    text = text.strip()  # Local: sets text for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.

    # Local: defines the clean_number function. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    def clean_number(num_str):
        # Local: executes this statement in the current code path. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        """Remove currency symbols, commas, and whitespace."""
        # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        num_str = re.sub(r'[$€£¥]', '', num_str)
        # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        num_str = re.sub(r',', '', num_str)
        # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        num_str = re.sub(r'\s', '', num_str)
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return num_str

    ### Several Corner Cases ###
    # 1. \boxed{}
    # Local: sets boxed_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    boxed_pattern = r"\\boxed\{\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*\}"
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if match:
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return clean_number(match.group(1))
    
    # 2. Answer:
    # Local: sets answer_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    answer_pattern = r"Answer:\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    match = re.search(answer_pattern, text, re.IGNORECASE)
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if match:
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return clean_number(match.group(1))
    
    # 3. =
    # Local: sets equals_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    equals_pattern = r"=\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    match = re.search(equals_pattern, text)
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if match:
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return clean_number(match.group(1))

    # 4. With currency unit
    # Local: sets currency_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    currency_pattern = r"is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*(?:dollars|euros|pounds|yen)"
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    match = re.search(currency_pattern, text, re.IGNORECASE)
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if match:
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return clean_number(match.group(1))

    # 5. Search from the last line of the text upwards, matching independent numbers
    # Local: sets lines for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    lines = text.split('\n')
    # Local: iterates through the current collection. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    for line in reversed(lines):
        # Local: sets line for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        line = line.strip()
        # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        if line:
            # Local: sets final_num_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            final_num_pattern = r"([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*$"
            # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            match = re.search(final_num_pattern, line)
            # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
            if match:
                # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
                return clean_number(match.group(1))

    # 6. Returns the last matching number in the text
    # Local: sets number_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    number_pattern = r"([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    matches = re.findall(number_pattern, text)
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if matches:
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return clean_number(matches[-1])

    return None  # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.


# Local: defines the _extract_qwen25_1_5B_answer function. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
def _extract_qwen25_1_5B_answer(text):
    # Local: starts a multi-line text literal that Python treats as one value. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    """
    Extract numerical answer from generated text for Qwen-2.5 1.5B model.
    handling various edge cases.

    Args:
        text (str): Generated text to extract answer from.

    Returns:
        str or None: Extracted numerical answer, or None if not found.
    """
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if text is None:
        return None  # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.

    text = text.strip()  # Local: sets text for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.

    # Local: defines the clean_number function. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    def clean_number(num_str):
        # Local: executes this statement in the current code path. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        """Remove currency symbols, commas, and whitespace."""
        # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        num_str = re.sub(r'[$€£¥]', '', num_str)
        # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        num_str = re.sub(r',', '', num_str)
        # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        num_str = re.sub(r'\s', '', num_str)
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return num_str

    ### Several Corner Cases ###
    # 1. \boxed{}
    # Local: sets boxed_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    boxed_pattern = r"\\boxed\{\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*\}"
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if match:
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return clean_number(match.group(1))

    # 2. he answer is
    # Local: sets answer_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    answer_pattern = r"he answer is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    match = re.search(answer_pattern, text, re.IGNORECASE)
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if match:
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return clean_number(match.group(1))

    # 3. final answer is
    # Local: sets answer_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    answer_pattern = r"final answer is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    match = re.search(answer_pattern, text, re.IGNORECASE)
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if match:
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return clean_number(match.group(1))

    # 4. Returns the last matching number in the text
    # Local: sets number_pattern for later use in this scope. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    number_pattern = r'\d+(?:,\d+)*(?:\.\d+)?'
    # Local: applies a regular expression to normalize or extract answer text. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    matches = re.findall(number_pattern, text)
    # Local: opens a condition that selects behavior from current state. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
    if matches:
        # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
        return clean_number(matches[-1])

    return None  # Local: returns the computed result to the caller. Global: extracts and judges final answers so ALS accuracy can be measured consistently.
