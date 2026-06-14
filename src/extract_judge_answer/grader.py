# Local: file src/extract_judge_answer/grader.py provides first-party ALS source context. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
# Local: starts a multi-line text literal that Python treats as one value. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
"""

import re  # Local: imports re for this module. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
import regex  # Local: imports regex for this module. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
import multiprocessing  # Local: imports multiprocessing for this module. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
from math import isclose  # Local: imports selected helpers from math. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
from typing import Union  # Local: imports selected helpers from typing. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
# Local: imports selected helpers from collections. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
from collections import defaultdict

# Local: imports selected helpers from sympy. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
from sympy import simplify, N
# Local: imports selected helpers from sympy.parsing.sympy_parser. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
from sympy.parsing.sympy_parser import parse_expr
# Local: imports selected helpers from sympy.parsing.latex. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
from sympy.parsing.latex import parse_latex
# Local: imports selected helpers from latex2sympy2. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
from latex2sympy2 import latex2sympy

# from .parser import choice_answer_clean, strip_string
# from parser import choice_answer_clean


# Local: defines the choice_answer_clean function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
def choice_answer_clean(pred: str):
    # Local: sets pred for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    # Local: applies a regular expression to normalize or extract answer text. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if tmp:
        pred = tmp  # Local: sets pred for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    else:  # Local: handles the remaining branch after earlier checks fail. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: sets pred for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        pred = [pred.strip().strip(".")]
    pred = pred[-1]  # Local: sets pred for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    # Remove the period at the end, again!
    # Local: sets pred for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    pred = pred.rstrip(".").rstrip("/")
    return pred  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.


def parse_digits(num):  # Local: defines the parse_digits function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    # Local: sets num for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    num = regex.sub(",", "", str(num))
    # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    try:
        # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        return float(num)
    # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    except:
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if num.endswith("%"):
            num = num[:-1]  # Local: sets num for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            if num.endswith("\\"):
                # Local: sets num for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                num = num[:-1]
            # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            try:
                # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                return float(num) / 100
            # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            except:
                # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                pass
    return None  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.


def is_digit(num):  # Local: defines the is_digit function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    # paired with parse_digits
    # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    return parse_digits(num) is not None


# Local: defines the str_to_pmatrix function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
def str_to_pmatrix(input_str):
    # Local: sets input_str for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    input_str = input_str.strip()
    # Local: applies a regular expression to normalize or extract answer text. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    # Local: sets pmatrix_list for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    pmatrix_list = []

    for m in matrix_str:  # Local: iterates through the current collection. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        m = m.strip("{}")  # Local: sets m for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: sets pmatrix for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        pmatrix_list.append(pmatrix)

    # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    return ", ".join(pmatrix_list)


def math_equal(  # Local: defines the math_equal function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    # Local: adds an item or argument to the surrounding expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    prediction: Union[bool, float, str],
    # Local: adds an item or argument to the surrounding expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    reference: Union[float, str],
    # Local: sets include_percentage: bool for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    include_percentage: bool = True,
    # Local: sets is_close: bool for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    is_close: bool = True,
    # Local: sets timeout: bool for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    timeout: bool = False,
) -> bool:  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    # Local: starts a multi-line text literal that Python treats as one value. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if prediction is None or reference is None:
        return False  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if (
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        reference in ["A", "B", "C", "D", "E"]
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and choice_answer_clean(prediction) == reference
    ):  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        return True  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    try:  # 1. numerical equal
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if is_digit(prediction) and is_digit(reference):
            # Local: sets prediction for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            prediction = parse_digits(prediction)
            # Local: sets reference for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            reference = parse_digits(reference)
            # number questions
            # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            if include_percentage:
                # Local: sets gt_result for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                gt_result = [reference / 100, reference, reference * 100]
            # Local: handles the remaining branch after earlier checks fail. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            else:
                # Local: sets gt_result for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                gt_result = [reference]
            # Local: iterates through the current collection. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            for item in gt_result:
                # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                try:
                    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    if is_close:
                        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                        if numeric_equal(prediction, item):
                            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                            return True
                    # Local: handles the remaining branch after earlier checks fail. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    else:
                        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                        if item == prediction:
                            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                            return True
                # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                except Exception:
                    # Local: skips the rest of this iteration and moves to the next item. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    continue
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return False
    # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    except:
        pass  # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if not prediction and prediction not in [0, False]:
        return False  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # 2. symbolic equal
    # Local: sets reference for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    reference = str(reference).strip()
    # Local: sets prediction for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if "pmatrix" in prediction and not "pmatrix" in reference:
        # Local: sets reference for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    # Local: sets pred_str, ref_str for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    pred_str, ref_str = prediction, reference
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if (
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        prediction.startswith("[")
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and prediction.endswith("]")
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and not reference.startswith("(")
    ) or (  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        prediction.startswith("(")
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and prediction.endswith(")")
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and not reference.startswith("[")
    ):  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: sets pred_str for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        pred_str = pred_str.strip("[]()")
        # Local: sets ref_str for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        ref_str = ref_str.strip("[]()")
    # Local: iterates through the current collection. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    for s in ["{", "}", "(", ")"]:
        # Local: sets ref_str for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        ref_str = ref_str.replace(s, "")
        # Local: sets pred_str for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        pred_str = pred_str.replace(s, "")
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if pred_str.lower() == ref_str.lower():
        return True  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    ## [a, b] vs. [c, d], return a==c and b==d
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if (
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: sets pred_parts for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        pred_parts = prediction[1:-1].split(",")
        # Local: sets ref_parts for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        ref_parts = reference[1:-1].split(",")
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if len(pred_parts) == len(ref_parts):
            # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            if all(
                [  # Local: starts a multi-line list literal. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    # Local: starts a multi-line call or expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    math_equal(
                        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    # Local: iterates through the current collection. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    for i in range(len(pred_parts))
                ]
            ):  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                return True
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if (
        (  # Local: starts a multi-line call or expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            prediction.startswith("\\begin{pmatrix}")
            # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (  # Local: starts a multi-line call or expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            prediction.endswith("\\end{pmatrix}")
            # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            or prediction.endswith("\\end{bmatrix}")
        )
        and (  # Local: starts a multi-line call or expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            reference.startswith("\\begin{pmatrix}")
            # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            or reference.startswith("\\begin{bmatrix}")
        )
        and (  # Local: starts a multi-line call or expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: sets pred_lines for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        pred_lines = [
            # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            line.strip()
            # Local: iterates through the current collection. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            for line in prediction[
                # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            ].split("\\\\")
            # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            if line.strip()
        ]
        ref_lines = [  # Local: sets ref_lines for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            line.strip()
            # Local: iterates through the current collection. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            for line in reference[
                # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            ].split("\\\\")
            # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            if line.strip()
        ]
        matched = True  # Local: sets matched for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if len(pred_lines) == len(ref_lines):
            # Local: iterates through the current collection. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                # Local: sets pred_parts for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                pred_parts = pred_line.split("&")
                # Local: sets ref_parts for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                ref_parts = ref_line.split("&")
                # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                if len(pred_parts) == len(ref_parts):
                    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    if not all(
                        [  # Local: starts a multi-line list literal. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                            # Local: starts a multi-line call or expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                            math_equal(
                                # Local: adds an item or argument to the surrounding expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                                pred_parts[i],
                                # Local: adds an item or argument to the surrounding expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                                ref_parts[i],
                                # Local: adds an item or argument to the surrounding expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                                include_percentage,
                                # Local: adds an item or argument to the surrounding expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                                is_close,
                            )
                            # Local: iterates through the current collection. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                            for i in range(len(pred_parts))
                        ]
                    # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    ):
                        # Local: sets matched for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                        matched = False
                        # Local: exits the current loop once a stopping condition is met. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                        break
                # Local: handles the remaining branch after earlier checks fail. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                else:
                    # Local: sets matched for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    matched = False
                # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                if not matched:
                    # Local: exits the current loop once a stopping condition is met. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    break
        # Local: handles the remaining branch after earlier checks fail. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        else:
            # Local: sets matched for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            matched = False
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if matched:
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True

    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if prediction.count("=") == 1 and reference.count("=") == 1:
        # Local: sets pred for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        pred = prediction.split("=")
        # Local: sets pred for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        # Local: sets ref for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        ref = reference.split("=")
        # Local: sets ref for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True
    elif (  # Local: checks the next mutually exclusive condition. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        prediction.count("=") == 1
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and len(prediction.split("=")[0].strip()) <= 2
        # Local: sets and " for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and "=" not in reference
    ):  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if math_equal(
            # Local: sets prediction.split(" for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            prediction.split("=")[1], reference, include_percentage, is_close
        ):  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True
    elif (  # Local: checks the next mutually exclusive condition. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        reference.count("=") == 1
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and len(reference.split("=")[0].strip()) <= 2
        # Local: sets and " for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        and "=" not in prediction
    ):  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if math_equal(
            # Local: sets prediction, reference.split(" for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            prediction, reference.split("=")[1], include_percentage, is_close
        ):  # Local: closes the surrounding literal or call expression. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True

    # symbolic equal with sympy
    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if timeout:
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True
    else:  # Local: handles the remaining branch after earlier checks fail. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if symbolic_equal(prediction, reference):
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True

    return False  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.


# Local: defines the math_equal_process function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
def math_equal_process(param):
    # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    return math_equal(param[-2], param[-1])


# Local: defines the numeric_equal function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):  # Local: defines the symbolic_equal function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    def _parse(s):  # Local: defines the _parse function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        # Local: iterates through the current collection. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        for f in [parse_latex, parse_expr, latex2sympy]:
            # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            try:
                # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                return f(s.replace("\\\\", "\\"))
            # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            except:
                # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                try:
                    # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    return f(s)
                # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                except:
                    # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                    pass
        return s  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    a = _parse(a)  # Local: sets a for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    b = _parse(b)  # Local: sets b for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # direct equal
    # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    try:
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if str(a) == str(b) or a == b:
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True
    # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    except:
        pass  # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # simplify equal
    # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    try:
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if a.equals(b) or simplify(a - b) == 0:
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True
    # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    except:
        pass  # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # equation equal
    # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    try:
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True
    # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    except:
        pass  # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    try:
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if numeric_equal(float(N(a)), float(N(b))):
            # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            return True
    # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    except:
        pass  # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # matrix
    # Local: starts a protected operation that may fail on external or parsed input. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    try:
        # if a and b are matrix
        # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        if a.shape == b.shape:
            # Local: sets _a for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            _a = a.applyfunc(lambda x: round(x, 3))
            # Local: sets _b for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            _b = b.applyfunc(lambda x: round(x, 3))
            # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
            if _a.equals(_b):
                # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
                return True
    # Local: handles a recoverable failure from the protected operation. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    except:
        pass  # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    return False  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.


# Local: defines the symbolic_equal_process function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
def symbolic_equal_process(a, b, output_queue):
    # Local: sets result for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    result = symbolic_equal(a, b)
    # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    output_queue.put(result)


# Local: defines the call_with_timeout function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
def call_with_timeout(func, *args, timeout=1, **kwargs):
    # Local: sets output_queue for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    output_queue = multiprocessing.Queue()
    # Local: sets process_args for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    process_args = args + (output_queue,)
    # Local: sets process for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    process.start()
    # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    process.join(timeout)

    # Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    if process.is_alive():
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        process.terminate()
        # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
        process.join()
        return False  # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # Local: returns the computed result to the caller. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    return output_queue.get()

def _test_math_equal():  # Local: defines the _test_math_equal function. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    # print(math_equal("0.0833333333333333", "\\frac{1}{12}"))
    # print(math_equal("(1,4.5)", "(1,\\frac{9}{2})"))
    # print(math_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}", timeout=True))
    # print(math_equal("\\sec^2(y)", "\\tan^2(y)+1", timeout=True))
    # print(math_equal("\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\end{pmatrix}", "(\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix})", timeout=True))

    # pred = '\\begin{pmatrix}\\frac{1}{3x^{2/3}}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\end{pmatrix}'
    # gt = '(\\begin{pmatrix}\\frac{1}{3\\sqrt[3]{x}^2}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\\\\\end{pmatrix})'

    # pred= '-\\frac{8x^2}{9(x^2-2)^{5/3}}+\\frac{2}{3(x^2-2)^{2/3}}'
    # gt= '-\\frac{2(x^2+6)}{9(x^2-2)\\sqrt[3]{x^2-2}^2}'

    # pred =  '-34x-45y+20z-100=0'
    # gt = '34x+45y-20z+100=0'

    # pred = '\\frac{100}{3}'
    # gt = '33.3'

    # pred = '\\begin{pmatrix}0.290243531202435\\\\0.196008371385084\\\\-0.186381278538813\\end{pmatrix}'
    # gt = '(\\begin{pmatrix}0.29\\\\0.196\\\\-0.186\\\\\\end{pmatrix})'

    # pred = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{2\\sqrt{33}+15}'
    # gt = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{15+2\\sqrt{33}}'

    # pred = '(+5)(b+2)'
    # gt = '(a+5)(b+2)'

    # pred = '\\frac{1+\\sqrt{5}}{2}'
    # gt = '2'

    # pred = '\\frac{34}{16}+\\frac{\\sqrt{1358}}{16}', gt = '4'
    # pred = '1', gt = '1\\\\sqrt{19}'

    # pred = "(0.6,2.6667]"
    # gt = "(\\frac{3}{5},\\frac{8}{3}]"

    gt = "x+2n+1"  # Local: sets gt for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    pred = "x+1"  # Local: sets pred for later use in this scope. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.

    # Local: reports progress or diagnostics to the run log. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    print(math_equal(pred, gt, timeout=True))


# Local: opens a condition that selects behavior from current state. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
if __name__ == "__main__":
    # Local: executes this statement in the current code path. Global: provides math equivalence checks used by ALS evaluation on symbolic answers.
    _test_math_equal()
