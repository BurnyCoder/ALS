# Local: file src/extract_judge_answer/parse_utils_qwen.py provides first-party ALS source context. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
# Local: starts a multi-line text literal that Python treats as one value. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import random  # Local: imports random for this module. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
import regex  # Local: imports regex for this module. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
import re  # Local: imports re for this module. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
import sympy  # Local: imports sympy for this module. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
from latex2sympy2 import latex2sympy  # Local: imports selected helpers from latex2sympy2. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
# Local: imports selected helpers from typing. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
from typing import TypeVar, Iterable, List, Union, Any, Dict
from word2number import w2n  # Local: imports selected helpers from word2number. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


def _fix_fracs(string):  # Local: defines the _fix_fracs function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    substrs = string.split("\\frac")  # Local: sets substrs for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    new_str = substrs[0]  # Local: sets new_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if len(substrs) > 1:
        substrs = substrs[1:]  # Local: sets substrs for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        for substr in substrs:  # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            new_str += "\\frac"  # Local: updates new_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr  # Local: updates new_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                # Local: starts a protected operation that may fail on external or parsed input. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                try:
                    # Local: checks an invariant expected by this parsing path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    assert len(substr) >= 2
                # Local: handles a recoverable failure from the protected operation. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                except:
                    return string  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                a = substr[0]  # Local: sets a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                b = substr[1]  # Local: sets b for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                if b != "{":
                    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    if len(substr) > 2:
                        # Local: sets post_substr for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                        post_substr = substr[2:]
                        # Local: updates new_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    else:
                        # Local: updates new_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                        new_str += "{" + a + "}{" + b + "}"
                else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    if len(substr) > 2:
                        # Local: sets post_substr for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                        post_substr = substr[2:]
                        # Local: updates new_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                        new_str += "{" + a + "}" + b + post_substr
                    # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    else:
                        # Local: updates new_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                        new_str += "{" + a + "}" + b
    string = new_str  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    return string  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


def _fix_a_slash_b(string):  # Local: defines the _fix_a_slash_b function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if len(string.split("/")) != 2:
        return string  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    a = string.split("/")[0]  # Local: sets a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    b = string.split("/")[1]  # Local: sets b for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    try:  # Local: starts a protected operation that may fail on external or parsed input. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if "sqrt" not in a:
            a = int(a)  # Local: sets a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if "sqrt" not in b:
            b = int(b)  # Local: sets b for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: checks an invariant expected by this parsing path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        assert string == "{}/{}".format(a, b)
        # Local: sets new_string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    except:  # Local: handles a recoverable failure from the protected operation. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        return string  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


def _fix_sqrt(string):  # Local: defines the _fix_sqrt function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


# Local: defines the convert_word_number function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
def convert_word_number(text: str) -> str:
    try:  # Local: starts a protected operation that may fail on external or parsed input. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        text = str(w2n.word_to_num(text))  # Local: sets text for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    except:  # Local: handles a recoverable failure from the protected operation. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pass  # Local: executes this statement in the current code path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    return text  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


# units mainly from MathQA
unit_texts = [  # Local: sets unit_texts for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "east",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "degree",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "mph",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "kmph",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "ft",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "m sqaure",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    " m east",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "sq m",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "deg",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "mile",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "q .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "monkey",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "prime",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "ratio",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "profit of rs",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "rd",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "o",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "gm",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "p . m",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "lb",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "tile",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "per",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "dm",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "lt",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "gain",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "ab",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "way",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "west",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "a .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "b .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "c .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "d .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "e .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "f .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "g .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "h .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "t",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "a",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "h",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "no change",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "men",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "soldier",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "pie",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "bc",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "excess",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "st",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "inches",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "noon",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "percent",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "by",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "gal",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "kmh",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "c",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "acre",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "rise",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "a . m",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "th",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "π r 2",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "sq",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "mark",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "l",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "toy",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "coin",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "sq . m",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "gallon",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "° f",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "profit",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "minw",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "yr",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "women",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "feet",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "am",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "pm",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "hr",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "cu cm",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "square",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "v â € ™",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "are",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "rupee",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "rounds",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "cubic",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "cc",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "mtr",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "s",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "ohm",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "number",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "kmph",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "day",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "hour",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "minute",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "min",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "second",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "man",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "woman",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "sec",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "cube",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "mt",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "sq inch",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "mp",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "∏ cm ³",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "hectare",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "more",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "sec",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "unit",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "cu . m",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "cm 2",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "rs .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "rs",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "kg",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "g",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "month",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "km",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "m",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "cm",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "mm",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "apple",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "liter",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "loss",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "yard",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "pure",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "year",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "increase",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "decrease",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "d",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "less",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "Surface",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "litre",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "pi sq m",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "s .",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "metre",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "meter",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    "inch",  # Local: adds an item or argument to the surrounding expression. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
]

# Local: executes this statement in the current code path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string, skip_unit=False):  # Local: defines the strip_string function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = str(string).strip()  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # linebreaks
    string = string.replace("\n", "")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # right "."
    string = string.rstrip(".")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("tfrac", "frac")
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("dfrac", "frac")
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = (string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge"))

    # remove \left and \right
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("\\left", "")
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("\\}", "}")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # Remove unit: miles, dollars if after is not none
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):  # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                if _string != "":
                    string = _string  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # Remove circ (degrees)
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("^{\\circ}", "")
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("$", "")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        string = string.replace(key, "")
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("\\emptyset", r"{}")
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("\%", "")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("%", "")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace(" .", " 0.")
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if (string.startswith("{") and string.endswith("}") and string.isalnum() or
            # Local: executes this statement in the current code path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            string.startswith("(") and string.endswith(")") and string.isalnum() or
            # Local: executes this statement in the current code path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            string.startswith("[") and string.endswith("]") and string.isalnum()):
        string = string[1:-1]  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # inf
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("infinity", "\\infty")
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if "\\infty" not in string:
        # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        string = string.replace("inf", "\\infty")
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")  # Local: executes this statement in the current code path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string.replace('"', "")  # Local: executes this statement in the current code path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # i, j
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if "j" in string and "i" not in string:
        # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if len(string) == 0:
        return string  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if string[0] == ".":
        string = "0" + string  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if len(string.split("=")) == 2:
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if len(string.split("=")[0]) <= 2:
            # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            string = string.split("=")[1]

    string = _fix_sqrt(string)  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    string = string.replace(" ", "")  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)  # Local: sets string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    return string  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


# Local: sets direct_answer_trigger_for_fewshot for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
direct_answer_trigger_for_fewshot = ("choice is", "answer is")


def choice_answer_clean(pred: str):  # Local: defines the choice_answer_clean function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred = pred.strip("\n")  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False  # Local: sets ICL for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    for trigger in direct_answer_trigger_for_fewshot:
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if pred.count(trigger) > 1:
            ICL = True  # Local: sets ICL for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if ICL:  # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pred = pred.split("\n\n")[0]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # Split the trigger to find the answer.
    # Local: sets preds for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if len(preds) > 1:
        answer_flag = True  # Local: sets answer_flag for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pred = preds[-1]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        answer_flag = False  # Local: sets answer_flag for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:  # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pred = tmp  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pred = [pred.strip().strip(".")]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if len(pred) == 0:
        pred = ""  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # choose the last e
            pred = pred[-1]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    return pred  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


def find_box(pred_str: str):  # Local: defines the find_box function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    ans = pred_str.split("boxed")[-1]  # Local: sets ans for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if not ans:  # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        return ""  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if ans[0] == "{":
        stack = 1  # Local: sets stack for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        a = ""  # Local: sets a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        for c in ans[1:]:  # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            if c == "{":
                stack += 1  # Local: updates stack for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                a += c  # Local: updates a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            elif c == "}":  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                stack -= 1  # Local: updates stack for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                if stack == 0:
                    # Local: exits the current loop once a stopping condition is met. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    break
                a += c  # Local: updates a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                a += c  # Local: updates a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        a = ans.split("$")[0].strip()  # Local: sets a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    return a  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


def clean_units(pred_str: str):  # Local: defines the clean_units function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: executes this statement in the current code path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    """Clean the units in the number."""

    # Local: defines the convert_pi_to_number function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    def convert_pi_to_number(code_string):
        # Local: sets code_string for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        code_string = code_string.replace("\\pi", "π")
        # Replace \pi or π not preceded by a digit or } with 3.14
        # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
        # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
        return code_string  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # Local: sets pred_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred_str = convert_pi_to_number(pred_str)
    # Local: sets pred_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred_str = pred_str.replace("%", "/100")
    # Local: sets pred_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred_str = pred_str.replace("$", "")
    # Local: sets pred_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred_str = pred_str.replace("¥", "")
    # Local: sets pred_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred_str = pred_str.replace("°C", "")
    # Local: sets pred_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred_str = pred_str.replace(" C", "")
    # Local: sets pred_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred_str = pred_str.replace("°", "")
    return pred_str  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


# Local: defines the extract_theoremqa_answer function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if any([option in pred.lower() for option in ["yes", "true"]]):
        pred = "True"  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    elif any([option in pred.lower() for option in ["no", "false"]]):
        pred = "False"  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    elif any([option in pred.lower() for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]]):
        pass  # Local: executes this statement in the current code path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Some of the models somehow get used to boxed output from pre-training
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if "boxed" in pred:
            pred = find_box(pred)  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if answer_flag:
            # Extract the numbers out of the string
            # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            pred = pred.split("=")[-1].strip()
            pred = clean_units(pred)  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # Local: starts a protected operation that may fail on external or parsed input. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            try:
                # Local: sets tmp for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                tmp = str(latex2sympy(pred))
                pred = str(eval(tmp))  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # Local: handles a recoverable failure from the protected operation. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            except Exception:
                # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                if re.match(r"-?[\d\.]+\s\D+$", pred):
                    # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    pred = pred.split(" ")[0]
                # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                elif re.match(r"-?[\d\.]+\s[^\s]+$", pred):
                    # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    pred = pred.split(" ")[0]
        else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # desparate search over the last number
            # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            preds = re.findall(r"-?\d*\.?\d+", pred)
            # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            if len(preds) >= 1:
                pred = preds[-1]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                pred = ""  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    return pred  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


# Local: defines the extract_answer function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
def extract_answer(pred_str, data_name, use_last_number=True):
    # Local: sets pred_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred_str = pred_str.replace("\u043a\u0438", "")

    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        # Local: sets tmp for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        tmp = pred_str.split("final answer is $", 1)[1]
        # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        ans = pred_str.split("boxed")[-1]  # Local: sets ans for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if len(ans) == 0:
            return ""  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        elif ans[0] == "{":  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            stack = 1  # Local: sets stack for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            a = ""  # Local: sets a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            for c in ans[1:]:  # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                if c == "{":
                    stack += 1  # Local: updates stack for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    a += c  # Local: updates a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                elif c == "}":  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    stack -= 1  # Local: updates stack for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    if stack == 0:
                        # Local: exits the current loop once a stopping condition is met. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                        break
                    a += c  # Local: updates a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                    a += c  # Local: updates a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            a = ans.split("$")[0].strip()  # Local: sets a for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pred = a  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    elif "he answer is" in pred_str:
        # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pred = pred_str.split("he answer is")[-1].strip()
    # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    elif "final answer is" in pred_str:
        # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Handle Chinese few-shot multiple choice problem answer extraction
        # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    else:  # use the last number
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if use_last_number:
            pattern = "-?\d*\.?\d+"  # Local: sets pattern for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            pred = re.findall(pattern, pred_str.replace(",", ""))
            # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            if len(pred) >= 1:
                pred = pred[-1]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                pred = ""  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            pred = ""  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.

    # multiple line
    # pred = pred.split("\n")[0]
    # Local: applies a regular expression to normalize or extract answer text. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred = re.sub(r"\n\s*", "", pred)
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if pred != "" and pred[0] == ":":
        pred = pred[1:]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]  # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: sets pred for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


# Local: sets STRIP_EXCEPTIONS for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]


# Local: defines the parse_ground_truth function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
def parse_ground_truth(groudtruth_solution: str, data_name):
    # Local: sets gt_ans for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    gt_ans = extract_answer(groudtruth_solution, data_name)
    return gt_ans  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.


def parse_question(example, data_name):  # Local: defines the parse_question function. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    question = ""  # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if data_name == "asdiv":
        # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        question = f"{example['body'].strip()} {example['question'].strip()}"
    elif data_name == "svamp":  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        body = example["Body"].strip()  # Local: sets body for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if not body.endswith("."):
            body = body + "."  # Local: sets body for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        question = f'{body} {example["Question"].strip()}'
    elif data_name == "tabmwp":  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: sets title_str for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        title_str = (f'regarding "{example["table_title"]}" ' if example["table_title"] else "")
        # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        question = f"Read the following table {title_str}and answer a question:\n"
        # Local: updates question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        question += f'{example["table"]}\n{example["question"]}'
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if example["choices"]:
            # Local: updates question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            question += (f' Please select from the following options: {example["choices"]}')
    elif data_name == "carp_en":  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        question = example["content"]  # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    elif data_name == "mmlu_stem":  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        options = example["choices"]  # Local: sets options for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: checks an invariant expected by this parsing path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        assert len(options) == 4
        # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        for i, (label, option) in enumerate(zip("ABCD", options)):
            # Local: sets options[i] for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            options[i] = f"({label}) {str(option).strip()}"
        options = " ".join(options)  # Local: sets options for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
        # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    elif data_name == "sat_math":  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: sets options for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        options = example["options"].strip()
        # Local: checks an invariant expected by this parsing path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        assert "A" == options[0]
        options = "(" + options  # Local: sets options for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        for ch in "BCD":  # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            if f" {ch}) " in options:
                # Local: sets options for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                options = regex.sub(f" {ch}\) ", f" ({ch}) ", options)
        # question = f"{example['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
        # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    elif "aqua" in data_name:  # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        options = example["options"]  # Local: sets options for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: sets choice for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        choice = "(" + "(".join(options)
        # Local: sets choice for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        choice = choice.replace("(", " (").replace(")", ") ").strip()
        # Local: sets choice for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        choice = "\nAnswer Choices: " + choice
        # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        question = example["question"].strip() + choice
    # Local: checks the next mutually exclusive condition. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    elif data_name == "gaokao_math_qa":
        # Local: sets options_dict for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        options_dict = example["options"]
        options = []  # Local: sets options for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        for key in options_dict:  # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            # Local: executes this statement in the current code path. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            options.append(f"({key}) {options_dict[key]}")
        options = " ".join(options)  # Local: sets options for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        question = f"{example['question'].strip()}\n选项: {options}"
    else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: iterates through the current collection. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        for key in ["question", "problem", "Question", "input"]:
            # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            if key in example:
                # Local: sets question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
                question = example[key]
                break  # Local: exits the current loop once a stopping condition is met. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    # assert question != ""
    # Yes or No question
    # Local: sets _, gt_ans for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    _, gt_ans = parse_ground_truth(example, data_name)
    # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
    if isinstance(gt_ans, str):
        gt_lower = gt_ans.lower()  # Local: sets gt_lower for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if gt_lower in ["true", "false"]:
            # Local: updates question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            question += " (True or False)"
        # Local: opens a condition that selects behavior from current state. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
        if gt_lower in ["yes", "no"]:
            # Local: updates question for later use in this scope. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
            question += " (Yes or No)"
    return question.strip()  # Local: returns the computed result to the caller. Global: normalizes Qwen-style math outputs for ALS benchmark scoring.
