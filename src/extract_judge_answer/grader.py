"""Robust mathematical answer equivalence checks for ALS evaluation.

This file is largely copied from established math-evaluation utilities and is
used by `utils.judge_answer` as one MATH-500 correctness path. It combines
exact matching, numeric tolerance, symbolic simplification, equation handling,
tuple/matrix recursion, and optional timeout isolation.
"""

# Regular expressions handle lightweight answer cleanup and structural matching.
import re
# The `regex` package supports richer matching used by copied evaluation logic.
import regex
# Multiprocessing provides timeout isolation for potentially slow symbolic checks.
import multiprocessing
# `isclose` implements tolerant floating-point equality.
from math import isclose
# `Union` documents accepted answer input types.
from typing import Union
# `defaultdict` is imported from the source utility set and kept for compatibility.
from collections import defaultdict

# SymPy simplification and numeric evaluation power symbolic equivalence checks.
from sympy import simplify, N
# `parse_expr` parses Python/SymPy-style expressions.
from sympy.parsing.sympy_parser import parse_expr
# `parse_latex` parses LaTeX expressions when available.
from sympy.parsing.latex import parse_latex
# `latex2sympy` is a fallback parser for LaTeX expressions.
from latex2sympy2 import latex2sympy

# The copied source had parser imports here; they remain commented because local code defines needed helpers.
# from .parser import choice_answer_clean, strip_string
# from parser import choice_answer_clean


# This helper extracts multiple-choice letters from model output.
def choice_answer_clean(pred: str):
    """Normalize a predicted multiple-choice answer.

    Locally, this removes punctuation and selects the last A-E choice if one is
    present. Globally, it lets the generic math judge handle datasets or answers
    that use choice labels.
    """
    # Strip newlines, trailing periods/slashes, spaces, and leading colons from the prediction.
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Find standalone A-E option letters in uppercase-normalized text.
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    # If option letters exist, use them as candidate answers.
    if tmp:
        # Store all found choices so the last one can be selected below.
        pred = tmp
    # If no choice letters exist, keep the cleaned text as a one-item list.
    else:
        # Strip another layer of spaces/periods for non-choice answers.
        pred = [pred.strip().strip(".")]
    # Use the last candidate because final answers usually appear last.
    pred = pred[-1]
    # Remove a final period or slash again after candidate selection.
    pred = pred.rstrip(".").rstrip("/")
    # Return the cleaned choice or text answer.
    return pred


# This helper parses numbers and percentages into floats.
def parse_digits(num):
    """Return a float for numeric or percentage strings, otherwise None."""
    # Remove comma thousands separators before float parsing.
    num = regex.sub(",", "", str(num))
    # Try direct floating-point conversion first.
    try:
        # Successful conversion returns a numeric value.
        return float(num)
    # If direct conversion fails, try percentage-specific handling.
    except:
        # Percentage answers are normalized by dividing by 100.
        if num.endswith("%"):
            # Drop the percent sign.
            num = num[:-1]
            # Some LaTeX outputs include a trailing backslash before `%`; remove it.
            if num.endswith("\\"):
                # Drop the trailing backslash.
                num = num[:-1]
            # Try converting the remaining percentage number.
            try:
                # Divide by 100 to convert percent to ratio.
                return float(num) / 100
            # If percentage conversion fails, fall through to None.
            except:
                # Preserve original fallback behavior.
                pass
    # Non-numeric strings return None.
    return None


# This helper checks whether a value can be parsed by `parse_digits`.
def is_digit(num):
    """Return True when `num` can be interpreted as numeric text."""
    # Any non-None parse result means numeric equality can be attempted.
    return parse_digits(num) is not None


# This helper converts simple brace matrices into LaTeX pmatrix text.
def str_to_pmatrix(input_str):
    """Convert `{a,b}`-style matrix snippets into `pmatrix` strings.

    Locally, this supports matrix comparison when predictions omit explicit
    LaTeX matrix wrappers. Globally, it improves MATH-style answer judging for
    vector and matrix outputs.
    """
    # Strip surrounding whitespace before regex matching.
    input_str = input_str.strip()
    # Find brace groups containing commas as candidate matrix rows.
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    # Accumulate converted pmatrix snippets.
    pmatrix_list = []

    # Convert each matched brace group.
    for m in matrix_str:
        # Remove the outer braces.
        m = m.strip("{}")
        # Build a pmatrix, replacing commas with LaTeX row separators.
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        # Store the converted matrix string.
        pmatrix_list.append(pmatrix)

    # Join converted matrices with comma separators.
    return ", ".join(pmatrix_list)


# This is the main generic equivalence function used by `math_equal_process`.
def math_equal(
    # The model prediction can be boolean, numeric, or string-like.
    prediction: Union[bool, float, str],
    # The reference answer can be numeric or string-like.
    reference: Union[float, str],
    # Percentage matching optionally checks x, x/100, and 100x variants.
    include_percentage: bool = True,
    # Approximate numeric equality can be enabled for floats.
    is_close: bool = True,
    # Timeout mode runs symbolic checks in a subprocess.
    timeout: bool = False,
) -> bool:
    """Return True if prediction and reference are mathematically equivalent.

    Locally, the function tries exact string matching, numeric comparison,
    structural tuple/matrix comparison, equation normalization, and symbolic
    equality. Globally, this is one of the MATH-500 judges used to score ALS and
    baseline outputs.
    """
    # Missing prediction or reference cannot be judged equivalent.
    if prediction is None or reference is None:
        # Return False conservatively for missing values.
        return False
    # Case-insensitive exact string match accepts identical textual answers.
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        # Return True before slower parsing.
        return True
    # Multiple-choice references are handled by extracting A-E choices from the prediction.
    if (
        # The reference must be one of the supported choice labels.
        reference in ["A", "B", "C", "D", "E"]
        # The cleaned predicted choice must equal the reference.
        and choice_answer_clean(prediction) == reference
    ):
        # Return True for matching choice labels.
        return True

    # Numeric equality is attempted before symbolic parsing for speed.
    try:
        # Both sides must parse as numeric strings.
        if is_digit(prediction) and is_digit(reference):
            # Convert prediction to float.
            prediction = parse_digits(prediction)
            # Convert reference to float.
            reference = parse_digits(reference)
            # Percentage-inclusive mode tests reference, reference/100, and reference*100.
            if include_percentage:
                # Build candidate ground-truth numeric variants.
                gt_result = [reference / 100, reference, reference * 100]
            # Strict numeric mode only checks the reference itself.
            else:
                # Build a one-item candidate list.
                gt_result = [reference]
            # Compare prediction against each numeric reference variant.
            for item in gt_result:
                # Individual numeric comparisons can fail on unusual values, so isolate them.
                try:
                    # Approximate mode uses relative tolerance.
                    if is_close:
                        # Accept if prediction and item are close enough.
                        if numeric_equal(prediction, item):
                            # Return True on the first accepted numeric comparison.
                            return True
                    # Exact numeric mode requires equality.
                    else:
                        # Accept if floats match exactly.
                        if item == prediction:
                            # Return True on exact numeric match.
                            return True
                # Ignore failures and try the next numeric variant.
                except Exception:
                    # Continue through candidate numeric variants.
                    continue
            # If no numeric variant matched, numeric checking rejects the answer.
            return False
    # If numeric parsing/comparison fails, fall through to symbolic checks.
    except:
        # Preserve original permissive fallback behavior.
        pass

    # Empty predictions are rejected except valid falsey answers like 0 or False.
    if not prediction and prediction not in [0, False]:
        # Return False because there is no answer content to compare.
        return False

    # Convert both answers to stripped strings for structural and symbolic processing.
    reference = str(reference).strip()
    # Convert the prediction to stripped string form.
    prediction = str(prediction).strip()

    # If the prediction has a pmatrix but the reference does not, convert reference matrix shorthand.
    if "pmatrix" in prediction and not "pmatrix" in reference:
        # Build a pmatrix string from brace/comma matrix syntax.
        reference = str_to_pmatrix(reference)

    # Initialize simplified structural strings for bracket/brace normalization.
    pred_str, ref_str = prediction, reference
    # If one answer uses brackets and the other parentheses, compare inner content.
    if (
        # Prediction is bracketed while reference is not parenthesized.
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        # Prediction is parenthesized while reference is not bracketed.
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        # Strip one layer of brackets/parentheses from prediction.
        pred_str = pred_str.strip("[]()")
        # Strip one layer of brackets/parentheses from reference.
        ref_str = ref_str.strip("[]()")
    # Remove common grouping characters for a coarse textual comparison.
    for s in ["{", "}", "(", ")"]:
        # Remove the character from the reference string.
        ref_str = ref_str.replace(s, "")
        # Remove the character from the prediction string.
        pred_str = pred_str.replace(s, "")
    # Case-insensitive match after grouping removal accepts simple formatting differences.
    if pred_str.lower() == ref_str.lower():
        # Return True before deeper parsing.
        return True

    # Tuple/list interval answers can be compared elementwise.
    if (
        # Prediction must look bracketed or parenthesized.
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        # Reference must also look bracketed or parenthesized.
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        # Split prediction contents by commas.
        pred_parts = prediction[1:-1].split(",")
        # Split reference contents by commas.
        ref_parts = reference[1:-1].split(",")
        # Elementwise comparison requires the same arity.
        if len(pred_parts) == len(ref_parts):
            # Recursively compare every corresponding part.
            if all(
                [
                    # Recursive calls reuse the same percentage and tolerance settings.
                    math_equal(
                        # Current prediction element.
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    # Iterate through all element positions.
                    for i in range(len(pred_parts))
                ]
            ):
                # Return True only if all tuple/list elements match.
                return True
    # Matrix answers in pmatrix/bmatrix environments get row/column elementwise comparison.
    if (
        (
            # Prediction may start with pmatrix.
            prediction.startswith("\\begin{pmatrix}")
            # Or prediction may start with bmatrix.
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            # Prediction may end with pmatrix.
            prediction.endswith("\\end{pmatrix}")
            # Or prediction may end with bmatrix.
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            # Reference may start with pmatrix.
            reference.startswith("\\begin{pmatrix}")
            # Or reference may start with bmatrix.
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            # Reference may end with pmatrix.
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        # Split prediction rows inside the matrix environment.
        pred_lines = [
            # Strip whitespace from each row.
            line.strip()
            # Slice off the matrix environment wrappers and split rows on `\\`.
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            # Keep non-empty rows only.
            if line.strip()
        ]
        # Split reference rows inside the matrix environment.
        ref_lines = [
            # Strip whitespace from each row.
            line.strip()
            # Slice off the matrix environment wrappers and split rows on `\\`.
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            # Keep non-empty rows only.
            if line.strip()
        ]
        # Assume matched until a shape or element mismatch is found.
        matched = True
        # Row counts must match.
        if len(pred_lines) == len(ref_lines):
            # Compare rows pairwise.
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                # Matrix row entries are separated by ampersands.
                pred_parts = pred_line.split("&")
                # Matrix row entries are separated by ampersands.
                ref_parts = ref_line.split("&")
                # Column counts must match.
                if len(pred_parts) == len(ref_parts):
                    # Recursively compare each matrix entry.
                    if not all(
                        [
                            # Recursive calls handle numeric/symbolic entry equality.
                            math_equal(
                                # Current prediction entry.
                                pred_parts[i],
                                # Current reference entry.
                                ref_parts[i],
                                # Preserve percentage handling.
                                include_percentage,
                                # Preserve numeric tolerance setting.
                                is_close,
                            )
                            # Iterate over all columns in this row.
                            for i in range(len(pred_parts))
                        ]
                    ):
                        # Mark mismatch if any entry fails.
                        matched = False
                        # Stop row comparison after failure.
                        break
                # Different column counts mean matrix mismatch.
                else:
                    # Mark mismatch for shape difference.
                    matched = False
                # Stop outer loop after mismatch.
                if not matched:
                    # Break because final result is already false.
                    break
        # Different row counts mean matrix mismatch.
        else:
            # Mark mismatch for shape difference.
            matched = False
        # If every row and entry matched, accept the matrix answer.
        if matched:
            # Return True for matrix equivalence.
            return True

    # Equations with one equals sign on both sides can be compared as lhs-rhs expressions.
    if prediction.count("=") == 1 and reference.count("=") == 1:
        # Split prediction into left and right sides.
        pred = prediction.split("=")
        # Rewrite prediction equation as a zero-expression.
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        # Split reference into left and right sides.
        ref = reference.split("=")
        # Rewrite reference equation as a zero-expression.
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        # Compare equations directly or up to sign.
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            # Return True if the symbolic equation forms match.
            return True
    # If prediction is a short assignment like `x=3` and reference is just `3`, compare RHS.
    elif (
        # Prediction must have exactly one equals sign.
        prediction.count("=") == 1
        # Left-hand variable must be short.
        and len(prediction.split("=")[0].strip()) <= 2
        # Reference must not itself be an equation.
        and "=" not in reference
    ):
        # Compare the prediction right-hand side to the reference.
        if math_equal(
            # Prediction RHS.
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            # Return True when RHS matches.
            return True
    # If reference is a short assignment and prediction is just the RHS, compare prediction to reference RHS.
    elif (
        # Reference must have exactly one equals sign.
        reference.count("=") == 1
        # Left-hand variable must be short.
        and len(reference.split("=")[0].strip()) <= 2
        # Prediction must not itself be an equation.
        and "=" not in prediction
    ):
        # Compare prediction to the reference right-hand side.
        if math_equal(
            # Prediction value.
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            # Return True when prediction matches assignment RHS.
            return True

    # Symbolic equality is the final expensive comparison path.
    if timeout:
        # Timeout mode isolates symbolic equality in a subprocess.
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            # Return True if the subprocess accepted equivalence.
            return True
    # Non-timeout mode calls symbolic equality directly.
    else:
        # Direct symbolic equality may parse and simplify expressions in-process.
        if symbolic_equal(prediction, reference):
            # Return True if symbolic comparison accepted equivalence.
            return True

    # No comparison path accepted equivalence.
    return False


# This wrapper accepts the tuple shape used by callers in `utils.py`.
def math_equal_process(param):
    """Unpack a tuple-like parameter and call `math_equal`.

    Locally, `param[-2]` and `param[-1]` are reference and prediction. Globally,
    this shape matches copied evaluator call sites.
    """
    # Compare the final two tuple items with the generic equivalence function.
    return math_equal(param[-2], param[-1])


# This helper implements tolerant float equality.
def numeric_equal(prediction: float, reference: float):
    """Return whether two floats are close under the evaluator tolerance."""
    # Relative tolerance accepts small floating-point differences in numeric answers.
    return isclose(reference, prediction, rel_tol=1e-4)


# This helper attempts symbolic equality through several parsers and simplifiers.
def symbolic_equal(a, b):
    """Return whether two expressions are symbolically equivalent.

    Locally, strings are parsed through LaTeX, SymPy expression, and latex2sympy
    parsers, then compared directly, by simplification, by equation structure,
    by numeric evaluation, and by matrix equality. Globally, this handles
    symbolic MATH-500 answers generated by ALS and baselines.
    """
    # Nested parser tries several expression parsers in priority order.
    def _parse(s):
        # Try LaTeX parser, SymPy parser, then latex2sympy parser.
        for f in [parse_latex, parse_expr, latex2sympy]:
            # First try after normalizing double backslashes.
            try:
                # Return the parsed expression if successful.
                return f(s.replace("\\\\", "\\"))
            # If normalized parsing fails, try the raw string.
            except:
                # Raw parsing may work for parser-specific syntax.
                try:
                    # Return raw parsed expression if successful.
                    return f(s)
                # If this parser fails, try the next parser.
                except:
                    # Continue to the next parser.
                    pass
        # If all parsers fail, return the original string.
        return s

    # Parse the first expression.
    a = _parse(a)
    # Parse the second expression.
    b = _parse(b)

    # Direct equality is the cheapest symbolic comparison.
    try:
        # String equality or object equality accepts identical parsed expressions.
        if str(a) == str(b) or a == b:
            # Return True for direct equality.
            return True
    # Ignore equality failures from unusual parsed objects.
    except:
        # Fall through to simplification.
        pass

    # Simplification can prove equivalent algebraic expressions.
    try:
        # `.equals` and `simplify(a - b) == 0` are two SymPy equivalence strategies.
        if a.equals(b) or simplify(a - b) == 0:
            # Return True when simplification proves equality.
            return True
    # Ignore simplification failures and try other comparison forms.
    except:
        # Fall through to equation comparison.
        pass

    # Equation objects can be compared by absolute lhs-rhs distance.
    try:
        # Equivalent equations have the same absolute residual expression.
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            # Return True when equation residuals match.
            return True
    # Non-equation objects or failed equation comparison fall through.
    except:
        # Fall through to numeric evaluation.
        pass

    # Numeric evaluation can catch expressions that simplify to the same float.
    try:
        # Convert SymPy expressions to floats and compare with tolerance.
        if numeric_equal(float(N(a)), float(N(b))):
            # Return True for tolerant numeric equality.
            return True
    # Ignore numeric conversion failures.
    except:
        # Fall through to matrix comparison.
        pass

    # Matrix expressions can be compared elementwise after rounding.
    try:
        # Matching shapes are required before elementwise equality.
        if a.shape == b.shape:
            # Round entries in the first matrix to three decimals.
            _a = a.applyfunc(lambda x: round(x, 3))
            # Round entries in the second matrix to three decimals.
            _b = b.applyfunc(lambda x: round(x, 3))
            # SymPy matrix equality checks all entries.
            if _a.equals(_b):
                # Return True for rounded matrix equality.
                return True
    # Non-matrix objects or failed matrix comparison fall through.
    except:
        # Fall through to final rejection.
        pass

    # No symbolic strategy accepted equivalence.
    return False


# This subprocess target writes symbolic equality result to a queue.
def symbolic_equal_process(a, b, output_queue):
    """Run `symbolic_equal` and put its result on a multiprocessing queue."""
    # Compute symbolic equality in the child process.
    result = symbolic_equal(a, b)
    # Put the boolean result where the parent can read it.
    output_queue.put(result)


# This helper runs a function in a subprocess with a timeout.
def call_with_timeout(func, *args, timeout=1, **kwargs):
    """Return a subprocess result or False if the function times out.

    Locally, this prevents expensive symbolic parsing from hanging evaluation.
    Globally, it protects ALS benchmark runs from one pathological answer
    consuming unbounded time.
    """
    # A queue carries the child process result back to the parent.
    output_queue = multiprocessing.Queue()
    # Append the queue to positional arguments expected by `symbolic_equal_process`.
    process_args = args + (output_queue,)
    # Create a child process for the target function.
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    # Start the child process.
    process.start()
    # Wait up to the timeout for completion.
    process.join(timeout)

    # If the process is still alive, it exceeded the timeout.
    if process.is_alive():
        # Terminate the slow child process.
        process.terminate()
        # Join after termination to clean up process resources.
        process.join()
        # Timeout is treated as failed equivalence.
        return False

    # Return the result written by the child process.
    return output_queue.get()


# This manual test helper exercises selected equivalence cases during standalone debugging.
def _test_math_equal():
    """Run an ad hoc local check of `math_equal` when the file is executed."""
    # The following commented examples document historical edge cases for the grader.
    # print(math_equal("0.0833333333333333", "\\frac{1}{12}"))
    # print(math_equal("(1,4.5)", "(1,\\frac{9}{2})"))
    # print(math_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}", timeout=True))
    # print(math_equal("\\sec^2(y)", "\\tan^2(y)+1", timeout=True))
    # print(math_equal("\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\end{pmatrix}", "(\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix})", timeout=True))

    # This commented prediction is a matrix derivative-style example.
    # pred = '\\begin{pmatrix}\\frac{1}{3x^{2/3}}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\end{pmatrix}'
    # This commented ground truth is the corresponding equivalent matrix.
    # gt = '(\\begin{pmatrix}\\frac{1}{3\\sqrt[3]{x}^2}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\\\\\end{pmatrix})'

    # This commented prediction is an algebraic fraction expression.
    # pred= '-\\frac{8x^2}{9(x^2-2)^{5/3}}+\\frac{2}{3(x^2-2)^{2/3}}'
    # This commented ground truth is an equivalent algebraic fraction expression.
    # gt= '-\\frac{2(x^2+6)}{9(x^2-2)\\sqrt[3]{x^2-2}^2}'

    # This commented prediction is a plane equation with opposite sign.
    # pred =  '-34x-45y+20z-100=0'
    # This commented ground truth is the sign-flipped plane equation.
    # gt = '34x+45y-20z+100=0'

    # This commented prediction is a fraction.
    # pred = '\\frac{100}{3}'
    # This commented ground truth is a rounded decimal.
    # gt = '33.3'

    # This commented prediction is a numeric matrix.
    # pred = '\\begin{pmatrix}0.290243531202435\\\\0.196008371385084\\\\-0.186381278538813\\end{pmatrix}'
    # This commented ground truth is the rounded matrix form.
    # gt = '(\\begin{pmatrix}0.29\\\\0.196\\\\-0.186\\\\\\end{pmatrix})'

    # This commented prediction is a radical fraction.
    # pred = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{2\\sqrt{33}+15}'
    # This commented ground truth has the denominator terms reordered.
    # gt = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{15+2\\sqrt{33}}'

    # This commented prediction is an expression missing a variable.
    # pred = '(+5)(b+2)'
    # This commented ground truth includes the variable.
    # gt = '(a+5)(b+2)'

    # This commented prediction is the golden-ratio expression.
    # pred = '\\frac{1+\\sqrt{5}}{2}'
    # This commented ground truth is a non-equivalent integer.
    # gt = '2'

    # This commented pair represents another radical-vs-integer case.
    # pred = '\\frac{34}{16}+\\frac{\\sqrt{1358}}{16}', gt = '4'
    # This commented pair represents a malformed radical case.
    # pred = '1', gt = '1\\\\sqrt{19}'

    # This commented prediction is an interval with decimals.
    # pred = "(0.6,2.6667]"
    # This commented ground truth is the same interval with fractions.
    # gt = "(\\frac{3}{5},\\frac{8}{3}]"

    # The active ground truth is a symbolic expression.
    gt = "x+2n+1"
    # The active prediction is a different symbolic expression.
    pred = "x+1"

    # Print the equivalence result for standalone debugging.
    print(math_equal(pred, gt, timeout=True))


# Running this file directly executes the ad hoc grader test.
if __name__ == "__main__":
    # Call the local test helper.
    _test_math_equal()
