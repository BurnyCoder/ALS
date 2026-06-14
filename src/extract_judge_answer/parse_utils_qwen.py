"""Qwen/OpenR-style answer parsing helpers for math benchmark outputs.

This file is largely borrowed from OpenR and supports MATH-style extraction in
`utils.extract_answer`. It normalizes LaTeX, units, boxed answers, theorem-style
answers, word numbers, and dataset-specific question/ground-truth shapes.
"""

# `random` is imported by the copied utility set and kept for compatibility.
import random
# The `regex` package supports matching patterns used by copied parsers.
import regex
# Standard regular expressions implement most normalization and extraction rules.
import re
# SymPy is used when converting parsed LaTeX/math text into evaluable expressions.
import sympy
# `latex2sympy` converts LaTeX answer fragments into SymPy expressions for cleanup.
from latex2sympy2 import latex2sympy
# Typing imports document helper signatures copied from OpenR.
from typing import TypeVar, Iterable, List, Union, Any, Dict
# `word2number` converts written numbers like "three" into digits.
from word2number import w2n


# This helper repairs shorthand LaTeX fractions such as `\frac12`.
def _fix_fracs(string):
    """Normalize malformed `\frac` expressions by adding braces when safe."""
    # Split text at every fraction command.
    substrs = string.split("\\frac")
    # Keep the prefix before the first fraction.
    new_str = substrs[0]
    # Only process if at least one fraction command was present.
    if len(substrs) > 1:
        # Drop the prefix so the loop sees fraction tails only.
        substrs = substrs[1:]
        # Repair each fraction tail.
        for substr in substrs:
            # Reattach the fraction command before the repaired tail.
            new_str += "\\frac"
            # Already braced fractions are preserved.
            if len(substr) > 0 and substr[0] == "{":
                # Append the tail unchanged.
                new_str += substr
            # Unbraced fractions need numerator/denominator reconstruction.
            else:
                # Need at least two characters for numerator and denominator.
                try:
                    # Assert the minimal length required for repair.
                    assert len(substr) >= 2
                # Malformed fractions fall back to the original string.
                except:
                    # Return unchanged rather than guessing.
                    return string
                # First character is treated as numerator.
                a = substr[0]
                # Second character is treated as denominator or opening brace.
                b = substr[1]
                # If denominator is not braced, brace both single-character terms.
                if b != "{":
                    # Preserve any remaining suffix after numerator/denominator.
                    if len(substr) > 2:
                        # Store the suffix after the first two characters.
                        post_substr = substr[2:]
                        # Add braces around numerator and denominator.
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    # Minimal two-character fraction tail.
                    else:
                        # Add braces around numerator and denominator.
                        new_str += "{" + a + "}{" + b + "}"
                # If denominator already starts a brace, only brace numerator.
                else:
                    # Preserve remaining denominator tail when present.
                    if len(substr) > 2:
                        # Store the suffix after the brace marker.
                        post_substr = substr[2:]
                        # Add braces around numerator and preserve denominator text.
                        new_str += "{" + a + "}" + b + post_substr
                    # Minimal numerator plus brace tail.
                    else:
                        # Add braces around numerator and append the brace marker.
                        new_str += "{" + a + "}" + b
    # Assign the rebuilt string to the local variable for return.
    string = new_str
    # Return the normalized expression.
    return string


# This helper converts simple slash fractions to LaTeX fractions.
def _fix_a_slash_b(string):
    """Convert simple `a/b` answers into `\frac{a}{b}` when safe."""
    # Only exactly one slash is considered a simple fraction.
    if len(string.split("/")) != 2:
        # Return unchanged for complex expressions.
        return string
    # Candidate numerator is before the slash.
    a = string.split("/")[0]
    # Candidate denominator is after the slash.
    b = string.split("/")[1]
    # Conversion is attempted only for integer-like sides, except square-root fragments.
    try:
        # Numerators containing sqrt are left as strings.
        if "sqrt" not in a:
            # Convert non-sqrt numerator to integer.
            a = int(a)
        # Denominators containing sqrt are left as strings.
        if "sqrt" not in b:
            # Convert non-sqrt denominator to integer.
            b = int(b)
        # Ensure reconstruction equals the original text.
        assert string == "{}/{}".format(a, b)
        # Build canonical LaTeX fraction text.
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        # Return the converted fraction.
        return new_string
    # Any failed parse or assertion leaves the string unchanged.
    except:
        # Return original text for non-simple slash expressions.
        return string


# This helper braces shorthand square-root commands.
def _fix_sqrt(string):
    """Normalize `\sqrtx` into `\sqrt{x}` with a regex substitution."""
    # Regex captures the word after `\sqrt` and wraps it in braces.
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    # Return the normalized root string.
    return _string


# This helper converts written-out numbers into digits when possible.
def convert_word_number(text: str) -> str:
    """Convert word numbers like `three` to `3` when `word2number` can parse."""
    # Conversion can fail for non-number text, so keep a broad fallback.
    try:
        # Convert parsed word number to string form.
        text = str(w2n.word_to_num(text))
    # Non-number text is left unchanged.
    except:
        # Preserve original behavior by ignoring conversion failures.
        pass
    # Return converted or original text.
    return text


# This list contains unit words and fragments removed from extracted numeric answers.
unit_texts = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m",
    "deg", "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd",
    "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab",
    "way", "west", "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .",
    "t", "a", "h", "no change", "men", "soldier", "pie", "bc", "excess",
    "st", "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre",
    "rise", "a . m", "th", "π r 2", "sq", "mark", "l", "toy", "coin",
    "sq . m", "gallon", "° f", "profit", "minw", "yr", "women", "feet",
    "am", "pm", "hr", "cu cm", "square", "v â € ™", "are", "rupee",
    "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day",
    "hour", "minute", "min", "second", "man", "woman", "sec", "cube",
    "mt", "sq inch", "mp", "∏ cm ³", "hectare", "more", "sec", "unit",
    "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month", "km", "m",
    "cm", "mm", "apple", "liter", "loss", "yard", "pure", "year",
    "increase", "decrease", "d", "less", "Surface", "litre", "pi sq m",
    "s .", "metre", "meter", "inch",
]

# Plural variants are added because model outputs often include units in plural form.
unit_texts.extend([t + "s" for t in unit_texts])


# This function applies the main answer-string normalization pipeline.
def strip_string(string, skip_unit=False):
    """Normalize a predicted or reference math answer string.

    Locally, this removes whitespace, units, delimiters, LaTeX size commands,
    percentage/currency markers, and shorthand fractions/roots. Globally, it
    makes extracted MATH answers comparable across ALS and baseline outputs.
    """
    # Convert to string and trim outer whitespace.
    string = str(string).strip()
    # Remove line breaks so multi-line answers compare as one string.
    string = string.replace("\n", "")

    # Remove a trailing period from sentence-style answers.
    string = string.rstrip(".")

    # Remove LaTeX inverse-space commands.
    string = string.replace("\\!", "")

    # Normalize array environments into pmatrix environments.
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    # Normalize array endings into pmatrix endings.
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    # Treat bmatrix as pmatrix for comparison purposes.
    string = string.replace("bmatrix", "pmatrix")

    # Normalize text-style fractions to standard `frac`.
    string = string.replace("tfrac", "frac")
    # Normalize display-style fractions to standard `frac`.
    string = string.replace("dfrac", "frac")
    # Normalize common comparison operators to shorter LaTeX commands.
    string = (string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge"))

    # Remove LaTeX left delimiter sizing commands.
    string = string.replace("\\left", "")
    # Remove LaTeX right delimiter sizing commands.
    string = string.replace("\\right", "")
    # Unescape left braces.
    string = string.replace("\\{", "{")
    # Unescape right braces.
    string = string.replace("\\}", "}")

    # Remove trailing `\text{...}` units when present.
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    # If unit removal left nonempty text and changed the string, use it.
    if _string != "" and _string != string:
        # Assign the unit-stripped string.
        string = _string

    # Unless instructed otherwise, remove known unit words/fragments.
    if not skip_unit:
        # Repeat twice because removing one unit can expose another unit boundary.
        for _ in range(2):
            # Test every known unit fragment.
            for unit_text in unit_texts:
                # Use word/non-word boundaries so unit removal does not erase arbitrary substrings.
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                # Keep nonempty removals.
                if _string != "":
                    # Update the string after this unit removal pass.
                    string = _string

    # Remove degree markers written as braced superscript circ.
    string = string.replace("^{\\circ}", "")
    # Remove degree markers written as unbraced circ.
    string = string.replace("^\\circ", "")

    # Remove escaped dollar signs.
    string = string.replace("\\$", "")
    # Remove literal dollar signs.
    string = string.replace("$", "")
    # Remove LaTeX inline math open/close delimiters.
    string = string.replace("\\(", "").replace("\\)", "")

    # Convert written numbers to digits when possible.
    string = convert_word_number(string)

    # Replace `\text{...}` wrappers with their contents.
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    # Remove short variable assignment prefixes and set-membership prefixes.
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        # Removing these prefixes leaves the answer value.
        string = string.replace(key, "")
    # Normalize empty set spelling.
    string = string.replace("\\emptyset", r"{}")
    # Normalize full real-line interval to blackboard R.
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # Remove escaped percent commands.
    string = string.replace("\\%", "")
    # Remove backslash-percent text.
    string = string.replace("\%", "")
    # Remove literal percent signs.
    string = string.replace("%", "")

    # Normalize decimal strings missing a leading zero after spaces.
    string = string.replace(" .", " 0.")
    # Normalize decimal strings missing a leading zero after braces.
    string = string.replace("{.", "{0.")

    # Remove one layer of grouping around alphanumeric atoms when present.
    if (string.startswith("{") and string.endswith("}") and string.isalnum() or
            string.startswith("(") and string.endswith(")") and string.isalnum() or
            string.startswith("[") and string.endswith("]") and string.isalnum()):
        # Strip the first and last grouping characters.
        string = string[1:-1]

    # Normalize spelled-out infinity.
    string = string.replace("infinity", "\\infty")
    # Normalize `inf` when the string does not already contain `\infty`.
    if "\\infty" not in string:
        # Replace `inf` with LaTeX infinity.
        string = string.replace("inf", "\\infty")
    # Fix a common misspelling of infinity.
    string = string.replace("+\\inity", "\\infty")

    # Remove the word "and" from compound final-answer phrases.
    string = string.replace("and", "")
    # Remove LaTeX boldface command text.
    string = string.replace("\\mathbf", "")

    # Remove mbox text blocks.
    string = re.sub(r"\\mbox{.*?}", "", string)

    # Preserve original copied behavior: these replace calls are not assigned.
    string.replace("'", "")
    # Preserve original copied behavior: these replace calls are not assigned.
    string.replace('"', "")

    # Use `i` instead of `j` for imaginary unit if `i` is absent.
    if "j" in string and "i" not in string:
        # Replace `j` with `i`.
        string = string.replace("j", "i")

    # Remove redundant `.000` before non-digits.
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    # Remove redundant `.000` at the end.
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # Empty strings are returned before indexing.
    if len(string) == 0:
        # Return the empty normalized string.
        return string
    # A leading decimal point gets an explicit zero.
    if string[0] == ".":
        # Prefix zero to decimals such as `.5`.
        string = "0" + string

    # Short assignments like `x=3` are reduced to the right-hand side.
    if len(string.split("=")) == 2:
        # Only strip if the left side is short enough to be a variable.
        if len(string.split("=")[0]) <= 2:
            # Keep the answer side of the assignment.
            string = string.split("=")[1]

    # Normalize shorthand square roots.
    string = _fix_sqrt(string)
    # Remove all spaces.
    string = string.replace(" ", "")

    # Normalize shorthand LaTeX fractions.
    string = _fix_fracs(string)

    # Normalize simple slash fractions to LaTeX fractions.
    string = _fix_a_slash_b(string)

    # Return the fully normalized string.
    return string


# These phrases trigger direct-answer extraction in few-shot-style outputs.
direct_answer_trigger_for_fewshot = ("choice is", "answer is")


# This helper cleans multiple-choice answer text.
def choice_answer_clean(pred: str):
    """Extract a clean A-E choice or answer phrase from prediction text."""
    # Remove leading/trailing newlines first.
    pred = pred.strip("\n")

    # Track whether the output looks like in-context-learning examples.
    ICL = False
    # Repeated direct-answer triggers imply multiple demonstrations.
    for trigger in direct_answer_trigger_for_fewshot:
        # More than one trigger marks ICL-style output.
        if pred.count(trigger) > 1:
            # Enable ICL trimming.
            ICL = True
    # If ICL-like, keep only the first chunk before a blank-line separator.
    if ICL:
        # Split off the first response chunk.
        pred = pred.split("\n\n")[0]

    # Split on direct-answer trigger phrases.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    # If a trigger appeared, the last segment is likely the answer.
    if len(preds) > 1:
        # Mark that answer-trigger mode was used.
        answer_flag = True
        # Keep the text after the final trigger.
        pred = preds[-1]
    # If no trigger appeared, use generic cleanup.
    else:
        # Mark that no explicit answer trigger was found.
        answer_flag = False

    # Strip punctuation and whitespace around the candidate.
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Find standalone A-E option letters.
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    # Use found choices if any exist.
    if tmp:
        # Store all candidate choices.
        pred = tmp
    # Otherwise keep the cleaned text as one candidate.
    else:
        # Strip another layer of period punctuation.
        pred = [pred.strip().strip(".")]

    # Empty candidate lists become empty strings.
    if len(pred) == 0:
        # Store empty answer.
        pred = ""
    # Nonempty candidates are selected according to trigger mode.
    else:
        # Trigger mode chooses the first candidate after the answer phrase.
        if answer_flag:
            # Choose first candidate.
            pred = pred[0]
        # Generic mode chooses the last candidate.
        else:
            # Choose last candidate.
            pred = pred[-1]

    # Remove final period or slash after candidate selection.
    pred = pred.rstrip(".").rstrip("/")

    # Return cleaned answer text.
    return pred


# This helper extracts content from the last `boxed` occurrence.
def find_box(pred_str: str):
    """Return the content inside the last boxed expression."""
    # Split at the last textual `boxed` marker.
    ans = pred_str.split("boxed")[-1]
    # Empty tail means no content after boxed.
    if not ans:
        # Return empty extraction.
        return ""
    # Braced boxed content is parsed by balancing nested braces.
    if ans[0] == "{":
        # Stack depth starts after the first opening brace.
        stack = 1
        # Accumulate content inside the outer braces.
        a = ""
        # Iterate through characters after the opening brace.
        for c in ans[1:]:
            # Nested opening braces increase depth.
            if c == "{":
                # Increase brace stack.
                stack += 1
                # Keep nested brace content.
                a += c
            # Closing braces decrease depth.
            elif c == "}":
                # Decrease brace stack.
                stack -= 1
                # Depth zero means the outer boxed content ended.
                if stack == 0:
                    # Stop consuming boxed content.
                    break
                # Keep inner closing braces.
                a += c
            # Ordinary characters are part of the boxed content.
            else:
                # Append the character to the extraction.
                a += c
    # Unbraced boxed content is read until a dollar sign.
    else:
        # Split on dollar delimiter and strip surrounding whitespace.
        a = ans.split("$")[0].strip()
    # Return extracted boxed content.
    return a


# This helper removes or normalizes units and constants in a prediction string.
def clean_units(pred_str: str):
    """Clean units and convert pi/percent syntax in extracted answers."""

    # Nested helper rewrites pi tokens to numeric approximations where evaluable.
    def convert_pi_to_number(code_string):
        """Convert standalone pi tokens to 3.14-compatible numeric text."""
        # Normalize LaTeX pi to Unicode pi first.
        code_string = code_string.replace("\\pi", "π")
        # Replace standalone pi not preceded by a digit or closing brace.
        code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
        # Replace implicit multiplication such as `3π`.
        code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
        # Replace braced pi with numeric pi.
        code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
        # Replace explicit multiplication by pi.
        code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
        # Return converted code text.
        return code_string

    # Convert pi tokens before other unit cleanup.
    pred_str = convert_pi_to_number(pred_str)
    # Convert percentage syntax into division by 100.
    pred_str = pred_str.replace("%", "/100")
    # Remove dollar signs.
    pred_str = pred_str.replace("$", "")
    # Remove yen signs.
    pred_str = pred_str.replace("¥", "")
    # Remove Celsius unit marker.
    pred_str = pred_str.replace("°C", "")
    # Remove spaced Celsius marker.
    pred_str = pred_str.replace(" C", "")
    # Remove degree symbol.
    pred_str = pred_str.replace("°", "")
    # Return cleaned unit string.
    return pred_str


# This helper extracts theorem-style answers, including booleans and choices.
def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    """Extract an answer from theorem/QA-style prediction text."""
    # Yes/true outputs normalize to Python-style True.
    if any([option in pred.lower() for option in ["yes", "true"]]):
        # Store canonical True.
        pred = "True"
    # No/false outputs normalize to Python-style False.
    elif any([option in pred.lower() for option in ["no", "false"]]):
        # Store canonical False.
        pred = "False"
    # Multiple-choice markers are left for downstream logic.
    elif any([option in pred.lower() for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]]):
        # Preserve copied no-op behavior.
        pass
    # Otherwise use boxed/evaluable/last-number extraction.
    else:
        # Boxed output is common from pretrained math models.
        if "boxed" in pred:
            # Extract boxed content.
            pred = find_box(pred)

        # In answer mode, try to evaluate the candidate expression.
        if answer_flag:
            # Keep text after the last equals sign.
            pred = pred.split("=")[-1].strip()
            # Remove units and normalize pi/percent.
            pred = clean_units(pred)
            # Try converting LaTeX to SymPy, then evaluating.
            try:
                # Convert LaTeX-like expression to a SymPy string.
                tmp = str(latex2sympy(pred))
                # Evaluate the expression string.
                pred = str(eval(tmp))
            # If evaluation fails, fall back to simple numeric stripping.
            except Exception:
                # Number followed by non-space unit can be split.
                if re.match(r"-?[\d\.]+\s\D+$", pred):
                    # Keep the numeric prefix.
                    pred = pred.split(" ")[0]
                # Number followed by one unit token can be split.
                elif re.match(r"-?[\d\.]+\s[^\s]+$", pred):
                    # Keep the numeric prefix.
                    pred = pred.split(" ")[0]
        # In non-answer mode, search for the last number.
        else:
            # Find all integer/decimal numeric substrings.
            preds = re.findall(r"-?\d*\.?\d+", pred)
            # If at least one number exists, use the last one.
            if len(preds) >= 1:
                # Store the last numeric candidate.
                pred = preds[-1]
            # If no numbers exist, extraction returns empty text.
            else:
                # Store empty answer.
                pred = ""

    # Return extracted theorem-style answer.
    return pred


# This is the main prediction-answer extractor used for MATH boxed outputs.
def extract_answer(pred_str, data_name, use_last_number=True):
    """Extract the final answer from a generated math response.

    Locally, this prioritizes Minerva-style final answer markers, boxed content,
    final-answer phrases, Chinese answer markers, and finally the last number.
    Globally, `utils.extract_answer` uses it for MATH-500 prompt 0 scoring.
    """
    # Remove stray Cyrillic characters observed in some outputs.
    pred_str = pred_str.replace("\u043a\u0438", "")

    # Minerva-style outputs wrap the final answer in dollar signs and a fixed phrase.
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # Keep text after the final-answer phrase.
        tmp = pred_str.split("final answer is $", 1)[1]
        # Keep text before the closing phrase.
        pred = tmp.split("$. I hope", 1)[0].strip()
    # Boxed answers are the most common math final-answer format.
    elif "boxed" in pred_str:
        # Keep text after the last `boxed` marker.
        ans = pred_str.split("boxed")[-1]
        # Empty boxed tail means extraction fails to empty string.
        if len(ans) == 0:
            # Return empty answer.
            return ""
        # Braced boxed content is parsed by balancing braces.
        elif ans[0] == "{":
            # Stack depth starts after the first opening brace.
            stack = 1
            # Accumulate boxed content.
            a = ""
            # Iterate through characters after the opening brace.
            for c in ans[1:]:
                # Nested opening braces increase depth.
                if c == "{":
                    # Increase depth.
                    stack += 1
                    # Keep nested opening brace.
                    a += c
                # Closing braces decrease depth.
                elif c == "}":
                    # Decrease depth.
                    stack -= 1
                    # Depth zero means outer boxed content ended.
                    if stack == 0:
                        # Stop scanning boxed content.
                        break
                    # Keep inner closing brace.
                    a += c
                # Ordinary characters are part of the boxed answer.
                else:
                    # Append character to boxed extraction.
                    a += c
        # Unbraced boxed content is read to the next dollar sign.
        else:
            # Split on dollar delimiter and trim whitespace.
            a = ans.split("$")[0].strip()
        # Store boxed extraction as prediction.
        pred = a
    # English phrase "the answer is" can appear as "he answer is" after splitting.
    elif "he answer is" in pred_str:
        # Keep text after the phrase.
        pred = pred_str.split("he answer is")[-1].strip()
    # Explicit "final answer is" phrase is another common output.
    elif "final answer is" in pred_str:
        # Keep text after the phrase.
        pred = pred_str.split("final answer is")[-1].strip()
    # Chinese final-answer marker is supported by the copied parser.
    elif "答案是" in pred_str:
        # Keep text after the Chinese marker and before double newline.
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    # If no explicit marker exists, use the last number when allowed.
    else:
        # Last-number fallback is enabled by default.
        if use_last_number:
            # Match signed integers/decimals after removing comma separators.
            pattern = "-?\d*\.?\d+"
            # Collect numeric candidates.
            pred = re.findall(pattern, pred_str.replace(",", ""))
            # Use the last numeric candidate when present.
            if len(pred) >= 1:
                # Store the final number.
                pred = pred[-1]
            # No numeric candidates produces empty answer.
            else:
                # Store empty answer.
                pred = ""
        # If last-number fallback is disabled, extraction fails to empty answer.
        else:
            # Store empty answer.
            pred = ""

    # Remove line breaks and following spaces inside the extracted answer.
    pred = re.sub(r"\n\s*", "", pred)
    # Strip a leading colon left by answer markers.
    if pred != "" and pred[0] == ":":
        # Remove the colon.
        pred = pred[1:]
    # Strip a trailing period.
    if pred != "" and pred[-1] == ".":
        # Remove the period.
        pred = pred[:-1]
    # Strip a trailing slash.
    if pred != "" and pred[-1] == "/":
        # Remove the slash.
        pred = pred[:-1]
    # Normalize the extracted string, optionally preserving units for listed datasets.
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    # Return normalized answer text.
    return pred


# These datasets skip unit removal in `strip_string`.
STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]


# This helper extracts ground truth answer text using the same parser.
def parse_ground_truth(groudtruth_solution: str, data_name):
    """Parse a dataset ground-truth solution into a normalized answer."""
    # Reuse prediction extraction on the ground-truth solution.
    gt_ans = extract_answer(groudtruth_solution, data_name)
    # Return the parsed answer.
    return gt_ans


# This helper builds question text for many datasets from OpenR's parser suite.
def parse_question(example, data_name):
    """Return the question text for a dataset example.

    Locally, this handles several dataset schemas from the copied OpenR utility.
    Globally, only a subset is used by this repo, but preserving branches keeps
    the parser compatible with its source behavior.
    """
    # Start with empty question text until a dataset branch fills it.
    question = ""
    # ASDiv stores body and question separately.
    if data_name == "asdiv":
        # Concatenate body and question fields.
        question = f"{example['body'].strip()} {example['question'].strip()}"
    # SVAMP stores body and question with capitalized keys.
    elif data_name == "svamp":
        # Strip the body text.
        body = example["Body"].strip()
        # Ensure the body ends with a period before concatenation.
        if not body.endswith("."):
            # Add a period when missing.
            body = body + "."
        # Concatenate body and question.
        question = f'{body} {example["Question"].strip()}'
    # TabMWP includes table metadata and optional choices.
    elif data_name == "tabmwp":
        # Include table title when present.
        title_str = (f'regarding "{example["table_title"]}" ' if example["table_title"] else "")
        # Build the table-reading prompt prefix.
        question = f"Read the following table {title_str}and answer a question:\n"
        # Append table and question text.
        question += f'{example["table"]}\n{example["question"]}'
        # Append answer choices when available.
        if example["choices"]:
            # Add choices to the question text.
            question += (f' Please select from the following options: {example["choices"]}')
    # CARP English stores prompt content in `content`.
    elif data_name == "carp_en":
        # Use content field directly.
        question = example["content"]
    # MMLU STEM stores question and four choices.
    elif data_name == "mmlu_stem":
        # Extract choices list.
        options = example["choices"]
        # Assert four choices for A-D formatting.
        assert len(options) == 4
        # Add option labels to each choice.
        for i, (label, option) in enumerate(zip("ABCD", options)):
            # Format one labeled option.
            options[i] = f"({label}) {str(option).strip()}"
        # Join options into one line.
        options = " ".join(options)
        # Combine question and choices.
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    # SAT math has a single options string.
    elif data_name == "sat_math":
        # Strip options text.
        options = example["options"].strip()
        # The copied parser expects options to start with A.
        assert "A" == options[0]
        # Add opening parenthesis before A.
        options = "(" + options
        # Normalize spacing before B-D option labels.
        for ch in "BCD":
            # If the label appears with old spacing, replace it.
            if f" {ch}) " in options:
                # Add an opening parenthesis before the label.
                options = regex.sub(f" {ch}\) ", f" ({ch}) ", options)
        # Combine question and choices.
        question = f"{example['question'].strip()}\nAnswer Choices: {options}"
    # AQUA-style datasets store options separately.
    elif "aqua" in data_name:
        # Extract options list.
        options = example["options"]
        # Join options with parentheses.
        choice = "(" + "(".join(options)
        # Normalize parenthesis spacing.
        choice = choice.replace("(", " (").replace(")", ") ").strip()
        # Add answer choices header.
        choice = "\nAnswer Choices: " + choice
        # Combine question and choices.
        question = example["question"].strip() + choice
    # Gaokao math QA stores options as a dictionary.
    elif data_name == "gaokao_math_qa":
        # Extract option dictionary.
        options_dict = example["options"]
        # Accumulate formatted options.
        options = []
        # Iterate through option keys.
        for key in options_dict:
            # Append one formatted option.
            options.append(f"({key}) {options_dict[key]}")
        # Join formatted options.
        options = " ".join(options)
        # Combine question and Chinese option label.
        question = f"{example['question'].strip()}\n选项: {options}"
    # Generic branch tries common question keys.
    else:
        # Try each possible question field name.
        for key in ["question", "problem", "Question", "input"]:
            # If the key exists, use it.
            if key in example:
                # Store the question field.
                question = example[key]
                # Stop after the first matching key.
                break
    # Preserve copied behavior: parse ground truth to append yes/no or true/false hints.
    _, gt_ans = parse_ground_truth(example, data_name)
    # String answers can trigger a parenthetical answer-type hint.
    if isinstance(gt_ans, str):
        # Lowercase ground-truth answer for comparison.
        gt_lower = gt_ans.lower()
        # True/false labels add an explicit hint to the question.
        if gt_lower in ["true", "false"]:
            # Append true/false hint.
            question += " (True or False)"
        # Yes/no labels add an explicit hint to the question.
        if gt_lower in ["yes", "no"]:
            # Append yes/no hint.
            question += " (Yes or No)"
    # Return stripped question text.
    return question.strip()
