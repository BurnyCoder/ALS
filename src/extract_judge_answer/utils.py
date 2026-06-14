"""Answer extraction and correctness judging for ALS benchmark outputs.

Generation modes return raw model text, but metrics need normalized final
answers. This module extracts dataset/prompt-specific answers and combines
several math equivalence checks so ALS, LatentSeek, and baselines are scored
under the same rules.
"""

# JSON parsing handles prompt format P2, where models should emit `final answer` fields.
import json
# Regular expressions implement fallback extraction for boxed answers and numeric text.
import re
# `math_verify` provides one symbolic/numeric verifier for MATH-500 answers.
from math_verify import parse, verify
# The Qwen math grader provides another robust symbolic/numeric comparison path.
from .grader import math_equal_process
# The original MATH string normalizer provides a final equivalence fallback.
from .math_equivalent_MATH import is_equiv
# Qwen/OpenR-style extraction handles boxed symbolic answers for MATH-style outputs.
from .parse_utils_qwen import extract_answer as extract_fn


# This helper converts raw dataset labels into direct final-answer strings.
def extract_true_answer(text, name="gsm8k"):
    """Extract the ground-truth final answer from a dataset label field.

    Locally, GSM8K labels contain reasoning plus a `####` answer delimiter,
    while MATH-500 and AIME labels are already answer strings. Globally, this
    gives every generation mode the same target value for judging.
    """
    # GSM8K labels store the final answer after `#### `.
    if "gsm8k" in name:
        # Splitting on the delimiter isolates the official numeric label.
        label = text.split("#### ")[1]
        # Return the extracted GSM8K label for exact comparison.
        return label
    # MATH-500 labels are already final-answer strings.
    elif "MATH-500" in name:
        # Return unchanged so symbolic equivalence checkers can parse it later.
        return text
    # AIME labels are already final-answer strings.
    elif "AIME_2024" in name:
        # Return unchanged for exact numeric comparison.
        return text
    # Unsupported datasets have no known label format.
    else:
        # Raising prevents silent scoring against an unparsed label.
        raise ValueError(f"Unknown dataset name: {name}")


# This function scores one model output against one ground-truth label.
def judge_answer(input, label, data_name="gsm8k", extract=True, prompt_idx=0):
    """Return whether a model response is correct for the selected dataset.

    Locally, this optionally extracts the answer from raw response text and then
    applies dataset-specific equivalence rules. Globally, it is the accuracy
    gate used by `main.py` for ALS, ALS-Gated, LatentSeek, and baselines.
    """
    # GSM8K is judged by exact match after numeric extraction.
    if "gsm8k" in data_name:
        # Raw model output needs answer extraction unless the caller already supplied an answer.
        if extract:
            # Extract according to GSM8K prompt format.
            input = extract_answer(input, data_name="gsm8k", prompt_idx=prompt_idx)
        # Exact string equality is the GSM8K correctness rule in this codebase.
        return (input == label)
    # MATH-500 uses several increasingly permissive equivalence checks.
    elif "MATH-500" in data_name:
        # Raw model output needs MATH-specific answer extraction unless already extracted.
        if extract:
            # Extract boxed or JSON answer text depending on prompt index.
            input = extract_answer(input, data_name="MATH-500", prompt_idx=prompt_idx)

        # `math_verify.parse` converts the predicted answer into verifier input.
        hf_input = parse(input)
        # `verify` compares the parsed prediction against the reference label.
        hf_verifier_judge = verify(label, hf_input)
        # A successful Hugging Face verifier result accepts the answer.
        if hf_verifier_judge:
            # Return immediately because one accepted equivalence path is enough.
            return True

        # The Qwen math grader checks numeric, symbolic, tuple, matrix, and equation equivalences.
        qwen_verifier_judge = math_equal_process((label, input))
        # A successful Qwen-style equivalence result accepts the answer.
        if qwen_verifier_judge:
            # Return immediately because one accepted equivalence path is enough.
            return True

        # Exact string match catches simple cases not requiring parsing.
        exact_judge = (str(input) == str(label))
        # Exact matches are accepted.
        if exact_judge:
            # Return True for direct textual equality.
            return True

        # The original MATH normalizer catches formatting-only LaTeX differences.
        MATH_500_judge = is_equiv(str(label), str(input))
        # A successful normalized equality check accepts the answer.
        if MATH_500_judge:
            # Return True because the final fallback accepted equivalence.
            return True
        # If no equivalence path accepts the answer, mark it incorrect.
        return False

    # AIME uses exact string comparison after extraction.
    elif "AIME_2024" in data_name:
        # Raw model output needs AIME-specific extraction unless already extracted.
        if extract:
            # Extract boxed or JSON answer text depending on prompt index.
            input = extract_answer(input, data_name="AIME_2024", prompt_idx=prompt_idx)
            # Convert prediction to string for exact comparison.
            input = str(input)
            # Convert label to string for exact comparison.
            label = str(label)
        # AIME answers are exact numeric strings in this evaluation path.
        return (input == label)

    # Unsupported datasets cannot be judged safely.
    else:
        # Raise with dataset context for easier debugging.
        raise ValueError(f"Unknown dataset name: {data_name} for judge answer")


# This dispatcher extracts answer text according to dataset and prompt format.
def extract_answer(text, data_name="gsm8k", prompt_idx=0, model_name="Qwen2.5-7B-Instruct"):
    """Extract a final answer from raw model response text.

    Locally, each branch handles the expected boxed or JSON output format for a
    dataset. Globally, consistent extraction makes accuracy comparisons fair
    across ALS and all baselines.
    """
    # GSM8K answers are expected to be plain numbers after extraction.
    if "gsm8k" in data_name:
        # Prompt 0 expects boxed/free-form output.
        if prompt_idx == 0:
            # Qwen 2.5 1.5B has a known output quirk handled by a special extractor.
            if "qwen2.5-1.5b-instruct" in model_name.lower():
                # The special extractor catches phrasing like "he answer is".
                temp = _extract_qwen25_1_5B_answer(text)
            # Other models use the generic numeric extractor.
            else:
                # The generic extractor checks boxed, answer-prefix, equation, and last-number patterns.
                temp = _extract_answer(text)
            # Return the extracted numeric answer or None.
            return temp

        # Prompt 1 expects JSON with a `final answer` key.
        elif prompt_idx == 1:
            # Try strict JSON parsing first because valid P2 output should parse directly.
            try:
                # Strip common markdown fences/backticks and parse the JSON object.
                answer = json.loads(text.strip('` \n'))
                # Read the final-answer field, defaulting to empty text if missing.
                final_answer = answer.get('final answer', '')
                # Non-string final answers are converted so the numeric extractor can process them.
                if not isinstance(final_answer, str):
                    # Convert numbers or other JSON scalars to strings.
                    final_answer = str(final_answer)
                # Extract the numeric answer from the final-answer field.
                temp = _extract_answer(final_answer)
                # Return the normalized numeric string or None.
                return temp

            # Malformed JSON falls back to regex extraction.
            except json.JSONDecodeError:
                # The regex looks for final-answer-like text up to a brace or tag boundary.
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)[}<]'

                # Search case-insensitively across multiline output.
                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL)

                # If the regex found a final-answer field, extract a number from that group.
                if match:
                    # Generic numeric extraction handles currency, commas, and last numbers.
                    temp = _extract_answer(match.group(1))
                    # Return the extracted answer.
                    return temp
                # If no field-like pattern appears, fall back to scanning the whole response.
                else:
                    # Generic numeric extraction uses the last plausible number when needed.
                    temp = _extract_answer(text)
                    # Return the extracted answer or None.
                    return temp

        # Prompt indices outside 0/1 have no defined extraction rule.
        else:
            # Raise with prompt context for debugging.
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")

    # MATH-500 can contain symbolic answers and LaTeX, so extraction differs from GSM8K.
    elif "MATH-500" in data_name:
        # Prompt 0 expects boxed/free-form mathematical output.
        if prompt_idx == 0:
            # The Qwen/OpenR extractor handles boxed symbolic and numeric math answers.
            temp = extract_fn(text, data_name='math')
            # Return the extracted math answer string.
            return temp

        # Prompt 1 expects JSON output.
        elif prompt_idx == 1:
            # Try strict JSON parsing first for valid P2 output.
            try:
                # Strip common markdown fences/backticks and parse the JSON object.
                answer = json.loads(text.strip('` \n'))
                # Read the final-answer field, defaulting to empty string if missing.
                final_answer = answer.get('final answer', '')
                # Convert non-string JSON values to strings for consistent judging.
                if not isinstance(final_answer, str):
                    # Convert numbers or other scalar values to strings.
                    final_answer = str(final_answer)
                # Remove line breaks from symbolic answer text.
                final_answer = final_answer.replace("\n", "")
                # Remove double quotes that may be nested in the field value.
                final_answer = final_answer.replace("\"", "")
                # Remove single quotes that may be nested in the field value.
                final_answer = final_answer.replace("\'", "")
                # Return the cleaned symbolic/numeric answer.
                return final_answer

            # Malformed JSON falls back to regex extraction.
            except json.JSONDecodeError:
                # Remove newlines so the fallback regex sees a compact field.
                text = text.replace("\n", "")
                # The regex captures after final-answer-like labels up to JSON/tag boundaries.
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)(}<|<\|)'


                # Search case-insensitively across the compacted text.
                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL)

                # If a fallback match exists, clean its captured field.
                if match:
                    # Use the first capture group as the candidate answer.
                    temp = match.group(1)
                    # Remove line breaks defensively.
                    temp = temp.replace("\n", "")
                    # Remove nested double quotes.
                    temp = temp.replace("\"", "")
                    # Remove nested single quotes.
                    temp = temp.replace("\'", "")
                    # Return the cleaned fallback answer.
                    return temp
                # If no final-answer-like field is found, extraction fails.
                else:
                    # Returning None lets callers mark the generation incorrect.
                    return None

    # AIME uses numeric answers and follows the GSM8K-style extraction paths.
    elif "AIME_2024" in data_name:
        # Prompt 0 expects boxed/free-form numeric output.
        if prompt_idx == 0:
            # Use the generic numeric extractor.
            temp = _extract_answer(text)
            # Return the extracted answer or None.
            return temp

        # Prompt 1 expects JSON output.
        elif prompt_idx == 1:
            # Try strict JSON parsing first.
            try:
                # Strip common markdown fences/backticks and parse the JSON object.
                answer = json.loads(text.strip('` \n'))
                # Read the final-answer field, defaulting to empty string if missing.
                final_answer = answer.get('final answer', '')
                # Convert non-string JSON values to strings.
                if not isinstance(final_answer, str):
                    # Convert numbers or other scalar values to strings.
                    final_answer = str(final_answer)
                # Extract a numeric answer from the field.
                temp = _extract_answer(final_answer)
                # Return the extracted numeric string or None.
                return temp

            # Malformed JSON falls back to regex and then whole-text extraction.
            except json.JSONDecodeError:
                # The regex looks for final-answer-like text up to a brace or tag boundary.
                pattern = r'(?:final answer|my answer)"?:?\s*(.*?)[}<]'

                # Search case-insensitively across multiline output.
                match = re.search(pattern, text, flags=re.I | re.M | re.DOTALL)

                # If a final-answer-like field exists, extract a number from it.
                if match:
                    # Generic numeric extraction handles the captured field.
                    temp = _extract_answer(match.group(1))
                    # Return the extracted numeric answer.
                    return temp
                # Otherwise scan the whole response.
                else:
                    # Generic numeric extraction uses the last plausible number.
                    temp = _extract_answer(text)
                    # Return the extracted answer or None.
                    return temp

        # Prompt indices outside 0/1 have no AIME extraction rule.
        else:
            # Raise with prompt context for debugging.
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")
    # Unsupported datasets cannot be extracted safely.
    else:
        # Raise with dataset context for debugging.
        raise ValueError(f"Unknown dataset name: {data_name} for extract answer")


######################
#       MATH         #
######################

# This helper extracts boxed answers from model responses with assistant wrappers.
def extract_MATH_solution(solution_str: str):
    """Extract the final boxed answer from a MATH-style solution string.

    Locally, it strips assistant prefixes and searches XML-style or plain boxed
    patterns. Globally, it supports MATH answer normalization paths used during
    benchmark scoring.
    """
    # If a plain assistant prefix exists, keep only the assistant response text.
    if "Assistant:" in solution_str:
        # Splitting once avoids losing content after later occurrences.
        processed_str = solution_str.split("Assistant:", 1)[1]
    # ChatML-style assistant prefixes are handled separately.
    elif "<|im_start|>assistant" in solution_str:
        # Splitting once keeps the assistant content after the special token.
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # If no assistant marker exists, the whole string is treated as the response.
    else:
        # Use the original string unchanged.
        processed_str = solution_str

    # Prefer boxed answers inside XML-style `<answer>` tags.
    answer_pattern = r'<answer>.*?(\\boxed{.*}).*?</answer>'
    # Collect all matches so the last answer can be selected.
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    # If no XML-style boxed answer is found, try any boxed expression.
    if not matches:
        # This pattern captures the contents of a `\boxed{...}` expression.
        answer_pattern = r'\\boxed{(.*)}'
        # Collect all boxed matches in the processed response.
        matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    # If neither pattern matches, extraction fails.
    if not matches:
        # Print an error so failed extraction is visible in logs.
        print("[Error] No valid answer tags found")
        # Return None so callers can mark the answer incorrect.
        return None
    # The final match is treated as the model's final answer.
    final_answer = matches[-1].group(1).strip()
    # Return the stripped boxed-answer content.
    return final_answer


# This generic helper extracts numeric answers from free-form text.
def _extract_answer(text):
    """Extract a numeric final answer from generated text.

    Locally, the function tries boxed answers, explicit answer prefixes,
    equations, currency phrases, line endings, and finally the last number.
    Globally, it supports GSM8K and AIME exact-match judging.
    """
    # Missing text cannot yield an answer.
    if text is None:
        # Return None so callers count the output as unparseable.
        return None

    # Remove leading/trailing whitespace before regex matching.
    text = text.strip()

    # This nested helper canonicalizes numeric strings after regex capture.
    def clean_number(num_str):
        """Remove currency symbols, commas, and whitespace from one number."""
        # Drop common currency symbols before exact numeric comparison.
        num_str = re.sub(r'[$€£¥]', '', num_str)
        # Drop thousands separators.
        num_str = re.sub(r',', '', num_str)
        # Drop whitespace inside the captured number.
        num_str = re.sub(r'\s', '', num_str)
        # Return the cleaned numeric string.
        return num_str

    # Pattern 1 captures numeric content inside `\boxed{...}`.
    boxed_pattern = r"\\boxed\{\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*\}"
    # Search boxed answer case-insensitively.
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    # A boxed numeric answer is treated as highest-priority final answer.
    if match:
        # Clean and return the captured number.
        return clean_number(match.group(1))

    # Pattern 2 captures numbers after an `Answer:` prefix.
    answer_pattern = r"Answer:\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # Search the explicit answer prefix case-insensitively.
    match = re.search(answer_pattern, text, re.IGNORECASE)
    # An explicit `Answer:` match is accepted before looser patterns.
    if match:
        # Clean and return the captured number.
        return clean_number(match.group(1))

    # Pattern 3 captures a number after an equals sign.
    equals_pattern = r"=\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # Search for equation-style final numeric output.
    match = re.search(equals_pattern, text)
    # An equals-match is accepted when no boxed or answer-prefix match appeared.
    if match:
        # Clean and return the captured number.
        return clean_number(match.group(1))

    # Pattern 4 captures numbers in currency phrases like "is $5 dollars".
    currency_pattern = r"is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*(?:dollars|euros|pounds|yen)"
    # Search the currency phrase case-insensitively.
    match = re.search(currency_pattern, text, re.IGNORECASE)
    # A currency match is accepted before line/last-number fallbacks.
    if match:
        # Clean and return the captured number.
        return clean_number(match.group(1))

    # Split into lines to search from the final line upward.
    lines = text.split('\n')
    # Later lines are more likely to contain final answers, so iterate in reverse.
    for line in reversed(lines):
        # Strip whitespace around each candidate line.
        line = line.strip()
        # Empty lines cannot contain final numeric answers.
        if line:
            # Capture a number at the end of the line.
            final_num_pattern = r"([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*$"
            # Search this line for an ending numeric answer.
            match = re.search(final_num_pattern, line)
            # A line-ending number is accepted as a final-answer fallback.
            if match:
                # Clean and return the captured number.
                return clean_number(match.group(1))

    # Final fallback captures every plausible number in the text.
    number_pattern = r"([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # `findall` returns all numeric candidates in reading order.
    matches = re.findall(number_pattern, text)
    # If any numbers exist, the final one is treated as the answer.
    if matches:
        # Clean and return the last captured number.
        return clean_number(matches[-1])

    # No numeric pattern matched, so extraction failed.
    return None


# This specialized helper handles Qwen-2.5 1.5B answer phrasing quirks.
def _extract_qwen25_1_5B_answer(text):
    """Extract a numeric answer from Qwen-2.5 1.5B output.

    Locally, it checks boxed answers plus phrases observed in that model's
    generations. Globally, it keeps one model's formatting quirk from unfairly
    lowering GSM8K/AIME exact-match accuracy.
    """
    # Missing text cannot yield an answer.
    if text is None:
        # Return None so callers count the output as unparseable.
        return None

    # Remove leading/trailing whitespace before regex matching.
    text = text.strip()

    # This nested helper canonicalizes numeric strings after regex capture.
    def clean_number(num_str):
        """Remove currency symbols, commas, and whitespace from one number."""
        # Drop common currency symbols before exact numeric comparison.
        num_str = re.sub(r'[$€£¥]', '', num_str)
        # Drop thousands separators.
        num_str = re.sub(r',', '', num_str)
        # Drop whitespace inside the captured number.
        num_str = re.sub(r'\s', '', num_str)
        # Return the cleaned numeric string.
        return num_str

    # Pattern 1 captures numeric content inside `\boxed{...}`.
    boxed_pattern = r"\\boxed\{\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*\}"
    # Search boxed answer case-insensitively.
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    # A boxed numeric answer is treated as highest-priority final answer.
    if match:
        # Clean and return the captured number.
        return clean_number(match.group(1))

    # Pattern 2 intentionally matches the tail of "the answer is" even if the leading `t` is missing.
    answer_pattern = r"he answer is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # Search the model-specific phrase case-insensitively.
    match = re.search(answer_pattern, text, re.IGNORECASE)
    # If found, accept this as the answer.
    if match:
        # Clean and return the captured number.
        return clean_number(match.group(1))

    # Pattern 3 captures explicit "final answer is" phrasing.
    answer_pattern = r"final answer is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    # Search the final-answer phrase case-insensitively.
    match = re.search(answer_pattern, text, re.IGNORECASE)
    # If found, accept this as the answer.
    if match:
        # Clean and return the captured number.
        return clean_number(match.group(1))

    # Final fallback captures every unsigned numeric candidate in the response.
    number_pattern = r'\d+(?:,\d+)*(?:\.\d+)?'
    # `findall` returns all numeric candidates in reading order.
    matches = re.findall(number_pattern, text)
    # If any numbers exist, use the final one.
    if matches:
        # Clean and return the last captured number.
        return clean_number(matches[-1])

    # No numeric pattern matched, so extraction failed.
    return None
