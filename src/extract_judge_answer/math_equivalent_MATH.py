"""Lightweight string-normalization equivalence for MATH-500 answers.

This file is copied from the original MATH evaluation logic and is used as one
fallback judge in ALS evaluation. It normalizes common LaTeX/string variants so
answers that are textually different but syntactically equivalent can match.
"""


# This helper repairs shorthand `\frac12`-style LaTeX fractions.
def _fix_fracs(string):
    """Normalize malformed LaTeX fraction commands.

    Locally, the function rewrites unbraced numerator/denominator patterns into
    `\frac{a}{b}`. Globally, this improves MATH-500 judging after model outputs
    lose braces during generation.
    """
    # Splitting on `\frac` isolates text before and after every fraction command.
    substrs = string.split("\\frac")
    # The output starts with the text before the first fraction command.
    new_str = substrs[0]
    # Only strings containing at least one fraction need repair.
    if len(substrs) > 1:
        # Drop the pre-fraction prefix so the loop sees only fraction tails.
        substrs = substrs[1:]
        # Process each fraction tail independently.
        for substr in substrs:
            # Reinsert the fraction command before its repaired tail.
            new_str += "\\frac"
            # Already braced fractions are preserved unchanged.
            if substr[0] == "{":
                # Appending the tail keeps correct LaTeX as-is.
                new_str += substr
            # Unbraced fractions need numerator/denominator reconstruction.
            else:
                # Short tails cannot supply both numerator and denominator characters.
                try:
                    # The assertion checks that at least two characters exist.
                    assert len(substr) >= 2
                # Malformed short fractions fall back to the original string.
                except:
                    # Returning the input avoids making an unsafe repair.
                    return string
                # The first character is treated as the numerator.
                a = substr[0]
                # The second character is treated as the denominator or opening brace.
                b = substr[1]
                # If the denominator is not already braced, brace both single-character parts.
                if b != "{":
                    # Preserve any characters after the numerator/denominator pair.
                    if len(substr) > 2:
                        # `post_substr` carries the rest of the expression after the fraction.
                        post_substr = substr[2:]
                        # Add braces around numerator and denominator, then append the rest.
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    # A two-character tail becomes exactly `\frac{a}{b}`.
                    else:
                        # Add braces around numerator and denominator.
                        new_str += "{" + a + "}{" + b + "}"
                # If the denominator starts with `{`, only the numerator needs braces.
                else:
                    # Preserve the rest of the denominator-braced tail when present.
                    if len(substr) > 2:
                        # `post_substr` carries everything after the opening denominator brace.
                        post_substr = substr[2:]
                        # Add braces around numerator and preserve the denominator tail.
                        new_str += "{" + a + "}" + b + post_substr
                    # A minimal numerator plus denominator brace tail is repaired directly.
                    else:
                        # Add braces around numerator and append the existing brace.
                        new_str += "{" + a + "}" + b
    # Assigning the rebuilt string keeps the original function structure.
    string = new_str
    # The normalized fraction string is returned for later equivalence comparison.
    return string


# This helper converts simple `a/b` numeric strings into LaTeX fractions.
def _fix_a_slash_b(string):
    """Convert simple integer slash fractions to `\frac{a}{b}`.

    Locally, only strings with exactly one slash and integer sides are changed.
    Globally, this lets model outputs like `1/2` match labels written as
    `\frac{1}{2}`.
    """
    # More or fewer than one slash means this is not a simple fraction.
    if len(string.split("/")) != 2:
        # Return unchanged to avoid altering complex expressions.
        return string
    # Text before the slash is the candidate numerator.
    a = string.split("/")[0]
    # Text after the slash is the candidate denominator.
    b = string.split("/")[1]
    # The conversion is only safe if both sides parse as integers and reconstruct exactly.
    try:
        # Parse numerator as an integer.
        a = int(a)
        # Parse denominator as an integer.
        b = int(b)
        # Ensure the original string is exactly the simple integer fraction.
        assert string == "{}/{}".format(a, b)
        # Build the equivalent LaTeX fraction.
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        # Return the normalized fraction.
        return new_string
    # Any parse or assertion failure means the string should remain unchanged.
    except:
        # Return original text for non-simple slash expressions.
        return string


# This helper strips units written in a right-side `\text{ ...}` block.
def _remove_right_units(string):
    """Remove trailing unit text from MATH-style answers.

    Locally, this splits on the MATH convention for right-side units. Globally,
    it keeps unit wording from preventing equivalent numeric/symbolic answers
    from matching.
    """
    # The MATH validation set uses `\text{ ` in this specific unit pattern.
    if "\\text{ " in string:
        # Splitting separates the mathematical answer from unit text.
        splits = string.split("\\text{ ")
        # The original evaluator expects exactly one unit split.
        assert len(splits) == 2
        # Return only the mathematical portion before the units.
        return splits[0]
    # Strings without that unit marker are already unit-free for this rule.
    else:
        # Return unchanged when there is no matching unit block.
        return string


# This helper repairs shorthand square-root commands.
def _fix_sqrt(string):
    """Normalize `\sqrt3` into `\sqrt{3}` form.

    Locally, it scans every `\sqrt` occurrence and braces the next character
    when missing. Globally, this improves equivalence for common model LaTeX
    shorthand.
    """
    # Strings without square-root commands need no repair.
    if "\\sqrt" not in string:
        # Return unchanged when no square-root command is present.
        return string
    # Split on the square-root command to examine each following tail.
    splits = string.split("\\sqrt")
    # The rebuilt string starts with the prefix before the first root.
    new_string = splits[0]
    # Each tail corresponds to one root command occurrence.
    for split in splits[1:]:
        # If the tail does not already start with `{`, brace the first character.
        if split[0] != "{":
            # The first character becomes the root argument.
            a = split[0]
            # Build a braced root argument and append the rest of the tail.
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        # Already braced root arguments are preserved.
        else:
            # Reattach the `\sqrt` command to the unchanged tail.
            new_substr = "\\sqrt" + split
        # Append this repaired root segment to the output.
        new_string += new_substr
    # Return the fully repaired root string.
    return new_string


# This function applies the full MATH answer-normalization pipeline.
def _strip_string(string):
    """Normalize a MATH answer string before exact comparison.

    Locally, this removes formatting noise, repairs common LaTeX shorthand, and
    canonicalizes simple fractions. Globally, it is the final fallback after
    stronger symbolic/numeric judges in `utils.judge_answer`.
    """
    # Remove line breaks so multi-line model answers can be compared as one string.
    string = string.replace("\n", "")

    # Remove LaTeX inverse-space commands that do not change mathematical meaning.
    string = string.replace("\\!", "")

    # Collapse escaped backslashes to the single LaTeX command slash expected by later rules.
    string = string.replace("\\\\", "\\")

    # Normalize text-style fractions to standard `frac`.
    string = string.replace("tfrac", "frac")
    # Normalize display-style fractions to standard `frac`.
    string = string.replace("dfrac", "frac")

    # Remove left delimiter sizing commands.
    string = string.replace("\\left", "")
    # Remove right delimiter sizing commands.
    string = string.replace("\\right", "")

    # Remove degree markers written with braced superscript circ.
    string = string.replace("^{\\circ}", "")
    # Remove degree markers written with unbraced circ.
    string = string.replace("^\\circ", "")

    # Remove escaped dollar signs used for currency or LaTeX delimiters.
    string = string.replace("\\$", "")

    # Strip right-side unit annotations before further normalization.
    string = _remove_right_units(string)

    # Remove escaped percent markers.
    string = string.replace("\\%", "")
    # Remove literal backslash-percent markers.
    string = string.replace("\%", "")

    # Convert `" ."` to `" 0."` so decimal strings have explicit leading zero.
    string = string.replace(" .", " 0.")
    # Convert `"{."` to `"{0."` for decimals after braces.
    string = string.replace("{.", "{0.")
    # Empty strings should return before indexing character zero.
    if len(string) == 0:
        # Return empty normalized string unchanged.
        return string
    # A leading decimal point is normalized with an explicit zero.
    if string[0] == ".":
        # Prefix zero to make `.5` comparable with `0.5`.
        string = "0" + string

    # Short variable assignments like `k = 3` are reduced to their right-hand side.
    if len(string.split("=")) == 2:
        # The left side length check avoids removing substantive equations.
        if len(string.split("=")[0]) <= 2:
            # Keep only the answer side of simple assignments.
            string = string.split("=")[1]

    # Repair shorthand square roots before removing spaces.
    string = _fix_sqrt(string)

    # Remove spaces so formatting does not affect exact comparison.
    string = string.replace(" ", "")

    # Repair shorthand LaTeX fractions such as `\frac12`.
    string = _fix_fracs(string)

    # Normalize the common decimal half to a fraction used in MATH labels.
    if string == "0.5":
        # Convert `0.5` to the canonical half fraction.
        string = "\\frac{1}{2}"

    # Convert simple `X/Y` outputs into LaTeX fraction form when safe.
    string = _fix_a_slash_b(string)

    # Return the final normalized string for exact comparison.
    return string


# This public helper compares two MATH answer strings after normalization.
def is_equiv(str1, str2, verbose=False):
    """Return whether two answer strings are equivalent under MATH normalization.

    Locally, both strings are stripped through `_strip_string` and compared.
    Globally, this provides a permissive fallback judge for MATH-500 outputs
    when parser-based numeric or symbolic verification did not already accept.
    """
    # Two missing answers are treated as equivalent only for compatibility with the original evaluator.
    if str1 is None and str2 is None:
        # The warning highlights this unusual equivalence case.
        print("WARNING: Both None")
        # Return True to preserve original MATH equivalence behavior.
        return True
    # If only one answer is missing, they cannot be equivalent.
    if str1 is None or str2 is None:
        # Return False for one-sided missing answers.
        return False

    # Normalization can fail on malformed strings, so preserve a fallback.
    try:
        # Normalize the first answer string.
        ss1 = _strip_string(str1)
        # Normalize the second answer string.
        ss2 = _strip_string(str2)
        # Verbose mode prints normalized forms for debugging judge decisions.
        if verbose:
            # Printing both normalized strings shows what exact comparison will use.
            print(ss1, ss2)
        # Equivalence is exact string equality after normalization.
        return ss1 == ss2
    # If normalization itself fails, use raw string equality as a conservative fallback.
    except:
        # Return whether the original strings match exactly.
        return str1 == str2
