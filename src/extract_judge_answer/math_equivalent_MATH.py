# Local: file src/extract_judge_answer/math_equivalent_MATH.py provides first-party ALS source context. Global: normalizes MATH answers for ALS benchmark scoring.
# Local: starts a multi-line text literal that Python treats as one value. Global: normalizes MATH answers for ALS benchmark scoring.
"""
Copied from MATH_500 original paper 

- https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
"""

def _fix_fracs(string):  # Local: defines the _fix_fracs function. Global: normalizes MATH answers for ALS benchmark scoring.
    substrs = string.split("\\frac")  # Local: sets substrs for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    new_str = substrs[0]  # Local: sets new_str for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    if len(substrs) > 1:  # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
        substrs = substrs[1:]  # Local: sets substrs for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
        for substr in substrs:  # Local: iterates through the current collection. Global: normalizes MATH answers for ALS benchmark scoring.
            new_str += "\\frac"  # Local: updates new_str for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
            # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
            if substr[0] == "{":
                new_str += substr  # Local: updates new_str for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
            else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes MATH answers for ALS benchmark scoring.
                # Local: starts a protected operation that may fail on external or parsed input. Global: normalizes MATH answers for ALS benchmark scoring.
                try:
                    # Local: checks an invariant expected by this parsing path. Global: normalizes MATH answers for ALS benchmark scoring.
                    assert len(substr) >= 2
                except:  # Local: handles a recoverable failure from the protected operation. Global: normalizes MATH answers for ALS benchmark scoring.
                    return string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.
                a = substr[0]  # Local: sets a for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
                b = substr[1]  # Local: sets b for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
                if b != "{":  # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
                    # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
                    if len(substr) > 2:
                        # Local: sets post_substr for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
                        post_substr = substr[2:]
                        # Local: updates new_str for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes MATH answers for ALS benchmark scoring.
                        # Local: updates new_str for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
                        new_str += "{" + a + "}{" + b + "}"
                else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes MATH answers for ALS benchmark scoring.
                    # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
                    if len(substr) > 2:
                        # Local: sets post_substr for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
                        post_substr = substr[2:]
                        # Local: updates new_str for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
                        new_str += "{" + a + "}" + b + post_substr
                    else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes MATH answers for ALS benchmark scoring.
                        # Local: updates new_str for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
                        new_str += "{" + a + "}" + b
    string = new_str  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    return string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.

def _fix_a_slash_b(string):  # Local: defines the _fix_a_slash_b function. Global: normalizes MATH answers for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
    if len(string.split("/")) != 2:
        return string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.
    a = string.split("/")[0]  # Local: sets a for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    b = string.split("/")[1]  # Local: sets b for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    try:  # Local: starts a protected operation that may fail on external or parsed input. Global: normalizes MATH answers for ALS benchmark scoring.
        a = int(a)  # Local: sets a for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
        b = int(b)  # Local: sets b for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
        # Local: checks an invariant expected by this parsing path. Global: normalizes MATH answers for ALS benchmark scoring.
        assert string == "{}/{}".format(a, b)
        # Local: sets new_string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.
    except:  # Local: handles a recoverable failure from the protected operation. Global: normalizes MATH answers for ALS benchmark scoring.
        return string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.

def _remove_right_units(string):  # Local: defines the _remove_right_units function. Global: normalizes MATH answers for ALS benchmark scoring.
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:  # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
        splits = string.split("\\text{ ")  # Local: sets splits for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
        assert len(splits) == 2  # Local: checks an invariant expected by this parsing path. Global: normalizes MATH answers for ALS benchmark scoring.
        return splits[0]  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.
    else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes MATH answers for ALS benchmark scoring.
        return string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.

def _fix_sqrt(string):  # Local: defines the _fix_sqrt function. Global: normalizes MATH answers for ALS benchmark scoring.
    if "\\sqrt" not in string:  # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
        return string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.
    splits = string.split("\\sqrt")  # Local: sets splits for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    new_string = splits[0]   # Local: sets new_string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    for split in splits[1:]:  # Local: iterates through the current collection. Global: normalizes MATH answers for ALS benchmark scoring.
        if split[0] != "{":  # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
            a = split[0]  # Local: sets a for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
            # Local: sets new_substr for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:  # Local: handles the remaining branch after earlier checks fail. Global: normalizes MATH answers for ALS benchmark scoring.
            new_substr = "\\sqrt" + split  # Local: sets new_substr for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
        new_string += new_substr  # Local: updates new_string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    return new_string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.

def _strip_string(string):  # Local: defines the _strip_string function. Global: normalizes MATH answers for ALS benchmark scoring.
    # linebreaks  
    string = string.replace("\n", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    string = string.replace("dfrac", "frac")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    string = string.replace("\\right", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    string = string.replace("^\\circ", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    # remove dollar signs
    string = string.replace("\\$", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    
    # remove units (on the right)
    string = _remove_right_units(string)  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    # remove percentage
    string = string.replace("\\%", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    string = string.replace("\%", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    string = string.replace("{.", "{0.")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
    # if empty, return empty string
    if len(string) == 0:  # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
        return string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.
    if string[0] == ".":  # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
        string = "0" + string  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
    if len(string.split("=")) == 2:
        # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    # remove spaces
    string = string.replace(" ", "")  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":  # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
        string = "\\frac{1}{2}"  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)  # Local: sets string for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.

    return string  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.

def is_equiv(str1, str2, verbose=False):  # Local: defines the is_equiv function. Global: normalizes MATH answers for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
    if str1 is None and str2 is None:
        print("WARNING: Both None")  # Local: reports progress or diagnostics to the run log. Global: normalizes MATH answers for ALS benchmark scoring.
        return True  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.
    # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
    if str1 is None or str2 is None:
        return False  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.

    try:  # Local: starts a protected operation that may fail on external or parsed input. Global: normalizes MATH answers for ALS benchmark scoring.
        ss1 = _strip_string(str1)  # Local: sets ss1 for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
        ss2 = _strip_string(str2)  # Local: sets ss2 for later use in this scope. Global: normalizes MATH answers for ALS benchmark scoring.
        if verbose:  # Local: opens a condition that selects behavior from current state. Global: normalizes MATH answers for ALS benchmark scoring.
            print(ss1, ss2)  # Local: reports progress or diagnostics to the run log. Global: normalizes MATH answers for ALS benchmark scoring.
        return ss1 == ss2  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.
    except:  # Local: handles a recoverable failure from the protected operation. Global: normalizes MATH answers for ALS benchmark scoring.
        return str1 == str2  # Local: returns the computed result to the caller. Global: normalizes MATH answers for ALS benchmark scoring.
