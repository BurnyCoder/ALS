"""Verifier prompt templates used by the reward model.

The prompts are adapted from multi-verifier test-time-compute work. In this
repository they serve two ALS-related roles: labeling hidden states as good/bad
during offline collection and providing reward for the LatentSeek baseline.
"""

# The reward parser searches for this marker before reading a verifier's True/False decision.
VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER IS:"


# This function selects the verifier-specific prompt for one question/solution pair.
def get_vera_prompt(vera_name, question, solution):
    """Build the prompt for one named mathematical verifier.

    Locally, this interpolates the task question and proposed solution into one
    verifier instruction. Globally, verifier outputs become either binary labels
    for ALS state collection or scalar reward components for LatentSeek.
    """
    # The shared verifier role tells the model to evaluate rather than solve.
    system_str_math = (
        # This sentence establishes a critical-verifier role.
        "You are a critical verifier tasked with evaluating mathematical problem-solving. "
        # This sentence defines the two inputs the verifier will inspect.
        "You will be presented with a question and a proposed solution. "
        # This sentence asks for careful analysis under the following verifier instructions.
        "Your job is to carefully go over and analyze the solution. Follow the instructions."
    )

    # The prefix injects the shared role, original question, and candidate solution into every verifier.
    math_prefix = f"""{system_str_math}\n\n
    QUESTION:
    {question}\n\n
    PROPOSED SOLUTION:
    {solution}\n\n"""

    # The dictionary maps verifier names from `RewardModel.load_domain_specific_verifiers` to full prompts.
    vera_names_to_prompts = {
        # This verifier checks arithmetic and algebraic calculations inside the solution.
        "calculation_check": (
            # The common prefix gives the verifier the task and candidate solution.
            f"{math_prefix}"
            # The instruction header separates context from verifier-specific rules.
            "INSTRUCTIONS:\n"
            # Step 1 tells the verifier to identify calculations before judging them.
            "1. EXTRACT CALCULATION EXPRESSIONS: Extract all the mathematical calculations from the PROPOSED SOLUTION.\n"
            # Step 2 asks for independent recomputation rather than trusting the solution trace.
            "2. INDEPENDENT RECOMPUTATION: Break down the calculations step-by-step and recompute them.\n"
            # Step 3 requires the fixed marker plus True/False so `extract_verifier_approval` can parse it.
            f"3. VERIFY: Compare your recomputation with the PROPOSED SOLUTION. If any discrepancy is found, output '{VERA_ANSWER_SYMBOL}False'. If all steps are correct, output '{VERA_ANSWER_SYMBOL}True'.\n\n"
            # The note narrows the verifier to actual computations rather than every standalone number.
            "NOTE: You ONLY need to check calculations(like 1 + 1 = 2, 2 * 3 = 6, etc). Ignore standalone numbers(like 1, 2, 3, etc) that are not part of a computation.\n\n"
        ),

        # This verifier checks final-answer correctness without grading the reasoning path.
        "answer_correct": (
            # The common prefix supplies the question and proposed solution.
            f"{math_prefix}"
            # The instruction header begins the verifier-specific contract.
            "INSTRUCTIONS:\n"
            # This line defines the verifier's task as final-answer correctness.
            "Your task is to determine whether the provided answer is correct.\n"
            # This line asks the verifier to reason before producing the marker.
            "Think through the verification process carefully and logically.\n"
            # This line introduces constraints that keep the verifier focused.
            "IMPORTANT RULES:\n"
            # Rule 1 prevents step-quality judgments from affecting this verifier.
            "1. Do NOT analyze the steps or methods used to arrive at the answer.\n"
            # Rule 2 scopes the verifier to final answer correctness only.
            "2. Only evaluate the final answer's correctness.\n"
            # Rule 3 enforces the parser-readable output format.
            "3. Your response must strictly follow the required format:\n"
            # The positive case includes the required marker plus True.
            f"- If the answer is correct, respond with: '{VERA_ANSWER_SYMBOL}True'.\n"
            # The negative case includes the required marker plus False.
            f"- If the answer is incorrect, respond with: '{VERA_ANSWER_SYMBOL}False'.\n"
        ),
        # This verifier checks whether a solution actually reaches a final answer.
        "answer_completeness": (
            # The common prefix supplies the question and candidate solution.
            f"{math_prefix}"
            # The instruction header begins the completeness contract.
            "INSTRUCTIONS:\n"
            # This line defines completeness as a final-answer property.
            "Your task is to verify whether the solution provides a complete and final answer.\n"
            # This line asks the verifier to follow explicit checks.
            "Follow these rules carefully:\n"
            # Rule 1 checks that the solution reaches a clear conclusion.
            "1. Check if the solution reaches a clear and definitive final answer.\n"
            # Rule 2 introduces incomplete-output examples.
            "2. The answer must not be left incomplete, such as:\n"
            # This subcase rejects unresolved formulas.
            "   - Ending with an unresolved expression or formula instead of a computed result.\n"
            # This subcase rejects missing final statements.
            "   - Missing a conclusion or final statement explicitly stating the final answer.\n"
            # Rule 3 tells the verifier to emit False immediately for incomplete outputs.
            "3. If the solution is incomplete or lacks a final answer, immediately stop checking further and respond in the exact format:\n"
            # The false marker is parser-readable by the reward model.
            f"   - '{VERA_ANSWER_SYMBOL}False'\n"
            # Rule 4 tells the verifier to emit True for complete final answers.
            "4. If the solution is complete and provides a final, explicit answer, respond in the exact format:\n"
            # The true marker is parser-readable by the reward model.
            f"   - '{VERA_ANSWER_SYMBOL}True'\n"

            # Examples calibrate the verifier's interpretation of completeness.
            "Examples:\n"
            # Example 1 introduces a complete numeric final answer.
            "Example 1:\n"
            # The example final answer is short and definitive.
            "final answer: 8.\n"
            # The expected verifier response is True for a complete final answer.
            f"Your response: '{VERA_ANSWER_SYMBOL}True' (The solution provides a final, definitive answer of 8.)\n"

            # Example 2 introduces an unresolved formula with a parameter.
            "Example 2:\n"
            # The formula still depends on `r`, so it is incomplete.
            "final answer: The area of the circle is πr², where r = 4.\n"
            # The expected verifier response is False for unresolved output.
            f"Your response: '{VERA_ANSWER_SYMBOL}False' (The answer ends with an unresolved formula, not a computed result.)\n"

            # Example 3 introduces a refusal or missing answer.
            "Example 3:\n"
            # The answer text does not provide a solution.
            "final answer: This question does not have an answer./I cannot solve this problem.\n"
            # The expected verifier response is False for missing final content.
            f"Your response: '{VERA_ANSWER_SYMBOL}False' (The solution lacks a clear, final answer.)\n"
        ),
       # This verifier checks whether the solution understood and addressed the problem.
       "understanding_check": (
            # The common prefix supplies the question and proposed solution.
            f"{math_prefix}"
            # The instruction header begins the understanding contract.
            "INSTRUCTIONS:\n"
            # Section 1 scopes the verifier to problem interpretation.
            "1. PROBLEM INTERPRETATION:\n"
            # This line asks whether the solution understood the statement.
            "   - Assess if the proposed solution clearly understands the problem statement.\n"
            # This line asks whether the solution covers all relevant details.
            "   - Ensure that the proposed solution addresses all relevant aspects of the problem, without ignoring any key detail.\n"
            # This line flags misinterpretation or missing scope.
            "   - Flag if the solution misinterprets or overlooks the problem's core requirements or scope.\n\n"

            # Section 2 scopes the verifier to task alignment.
            "2. ALIGNMENT WITH THE TASK:\n"
            # This line checks that the solution answers the asked question.
            "   - Verify that the solution responds to the specific question or task outlined in the problem statement.\n"
            # This line rejects unrelated or contextually wrong answers.
            "   - Ensure that the solution does not deviate from the problem’s context or provides an unrelated answer.\n"
            # This line asks for critical misinterpretation checks.
            "   - Check if any critical parts of the problem have been misinterpreted or neglected.\n\n"

            # Section 3 defines exactly how to terminate with a parser-readable marker.
            "3. TERMINATION PROTOCOL:\n"
            # This line defines the false condition.
            "   - If the solution clearly misinterprets or fails to address the problem correctly, stop and respond in the exact format:\n"
            # The false marker is parser-readable by the reward model.
            f"     - '{VERA_ANSWER_SYMBOL}False'\n"
            # This line defines the true condition.
            "   - If the solution accurately captures the problem statement and aligns with the required solution, respond in the exact format:\n"
            # The true marker is parser-readable by the reward model.
            f"     - '{VERA_ANSWER_SYMBOL}True'\n"

            # Examples calibrate the verifier's notion of misunderstanding.
            "EXAMPLES:\n"
            # Case 1 is a multiplier misunderstanding.
            "[Case 1] Problem: A shop is selling a drink at 1.5 times the original price. If the original price is $10, what is the new price?\n"
            # The example solution uses 1.15 instead of 1.5.
            "  Solution: The new price is 1.15 * $10 = $11.50.\n"
            # The assessment explains why the solution misread the problem.
            "  Assessment: The solution misinterprets the problem by calculating 1.15 times the original price instead of 1.5 times.\n"
            # The expected result is False for misinterpretation.
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            # Case 2 is a correctly interpreted half-price problem.
            "[Case 2] Problem: The second cup of coffee is half price. If the first cup costs $5, how much is the second cup?\n"
            # The example solution applies the half-price rule correctly.
            "  Solution: The second cup costs $5 * 0.5 = $2.50.\n"
            # The assessment explains why the interpretation is correct.
            "  Assessment: The solution correctly interprets the price as half the original price for the second cup.\n"
            # The expected result is True for correct interpretation.
            f"  Result: '{VERA_ANSWER_SYMBOL}True'\n\n"

            # Case 3 is a radius/formula misunderstanding.
            "[Case 3] Problem: A pizza has a radius of 8 inches. What is the area of the pizza?\n"
            # The example solution uses the wrong radius value.
            "  Solution: The area is π * r², where r = 4 inches. The area is 16π square inches.\n"
            # The assessment explains the formula/application error.
            "  Assessment: The solution misinterprets the formula for the area of a circle by using the radius incorrectly.\n"
            # The expected result is False for misunderstanding.
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            # Case 4 is a direction/velocity misunderstanding.
            "[Case 4] Problem: A train is moving at 60 km/h towards the east. What is its velocity after 2 hours?\n"
            # The example solution changes direction incorrectly.
            "  Solution: The velocity is 120 km/h west.\n"
            # The assessment explains that speed calculation is not enough if direction is wrong.
            "  Assessment: The solution correctly calculates the speed, but misinterprets the direction as west instead of east.\n"
            # The expected result is False for misinterpreting direction.
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            # This header reinforces hard requirements after examples.
            "CRITICAL REQUIREMENTS:\n"
            # This line asks the verifier to check all problem parts.
            "- Assess whether the solution addresses all parts of the problem.\n"
            # This line asks the verifier to reject deviation from intent.
            "- Ensure the solution does not deviate from the problem’s intent.\n"
            # This line enforces exact marker output for downstream parsing.
            "- Use exact output formats specified, showing no tolerance for misinterpretations."
        ),

    }
    # Returning by key lets `RewardModel` request each verifier by name.
    return vera_names_to_prompts[vera_name]
