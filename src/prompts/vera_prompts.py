# Local: file src/prompts/vera_prompts.py provides first-party ALS source context. Global: defines verifier prompts used to label good and bad reasoning trajectories.
# Local: starts a multi-line text literal that Python treats as one value. Global: defines verifier prompts used to label good and bad reasoning trajectories.
"""
Verification prompt from Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers.

"""
# Local: sets VERA_ANSWER_SYMBOL for later use in this scope. Global: defines verifier prompts used to label good and bad reasoning trajectories.
VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER IS:"

# Local: defines the get_vera_prompt function. Global: defines verifier prompts used to label good and bad reasoning trajectories.
def get_vera_prompt(vera_name, question, solution):
    # Local: starts a multi-line text literal that Python treats as one value. Global: defines verifier prompts used to label good and bad reasoning trajectories.
    '''
    Get prompt used for verifications.
    Args:
        vera_name: str, name of the verifier.
        question: str, the question to be verified.
        solution: str, the proposed solution to the question.
    '''
    # Local: sets system_str_math for later use in this scope. Global: defines verifier prompts used to label good and bad reasoning trajectories.
    system_str_math = (
        # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
        "You are a critical verifier tasked with evaluating mathematical problem-solving. "
        # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
        "You will be presented with a question and a proposed solution. "
        # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
        "Your job is to carefully go over and analyze the solution. Follow the instructions."
    )

    # Local: sets math_prefix for later use in this scope. Global: defines verifier prompts used to label good and bad reasoning trajectories.
    math_prefix = f"""{system_str_math}\n\n
    QUESTION:
    {question}\n\n
    PROPOSED SOLUTION:
    {solution}\n\n"""

   
    # Local: sets vera_names_to_prompts for later use in this scope. Global: defines verifier prompts used to label good and bad reasoning trajectories.
    vera_names_to_prompts = {
        # Local: starts a multi-line call or expression. Global: defines verifier prompts used to label good and bad reasoning trajectories.
        "calculation_check": (
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"{math_prefix}"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "INSTRUCTIONS:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "1. EXTRACT CALCULATION EXPRESSIONS: Extract all the mathematical calculations from the PROPOSED SOLUTION.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "2. INDEPENDENT RECOMPUTATION: Break down the calculations step-by-step and recompute them.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"3. VERIFY: Compare your recomputation with the PROPOSED SOLUTION. If any discrepancy is found, output '{VERA_ANSWER_SYMBOL}False'. If all steps are correct, output '{VERA_ANSWER_SYMBOL}True'.\n\n"
            # Local: sets "NOTE: You ONLY need to check calculations(like 1 + 1 for later use in this scope. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "NOTE: You ONLY need to check calculations(like 1 + 1 = 2, 2 * 3 = 6, etc). Ignore standalone numbers(like 1, 2, 3, etc) that are not part of a computation.\n\n"
        ),

        # Local: starts a multi-line call or expression. Global: defines verifier prompts used to label good and bad reasoning trajectories.
        "answer_correct": (
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"{math_prefix}"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "INSTRUCTIONS:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "Your task is to determine whether the provided answer is correct.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "Think through the verification process carefully and logically.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "IMPORTANT RULES:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "1. Do NOT analyze the steps or methods used to arrive at the answer.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "2. Only evaluate the final answer's correctness.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "3. Your response must strictly follow the required format:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"- If the answer is correct, respond with: '{VERA_ANSWER_SYMBOL}True'.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"- If the answer is incorrect, respond with: '{VERA_ANSWER_SYMBOL}False'.\n"
        ),
        # Local: starts a multi-line call or expression. Global: defines verifier prompts used to label good and bad reasoning trajectories.
        "answer_completeness": (
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"{math_prefix}"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "INSTRUCTIONS:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "Your task is to verify whether the solution provides a complete and final answer.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "Follow these rules carefully:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "1. Check if the solution reaches a clear and definitive final answer.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "2. The answer must not be left incomplete, such as:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - Ending with an unresolved expression or formula instead of a computed result.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - Missing a conclusion or final statement explicitly stating the final answer.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "3. If the solution is incomplete or lacks a final answer, immediately stop checking further and respond in the exact format:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"   - '{VERA_ANSWER_SYMBOL}False'\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "4. If the solution is complete and provides a final, explicit answer, respond in the exact format:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"   - '{VERA_ANSWER_SYMBOL}True'\n"
            
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "Examples:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "Example 1:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "final answer: 8.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"Your response: '{VERA_ANSWER_SYMBOL}True' (The solution provides a final, definitive answer of 8.)\n"

            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "Example 2:\n"
            # Local: sets "final answer: The area of the circle is πr², where r for later use in this scope. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "final answer: The area of the circle is πr², where r = 4.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"Your response: '{VERA_ANSWER_SYMBOL}False' (The answer ends with an unresolved formula, not a computed result.)\n"

            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "Example 3:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "final answer: This question does not have an answer./I cannot solve this problem.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"Your response: '{VERA_ANSWER_SYMBOL}False' (The solution lacks a clear, final answer.)\n"
        ),
       # Local: starts a multi-line call or expression. Global: defines verifier prompts used to label good and bad reasoning trajectories.
       "understanding_check": (
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"{math_prefix}"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "INSTRUCTIONS:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "1. PROBLEM INTERPRETATION:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - Assess if the proposed solution clearly understands the problem statement.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - Ensure that the proposed solution addresses all relevant aspects of the problem, without ignoring any key detail.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - Flag if the solution misinterprets or overlooks the problem's core requirements or scope.\n\n"

            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "2. ALIGNMENT WITH THE TASK:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - Verify that the solution responds to the specific question or task outlined in the problem statement.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - Ensure that the solution does not deviate from the problem’s context or provides an unrelated answer.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - Check if any critical parts of the problem have been misinterpreted or neglected.\n\n"

            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "3. TERMINATION PROTOCOL:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - If the solution clearly misinterprets or fails to address the problem correctly, stop and respond in the exact format:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"     - '{VERA_ANSWER_SYMBOL}False'\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "   - If the solution accurately captures the problem statement and aligns with the required solution, respond in the exact format:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"     - '{VERA_ANSWER_SYMBOL}True'\n"

            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "EXAMPLES:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "[Case 1] Problem: A shop is selling a drink at 1.5 times the original price. If the original price is $10, what is the new price?\n"
            # Local: sets "  Solution: The new price is 1.15 * $10 for later use in this scope. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "  Solution: The new price is 1.15 * $10 = $11.50.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "  Assessment: The solution misinterprets the problem by calculating 1.15 times the original price instead of 1.5 times.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "[Case 2] Problem: The second cup of coffee is half price. If the first cup costs $5, how much is the second cup?\n"
            # Local: sets "  Solution: The second cup costs $5 * 0.5 for later use in this scope. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "  Solution: The second cup costs $5 * 0.5 = $2.50.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "  Assessment: The solution correctly interprets the price as half the original price for the second cup.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"  Result: '{VERA_ANSWER_SYMBOL}True'\n\n"

            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "[Case 3] Problem: A pizza has a radius of 8 inches. What is the area of the pizza?\n"
            # Local: sets "  Solution: The area is π * r², where r for later use in this scope. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "  Solution: The area is π * r², where r = 4 inches. The area is 16π square inches.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "  Assessment: The solution misinterprets the formula for the area of a circle by using the radius incorrectly.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "[Case 4] Problem: A train is moving at 60 km/h towards the east. What is its velocity after 2 hours?\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "  Solution: The velocity is 120 km/h west.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "  Assessment: The solution correctly calculates the speed, but misinterprets the direction as west instead of east.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            f"  Result: '{VERA_ANSWER_SYMBOL}False'\n\n"

            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "CRITICAL REQUIREMENTS:\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "- Assess whether the solution addresses all parts of the problem.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "- Ensure the solution does not deviate from the problem’s intent.\n"
            # Local: executes this statement in the current code path. Global: defines verifier prompts used to label good and bad reasoning trajectories.
            "- Use exact output formats specified, showing no tolerance for misinterpretations."
        ),

    }
    # Local: returns the computed result to the caller. Global: defines verifier prompts used to label good and bad reasoning trajectories.
    return vera_names_to_prompts[vera_name]
