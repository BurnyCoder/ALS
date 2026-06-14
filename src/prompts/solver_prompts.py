"""Solver prompts for the math reasoning datasets used by ALS.

The paper evaluates two prompt formats per dataset: prompt index 0 asks for
free-form chain-of-thought with a boxed answer, and prompt index 1 asks for a
JSON object with separate reasoning and final-answer fields.
"""


# GSM8K uses arithmetic word problems, so its JSON prompt asks for pure numeric answers.
def gsm8k_prompt(q, prompt_idx=0):
    """Return the GSM8K chat messages for the requested prompt format.

    Locally, this builds role/content dictionaries consumed by the tokenizer's
    chat template. Globally, matching prompt indices keep offline ALS state
    collection aligned with online evaluation.
    """
    # The outer list stores all supported prompt variants in prompt-index order.
    prompt = [
            # Prompt 0 is the boxed free-form CoT format from the paper's P1 setup.
            [
                # The system message instructs step-by-step reasoning and final boxed answer extraction.
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                # The user message supplies the raw GSM8K question.
                {"role": "user", "content": q},
                ],

            # Prompt 1 is the structured JSON format from the paper's P2 setup.
            [
                # The system message frames the model as a precise math solver.
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                # The user message combines the question with the required JSON schema.
                {"role": "user", "content":
                 # The formatted question prefix makes the problem boundary explicit.
                 f"QUESTION: {q} \n"
                 # This sentence asks the model to produce a reasoning trace.
                 "Let's think step by step. "
                 # This sentence requires separate reasoning and answer fields.
                 "Please provide your thought process and your final answer separately and response in json format "
                 # This sentence names the exact JSON keys used by answer extraction.
                 "containing the keys \"thought process\" and \"final answer\". "
                 # This sentence introduces a concrete schema example.
                 "For example your response should be "
                 # The example shows the expected JSON object shape.
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 # GSM8K labels are numeric, so this narrows final answer content for exact matching.
                 "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "}
                ],

    ]
    # Indexing returns exactly one role-message list for the requested prompt variant.
    return prompt[prompt_idx]


# MATH-500 uses symbolic and competition-style math, so its JSON prompt is less number-only constrained.
def MATH_500_prompt(q, prompt_idx=0):
    """Return the MATH-500 chat messages for the requested prompt format.

    Locally, this mirrors the two-format evaluation used for GSM8K. Globally,
    the function lets ALS compare steering behavior under free-form and
    structured-output constraints on harder symbolic problems.
    """
    # The outer list stores boxed and JSON prompt variants in prompt-index order.
    prompt = [
            # Prompt 0 asks for free-form reasoning with a boxed answer.
            [
                # The system message requests step-by-step reasoning and a boxed final answer.
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                # The user message supplies the raw MATH-500 problem.
                {"role": "user", "content": q},
                ],
            # Prompt 1 asks for the structured JSON output schema.
            [
                # The system message frames the model as a precise solver.
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                # The user message gives the problem and schema instruction.
                {"role": "user", "content":
                 # The formatted question prefix makes the problem boundary explicit.
                 f"QUESTION: {q} \n"
                 # This sentence asks for step-by-step reasoning.
                 "Let's think step by step. "
                 # This sentence requires reasoning and final answer to be separate.
                 "Please provide your thought process and your final answer separately and response in json format "
                 # This sentence names the keys that extraction code later reads.
                 "containing the keys \"thought process\" and \"final answer\". "
                 # This sentence introduces the example JSON schema.
                 "For example your response should be "
                 # The example object anchors the expected structured-output format.
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 }
                ],
            ]
    # Indexing returns the prompt variant requested by state collection or evaluation.
    return prompt[prompt_idx]


# AIME_2024 follows the same two-format structure and also expects numeric final answers.
def AIME_2024_prompt(q, prompt_idx=0):
    """Return the AIME 2024 chat messages for the requested prompt format.

    Locally, this uses the same boxed/JSON patterns as the other datasets.
    Globally, it lets the evaluation code reuse one prompt-index convention
    across all supported math benchmarks.
    """
    # The outer list stores all supported prompt variants in prompt-index order.
    prompt = [
            # Prompt 0 is the boxed free-form CoT format.
            [
                # The system message requests step-by-step reasoning and a boxed final answer.
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                # The user message supplies the raw AIME problem.
                {"role": "user", "content": q},
                ],
            # Prompt 1 is the structured JSON format.
            [
                # The system message frames the model as a precise solver.
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                # The user message gives the question and the JSON answer contract.
                {"role": "user", "content":
                 # The formatted question prefix makes the problem boundary explicit.
                 f"QUESTION: {q} \n"
                 # This sentence asks for chain-of-thought style reasoning.
                 "Let's think step by step. "
                 # This sentence requires separate thought and final answer fields.
                 "Please provide your thought process and your final answer separately and response in json format "
                 # This sentence names the keys used by extraction.
                 "containing the keys \"thought process\" and \"final answer\". "
                 # This sentence introduces the example schema.
                 "For example your response should be "
                 # The example object shows exactly what the JSON shape should look like.
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 # AIME labels are numeric, so this narrows final answer content for exact matching.
                 "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "}
                ],


    ]
    # Indexing returns the requested boxed or JSON prompt.
    return prompt[prompt_idx]

