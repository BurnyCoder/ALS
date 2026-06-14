# Local: file src/prompts/solver_prompts.py provides first-party ALS source context. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
# Local: starts a multi-line text literal that Python treats as one value. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
"""
Solver prompts for different math datasets.
"""

def gsm8k_prompt(q, prompt_idx=0):  # Local: defines the gsm8k_prompt function. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
    # Local: starts a multi-line text literal that Python treats as one value. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """     
    prompt = [  # Local: sets prompt for later use in this scope. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
            # idx 0: boxed
            [  # Local: starts a multi-line list literal. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                # Local: adds an item or argument to the surrounding expression. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                # Local: adds an item or argument to the surrounding expression. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "user", "content": q},
                ],

            # idx 1: json
            [  # Local: starts a multi-line list literal. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                # Local: adds an item or argument to the surrounding expression. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "user", "content": 
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 f"QUESTION: {q} \n"
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "Let's think step by step. "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "Please provide your thought process and your final answer separately and response in json format "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "containing the keys \"thought process\" and \"final answer\". "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "For example your response should be "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "}
                ],

    ]
    # Local: returns the computed result to the caller. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
    return prompt[prompt_idx] 


# Local: defines the MATH_500_prompt function. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
def MATH_500_prompt(q, prompt_idx=0):
    # Local: starts a multi-line text literal that Python treats as one value. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
    """
    Args:
        q (str): The question to be solved.
    """
    prompt = [  # Local: sets prompt for later use in this scope. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
            # idx 0: boxed
            [  # Local: starts a multi-line list literal. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                # Local: adds an item or argument to the surrounding expression. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                # Local: adds an item or argument to the surrounding expression. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "user", "content": q},
                ],
            # idx 1: json
            [  # Local: starts a multi-line list literal. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                # Local: adds an item or argument to the surrounding expression. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "user", "content": 
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 f"QUESTION: {q} \n"
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "Let's think step by step. "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "Please provide your thought process and your final answer separately and response in json format "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "containing the keys \"thought process\" and \"final answer\". "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "For example your response should be "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 }
                ],
            ]
    return prompt[prompt_idx]  # Local: returns the computed result to the caller. Global: defines the boxed and JSON prompt conditions used in ALS experiments.


# Local: defines the AIME_2024_prompt function. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
def AIME_2024_prompt(q, prompt_idx=0):
    # Local: starts a multi-line text literal that Python treats as one value. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """     
    prompt = [  # Local: sets prompt for later use in this scope. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
            # idx 0: boxed
            [  # Local: starts a multi-line list literal. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                # Local: adds an item or argument to the surrounding expression. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                # Local: adds an item or argument to the surrounding expression. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "user", "content": q},
                ],
            # idx 1: json 
            [  # Local: starts a multi-line list literal. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                # Local: adds an item or argument to the surrounding expression. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
                # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                {"role": "user", "content": 
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 f"QUESTION: {q} \n"
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "Let's think step by step. "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "Please provide your thought process and your final answer separately and response in json format "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "containing the keys \"thought process\" and \"final answer\". "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "For example your response should be "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                 # Local: executes this statement in the current code path. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
                 "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "}
                ],


    ]
    # Local: returns the computed result to the caller. Global: defines the boxed and JSON prompt conditions used in ALS experiments.
    return prompt[prompt_idx] 


