# Local: file src/data.py provides first-party ALS source context. Global: formats benchmark examples into the prompt styles used by ALS experiments.
# Local: starts a multi-line text literal that Python treats as one value. Global: formats benchmark examples into the prompt styles used by ALS experiments.
"""
Data api
"""
# Local: imports selected helpers from datasets. Global: formats benchmark examples into the prompt styles used by ALS experiments.
from datasets import load_dataset, load_from_disk
# Local: imports selected helpers from prompts. Global: formats benchmark examples into the prompt styles used by ALS experiments.
from prompts import gsm8k_prompt, MATH_500_prompt, AIME_2024_prompt

# Local: defines the get_dataset function. Global: formats benchmark examples into the prompt styles used by ALS experiments.
def get_dataset(data_name_or_path, tokenizer, prompt_idx, split="test"):
    # Local: starts a multi-line text literal that Python treats as one value. Global: formats benchmark examples into the prompt styles used by ALS experiments.
    """
    Args:
        data_name_or_path: dataset name or path
        tokenizer: tokenizer
        prompt_idx: which query prompt to use
    Returns:
        dataset: dataset
    """

    ### Load dataset ### 
    # Local: opens a condition that selects behavior from current state. Global: formats benchmark examples into the prompt styles used by ALS experiments.
    if "gsm8k" in data_name_or_path:
        # Local: starts a protected operation that may fail on external or parsed input. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        try:
            # Local: loads benchmark data from disk or Hugging Face datasets. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            dataset = load_from_disk(data_name_or_path)[split]
        # Local: handles a recoverable failure from the protected operation. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        except:
            # Local: loads benchmark data from disk or Hugging Face datasets. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            dataset = load_dataset("openai/gsm8k", "socratic")["test"]
        # Local: sets question_col for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        question_col = "question"
        # Local: sets answer_col for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        answer_col = "answer"

    # Local: checks the next mutually exclusive condition. Global: formats benchmark examples into the prompt styles used by ALS experiments.
    elif "MATH-500" in data_name_or_path:
        # Local: starts a protected operation that may fail on external or parsed input. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        try:
            # Local: loads benchmark data from disk or Hugging Face datasets. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            dataset = load_from_disk(data_name_or_path)[split]
        # Local: handles a recoverable failure from the protected operation. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        except:
            # Local: loads benchmark data from disk or Hugging Face datasets. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        # Local: sets question_col for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        question_col = "problem"
        # Local: sets answer_col for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        answer_col = "answer"

    # Local: checks the next mutually exclusive condition. Global: formats benchmark examples into the prompt styles used by ALS experiments.
    elif "AIME_2024" in data_name_or_path:
        # Local: starts a protected operation that may fail on external or parsed input. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        try:
            # Local: loads benchmark data from disk or Hugging Face datasets. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            dataset = load_from_disk(data_name_or_path)[split]
        # Local: handles a recoverable failure from the protected operation. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        except:
            # Local: loads benchmark data from disk or Hugging Face datasets. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            dataset = load_dataset("Maxwell-Jia/AIME_2024")["test"]
        # Local: sets question_col for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        question_col = "Problem"
        # Local: sets answer_col for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        answer_col = "Answer"

    else:  # Local: handles the remaining branch after earlier checks fail. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        # Local: stops execution with an explicit error for invalid state. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        raise ValueError(f"Unsupported dataset: {data_name_or_path}")

    # preprocess dataset
    # Local: defines the preprocess_function function. Global: formats benchmark examples into the prompt styles used by ALS experiments.
    def preprocess_function(examples):
        # Local: starts a multi-line text literal that Python treats as one value. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        '''
        Preprocess dataset

        Args:
            examples: dataset examples

        Returns:
            formatted: formatted dataset
        '''
        formatted = []  # Local: sets formatted for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        # Local: sets questions for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        questions = examples[question_col]
        # Local: iterates through the current collection. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        for q in questions:
            # Local: opens a condition that selects behavior from current state. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            if "gsm8k" in data_name_or_path:
                # Local: sets messages for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
                messages = gsm8k_prompt(q, prompt_idx)
            # Local: checks the next mutually exclusive condition. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            elif "MATH-500" in data_name_or_path:
                # Local: sets messages for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
                messages = MATH_500_prompt(q, prompt_idx)
            # Local: checks the next mutually exclusive condition. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            elif "AIME_2024" in data_name_or_path:
                # Local: sets messages for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
                messages = AIME_2024_prompt(q, prompt_idx)
            # Local: handles the remaining branch after earlier checks fail. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            else:
                # Local: stops execution with an explicit error for invalid state. Global: formats benchmark examples into the prompt styles used by ALS experiments.
                raise ValueError(f"Unsupported dataset: {data_name_or_path}")

            # Local: formats chat messages using the model tokenizer template. Global: formats benchmark examples into the prompt styles used by ALS experiments.
            formatted.append(tokenizer.apply_chat_template(
                # Local: sets messages, tokenize for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
                messages, tokenize=False, add_generation_prompt=True
            ))  # Local: closes the surrounding literal or call expression. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        # Local: returns the computed result to the caller. Global: formats benchmark examples into the prompt styles used by ALS experiments.
        return {"formatted": formatted, "question": questions, "answer": examples[answer_col]}

    # Local: sets dataset for later use in this scope. Global: formats benchmark examples into the prompt styles used by ALS experiments.
    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
    return dataset  # Local: returns the computed result to the caller. Global: formats benchmark examples into the prompt styles used by ALS experiments.

