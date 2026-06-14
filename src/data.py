"""Dataset loading and prompt formatting for ALS experiments.

This module hides dataset-specific column names and prompt templates behind one
function. Globally, it ensures that state collection, ALS evaluation, LatentSeek,
and baselines all see the same formatted problem text for each prompt index.
"""

# Hugging Face `load_dataset` downloads public benchmarks; `load_from_disk` supports local copies.
from datasets import load_dataset, load_from_disk
# Prompt builders define the boxed and JSON chat messages for each supported benchmark.
from prompts import gsm8k_prompt, MATH_500_prompt, AIME_2024_prompt


# This function returns a mapped dataset with formatted prompt, question, and answer fields.
def get_dataset(data_name_or_path, tokenizer, prompt_idx, split="test"):
    """Load a supported math dataset and attach model-ready chat prompts.

    Locally, the function selects dataset columns, applies a prompt template,
    and calls the tokenizer chat template. Globally, it keeps offline state
    collection and online evaluation aligned on identical input formatting.
    """

    # GSM8K uses `question`/`answer` columns and the Socratic configuration in this codebase.
    if "gsm8k" in data_name_or_path:
        # Try a local `datasets`-saved path first so experiments can use cached or custom copies.
        try:
            # Local datasets are indexed by split after loading from disk.
            dataset = load_from_disk(data_name_or_path)[split]
        # If local loading fails, fall back to the public Hugging Face dataset.
        except:
            # The fallback currently loads the public test split used by evaluations.
            dataset = load_dataset("openai/gsm8k", "socratic")["test"]
        # GSM8K problem text lives in the `question` column.
        question_col = "question"
        # GSM8K labels live in the `answer` column with `####` final-answer text.
        answer_col = "answer"

    # MATH-500 uses `problem`/`answer` columns.
    elif "MATH-500" in data_name_or_path:
        # Try a local saved dataset first for reproducibility/offline use.
        try:
            # Local datasets are indexed by requested split.
            dataset = load_from_disk(data_name_or_path)[split]
        # If local loading fails, fall back to the public benchmark.
        except:
            # The fallback loads HuggingFaceH4/MATH-500 test examples.
            dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        # MATH-500 problem text lives in the `problem` column.
        question_col = "problem"
        # MATH-500 labels are already final-answer strings in `answer`.
        answer_col = "answer"

    # AIME_2024 uses capitalized column names in its source dataset.
    elif "AIME_2024" in data_name_or_path:
        # Try local saved data before downloading from Hugging Face.
        try:
            # Local datasets are indexed by requested split.
            dataset = load_from_disk(data_name_or_path)[split]
        # If local loading fails, fall back to the public AIME dataset.
        except:
            # The fallback loads the test split used by the benchmark.
            dataset = load_dataset("Maxwell-Jia/AIME_2024")["test"]
        # AIME problem text uses a capitalized `Problem` column.
        question_col = "Problem"
        # AIME labels use a capitalized `Answer` column.
        answer_col = "Answer"

    # Any unsupported name cannot be formatted or judged safely.
    else:
        # Raising here prevents later key errors in preprocessing or answer extraction.
        raise ValueError(f"Unsupported dataset: {data_name_or_path}")

    # This nested function is passed to Hugging Face `map` and processes batches of examples.
    def preprocess_function(examples):
        """Format a batch of raw dataset rows into model chat prompts.

        Locally, this selects the right prompt template for each question and
        applies the tokenizer's chat template. Globally, the resulting
        `formatted` text is the exact input used by all ALS and baseline modes.
        """
        # `formatted` accumulates one prompt string per question in the batch.
        formatted = []
        # `questions` references the dataset-specific problem column selected above.
        questions = examples[question_col]
        # Each question is converted independently because prompt builders take one string.
        for q in questions:
            # GSM8K questions receive the GSM8K-specific boxed or JSON solver prompt.
            if "gsm8k" in data_name_or_path:
                # The prompt index selects between prompt format P1 and P2.
                messages = gsm8k_prompt(q, prompt_idx)
            # MATH-500 questions receive the MATH-specific prompt.
            elif "MATH-500" in data_name_or_path:
                # The prompt index selects between prompt format P1 and P2.
                messages = MATH_500_prompt(q, prompt_idx)
            # AIME questions receive the AIME-specific prompt.
            elif "AIME_2024" in data_name_or_path:
                # The prompt index selects between prompt format P1 and P2.
                messages = AIME_2024_prompt(q, prompt_idx)
            # This branch mirrors the outer guard in case the function is reused unexpectedly.
            else:
                # Raising avoids silently formatting unsupported datasets with the wrong prompt.
                raise ValueError(f"Unsupported dataset: {data_name_or_path}")

            # The tokenizer converts role/content messages into the model's chat prompt string.
            formatted.append(tokenizer.apply_chat_template(
                # `messages` contains system/user messages for one math problem.
                messages, tokenize=False, add_generation_prompt=True
            ))
        # The mapped dataset exposes formatted input plus raw question and answer for generation/judging.
        return {"formatted": formatted, "question": questions, "answer": examples[answer_col]}

    # Mapping attaches formatted prompts to every row and disables cache reuse because prompt index can change.
    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
    # The returned dataset is ready for state collection or evaluation loops.
    return dataset
