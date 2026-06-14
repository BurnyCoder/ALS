# Local: file src/baselines.py provides first-party ALS source context. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
import torch  # Local: imports torch for this module. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
# Local: imports selected helpers from transformers. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
from transformers import PreTrainedModel, PreTrainedTokenizer
# Local: imports selected helpers from collections. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
from collections import Counter
# Local: imports selected helpers from extract_judge_answer. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
from extract_judge_answer import extract_answer

# Local: sets stop_words for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

# Local: defines the greedy_cot_generation function. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
def greedy_cot_generation(
    # Local: adds an item or argument to the surrounding expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    model: PreTrainedModel,
    # Local: adds an item or argument to the surrounding expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    tokenizer: PreTrainedTokenizer,
    # Local: adds an item or argument to the surrounding expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    input_text: str,
    # Local: sets max_new_tokens: int for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    max_new_tokens: int = 1024,
    # Local: sets device: str for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    device: str = "cuda",
    # Local: sets do_sample: bool for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    do_sample: bool = False,
    # Local: sets temperature: float for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    temperature: float = 0.7,
# Local: closes the surrounding literal or call expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
) -> str:
    # Local: starts a multi-line text literal that Python treats as one value. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    """
    Generates a single, greedy response from the model (standard Chain of Thought).
    Can also be used for sampling if do_sample is True.
    """
    # Local: sets inputs for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    
    # ensure pad_token_id is set to avoid warnings
    # Local: opens a condition that selects behavior from current state. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    if tokenizer.pad_token_id is None:
        # Local: sets tokenizer.pad_token_id for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Local: enters a managed runtime context. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    with torch.no_grad():
        # Local: asks the model to generate continuation tokens. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
        outputs = model.generate(
            # Local: adds an item or argument to the surrounding expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            **inputs,
            # Local: sets max_new_tokens for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            max_new_tokens=max_new_tokens,
            # Local: sets do_sample for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            do_sample=do_sample,
            # Local: sets temperature for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            temperature=temperature if do_sample else None,
            # Local: sets pad_token_id for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            pad_token_id=tokenizer.pad_token_id
        )

    # slice the output to get only the generated tokens (excluding prompt)
    # Local: sets generated_ids for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    # Local: returns the computed result to the caller. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# Local: defines the self_consistency_generation function. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
def self_consistency_generation(
    # Local: adds an item or argument to the surrounding expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    model: PreTrainedModel,
    # Local: adds an item or argument to the surrounding expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    tokenizer: PreTrainedTokenizer,
    # Local: adds an item or argument to the surrounding expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    input_text: str,
    # Local: sets k: int for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    k: int = 5,
    # Local: sets max_new_tokens: int for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    max_new_tokens: int = 1024,
    # Local: sets device: str for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    device: str = "cuda",
    # Local: sets data_name: str for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    data_name: str = "",
    # Local: sets prompt_idx: int for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    prompt_idx: int = 0,
    # Local: sets model_name: str for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    model_name: str = ""
# Local: closes the surrounding literal or call expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
) -> str:
    # Local: starts a multi-line text literal that Python treats as one value. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    """
    Generates k responses and returns the one with the majority vote answer.
    """
    # Local: sets generations for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    generations = []
    # Local: iterates through the current collection. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    for _ in range(k):
        # Local: sets gen for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
        gen = greedy_cot_generation(
            # Local: sets model for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            model=model,
            # Local: sets tokenizer for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            tokenizer=tokenizer,
            # Local: sets input_text for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            input_text=input_text,
            # Local: sets max_new_tokens for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            max_new_tokens=max_new_tokens,
            # Local: sets device for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            device=device,
            # Local: sets do_sample for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            do_sample=True,
            # Local: sets temperature for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            temperature=0.7
        )
        # Local: executes this statement in the current code path. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
        generations.append(gen)

    # Local: sets answers for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    answers = []
    # Local: iterates through the current collection. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    for gen in generations:
        # Local: sets ans for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
        ans = extract_answer(
            # Local: adds an item or argument to the surrounding expression. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            gen,
            # Local: sets data_name for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            data_name=data_name,
            # Local: sets prompt_idx for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            prompt_idx=prompt_idx,
            # Local: sets model_name for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            model_name=model_name
        )
        # Local: executes this statement in the current code path. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
        answers.append(str(ans) if ans is not None else None)

    # Local: opens a condition that selects behavior from current state. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    if not answers:
        # Local: returns the computed result to the caller. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
        return generations[0] if generations else ""

    # Local: sets majority_vote for later use in this scope. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    majority_vote = Counter(answers).most_common(1)[0][0]

    # Local: iterates through the current collection. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    for i, ans in enumerate(answers):
        # Local: opens a condition that selects behavior from current state. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
        if ans == majority_vote:
            # Local: returns the computed result to the caller. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
            return generations[i]
    
    # Local: returns the computed result to the caller. Global: connects ALS results to greedy CoT and self-consistency baselines for fair evaluation.
    return generations[0]