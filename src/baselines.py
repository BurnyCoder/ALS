import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from collections import Counter
from extract_judge_answer import extract_answer

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

def greedy_cot_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    max_new_tokens: int = 1024,
    device: str = "cuda",
    do_sample: bool = False,
    temperature: float = 0.7,
) -> str:
    """
    Generates a single, greedy response from the model (standard Chain of Thought).
    Can also be used for sampling if do_sample is True.
    """
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    
    # ensure pad_token_id is set to avoid warnings
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id
        )

    # slice the output to get only the generated tokens (excluding prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def self_consistency_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    k: int = 5,
    max_new_tokens: int = 1024,
    device: str = "cuda",
    data_name: str = "",
    prompt_idx: int = 0,
    model_name: str = ""
) -> str:
    """
    Generates k responses and returns the one with the majority vote answer.
    """
    generations = []
    for _ in range(k):
        gen = greedy_cot_generation(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            device=device,
            do_sample=True,
            temperature=0.7
        )
        generations.append(gen)

    answers = []
    for gen in generations:
        ans = extract_answer(
            gen,
            data_name=data_name,
            prompt_idx=prompt_idx,
            model_name=model_name
        )
        answers.append(str(ans) if ans is not None else None)

    if not answers:
        return generations[0] if generations else ""

    majority_vote = Counter(answers).most_common(1)[0][0]

    for i, ans in enumerate(answers):
        if ans == majority_vote:
            return generations[i]
    
    return generations[0]