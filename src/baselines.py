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
) -> str:
    """
    Generates a single, greedy response from the model (standard Chain of Thought).
    """
    eos_token = tokenizer.eos_token
    if eos_token not in stop_words:
        stop_words.append(eos_token)

    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    generated_ids = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            new_token = tokenizer.decode(next_token_id.item())
            if new_token in stop_words:
                break
            
            generated_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

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
            device=device
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