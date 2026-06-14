"""Baseline generation methods used to compare against ALS.

The ALS experiments compare the fixed-vector steering path to standard greedy
Chain-of-Thought and Self-Consistency. These functions intentionally use
Hugging Face `generate` instead of hidden-state intervention so they represent
ordinary decoding baselines.
"""

# PyTorch disables gradients during baseline generation.
import torch
# Type hints document the Hugging Face model/tokenizer interfaces expected here.
from transformers import PreTrainedModel, PreTrainedTokenizer
# `Counter` implements majority voting over extracted answers for Self-Consistency.
from collections import Counter
# Answer extraction normalizes generations before the majority vote is computed.
from extract_judge_answer import extract_answer

# These stop words are kept for consistency with manual generation modules, although `generate` handles EOS.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


# This function provides the ordinary CoT decoding baseline.
def greedy_cot_generation(
    # The model is the same causal LM evaluated by ALS and LatentSeek.
    model: PreTrainedModel,
    # The tokenizer encodes the formatted prompt and decodes generated ids.
    tokenizer: PreTrainedTokenizer,
    # `input_text` already includes the selected solver prompt and chat template.
    input_text: str,
    # This cap bounds generation latency and output length across baselines.
    max_new_tokens: int = 1024,
    # The device places prompt tensors next to the model.
    device: str = "cuda",
    # `do_sample=False` makes this greedy CoT; `True` lets Self-Consistency draw samples.
    do_sample: bool = False,
    # Temperature controls diversity only when sampling is enabled.
    temperature: float = 0.7,
) -> str:
    """Generate one baseline response through Hugging Face `generate`.

    Locally, this is a thin wrapper around model-native decoding. Globally, it
    gives the experiments a non-steered baseline against which ALS speed and
    accuracy are compared.
    """
    # Tokenizing as a batch of one produces input ids and attention masks for `generate`.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)

    # Some tokenizers omit a padding id, and `generate` warns unless one is set.
    if tokenizer.pad_token_id is None:
        # Using EOS as padding is standard for decoder-only inference when no pad token exists.
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # No gradients are needed because baseline decoding does not optimize parameters or activations.
    with torch.no_grad():
        # `generate` performs the full autoregressive decoding loop inside transformers.
        outputs = model.generate(
            # Expanding `inputs` passes input ids and attention masks by name.
            **inputs,
            # This limits the number of tokens after the prompt.
            max_new_tokens=max_new_tokens,
            # Sampling is disabled for greedy CoT and enabled for Self-Consistency samples.
            do_sample=do_sample,
            # Temperature is meaningful only in sampling mode, so greedy mode passes `None`.
            temperature=temperature if do_sample else None,
            # The pad id prevents generation warnings for models without a dedicated pad token.
            pad_token_id=tokenizer.pad_token_id
        )

    # Slicing removes the prompt tokens so evaluation sees only the generated answer.
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    # Decoding converts token ids to text while removing tokenizer special markers.
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# This function implements the k-sample Self-Consistency baseline.
def self_consistency_generation(
    # The model is reused for all sampled reasoning paths.
    model: PreTrainedModel,
    # The tokenizer encodes the prompt and decodes each sampled answer.
    tokenizer: PreTrainedTokenizer,
    # `input_text` is the same formatted problem prompt used by other modes.
    input_text: str,
    # `k` controls how many independent sampled completions vote.
    k: int = 5,
    # The token cap is passed to each sampled generation.
    max_new_tokens: int = 1024,
    # The device places prompt tensors on the same hardware as the model.
    device: str = "cuda",
    # The dataset name selects answer-extraction rules for voting.
    data_name: str = "",
    # The prompt index tells extraction whether to expect boxed or JSON format.
    prompt_idx: int = 0,
    # The model name enables model-specific extraction fallbacks.
    model_name: str = ""
) -> str:
    """Sample `k` responses and return the generation with the majority answer.

    Locally, the function samples multiple CoT traces and votes over normalized
    final answers. Globally, it represents a token-space test-time-compute
    baseline that ALS aims to match or beat with lower online cost.
    """
    # This list stores the raw sampled generations so the winning text can be returned.
    generations = []
    # The loop draws `k` independently sampled completions.
    for _ in range(k):
        # Sampling is delegated to the same CoT wrapper with `do_sample=True`.
        gen = greedy_cot_generation(
            # The same model generates every sample.
            model=model,
            # The same tokenizer keeps formatting and decoding consistent.
            tokenizer=tokenizer,
            # Each sample starts from the identical prompt.
            input_text=input_text,
            # The per-sample length cap matches the requested evaluation limit.
            max_new_tokens=max_new_tokens,
            # The device is forwarded to tokenization and generation.
            device=device,
            # Enabling sampling creates diverse reasoning traces for voting.
            do_sample=True,
            # Temperature 0.7 matches the local self-consistency sampling setting.
            temperature=0.7
        )
        # The raw generation is saved for later return if its answer wins the vote.
        generations.append(gen)

    # This list stores normalized extracted answers parallel to `generations`.
    answers = []
    # Each generation is parsed into the answer format expected by the dataset/prompt pair.
    for gen in generations:
        # `extract_answer` removes CoT text, JSON wrappers, or boxed formatting as needed.
        ans = extract_answer(
            # The raw generation is the text to parse.
            gen,
            # Dataset-specific rules distinguish GSM8K numbers from MATH symbolic answers.
            data_name=data_name,
            # Prompt-specific rules distinguish boxed and JSON outputs.
            prompt_idx=prompt_idx,
            # Model-specific rules handle known extraction quirks.
            model_name=model_name
        )
        # Converting non-None answers to strings makes `Counter` grouping consistent.
        answers.append(str(ans) if ans is not None else None)

    # A defensive empty check handles cases where no generations were produced.
    if not answers:
        # Return the first raw generation if it exists, otherwise an empty answer.
        return generations[0] if generations else ""

    # `most_common(1)` returns the answer value with the largest vote count.
    majority_vote = Counter(answers).most_common(1)[0][0]

    # Returning the first generation with the winning answer preserves a complete CoT trace.
    for i, ans in enumerate(answers):
        # The first matching answer breaks ties by earliest sample.
        if ans == majority_vote:
            # The raw generation, not just the extracted answer, is returned for downstream evaluation.
            return generations[i]

    # This fallback is unreachable in normal cases but preserves a valid string return.
    return generations[0]
