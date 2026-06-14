"""Generate an unsteered greedy answer while recording hidden states.

This helper is used in two global roles: `collect_states.py` uses its hidden
states to build ALS good/bad latent pools, and `main.py` uses its answer plus
states as the starting point for LatentSeek optimization.
"""

# PyTorch supplies no-grad inference, argmax token selection, and tensor concatenation.
import torch

# These strings cover common end markers across the instruction-tuned models used in the paper.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


# This function performs plain greedy decoding and exposes the latent trajectory.
def original_generation(input_text, model, tokenizer, device):
    """Generate one answer and return token-level hidden states.

    Locally, the loop runs the base model one token at a time and stores the
    last hidden state before projecting to logits. Globally, those states are
    either averaged into the ALS steering-vector dataset or optimized by the
    LatentSeek baseline.
    """
    # The tokenizer's EOS token is the model-specific way to end generation.
    eos_token = tokenizer.eos_token
    # Appending EOS ensures this manual loop stops for the loaded tokenizer.
    stop_words.append(eos_token)
    # Tokenizing the prompt creates a batch of one autoregressive context.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    # A clone is extended with generated token ids without mutating `inputs`.
    new_input_ids = inputs.input_ids.clone()
    # Generated token ids are collected so the answer can be decoded at the end.
    answer = []
    # Hidden states are collected per generated token for ALS/LatentSeek downstream use.
    hidden_states_list = []

    # `cnt` provides a hard generation limit independent of EOS behavior.
    cnt = 0
    # The loop exits on stop token or after the maximum manual limit.
    while True:
        # The base forward pass does not need gradients during plain generation.
        with torch.no_grad():
            # Calling `model.model` returns decoder hidden states before the language-model head.
            outputs = model.model(new_input_ids, output_hidden_states=True)
        # `outputs[0][:, -1]` selects the final-token representation from the final hidden layer.
        hidden_states = outputs[0][:, -1]
        # A clone is stored so later tensor edits do not alter the collected trajectory.
        hidden_states_list.append(hidden_states.clone())
        # Detaching separates this hidden state from the no-grad forward graph.
        hidden_states = hidden_states.detach()
        # LatentSeek later expects optimizable hidden states, so this copy is marked gradient-capable.
        hidden_states.requires_grad = True

        # If a gradient tensor already exists, clearing it prevents stale gradients from leaking forward.
        if hidden_states.grad is not None:
            # Zeroing keeps this token state's gradient buffer clean.
            hidden_states.grad.zero_()

        # Token projection and selection are inference-only in the original generation path.
        with torch.no_grad():
            # The language-model head maps the hidden state to vocabulary logits.
            eval_logits = model.lm_head(hidden_states)
            # Greedy decoding selects the highest-logit token id.
            next_token_id = torch.argmax(eval_logits, dim = -1)
            # Decoding the single token lets the loop detect textual stop markers.
            new_token = tokenizer.decode(next_token_id.item())
            # The generated id is appended before stop checking, matching the original behavior.
            answer.append(next_token_id.item())

            # Stop markers end the answer-generation loop.
            if new_token in stop_words:
                # Breaking prevents the stop token from expanding the context further.
                break
            # The selected token becomes part of the context for the next autoregressive step.
            new_input_ids = torch.cat([new_input_ids, next_token_id.unsqueeze(0)], dim=-1)
        # The counter advances once per generated token attempt.
        cnt += 1
        # A hard cap prevents infinite loops when the model never emits a stop marker.
        if cnt > 1024:
            # Breaking returns the partial answer and collected states.
            break
    # Decoding all stored token ids produces the raw generated answer string.
    answer = tokenizer.decode(answer)
    # Returning final context ids lets LatentSeek know the prompt-plus-answer token sequence.
    return answer, hidden_states_list, new_input_ids
