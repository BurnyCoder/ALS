# Local: file src/gated_generation.py provides first-party ALS source context. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
import torch  # Local: imports torch for this module. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
import json  # Local: imports json for this module. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
# Local: imports selected helpers from transformers. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
from transformers import PreTrainedModel, PreTrainedTokenizer
# Local: imports selected helpers from transformers.modeling_attn_mask_utils. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

# Local: sets stop_words for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

# Local: defines the gated_steered_generation function. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
def gated_steered_generation(
    # Local: adds an item or argument to the surrounding expression. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    model: PreTrainedModel,
    # Local: adds an item or argument to the surrounding expression. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    tokenizer: PreTrainedTokenizer,
    # Local: adds an item or argument to the surrounding expression. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    input_text: str,
    # Local: adds an item or argument to the surrounding expression. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    steering_vector: torch.Tensor,
    # Local: sets alpha: float for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    alpha: float = 0.3,
    # Local: sets similarity_threshold: float for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    similarity_threshold: float = 0.1,
    # Local: sets max_new_tokens: int for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    max_new_tokens: int = 1024,
    # Local: sets device: str for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    device: str = "cuda"
) -> str:  # Local: closes the surrounding literal or call expression. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    # Local: starts a multi-line text literal that Python treats as one value. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    """
    Generates text using a steering vector that is 'gated' off if JSON syntax is broken.
    """
    # Local: sets eos_token for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    eos_token = tokenizer.eos_token
    # Local: opens a condition that selects behavior from current state. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    if eos_token not in stop_words:
        # Local: executes this statement in the current code path. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
        stop_words.append(eos_token)

    # Local: sets inputs for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    # Local: sets input_ids for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    input_ids = inputs.input_ids

    # Local: sets generated_ids for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    generated_ids = []
    # Local: normalizes the vector so steering strength is controlled by alpha. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    steering_vector = torch.nn.functional.normalize(steering_vector.to(device).squeeze(), dim=-1)

    # Local: iterates through the current collection. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    for token_idx in range(max_new_tokens):
        with torch.no_grad():  # Local: enters a managed runtime context. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            # Local: sets outputs for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            outputs = model(input_ids, output_hidden_states=True)
            # Local: sets next_token_logits for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            next_token_logits = outputs.logits[:, -1, :]
            # Local: sets all_hidden_states for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            all_hidden_states = outputs.hidden_states

            # Local: sets num_layers for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            num_layers = len(model.model.layers)
            # Local: sets target_layer_idx for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            target_layer_idx = num_layers - 1

            # Local: opens a condition that selects behavior from current state. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            if 0 <= target_layer_idx < len(all_hidden_states):
                # Local: sets target_hidden_state for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                target_hidden_state = all_hidden_states[target_layer_idx]
                # Local: sets last_token_hs for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                last_token_hs = target_hidden_state[:, -1, :]
                # Local: sets current_alpha for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                current_alpha = alpha

                # gating logic for json
                # Local: opens a condition that selects behavior from current state. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                if token_idx > 5:
                    # Local: sets current_gen_text for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    current_gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    # Local: starts a protected operation that may fail on external or parsed input. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    try:
                        # Local: sets json_start for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                        json_start = current_gen_text.find('{')
                        # Local: opens a condition that selects behavior from current state. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                        if json_start != -1:
                            # Local: parses generated text as JSON when the prompt requires structure. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                            _ = json.loads(current_gen_text[json_start:] + '}')
                    # Local: handles a recoverable failure from the protected operation. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    except json.JSONDecodeError:
                        # Local: sets current_alpha for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                        current_alpha = 0.0

                # Local: measures alignment between the current hidden state and steering vector. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                similarity = torch.nn.functional.cosine_similarity(last_token_hs.squeeze(), steering_vector, dim=-1)

                # Local: opens a condition that selects behavior from current state. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                if similarity < similarity_threshold:
                    # Local: sets nudge for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    nudge = (current_alpha * steering_vector).unsqueeze(0)

                    # Local: sets modified_hs for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    modified_hs = target_hidden_state.clone()
                    # Local: updates modified_hs[:, -1, :] for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    modified_hs[:, -1, :] += nudge

                    # args for the layer forward pass
                    # attention_mask needs to be updated for current input_ids
                    # Local: sets current_attention_mask for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    current_attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
                    # Local: sets seq_length for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    seq_length = input_ids.shape[1]
                    # Local: sets position_ids for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

                    # Local: sets causal_mask for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    causal_mask = _prepare_4d_causal_attention_mask(
                        # Local: executes this statement in the current code path. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                        current_attention_mask, (1, seq_length), modified_hs, 0
                    )

                    # pre-calculate position embeddings once
                    # Local: sets position_embeddings for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    position_embeddings = model.model.rotary_emb(all_hidden_states[0], position_ids)

                    # Local: sets temp_hs for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    temp_hs = modified_hs
                    # Local: iterates through the current collection. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    for i in range(target_layer_idx, num_layers):
                        # Local: sets layer_outputs for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                        layer_outputs = model.model.layers[i](
                            # Local: adds an item or argument to the surrounding expression. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                            temp_hs,
                            # Local: sets attention_mask for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                            attention_mask=causal_mask,
                            # Local: sets position_ids for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                            position_ids=position_ids,
                            # Local: sets position_embeddings for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                            position_embeddings=position_embeddings,
                        )
                        # Local: sets temp_hs for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                        temp_hs = layer_outputs[0]

                    # Local: sets temp_hs for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    temp_hs = model.model.norm(temp_hs)
                    # Ensure batch dimension is present
                    # Local: opens a condition that selects behavior from current state. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    if temp_hs.dim() == 2:
                        # Local: sets temp_hs for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                        temp_hs = temp_hs.unsqueeze(0) # Add batch_size dimension back
                    # Local: projects hidden states into vocabulary logits. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                    next_token_logits = model.lm_head(temp_hs)[:, -1, :]

            # Local: selects the highest-scoring next token. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Local: sets new_token for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            new_token = tokenizer.decode(next_token_id.item())
            # Local: opens a condition that selects behavior from current state. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            if new_token in stop_words:
                # Local: exits the current loop once a stopping condition is met. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
                break
            
            # Local: executes this statement in the current code path. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            generated_ids.append(next_token_id.item())
            # Local: sets input_ids for later use in this scope. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    # Local: returns the computed result to the caller. Global: implements ALS-Gated, suppressing steering when JSON structure is at risk.
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
