# Local: file src/steered_generation.py provides first-party ALS source context. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
import torch  # Local: imports torch for this module. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
# Local: imports selected helpers from transformers. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
from transformers import PreTrainedModel, PreTrainedTokenizer
# Local: imports selected helpers from transformers.modeling_attn_mask_utils. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

# Local: sets stop_words for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

def steered_generation(  # Local: defines the steered_generation function. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    # Local: adds an item or argument to the surrounding expression. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    model: PreTrainedModel,
    # Local: adds an item or argument to the surrounding expression. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    tokenizer: PreTrainedTokenizer,
    # Local: adds an item or argument to the surrounding expression. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    input_text: str,
    # Local: adds an item or argument to the surrounding expression. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    steering_vector: torch.Tensor,
    # Local: sets alpha: float for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    alpha: float = 0.3,
    # Local: sets similarity_threshold: float for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    similarity_threshold: float = 0.1,
    # Local: sets max_new_tokens: int for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    max_new_tokens: int = 1024,
    # Local: sets device: str for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    device: str = "cuda"
) -> str:  # Local: closes the surrounding literal or call expression. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    # Local: starts a multi-line text literal that Python treats as one value. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    """
    Generates text using a steering vector applied at a specific layer with a manual generation loop.
    """
    # Local: sets eos_token for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    eos_token = tokenizer.eos_token
    # Local: opens a condition that selects behavior from current state. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    if eos_token not in stop_words:
        # Local: executes this statement in the current code path. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
        stop_words.append(eos_token)

    # Local: sets inputs for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    # Local: sets input_ids for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    input_ids = inputs.input_ids

    # Local: sets generated_ids for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    generated_ids = []
    # Local: normalizes the vector so steering strength is controlled by alpha. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    steering_vector = torch.nn.functional.normalize(steering_vector.to(device).squeeze(), dim=-1)

    # Local: iterates through the current collection. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    for _ in range(max_new_tokens):
        with torch.no_grad():  # Local: enters a managed runtime context. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            # Local: sets outputs for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            outputs = model(input_ids, output_hidden_states=True)
            
            # Local: sets next_token_logits for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            next_token_logits = outputs.logits[:, -1, :]
            
            # Local: sets all_hidden_states for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            all_hidden_states = outputs.hidden_states

            # Local: sets num_layers for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            num_layers = len(model.model.layers)
            # Local: sets target_layer_idx for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            target_layer_idx = num_layers - 1

            # Local: opens a condition that selects behavior from current state. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            if 0 <= target_layer_idx < len(all_hidden_states):
                # Local: sets target_hidden_state for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                target_hidden_state = all_hidden_states[target_layer_idx]
                # Local: sets last_token_hs for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                last_token_hs = target_hidden_state[:, -1, :]

                # Local: measures alignment between the current hidden state and steering vector. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                similarity = torch.nn.functional.cosine_similarity(last_token_hs.squeeze(), steering_vector, dim=-1)

                # Local: opens a condition that selects behavior from current state. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                if similarity < similarity_threshold:
                    # Local: sets nudge for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    nudge = (alpha * steering_vector).unsqueeze(0)

                    # Local: sets modified_hs for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    modified_hs = target_hidden_state.clone()
                    # Local: updates modified_hs[:, -1, :] for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    modified_hs[:, -1, :] += nudge

                    # args for the layer forward pass
                    # attention_mask needs to be updated for current input_ids
                    # Local: sets current_attention_mask for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    current_attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
                    # Local: sets seq_length for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    seq_length = input_ids.shape[1]
                    # Local: sets position_ids for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

                    # Local: sets causal_mask for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    causal_mask = _prepare_4d_causal_attention_mask(
                        # Local: executes this statement in the current code path. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                        current_attention_mask, (1, seq_length), modified_hs, 0
                    )

                    # pre-calculate position embeddings once
                    # Local: sets position_embeddings for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    position_embeddings = model.model.rotary_emb(all_hidden_states[0], position_ids)

                    # Local: sets temp_hs for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    temp_hs = modified_hs
                    # Local: iterates through the current collection. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    for i in range(target_layer_idx, num_layers):
                        # Local: sets layer_outputs for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                        layer_outputs = model.model.layers[i](
                            # Local: adds an item or argument to the surrounding expression. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                            temp_hs,
                            # Local: sets attention_mask for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                            attention_mask=causal_mask,
                            # Local: sets position_ids for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                            position_ids=position_ids,
                            # Local: sets position_embeddings for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                            position_embeddings=position_embeddings,
                        )
                        # Local: sets temp_hs for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                        temp_hs = layer_outputs[0]

                    # Local: sets temp_hs for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    temp_hs = model.model.norm(temp_hs)
                    # Ensure batch dimension is present
                    # Local: opens a condition that selects behavior from current state. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    if temp_hs.dim() == 2:
                        # Local: sets temp_hs for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                        temp_hs = temp_hs.unsqueeze(0) # Add batch_size dimension back
                    # Local: projects hidden states into vocabulary logits. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                    next_token_logits = model.lm_head(temp_hs)[:, -1, :]

            # Local: selects the highest-scoring next token. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Local: sets new_token for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            new_token = tokenizer.decode(next_token_id.item())
            # Local: opens a condition that selects behavior from current state. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            if new_token in stop_words:
                # Local: exits the current loop once a stopping condition is met. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
                break
            
            # Local: executes this statement in the current code path. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            generated_ids.append(next_token_id.item())
            # Local: sets input_ids for later use in this scope. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    # Local: returns the computed result to the caller. Global: implements the ALS online intervention loop with cosine-thresholded nudges.
    return tokenizer.decode(generated_ids, skip_special_tokens=True)