"""Run ALS decoding with a JSON-format safety gate.

ALS-Gated follows the same hidden-state steering loop as standard ALS, but it
sets the current steering strength to zero when partial JSON appears malformed.
That matches the paper's structured-output variant: preserve the rigid answer
schema while still using the offline success direction when format state allows.
"""

# PyTorch supplies tensors, cosine similarity, vector addition, and greedy argmax decoding.
import torch
# The standard JSON parser is used as a lightweight syntax gate for structured prompts.
import json
# Type hints document that the function expects Hugging Face model/tokenizer objects.
from transformers import PreTrainedModel, PreTrainedTokenizer
# Manual layer replay needs the same 4D causal mask shape used by transformer internals.
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

# These strings cover common end-of-generation markers for the target instruction models.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


# This function is selected by `main.py --generation_mode als_gated`.
def gated_steered_generation(
    # The model provides logits, hidden states, decoder layers, normalization, and the output head.
    model: PreTrainedModel,
    # The tokenizer encodes the prompt and decodes both partial JSON and final answers.
    tokenizer: PreTrainedTokenizer,
    # `input_text` is a chat-template prompt, usually asking for JSON output.
    input_text: str,
    # `steering_vector` is the offline ALS direction computed from labeled hidden states.
    steering_vector: torch.Tensor,
    # `alpha` is the normal intervention strength before JSON gating adjusts it per token.
    alpha: float = 0.3,
    # The similarity threshold decides when a token state should be nudged toward success.
    similarity_threshold: float = 0.1,
    # The token limit prevents runaway generation when the model never emits a stop token.
    max_new_tokens: int = 1024,
    # The device keeps prompt tensors, hidden states, and the vector on the same accelerator.
    device: str = "cuda"
) -> str:
    """Generate with ALS, suppressing the nudge when partial JSON is broken.

    Locally, the function decodes one greedy token at a time and optionally
    replays downstream transformer layers from a nudged hidden state. Globally,
    the JSON gate prevents latent interventions from worsening structured-output
    validity on prompt format P2.
    """
    # The tokenizer supplies the model-specific EOS marker.
    eos_token = tokenizer.eos_token
    # Adding the EOS token to the stop list lets this manual loop mimic `generate` termination.
    if eos_token not in stop_words:
        # The append is conditional so repeated calls do not duplicate the same stop string.
        stop_words.append(eos_token)

    # Tokenization creates a one-example batch for the formatted prompt.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    # `input_ids` is extended with each generated token so the model conditions on its own output.
    input_ids = inputs.input_ids

    # Generated ids are tracked separately so the returned text excludes the prompt.
    generated_ids = []
    # Normalizing the vector makes cosine similarity meaningful and alpha the nudge scale.
    steering_vector = torch.nn.functional.normalize(steering_vector.to(device).squeeze(), dim=-1)

    # The loop index is used by the JSON gate to wait until a partial object could plausibly exist.
    for token_idx in range(max_new_tokens):
        # No gradients are needed because ALS-Gated steers activations without updating parameters.
        with torch.no_grad():
            # The full forward pass produces default logits and all hidden states for the context.
            outputs = model(input_ids, output_hidden_states=True)
            # Default next-token logits are preserved unless a gated steering replay replaces them.
            next_token_logits = outputs.logits[:, -1, :]
            # The hidden-state tuple contains candidate intervention points.
            all_hidden_states = outputs.hidden_states

            # The layer count gives the index of the last decoder block.
            num_layers = len(model.model.layers)
            # ALS is applied near the output side by targeting the final decoder layer.
            target_layer_idx = num_layers - 1

            # The guard avoids intervention if the model's hidden-state tuple is shorter than expected.
            if 0 <= target_layer_idx < len(all_hidden_states):
                # The selected representation has shape `[batch, sequence, hidden]`.
                target_hidden_state = all_hidden_states[target_layer_idx]
                # Only the last token representation influences the next-token logits.
                last_token_hs = target_hidden_state[:, -1, :]
                # `current_alpha` starts as the requested ALS strength and may be zeroed by the gate.
                current_alpha = alpha

                # The gate waits a few tokens so a JSON prefix has time to appear.
                if token_idx > 5:
                    # Decoding the partial answer gives the parser the current structured-output prefix.
                    current_gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    # Invalid partial JSON raises `JSONDecodeError`, which disables steering for this token.
                    try:
                        # The first object brace is the likely start of the answer schema.
                        json_start = current_gen_text.find('{')
                        # If a brace exists, parse the partial object with a temporary closing brace.
                        if json_start != -1:
                            # Successful parsing leaves `current_alpha` unchanged.
                            _ = json.loads(current_gen_text[json_start:] + '}')
                    # Parse failure indicates the partial schema is fragile, so avoid adding a latent nudge.
                    except json.JSONDecodeError:
                        # Zero strength makes the later nudge mathematically no-op while preserving control flow.
                        current_alpha = 0.0

                # Cosine similarity checks whether the current hidden state is aligned with the success direction.
                similarity = torch.nn.functional.cosine_similarity(last_token_hs.squeeze(), steering_vector, dim=-1)

                # Steering is considered only when alignment is below the threshold.
                if similarity < similarity_threshold:
                    # `current_alpha` is either the requested strength or zero from the JSON gate.
                    nudge = (current_alpha * steering_vector).unsqueeze(0)

                    # Cloning avoids changing the original hidden-state object returned by the model.
                    modified_hs = target_hidden_state.clone()
                    # Adding to the last token implements h'_t = h_t + alpha * v for the next-token decision.
                    modified_hs[:, -1, :] += nudge

                    # A fresh all-ones mask marks every current context token as valid.
                    current_attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
                    # The current sequence length determines attention and position tensor sizes.
                    seq_length = input_ids.shape[1]
                    # Position ids enumerate prompt and generated tokens from zero to length minus one.
                    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

                    # The 4D causal mask preserves autoregressive visibility during manual layer replay.
                    causal_mask = _prepare_4d_causal_attention_mask(
                        # The 2D attention mask identifies real tokens.
                        current_attention_mask, (1, seq_length), modified_hs, 0
                    )

                    # Rotary embeddings are recomputed for the full current context before replay.
                    position_embeddings = model.model.rotary_emb(all_hidden_states[0], position_ids)

                    # Replay starts from the nudged target-layer hidden state.
                    temp_hs = modified_hs
                    # The loop propagates the intervention through every remaining decoder layer.
                    for i in range(target_layer_idx, num_layers):
                        # Calling the layer manually mirrors the model forward path after the intervention point.
                        layer_outputs = model.model.layers[i](
                            # `temp_hs` carries the current replayed representation.
                            temp_hs,
                            # The causal mask prevents attention to future positions.
                            attention_mask=causal_mask,
                            # Position ids align token indices with the model's positional machinery.
                            position_ids=position_ids,
                            # Position embeddings provide rotary phases expected by the decoder layer.
                            position_embeddings=position_embeddings,
                        )
                        # The first tuple item is the hidden state for the next replay step.
                        temp_hs = layer_outputs[0]

                    # Final normalization prepares the replayed hidden state for projection into vocabulary logits.
                    temp_hs = model.model.norm(temp_hs)
                    # Some transformer paths may return `[seq, hidden]`; this restores `[batch, seq, hidden]`.
                    if temp_hs.dim() == 2:
                        # Adding axis zero restores the batch dimension for `lm_head`.
                        temp_hs = temp_hs.unsqueeze(0)
                    # These logits replace the default logits with logits derived from the gated ALS state.
                    next_token_logits = model.lm_head(temp_hs)[:, -1, :]

            # Greedy decoding picks the highest-probability token from the active logits.
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # The token string is needed to detect EOS and chat end markers.
            new_token = tokenizer.decode(next_token_id.item())
            # Stop markers terminate generation without returning the marker itself.
            if new_token in stop_words:
                # Breaking exits the manual decode loop cleanly.
                break

            # Store the token id for final answer decoding and partial JSON parsing.
            generated_ids.append(next_token_id.item())
            # Append the token to context so the next step conditions on everything generated so far.
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    # Decode only generated ids so the caller receives the answer rather than prompt plus answer.
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
