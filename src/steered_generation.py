"""Run standard ALS decoding with a fixed offline steering vector.

The ALS paper replaces LatentSeek's per-example backward-pass loop with a
constant-cost token loop: compare the current hidden state with the saved
success direction, add `alpha * v` only when similarity falls below a threshold,
and decode from the adjusted representation.
"""

# PyTorch supplies tensor operations for cosine similarity, vector addition, and greedy logits.
import torch
# Type hints document that this function expects Hugging Face causal-LM interfaces.
from transformers import PreTrainedModel, PreTrainedTokenizer
# This helper reconstructs the causal attention mask needed when replaying transformer layers manually.
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

# These strings cover common end-of-generation tokens used by Llama/Qwen-style chat models.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


# This function is the online ALS inference path used by `main.py --generation_mode als`.
def steered_generation(
    # The model supplies hidden states, transformer layers, normalization, and `lm_head` logits.
    model: PreTrainedModel,
    # The tokenizer formats the prompt and decodes generated token ids.
    tokenizer: PreTrainedTokenizer,
    # `input_text` is already chat-template formatted by `data.get_dataset`.
    input_text: str,
    # `steering_vector` is the offline `E[h_correct] - E[h_incorrect]` tensor.
    steering_vector: torch.Tensor,
    # `alpha` scales the additive intervention described as h'_t = h_t + alpha * v.
    alpha: float = 0.3,
    # The threshold decides when the current hidden state has drifted far enough to steer.
    similarity_threshold: float = 0.1,
    # The loop stops after this many generated tokens even if no EOS token appears.
    max_new_tokens: int = 1024,
    # The device identifies where prompt tensors and steering tensors should live.
    device: str = "cuda"
) -> str:
    """Generate a greedy response while applying ALS hidden-state nudges.

    Locally, the code performs a manual one-token-at-a-time decode. Globally,
    each iteration implements the paper's online intervention: monitor cosine
    similarity to the success direction and replay the downstream network from
    a nudged hidden state when the token is off-target.
    """
    # The tokenizer's EOS token is the model-specific stop marker.
    eos_token = tokenizer.eos_token
    # The global stop list is extended once per tokenizer so decoding stops on model-native EOS.
    if eos_token not in stop_words:
        # Appending keeps the shared stop list aware of this tokenizer's EOS string.
        stop_words.append(eos_token)

    # Tokenizing as a one-item batch creates the model input tensor for the prompt.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    # `input_ids` is extended after every generated token to maintain autoregressive context.
    input_ids = inputs.input_ids

    # Generated token ids are stored separately so the return value excludes the prompt.
    generated_ids = []
    # Normalizing `v` makes alpha control magnitude while cosine similarity uses a unit direction.
    steering_vector = torch.nn.functional.normalize(steering_vector.to(device).squeeze(), dim=-1)

    # The outer loop decodes at most `max_new_tokens` one token at a time.
    for _ in range(max_new_tokens):
        # ALS inference does not update model parameters, so gradients are disabled.
        with torch.no_grad():
            # The forward pass returns both logits and all hidden states for the current context.
            outputs = model(input_ids, output_hidden_states=True)

            # Default logits come from the unmodified model and are used if no steering happens.
            next_token_logits = outputs.logits[:, -1, :]

            # Hidden states expose the internal representation where the ALS vector is applied.
            all_hidden_states = outputs.hidden_states

            # The number of decoder layers determines the final layer index available for replay.
            num_layers = len(model.model.layers)
            # The implementation targets the last decoder layer, matching the paper's late-layer steering intent.
            target_layer_idx = num_layers - 1

            # This guard prevents invalid indexing if a model exposes an unexpected hidden-state layout.
            if 0 <= target_layer_idx < len(all_hidden_states):
                # The target hidden-state tensor has shape `[batch, sequence, hidden]`.
                target_hidden_state = all_hidden_states[target_layer_idx]
                # The last token is the only position used to choose the next generated token.
                last_token_hs = target_hidden_state[:, -1, :]

                # Cosine similarity measures alignment between the current token state and the success direction.
                similarity = torch.nn.functional.cosine_similarity(last_token_hs.squeeze(), steering_vector, dim=-1)

                # Steering is applied only when the current state falls below the chosen alignment threshold.
                if similarity < similarity_threshold:
                    # Scaling the unit vector creates the additive nudge used by the ALS update equation.
                    nudge = (alpha * steering_vector).unsqueeze(0)

                    # Cloning avoids mutating the model's original hidden-state tuple in place.
                    modified_hs = target_hidden_state.clone()
                    # Adding the nudge only to the last token changes the next-token decision while preserving prior context.
                    modified_hs[:, -1, :] += nudge

                    # A full attention mask is rebuilt for the current autoregressive context length.
                    current_attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
                    # `seq_length` drives mask and position-id shapes for replaying decoder layers.
                    seq_length = input_ids.shape[1]
                    # Position ids enumerate every token position in the current prompt-plus-generation context.
                    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

                    # The 4D causal mask lets replayed layers attend only to previous/current positions.
                    causal_mask = _prepare_4d_causal_attention_mask(
                        # The 2D mask says every token in `input_ids` is real, not padding.
                        current_attention_mask, (1, seq_length), modified_hs, 0
                    )

                    # Rotary position embeddings are recomputed once and reused during downstream layer replay.
                    position_embeddings = model.model.rotary_emb(all_hidden_states[0], position_ids)

                    # Replay begins from the nudged hidden state at the target layer.
                    temp_hs = modified_hs
                    # Each remaining layer transforms the nudged representation toward final logits.
                    for i in range(target_layer_idx, num_layers):
                        # The layer call mirrors the model forward path with the rebuilt masks/positions.
                        layer_outputs = model.model.layers[i](
                            # `temp_hs` is the current representation being propagated.
                            temp_hs,
                            # The causal mask preserves autoregressive attention semantics.
                            attention_mask=causal_mask,
                            # Position ids keep attention and rotary phases aligned with the context.
                            position_ids=position_ids,
                            # Position embeddings supply the rotary representation expected by newer transformers.
                            position_embeddings=position_embeddings,
                        )
                        # Transformer layers return tuples, with the next hidden state in position zero.
                        temp_hs = layer_outputs[0]

                    # Final model normalization converts replayed hidden states to the representation expected by `lm_head`.
                    temp_hs = model.model.norm(temp_hs)
                    # Some model internals can drop the batch dimension, so this guard restores `[batch, seq, hidden]`.
                    if temp_hs.dim() == 2:
                        # `unsqueeze(0)` adds the missing batch dimension back before projection.
                        temp_hs = temp_hs.unsqueeze(0)
                    # The language-model head converts the replayed final hidden state into next-token logits.
                    next_token_logits = model.lm_head(temp_hs)[:, -1, :]

            # Greedy decoding chooses the highest-logit token from steered or unsteered logits.
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Decoding this one token lets the loop detect model-specific stop strings.
            new_token = tokenizer.decode(next_token_id.item())
            # Stop tokens end generation without adding the EOS marker to the returned answer.
            if new_token in stop_words:
                # Breaking exits the token loop and returns the decoded answer so far.
                break

            # The raw id is retained so the final decode can skip special tokens consistently.
            generated_ids.append(next_token_id.item())
            # Appending the generated token to context makes the next iteration autoregressive.
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    # Decoding only generated ids returns the model answer without the original prompt.
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
