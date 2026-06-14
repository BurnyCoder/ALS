"""LatentSeek-style online hidden-state optimization baseline.

This module implements the expensive comparison point for ALS. Instead of using
one offline vector, it turns a slice of one answer's hidden states into trainable
parameters, uses verifier reward to define a loss, updates those hidden states,
and then decodes a new answer for the same problem.
"""

# PyTorch supplies trainable parameters, Adam optimization, gradients, and tensor operations.
import torch

# These tokens end manual decoding for the instruction models used in the experiments.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]


# This function performs per-query latent optimization, which ALS is designed to amortize away.
def optimized_generation(
        # `reward_model` scores generated answers through verifier prompts.
        reward_model, model, tokenizer, device,
        # The question and formatted prompt give the optimizer both raw task text and model input.
        question, input_text, original_answer,
        # The original hidden-state trajectory and ids seed the latent optimization.
        original_hidden_states_list, input_ids, start_index=0,
        # These hyperparameters control optimization iterations, learning rate, and generation length.
        max_num_steps=10, lr=0.03, max_new_tokens=1024,
        # Gradient clipping, update fraction, and reward threshold control search stability/stopping.
        grad_clip=None, k=0.1, reward_threshold=-0.2):
    """Optimize a slice of hidden states and decode a revised answer.

    Locally, the function promotes selected hidden states to a `Parameter`,
    improves them with a reward-weighted log-probability objective, and decodes
    from the optimized prefix. Globally, this is the LatentSeek baseline whose
    repeated backward passes motivate ALS's offline steering vector.
    """
    # The tokenizer EOS token is used as the manual generation stopping condition.
    eos_token = tokenizer.eos_token
    # Adding EOS to the stop list keeps this loop aligned with the loaded tokenizer.
    stop_words.append(eos_token)
    # Reward history records the initial and later verifier scores for analysis.
    reward_history = []
    # The original answer is scored before any hidden-state update to establish the starting point.
    initial_reward = reward_model.get_reward(question, original_answer)

    # This log links the baseline answer to the verifier score driving optimization.
    print(f"-- Original Output: {original_answer} -- Initial Reward: {initial_reward}")
    # The initial score is the first point in the optimization trace.
    reward_history.append(initial_reward)
    # `current_reward` is updated after each newly decoded answer.
    current_reward = initial_reward

    # The number of original hidden states bounds how many positions can be optimized.
    original_length = len(original_hidden_states_list)
    # The optimized answer length is filled in once a final candidate is chosen.
    optimized_length = 0

    # Tokenizing the original prompt reconstructs the prompt-only prefix.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    # A clone is used so later slicing and concatenation do not alter tokenizer output.
    base_input_ids = inputs.input_ids.clone()

    # `k` chooses a fraction of generated hidden states to optimize, capped at 300 for cost control.
    update_length = min(int(k * original_length), 300)
    # If the selected fraction is empty, there is no latent sequence to optimize.
    if update_length <= 0:
        # The warning explains why optimization short-circuited.
        print("Update Length Zero!!!")
        # With no update positions, the original answer remains the final answer.
        final_answer = original_answer
        # The return tuple keeps the same shape as successful optimization runs.
        return final_answer, reward_history, original_length, optimized_length, update_length

    # The selected hidden states become a single trainable tensor.
    optimized_hidden_states = torch.nn.Parameter(torch.stack(
        # Each state is cloned, detached from old graphs, and marked gradient-capable.
        [state.clone().detach().requires_grad_(True)
        # The slice selects the contiguous latent segment that LatentSeek will update.
        for state in original_hidden_states_list[start_index: min(start_index + update_length, len(original_hidden_states_list))]])
    )

    # Adam updates the latent states directly while leaving model parameters fixed.
    optimizer = torch.optim.Adam([optimized_hidden_states], lr=lr)

    # `original_seq` stores generated tokens before the optimized segment.
    original_seq = []
    # Extending from the original ids preserves any prefix tokens before `start_index`.
    original_seq.extend(input_ids[0][len(base_input_ids[-1]): len(base_input_ids[-1]) + start_index])

    # The active context is truncated to the prompt plus the unoptimized prefix.
    input_ids = input_ids[:, : len(base_input_ids[-1]) + start_index]
    # This clone becomes the reset point for every optimization iteration.
    base_input_ids = input_ids.clone()
    # `new_answer` is filled after the first optimized decode.
    new_answer = None

    # The outer loop performs the expensive per-instance optimization steps.
    for _ in range(max_num_steps):
        # Each iteration starts generation from the same prompt-plus-prefix context.
        input_ids = base_input_ids.clone()
        # If verifier reward is already good enough, the search stops early.
        if current_reward > reward_threshold:
            # Use the latest optimized answer when available, otherwise keep the original.
            final_answer = new_answer if new_answer is not None else original_answer
            # Encoding measures the token length of the final answer for reporting.
            optimized_length = len(tokenizer.encode(final_answer))
            # This log records the answer that satisfied the reward threshold.
            print(f"-- Final Answer: {final_answer}, -- Current Reward: {current_reward}")
            # Returning early avoids further backward passes once reward is sufficient.
            return final_answer, reward_history, original_length, optimized_length, update_length

        # Clearing optimizer gradients prevents accumulation across optimization steps.
        optimizer.zero_grad()

        # Projecting latent states through `lm_head` gives token logits for the optimized segment.
        logits = model.lm_head(optimized_hidden_states)
        # Softmax converts logits into token probabilities, with epsilon avoiding log zero.
        probs = torch.softmax(logits, dim=-1) + 1e-8

        # Greedy ids represent the current token choice implied by each optimized latent state.
        next_token_ids = torch.argmax(probs, dim=-1)
        # Removing the singleton vocabulary-choice dimension makes ids indexable by time step.
        next_token_ids = next_token_ids.squeeze(-1)
        # Log probabilities of the chosen ids form the policy-gradient-style objective.
        log_pi_xz = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)

        # The loss pushes chosen-token likelihood according to the signed verifier reward.
        loss = - current_reward * log_pi_xz.sum()
        # Logging the scalar loss exposes optimization progress/debugging information.
        print(f"-- Loss: {loss.item()}")
        # Backpropagation computes gradients with respect to the hidden-state parameter tensor.
        loss.backward(retain_graph=True)

        # Optional gradient clipping limits hidden-state update magnitude.
        if grad_clip:
            # Clipping is applied to the optimized hidden-state parameter only.
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], grad_clip)
        # Adam applies one latent update step.
        optimizer.step()

        # `generated_seq` starts with any original generated prefix before the optimized slice.
        generated_seq = []
        # Extending copies prefix token ids into the candidate output sequence.
        generated_seq.extend(original_seq)
        # Decoding the optimized segment itself does not require gradients.
        with torch.no_grad():
            # The optimized hidden states are projected to their current greedy token ids.
            next_tokens = torch.argmax(model.lm_head(optimized_hidden_states),
                                       # Taking argmax over vocabulary selects one token per hidden state.
                                       dim=-1)
            # Squeezing removes the singleton dimension to create a flat token sequence.
            next_tokens = next_tokens.squeeze(-1)
            # The optimized segment token ids are appended after the preserved prefix.
            generated_seq.extend(next_tokens.tolist())
            # The optimized tokens are appended to context before normal autoregressive continuation.
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)], dim=-1)

        # The rest of the answer is generated normally from the optimized prefix.
        with torch.no_grad():
            # The continuation counter enforces `max_new_tokens`.
            cnt = 0
            # Continue until EOS or the explicit length cap.
            while True:
                # The model consumes prompt plus optimized fraction plus generated continuation.
                outputs = model.model(input_ids, output_hidden_states=True)
                # The last final-layer hidden state determines the next continuation token.
                hidden_states = outputs[0][:, -1]
                # The language-model head turns that hidden state into vocabulary logits.
                logits = model.lm_head(hidden_states)
                # Greedy decoding selects the next continuation token.
                next_token_id = torch.argmax(logits, dim=-1)
                # Decoding the token string lets the loop detect EOS.
                new_token = tokenizer.decode(next_token_id.item())
                # The continuation token is appended to the candidate output sequence.
                generated_seq.append(next_token_id.item())
                # The token is appended to context for the next autoregressive step.
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
                # The continuation counter advances after each generated token.
                cnt += 1
                # EOS ends continuation generation.
                if new_token == eos_token:
                    # Breaking exits the continuation loop.
                    break
                # The length cap prevents infinite generation.
                if cnt > max_new_tokens:
                    # Breaking returns the capped candidate answer.
                    break
        # Deleting large tensors releases references before the next optimization iteration.
        del outputs, hidden_states, next_token_id, new_token
        # Deleting logits, segment tokens, and context reduces peak memory across iterations.
        del logits, next_tokens, input_ids
        # Clearing CUDA cache helps long per-query optimization loops avoid memory buildup.
        torch.cuda.empty_cache()

        # Decoding the candidate token sequence creates the text that verifiers can score.
        new_answer = tokenizer.decode(generated_seq)
        # The verifier reward for the candidate drives the next latent update.
        current_reward = reward_model.get_reward(question, new_answer)
        # Logging ties each candidate answer to its new reward.
        print(f"-- New Answer: {new_answer}, -- Current Reward: {current_reward}")

        # The reward trace stores the candidate score after this iteration.
        reward_history.append(current_reward)

    # If the loop exhausts all steps, the latest decoded candidate becomes final.
    final_answer = new_answer
    # The optimized answer length is measured for reporting and trade-off analysis.
    optimized_length = len(tokenizer.encode(final_answer))
    # Final logging exposes the answer returned after max steps.
    print(f"-- Final answer: {final_answer}")
    # The return tuple gives answer quality and optimization-cost metadata to callers.
    return final_answer, reward_history, original_length, optimized_length, update_length
