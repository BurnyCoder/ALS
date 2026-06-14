# Local: file src/opt_generation.py provides first-party ALS source context. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
import torch  # Local: imports torch for this module. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
# Local: sets stop_words for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
# Local: defines the optimized_generation function. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
def optimized_generation(
        # Local: adds an item or argument to the surrounding expression. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        reward_model, model, tokenizer, device,
        # Local: adds an item or argument to the surrounding expression. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        question, input_text, original_answer, 
        # Local: sets original_hidden_states_list, input_ids, start_index for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        original_hidden_states_list, input_ids, start_index=0, 
        # Local: sets max_num_steps for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        max_num_steps=10, lr=0.03, max_new_tokens=1024,
        # Local: sets grad_clip for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        grad_clip=None, k=0.1, reward_threshold=-0.2):
    # Local: starts a multi-line text literal that Python treats as one value. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    '''
    Generate answer using optimized generation process

    Args:
        reward_model: reward model
        model: language model
        tokenizer: tokenizer
        device: device to use
        question: question
        input_text: formatted prompt
        original_answer: original generated answer
        original_hidden_states_list: list of hidden states for each token
        input_ids: the input_ids of original generation
        start_index: the start index of the optimized hidden states
        max_num_steps: number of optimization steps
        lr: learning rate
        max_new_tokens: maximum number of new tokens to generate
        grad_clip: gradient clipping threshold
        k: ratio of update length to the total length of hidden states
        reward_threshold: threshold for the reward to stop optimization
        
    Returns:
        final_answer: the final generated answer
        reward_history: list of rewards during optimization
        original_length: length of the original answer
        optimized_length: length of the optimized answer
        update_length: length of the optimized hidden states
    '''
    # Local: sets eos_token for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    eos_token = tokenizer.eos_token
    # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    stop_words.append(eos_token)
    # Local: sets reward_history for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    reward_history = []
    # Local: sets initial_reward for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    initial_reward = reward_model.get_reward(question, original_answer)
    
    # Local: reports progress or diagnostics to the run log. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    print(f"-- Original Output: {original_answer} -- Initial Reward: {initial_reward}")
    # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    reward_history.append(initial_reward)
    # Local: sets current_reward for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    current_reward = initial_reward
    
    # Local: sets original_length for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    original_length = len(original_hidden_states_list)
    # Local: sets optimized_length for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    optimized_length = 0
    
    # Local: sets inputs for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    # Local: sets base_input_ids for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    base_input_ids = inputs.input_ids.clone()
    
    # grab update fraction
    # Local: sets update_length for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    update_length = min(int(k * original_length), 300)
    # Local: opens a condition that selects behavior from current state. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    if update_length <= 0:
        # Local: reports progress or diagnostics to the run log. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        print("Update Length Zero!!!")
        # Local: sets final_answer for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        final_answer = original_answer
        # Local: returns the computed result to the caller. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        return final_answer, reward_history, original_length, optimized_length, update_length

    # Local: sets optimized_hidden_states for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    optimized_hidden_states = torch.nn.Parameter(torch.stack(
        # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        [state.clone().detach().requires_grad_(True)
        # Local: iterates through the current collection. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        for state in original_hidden_states_list[start_index: min(start_index + update_length, len(original_hidden_states_list))]])
    )
    
    # configure optimizer
    # Local: sets optimizer for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    optimizer = torch.optim.Adam([optimized_hidden_states], lr=lr)
    
    # Local: sets original_seq for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    original_seq = []
    # the prompt
    # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    original_seq.extend(input_ids[0][len(base_input_ids[-1]): len(base_input_ids[-1]) + start_index])
    
    # Local: sets input_ids for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    input_ids = input_ids[:, : len(base_input_ids[-1]) + start_index]
    # Local: sets base_input_ids for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    base_input_ids = input_ids.clone()
    # Local: sets new_answer for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    new_answer = None
    
    # optimization loop
    # Local: iterates through the current collection. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    for _ in range(max_num_steps):
        # Local: sets input_ids for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        input_ids = base_input_ids.clone()
        # Local: opens a condition that selects behavior from current state. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        if current_reward > reward_threshold:
            # Local: sets final_answer for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            final_answer = new_answer if new_answer is not None else original_answer
            # Local: sets optimized_length for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            optimized_length = len(tokenizer.encode(final_answer))
            # Local: reports progress or diagnostics to the run log. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            print(f"-- Final Answer: {final_answer}, -- Current Reward: {current_reward}")
            # Local: returns the computed result to the caller. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            return final_answer, reward_history, original_length, optimized_length, update_length
        
        # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        optimizer.zero_grad()
        
        # Local: projects hidden states into vocabulary logits. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        logits = model.lm_head(optimized_hidden_states) #[update_length, 1, vocab_size]
        # Local: sets probs for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        probs = torch.softmax(logits, dim=-1) + 1e-8    #[update_length, 1, vocab_size]
        
        # Local: selects the highest-scoring next token. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        next_token_ids = torch.argmax(probs, dim=-1)    #[update_length, 1]
        # Local: sets next_token_ids for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        next_token_ids = next_token_ids.squeeze(-1)    #[update_length]
        # Local: sets log_pi_xz for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        log_pi_xz = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)
        
        # total loss
        # Local: sets loss for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        loss = - current_reward * log_pi_xz.sum()
        # Local: reports progress or diagnostics to the run log. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        print(f"-- Loss: {loss.item()}")
        # Local: sets loss.backward(retain_graph for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        loss.backward(retain_graph=True)
        
        # Local: opens a condition that selects behavior from current state. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        if grad_clip:
            # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], grad_clip)
        # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        optimizer.step()
        
        # update hidden states
        # Local: sets generated_seq for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        generated_seq = []
        # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        generated_seq.extend(original_seq)
        # Local: enters a managed runtime context. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        with torch.no_grad():
            # Local: projects hidden states into vocabulary logits. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            next_tokens = torch.argmax(model.lm_head(optimized_hidden_states), 
                                       # Local: sets dim for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                                       dim=-1) #[update_length, 1]
            # Local: sets next_tokens for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            next_tokens = next_tokens.squeeze(-1) #[update_length]
            # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            generated_seq.extend(next_tokens.tolist())
            # Local: sets input_ids for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)], dim=-1)
                
        # generate full answer
        # Local: enters a managed runtime context. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        with torch.no_grad():
            cnt = 0  # Local: sets cnt for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            # Local: continues a loop until its stopping condition is reached. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
            while True:
                # prompt + update fraction -> full model -> outputs
                # Local: runs the transformer backbone to expose hidden states. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                outputs = model.model(input_ids, output_hidden_states=True)
                # Local: sets hidden_states for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                hidden_states = outputs[0][:, -1]
                # Local: projects hidden states into vocabulary logits. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                logits = model.lm_head(hidden_states)
                # Local: selects the highest-scoring next token. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                next_token_id = torch.argmax(logits, dim=-1)
                # Local: sets new_token for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                new_token = tokenizer.decode(next_token_id.item())
                # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                generated_seq.append(next_token_id.item())
                # Local: sets input_ids for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
                # Local: updates cnt for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                cnt += 1
                # Local: opens a condition that selects behavior from current state. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                if new_token == eos_token:
                    # Local: exits the current loop once a stopping condition is met. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                    break
                # Local: opens a condition that selects behavior from current state. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                if cnt > max_new_tokens:
                    # Local: exits the current loop once a stopping condition is met. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
                    break
        # Local: releases local references so memory can be reclaimed. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        del outputs, hidden_states, next_token_id, new_token
        # Local: releases local references so memory can be reclaimed. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        del logits, next_tokens, input_ids
        # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        torch.cuda.empty_cache()

        # Local: sets new_answer for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        new_answer = tokenizer.decode(generated_seq)
        # Local: sets current_reward for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        current_reward = reward_model.get_reward(question, new_answer)
        # Local: reports progress or diagnostics to the run log. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        print(f"-- New Answer: {new_answer}, -- Current Reward: {current_reward}")
            
        # Local: executes this statement in the current code path. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
        reward_history.append(current_reward)
        
    # Local: sets final_answer for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    final_answer = new_answer
    # Local: sets optimized_length for later use in this scope. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    optimized_length = len(tokenizer.encode(final_answer))
    # Local: reports progress or diagnostics to the run log. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    print(f"-- Final answer: {final_answer}")
    # Local: returns the computed result to the caller. Global: implements the LatentSeek-style online optimization baseline ALS amortizes away.
    return final_answer, reward_history, original_length, optimized_length, update_length

