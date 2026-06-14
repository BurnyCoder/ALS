# Local: file src/ori_generation.py provides first-party ALS source context. Global: runs unsteered generation while collecting hidden states for downstream steering.
import torch  # Local: imports torch for this module. Global: runs unsteered generation while collecting hidden states for downstream steering.
# Local: sets stop_words for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
# Local: defines the original_generation function. Global: runs unsteered generation while collecting hidden states for downstream steering.
def original_generation(input_text, model, tokenizer, device):
    # Local: starts a multi-line text literal that Python treats as one value. Global: runs unsteered generation while collecting hidden states for downstream steering.
    '''
    Generate answer using original generation process

    Args:
        input_text
        tokenizer
        device

    Returns:
        answer: original generated answer
        hidden_states_list: list of hidden states for each token
        answer_start_index: index of the hidden state where the answer begins
    '''
    # Local: sets eos_token for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
    eos_token = tokenizer.eos_token
    # Local: executes this statement in the current code path. Global: runs unsteered generation while collecting hidden states for downstream steering.
    stop_words.append(eos_token)
    # Local: sets inputs for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    # Local: sets new_input_ids for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
    new_input_ids = inputs.input_ids.clone()
    answer = []  # Local: sets answer for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
    # Local: sets hidden_states_list for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
    hidden_states_list = []
    
    cnt = 0  # Local: sets cnt for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
    # Local: continues a loop until its stopping condition is reached. Global: runs unsteered generation while collecting hidden states for downstream steering.
    while True:
        # Local: enters a managed runtime context. Global: runs unsteered generation while collecting hidden states for downstream steering.
        with torch.no_grad():
            # Local: runs the transformer backbone to expose hidden states. Global: runs unsteered generation while collecting hidden states for downstream steering.
            outputs = model.model(new_input_ids, output_hidden_states=True)
        # Local: sets hidden_states for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
        hidden_states = outputs[0][:, -1] # representations for last token on the last hidden layer
        # Local: executes this statement in the current code path. Global: runs unsteered generation while collecting hidden states for downstream steering.
        hidden_states_list.append(hidden_states.clone())
        # Local: sets hidden_states for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
        hidden_states = hidden_states.detach()
        # Local: sets hidden_states.requires_grad for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
        hidden_states.requires_grad = True
        
        # Local: opens a condition that selects behavior from current state. Global: runs unsteered generation while collecting hidden states for downstream steering.
        if hidden_states.grad is not None:
            # Local: executes this statement in the current code path. Global: runs unsteered generation while collecting hidden states for downstream steering.
            hidden_states.grad.zero_()
        
        # generate next token
        # Local: enters a managed runtime context. Global: runs unsteered generation while collecting hidden states for downstream steering.
        with torch.no_grad():
            # Local: projects hidden states into vocabulary logits. Global: runs unsteered generation while collecting hidden states for downstream steering.
            eval_logits = model.lm_head(hidden_states)
            # Local: selects the highest-scoring next token. Global: runs unsteered generation while collecting hidden states for downstream steering.
            next_token_id = torch.argmax(eval_logits, dim = -1) # [1]
            # Local: sets new_token for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
            new_token = tokenizer.decode(next_token_id.item())
            # Local: executes this statement in the current code path. Global: runs unsteered generation while collecting hidden states for downstream steering.
            answer.append(next_token_id.item())
            
            # Local: opens a condition that selects behavior from current state. Global: runs unsteered generation while collecting hidden states for downstream steering.
            if new_token in stop_words:
                # Local: exits the current loop once a stopping condition is met. Global: runs unsteered generation while collecting hidden states for downstream steering.
                break
            # Local: sets new_input_ids for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
            new_input_ids = torch.cat([new_input_ids, next_token_id.unsqueeze(0)], dim=-1)
        cnt += 1  # Local: updates cnt for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
        # Local: opens a condition that selects behavior from current state. Global: runs unsteered generation while collecting hidden states for downstream steering.
        if cnt > 1024:
            # Local: exits the current loop once a stopping condition is met. Global: runs unsteered generation while collecting hidden states for downstream steering.
            break
    # Local: sets answer for later use in this scope. Global: runs unsteered generation while collecting hidden states for downstream steering.
    answer = tokenizer.decode(answer)
    # Local: returns the computed result to the caller. Global: runs unsteered generation while collecting hidden states for downstream steering.
    return answer, hidden_states_list, new_input_ids

