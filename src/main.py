# Local: file src/main.py provides first-party ALS source context. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
# Local: imports selected helpers from transformers. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  # Local: imports torch for this module. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from data import get_dataset  # Local: imports selected helpers from data. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from tqdm import tqdm  # Local: imports selected helpers from tqdm. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
# Local: imports selected helpers from rewards.reward. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from rewards.reward import RewardModel
# Local: imports selected helpers from ori_generation. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from ori_generation import original_generation
# Local: imports selected helpers from opt_generation. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from opt_generation import optimized_generation
# Local: imports selected helpers from steered_generation. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from steered_generation import steered_generation
# Local: imports selected helpers from gated_generation. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from gated_generation import gated_steered_generation
# Local: imports selected helpers from baselines. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from baselines import greedy_cot_generation, self_consistency_generation
import os  # Local: imports os for this module. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
# Local: imports selected helpers from extract_judge_answer. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
from extract_judge_answer import extract_answer, extract_true_answer, judge_answer
import argparse  # Local: imports argparse for this module. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
import numpy as np  # Local: imports numpy as np for this module. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
import random  # Local: imports random for this module. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
import time  # Local: imports time for this module. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.


def parse_args():  # Local: defines the parse_args function. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    # Local: sets parser for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser = argparse.ArgumentParser(description="Evaluate the model")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., 'openai/gsm8k', 'MATH-500')")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--prompts_to_run", type=str, default="all", help="Which prompts to run, e.g., '0', '1', or 'all'")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--end_data_idx", type=int, default=None, help="End index of the data to evaluate (exclusive)")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--fixed_subset_path", type=str, default=None, help="Path to a file with a fixed list of indices for evaluation.")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose print statements")

    # seed
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # generation mode
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--generation_mode", type=str, default="latentseek", choices=["latentseek", "als", "greedy_cot", "self_consistency", "als_gated"], help="Generation mode to use")

    # latentseek args
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--max_num_steps", type=int, default=10, help="Number of optimization iterations")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Number of generated tokens")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--reward_threshold", type=float, default=-0.2, help="Threshold for reward to stop optimization")

    # steered generation args
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--vector_name_template", type=str, default="./vectors/{model_name}_{dataset_name}_p{prompt_idx}.pt", help="Template for steering vector filenames.")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--alpha", type=float, default=0.3, help="Strength of the steering intervention")
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--similarity_threshold", type=float, default=0.1, help="Cosine similarity threshold to trigger steering")

    # self-consistency args
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--sc_k", type=int, default=5, help="Number of samples for self-consistency")

    # device
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--device", type=str, default=None)

    # format reward
    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--rule_format_string", type=str, default=None, help="the answer format that should follow")

    # Local: registers a command-line option for the script. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    return parser.parse_args()  # Local: returns the computed result to the caller. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.


def set_seed(seed):  # Local: defines the set_seed function. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    # Local: starts a multi-line text literal that Python treats as one value. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)  # Local: seeds PyTorch randomness for reproducibility. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    torch.cuda.manual_seed_all(seed)  # Local: seeds CUDA randomness for reproducibility. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    # Local: sets torch.backends.cudnn.deterministic for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    torch.backends.cudnn.deterministic = True
    # Local: sets torch.backends.cudnn.benchmark for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    torch.backends.cudnn.benchmark = False
    # Local: sets os.environ['PYTHONHASHSEED'] for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Local: seeds NumPy randomness for reproducibility. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    random.seed(seed)  # Local: seeds Python randomness for reproducibility. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.


# evaluate function 
def main(args):  # Local: defines the main function. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    # Local: starts a multi-line text literal that Python treats as one value. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    """
    Evaluate model
    """
    # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    if args.rule_format_string == "boxed":
        # Local: sets rule_format_string for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        rule_format_string = r'\\boxed{(.*)}'
    else:  # Local: handles the remaining branch after earlier checks fail. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        if args.rule_format_string:
            # Local: stops execution with an explicit error for invalid state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            raise ValueError("Unknown format")
        # Local: sets rule_format_string for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        rule_format_string = None
    
    if args.seed:  # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        set_seed(args.seed)  # Local: executes this statement in the current code path. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    
    # set device
    # Local: sets device for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    # Local: loads the causal language model weights. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    model = AutoModelForCausalLM.from_pretrained(
            # Local: adds an item or argument to the surrounding expression. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            args.model_name_or_path,
            # Local: sets torch_dtype for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            torch_dtype=torch.bfloat16,
            device_map=device  # Local: sets device_map for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    )
    model.eval()  # Local: executes this statement in the current code path. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    # Local: loads the matching tokenizer. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load reward model if in latentseek mode
    reward_model = None  # Local: sets reward_model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    if args.generation_mode == "latentseek":
        reward_model = RewardModel(  # Local: sets reward_model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                model=model,   # Local: sets model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                tokenizer=tokenizer,   # Local: sets tokenizer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                device=device,  # Local: sets device for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                # Local: sets data_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                data_name=args.dataset,
                # Local: sets rule_format_string for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                rule_format_string=rule_format_string
                )

    # --- main loop for prompts ---
    num_prompts = 2  # Local: sets num_prompts for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    if args.prompts_to_run == 'all':
        # Local: sets prompts_to_run for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        prompts_to_run = range(num_prompts)
    else:  # Local: handles the remaining branch after earlier checks fail. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        # Local: sets prompts_to_run for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        prompts_to_run = [int(p.strip()) for p in args.prompts_to_run.split(',')]

    final_results = {}  # Local: sets final_results for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.

    for prompt_idx in prompts_to_run:  # Local: iterates through the current collection. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        print(f"\n{'='*20} EVALUATING PROMPT {prompt_idx+1} {'='*20}")

        # Local: sets model_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        model_name = args.model_name_or_path.split("/")[-1]
        # Local: sets data_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        data_name = args.dataset.replace("/", "-")

        # for steered modes, check for vector and skip prompt if not found
        # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        if args.generation_mode in ["als", "als_gated"]:
            # Local: sets vector_path for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            vector_path = args.vector_name_template.format(
                # Local: sets model_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                model_name=model_name, 
                # Local: sets dataset_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                dataset_name=data_name, 
                prompt_idx=prompt_idx  # Local: sets prompt_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            )
            # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            if not os.path.exists(vector_path):
                # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                print(f"\nWarning: Could not find steering vector for prompt {prompt_idx} at {vector_path}. Skipping.")
                # Local: skips the rest of this iteration and moves to the next item. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                continue
            # Local: sets steering_vector for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            steering_vector = torch.load(vector_path)

        # load dataset for the specific prompt
        # Local: sets dataset for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        dataset = get_dataset(args.dataset, 
                              # Local: sets tokenizer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                              tokenizer=tokenizer,
                              # Local: sets prompt_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                              prompt_idx=prompt_idx,
                              # Local: sets split for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                              split=args.split)
        
        # filter dataset if a fixed subset is provided
        # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        if args.fixed_subset_path:
            # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            if not os.path.exists(args.fixed_subset_path):
                # Local: stops execution with an explicit error for invalid state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                raise FileNotFoundError(f"Fixed subset file not found at {args.fixed_subset_path}")
            # Local: enters a managed runtime context. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            with open(args.fixed_subset_path, 'r') as f:
                # Local: sets indices for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                indices = [int(line.strip()) for line in f]
            # Local: sets dataset for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            dataset = dataset.select(indices)
            # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            print(f"Evaluating on a fixed subset of {len(dataset)} samples from {args.fixed_subset_path}")

        # for logging
        results_history = []  # Local: sets results_history for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        time_history = []  # Local: sets time_history for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        
        # Local: sets base_output_dir for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        base_output_dir = f"{args.output_dir}/{model_name}-{data_name}"
        # Local: sets output_dir for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        output_dir = f"{base_output_dir}/{args.generation_mode}_eval/prompt{prompt_idx}"
        # Local: ensures the output directory exists. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        os.makedirs(output_dir, exist_ok=True)

        start_data_idx = 0  # Local: sets start_data_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        
        # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        if args.resume:
            # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            print(f"Resume from {output_dir}")
            # Local: sets logistics_path for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            logistics_path = f"{output_dir}/logistics.pt"
            # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            if os.path.exists(logistics_path):
                # Local: sets logistics for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                logistics = torch.load(logistics_path)
                # Local: sets results_history for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                results_history = logistics.get("results_history", [])
                # Local: sets time_history for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                time_history = logistics.get("time_history", [])
                # Local: sets start_data_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                start_data_idx = len(results_history)

        # if using a fixed subset, ignore start/end data_idx from args
        # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        if args.fixed_subset_path:
            start_data_idx = 0  # Local: sets start_data_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            # Local: sets end_data_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            end_data_idx = len(dataset)
        else:  # Local: handles the remaining branch after earlier checks fail. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            # Local: sets start_data_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            start_data_idx = max(start_data_idx, args.start_data_idx)
            # Local: sets end_data_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            end_data_idx = args.end_data_idx if args.end_data_idx is not None else len(dataset)
            # Local: sets end_data_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            end_data_idx = min(end_data_idx, len(dataset))
        
        # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        print(f"Start to evaluate {args.dataset} (Prompt {prompt_idx+1}) from {start_data_idx} to {end_data_idx}...")

        # Local: sets data_idx_list for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        data_idx_list = range(start_data_idx, end_data_idx)
        # Local: iterates through the current collection. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        for i in tqdm(data_idx_list, desc=f"Prompt {prompt_idx+1}"):
            example = dataset[i]  # Local: sets example for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            # Local: sets true_answer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            true_answer = extract_true_answer(example["answer"], name=args.dataset)

            # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            if true_answer is None:
                # Local: skips the rest of this iteration and moves to the next item. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                continue

            start_time = time.time()  # Local: sets start_time for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.

            # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            if args.generation_mode == "latentseek":
                # Local: sets original_output, hidden_states_list, input_ids for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                original_output, hidden_states_list, input_ids = original_generation(
                        # Local: sets input_text for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        input_text=example["formatted"],
                        model=model,  # Local: sets model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        # Local: sets tokenizer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        tokenizer=tokenizer,
                        device=device,)  # Local: sets device for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                
                # Local: sets final_output, _, _, _, _ for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                final_output, _, _, _, _ = optimized_generation(
                        # Local: sets reward_model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        reward_model=reward_model,
                        model=model,  # Local: sets model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        # Local: sets tokenizer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        tokenizer=tokenizer,
                        device=device,  # Local: sets device for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        # Local: sets question for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        question=example["question"],
                        # Local: sets input_text for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        input_text=example["formatted"],
                        # Local: sets original_answer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        original_answer=original_output,
                        # Local: sets original_hidden_states_list for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        original_hidden_states_list=hidden_states_list, 
                        # Local: sets input_ids for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        input_ids=input_ids,
                        # Local: sets max_num_steps for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        max_num_steps=args.max_num_steps,
                        lr=args.lr,  # Local: sets lr for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        # Local: sets grad_clip for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        grad_clip=args.grad_clip,
                        k=args.k,  # Local: sets k for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        # Local: sets reward_threshold for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        reward_threshold=args.reward_threshold,
                )
            # Local: checks the next mutually exclusive condition. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            elif args.generation_mode == "als":
                # Local: sets final_output for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                final_output = steered_generation(
                    model=model,  # Local: sets model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    # Local: sets tokenizer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    tokenizer=tokenizer,
                    # Local: sets input_text for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    input_text=example["formatted"],
                    # Local: sets steering_vector for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    steering_vector=steering_vector,
                    alpha=args.alpha,  # Local: sets alpha for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    # Local: sets similarity_threshold for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    similarity_threshold=args.similarity_threshold,
                    # Local: sets max_new_tokens for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    max_new_tokens=args.max_new_tokens,
                    device=device,  # Local: sets device for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                )
            # Local: checks the next mutually exclusive condition. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            elif args.generation_mode == "als_gated":
                # Local: sets final_output for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                final_output = gated_steered_generation(
                    model=model,  # Local: sets model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    # Local: sets tokenizer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    tokenizer=tokenizer,
                    # Local: sets input_text for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    input_text=example["formatted"],
                    # Local: sets steering_vector for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    steering_vector=steering_vector,
                    alpha=args.alpha,  # Local: sets alpha for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    # Local: sets similarity_threshold for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    similarity_threshold=args.similarity_threshold,
                    # Local: sets max_new_tokens for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    max_new_tokens=args.max_new_tokens,
                    device=device,  # Local: sets device for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                )
            # Local: checks the next mutually exclusive condition. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            elif args.generation_mode == "greedy_cot":
                # Local: sets final_output for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                final_output = greedy_cot_generation(
                    model=model,  # Local: sets model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    # Local: sets tokenizer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    tokenizer=tokenizer,
                    # Local: sets input_text for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    input_text=example["formatted"],
                    # Local: sets max_new_tokens for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    max_new_tokens=args.max_new_tokens,
                    device=device,  # Local: sets device for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                )
            # Local: checks the next mutually exclusive condition. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            elif args.generation_mode == "self_consistency":
                # Local: sets final_output for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                final_output = self_consistency_generation(
                    model=model,  # Local: sets model for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    # Local: sets tokenizer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    tokenizer=tokenizer,
                    # Local: sets input_text for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    input_text=example["formatted"],
                    k=args.sc_k,  # Local: sets k for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    # Local: sets max_new_tokens for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    max_new_tokens=args.max_new_tokens,
                    device=device,  # Local: sets device for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    # Local: sets data_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    data_name=args.dataset,
                    # Local: sets prompt_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    prompt_idx=prompt_idx,
                    # Local: sets model_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                    model_name=args.model_name_or_path
                )
            
            # Local: sets duration for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            duration = time.time() - start_time
            
            # Local: sets final_answer for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            final_answer = extract_answer(final_output, 
                                             # Local: sets data_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                                             data_name=args.dataset, 
                                             # Local: sets prompt_idx for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                                             prompt_idx=prompt_idx, 
                                             # Local: sets model_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                                             model_name=args.model_name_or_path)

            # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            if final_answer is not None:
                # Local: sets is_correct for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                is_correct = judge_answer(
                        # Local: sets final_output, true_answer, data_name for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                        final_output, true_answer, data_name=args.dataset, prompt_idx=prompt_idx)
            else:  # Local: handles the remaining branch after earlier checks fail. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                is_correct = False  # Local: sets is_correct for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.

            # Local: executes this statement in the current code path. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            results_history.append(is_correct)
            # Local: executes this statement in the current code path. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            time_history.append(duration)
            
            torch.save({  # Local: persists tensor or metric state to disk. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                # Local: adds an item or argument to the surrounding expression. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                "results_history": results_history,
                # Local: adds an item or argument to the surrounding expression. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
                "time_history": time_history,
            # Local: closes the surrounding literal or call expression. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            }, f"{output_dir}/logistics.pt")

        # Local: sets total_samples for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        total_samples = len(results_history)
        # Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        if total_samples > 0:
            # Local: sets final_accuracy for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            final_accuracy = sum(results_history) / total_samples
            # Local: sets avg_time for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            avg_time = sum(time_history) / total_samples
            # Local: sets final_results[prompt_idx] for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            final_results[prompt_idx] = {"accuracy": final_accuracy, "avg_time": avg_time}
            # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            print(f"Prompt {prompt_idx+1} Final accuracy: {final_accuracy:.4f}")
            # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            print(f"Prompt {prompt_idx+1} Average generation time: {avg_time:.4f} seconds")
        else:  # Local: handles the remaining branch after earlier checks fail. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
            print(f"No samples were evaluated for Prompt {prompt_idx+1}.")

    # --- print final summary table ---
    # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    print(f"\n{'='*20} FINAL SUMMARY ({args.generation_mode}) {'='*20}")
    # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    print(f"| {'Prompt':<10} | {'Accuracy':<10} | {'Avg. Time':<15} |")
    # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    print(f"|{'-'*12}|{'-'*12}|{'-'*17}|")
    # Local: iterates through the current collection. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    for prompt_idx, results in final_results.items():
        # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
        print(f"| {prompt_idx+1:<10} | {results['accuracy']:.4f}   | {results['avg_time']:.4f} sec      |")
    print(f"{ '='*55}")  # Local: reports progress or diagnostics to the run log. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.


# Local: opens a condition that selects behavior from current state. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
if __name__ == "__main__":
    args = parse_args()  # Local: sets args for later use in this scope. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
    main(args)  # Local: executes this statement in the current code path. Global: orchestrates ALS, LatentSeek, and baseline evaluation runs.
