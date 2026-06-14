# Local: file src/collect_states.py provides first-party ALS source context. Global: implements the paper's offline phase that labels generations and stores hidden states.
import argparse  # Local: imports argparse for this module. Global: implements the paper's offline phase that labels generations and stores hidden states.
import os  # Local: imports os for this module. Global: implements the paper's offline phase that labels generations and stores hidden states.
import random  # Local: imports random for this module. Global: implements the paper's offline phase that labels generations and stores hidden states.
import numpy as np  # Local: imports numpy as np for this module. Global: implements the paper's offline phase that labels generations and stores hidden states.
import torch  # Local: imports torch for this module. Global: implements the paper's offline phase that labels generations and stores hidden states.
# Local: imports selected helpers from tqdm. Global: implements the paper's offline phase that labels generations and stores hidden states.
from tqdm import tqdm

# Local: imports selected helpers from data. Global: implements the paper's offline phase that labels generations and stores hidden states.
from data import get_dataset
# Local: imports selected helpers from extract_judge_answer. Global: implements the paper's offline phase that labels generations and stores hidden states.
from extract_judge_answer import extract_true_answer
# Local: imports selected helpers from ori_generation. Global: implements the paper's offline phase that labels generations and stores hidden states.
from ori_generation import original_generation
# Local: imports selected helpers from rewards.reward. Global: implements the paper's offline phase that labels generations and stores hidden states.
from rewards.reward import RewardModel
# Local: imports selected helpers from transformers. Global: implements the paper's offline phase that labels generations and stores hidden states.
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(seed):  # Local: defines the set_seed function. Global: implements the paper's offline phase that labels generations and stores hidden states.
    # Local: executes this statement in the current code path. Global: implements the paper's offline phase that labels generations and stores hidden states.
    """Set random seed for reproducibility."""
    # Local: seeds PyTorch randomness for reproducibility. Global: implements the paper's offline phase that labels generations and stores hidden states.
    torch.manual_seed(seed)
    # Local: seeds CUDA randomness for reproducibility. Global: implements the paper's offline phase that labels generations and stores hidden states.
    torch.cuda.manual_seed_all(seed)
    # Local: sets torch.backends.cudnn.deterministic for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    torch.backends.cudnn.deterministic = True
    # Local: sets torch.backends.cudnn.benchmark for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    torch.backends.cudnn.benchmark = False
    # Local: sets os.environ['PYTHONHASHSEED'] for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Local: seeds NumPy randomness for reproducibility. Global: implements the paper's offline phase that labels generations and stores hidden states.
    np.random.seed(seed)
    # Local: seeds Python randomness for reproducibility. Global: implements the paper's offline phase that labels generations and stores hidden states.
    random.seed(seed)

# Local: defines the parse_arguments function. Global: implements the paper's offline phase that labels generations and stores hidden states.
def parse_arguments():
    # Local: executes this statement in the current code path. Global: implements the paper's offline phase that labels generations and stores hidden states.
    """Parse command-line arguments."""
    # Local: sets parser for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser = argparse.ArgumentParser(description="Collect hidden states for steering vector computation.")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., 'openai/gsm8k', 'MATH-500')")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--solver_prompt_idx", type=int, default=0, help="Index of the solver prompt to use")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to process")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--end_data_idx", type=int, default=None, help="End index of the data to process (exclusive)")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--good_example_threshold", type=int, default=4, choices=[1, 2, 3, 4], help="Number of verifiers that must pass for an example to be 'good'.")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
    # Local: registers a command-line option for the script. Global: implements the paper's offline phase that labels generations and stores hidden states.
    parser.add_argument("--resume", action="store_true", help="Resume from a previous run")
    # Local: returns the computed result to the caller. Global: implements the paper's offline phase that labels generations and stores hidden states.
    return parser.parse_args()

def main(args):  # Local: defines the main function. Global: implements the paper's offline phase that labels generations and stores hidden states.
    # Local: executes this statement in the current code path. Global: implements the paper's offline phase that labels generations and stores hidden states.
    """Main function to collect hidden states."""
    # Local: opens a condition that selects behavior from current state. Global: implements the paper's offline phase that labels generations and stores hidden states.
    if args.seed:
        # Local: executes this statement in the current code path. Global: implements the paper's offline phase that labels generations and stores hidden states.
        set_seed(args.seed)

    # Local: sets device for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
    print("Loading model and tokenizer...")
    # Local: loads the causal language model weights. Global: implements the paper's offline phase that labels generations and stores hidden states.
    model = AutoModelForCausalLM.from_pretrained(
        # Local: adds an item or argument to the surrounding expression. Global: implements the paper's offline phase that labels generations and stores hidden states.
        args.model_name_or_path,
        # Local: sets torch_dtype for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        torch_dtype=torch.bfloat16,
        # Local: sets device_map for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        device_map=device
    )
    # Local: executes this statement in the current code path. Global: implements the paper's offline phase that labels generations and stores hidden states.
    model.eval()
    # Local: loads the matching tokenizer. Global: implements the paper's offline phase that labels generations and stores hidden states.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
    print("Loading reward model...")
    # Local: sets reward_model for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    reward_model = RewardModel(
        # Local: sets model for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        model=model,
        # Local: sets tokenizer for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        tokenizer=tokenizer,
        # Local: sets device for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        device=device,
        # Local: sets data_name for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        data_name=args.dataset
    )

    # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
    print(f"Loading dataset: {args.dataset}, split: {args.split}")
    # Local: sets dataset for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    dataset = get_dataset(
        # Local: adds an item or argument to the surrounding expression. Global: implements the paper's offline phase that labels generations and stores hidden states.
        args.dataset,
        # Local: sets tokenizer for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        tokenizer=tokenizer,
        # Local: sets prompt_idx for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        prompt_idx=args.solver_prompt_idx,
        # Local: sets split for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        split=args.split,
    )

    # set up directories and logging
    # Local: sets model_name for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    model_name = args.model_name_or_path.split("/")[-1]
    # Local: sets data_name for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    data_name = args.dataset.replace("/", "-")
    # Local: sets base_output_dir for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    base_output_dir = f"{args.output_dir}/{model_name}-{data_name}"
    # Local: sets output_dir for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    output_dir = f"{base_output_dir}/state_collection/prompt{args.solver_prompt_idx}_thresh{args.good_example_threshold}"
    # Local: sets state_dir for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    state_dir = f"{output_dir}/hidden_states/"
    # Local: ensures the output directory exists. Global: implements the paper's offline phase that labels generations and stores hidden states.
    os.makedirs(state_dir, exist_ok=True)

    # Local: sets logistics_path for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    logistics_path = f"{output_dir}/logistics.pt"
    # Local: sets good_count for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    good_count = 0
    # Local: sets bad_count for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    bad_count = 0
    # Local: sets start_idx for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    start_idx = args.start_data_idx

    # Local: opens a condition that selects behavior from current state. Global: implements the paper's offline phase that labels generations and stores hidden states.
    if args.resume and os.path.exists(logistics_path):
        # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
        print(f"Resuming from {output_dir}")
        # Local: sets logistics for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        logistics = torch.load(logistics_path)
        # Local: sets good_count for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        good_count = logistics.get("good_count", 0)
        # Local: sets bad_count for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        bad_count = logistics.get("bad_count", 0)
        # Local: sets start_idx for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        start_idx = logistics.get("start_idx", 0)

    # determine data range
    # Local: sets end_idx for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    end_idx = args.end_data_idx if args.end_data_idx is not None else len(dataset)
    # Local: sets end_idx for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    end_idx = min(end_idx, len(dataset))

    # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
    print(f"Starting state collection from index {start_idx} to {end_idx}...")
    
    # Local: sets data_idx_list for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    data_idx_list = range(start_idx, end_idx)
    # Local: iterates through the current collection. Global: implements the paper's offline phase that labels generations and stores hidden states.
    for i in tqdm(data_idx_list, desc="Collecting States"):
        # Local: sets example for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        example = dataset[i]
        # Local: sets true_answer for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        true_answer = extract_true_answer(example["answer"], name=args.dataset)

        # Local: opens a condition that selects behavior from current state. Global: implements the paper's offline phase that labels generations and stores hidden states.
        if true_answer is None:
            # Local: skips the rest of this iteration and moves to the next item. Global: implements the paper's offline phase that labels generations and stores hidden states.
            continue

        # generate one solution
        # Local: sets original_output, hidden_states_list, _ for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        original_output, hidden_states_list, _ = original_generation(
            # Local: sets input_text for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
            input_text=example["formatted"],
            # Local: sets model for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
            model=model,
            # Local: sets tokenizer for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
            tokenizer=tokenizer,
            # Local: sets device for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
            device=device,
        )

        # Local: opens a condition that selects behavior from current state. Global: implements the paper's offline phase that labels generations and stores hidden states.
        if not hidden_states_list:
            # Local: skips the rest of this iteration and moves to the next item. Global: implements the paper's offline phase that labels generations and stores hidden states.
            continue

        # judge solution
        # Local: sets verifications for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        verifications = reward_model.get_verifications(example["question"], original_output)
        # Local: sets num_verifiers_passed for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        num_verifiers_passed = sum(verifications.values())
        # Local: executes this statement in the current code path. Global: implements the paper's offline phase that labels generations and stores hidden states.
        is_good = num_verifiers_passed >= args.good_example_threshold

        # save the averaged hidden state
        # Local: sets status for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        status = "good" if is_good else "bad"
        # Local: sets file_path for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        file_path = f"{state_dir}/idx_{i}_{status}.pt"
        # Local: sets avg_hidden_state for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
        avg_hidden_state = torch.stack(hidden_states_list).mean(dim=0)
        # Local: persists tensor or metric state to disk. Global: implements the paper's offline phase that labels generations and stores hidden states.
        torch.save(avg_hidden_state, file_path)

        # Local: opens a condition that selects behavior from current state. Global: implements the paper's offline phase that labels generations and stores hidden states.
        if is_good:
            # Local: updates good_count for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
            good_count += 1
        # Local: handles the remaining branch after earlier checks fail. Global: implements the paper's offline phase that labels generations and stores hidden states.
        else:
            # Local: updates bad_count for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
            bad_count += 1

        # save logistics
        # Local: persists tensor or metric state to disk. Global: implements the paper's offline phase that labels generations and stores hidden states.
        torch.save({
            # Local: adds an item or argument to the surrounding expression. Global: implements the paper's offline phase that labels generations and stores hidden states.
            "good_count": good_count,
            # Local: adds an item or argument to the surrounding expression. Global: implements the paper's offline phase that labels generations and stores hidden states.
            "bad_count": bad_count,
            # Local: adds an item or argument to the surrounding expression. Global: implements the paper's offline phase that labels generations and stores hidden states.
            "total": good_count + bad_count,
            # Local: adds an item or argument to the surrounding expression. Global: implements the paper's offline phase that labels generations and stores hidden states.
            "start_idx": i + 1,
        # Local: closes the surrounding literal or call expression. Global: implements the paper's offline phase that labels generations and stores hidden states.
        }, logistics_path)

    # Local: sets total_processed for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    total_processed = good_count + bad_count
    # Local: opens a condition that selects behavior from current state. Global: implements the paper's offline phase that labels generations and stores hidden states.
    if total_processed > 0:
        # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
        print("\nFinished collecting states.")
        # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
        print(f"Good examples: {good_count} ({good_count/total_processed:.2%})")
        # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
        print(f"Bad examples: {bad_count} ({bad_count/total_processed:.2%})")
    # Local: handles the remaining branch after earlier checks fail. Global: implements the paper's offline phase that labels generations and stores hidden states.
    else:
        # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
        print("\nNo states were collected.")

# Local: opens a condition that selects behavior from current state. Global: implements the paper's offline phase that labels generations and stores hidden states.
if __name__ == "__main__":
    # Local: sets args for later use in this scope. Global: implements the paper's offline phase that labels generations and stores hidden states.
    args = parse_arguments()
    # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
    print("--- Command Line Arguments ---")
    # Local: iterates through the current collection. Global: implements the paper's offline phase that labels generations and stores hidden states.
    for arg, value in vars(args).items():
        # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
        print(f"{arg}: {value}")
    # Local: reports progress or diagnostics to the run log. Global: implements the paper's offline phase that labels generations and stores hidden states.
    print("----------------------------")
    # Local: executes this statement in the current code path. Global: implements the paper's offline phase that labels generations and stores hidden states.
    main(args)