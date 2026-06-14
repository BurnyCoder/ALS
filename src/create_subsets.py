# Local: file src/create_subsets.py provides first-party ALS source context. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
import os  # Local: imports os for this module. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
import random  # Local: imports random for this module. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
import argparse  # Local: imports argparse for this module. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
from datasets import load_dataset  # Local: imports selected helpers from datasets. Global: keeps ALS ablations reproducible by fixing evaluation subsets.

# Local: defines the create_fixed_subset function. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
def create_fixed_subset(dataset_name, split, n_samples, output_dir):
    # Local: starts a multi-line text literal that Python treats as one value. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    """
    Creates a text file containing a fixed, random subset of indices from a dataset.
    """
    # Local: reports progress or diagnostics to the run log. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    print(f"Loading dataset {dataset_name} to create a subset of {n_samples} from the {split} split...")
    
    config_name = None  # Local: sets config_name for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    # Local: opens a condition that selects behavior from current state. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    if "gsm8k" in dataset_name:
        config_name = "socratic"  # Local: sets config_name for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    
    # Local: sets actual_dataset_name for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    actual_dataset_name = dataset_name
    # Local: opens a condition that selects behavior from current state. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    if dataset_name == "MATH-500":
        # Local: sets actual_dataset_name for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
        actual_dataset_name = "HuggingFaceH4/MATH-500"

    # Local: loads benchmark data from disk or Hugging Face datasets. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    dataset = load_dataset(actual_dataset_name, name=config_name, split=split)
    dataset_size = len(dataset)  # Local: sets dataset_size for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.

    # Local: opens a condition that selects behavior from current state. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    if n_samples > dataset_size:
        # Local: stops execution with an explicit error for invalid state. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
        raise ValueError(f"Requested sample size ({n_samples}) is larger than the dataset size ({dataset_size}).")

    # create a list of all possible indices and shuffle it
    # Local: sets all_indices for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    all_indices = list(range(dataset_size))
    # Local: executes this statement in the current code path. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    random.shuffle(all_indices)

    # take the first n_samples indices
    # Local: sets subset_indices for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    subset_indices = all_indices[:n_samples]

    os.makedirs(output_dir, exist_ok=True)  # Local: ensures the output directory exists. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    # Local: sets file_name for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    file_name = f"{dataset_name.replace('/', '-')}_{split}_{n_samples}.txt"
    # Local: sets save_path for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    save_path = os.path.join(output_dir, file_name)

    with open(save_path, 'w') as f:  # Local: enters a managed runtime context. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
        for index in subset_indices:  # Local: iterates through the current collection. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
            # Local: executes this statement in the current code path. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
            f.write(f"{index}\n")

    # Local: reports progress or diagnostics to the run log. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    print(f"Successfully created subset file at: {save_path}")

# Local: opens a condition that selects behavior from current state. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
if __name__ == "__main__":
    # Local: sets parser for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    parser = argparse.ArgumentParser(description="Create fixed random subsets of indices for evaluation.")
    # Local: registers a command-line option for the script. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the Hugging Face dataset (e.g., 'openai/gsm8k')")
    # Local: registers a command-line option for the script. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (e.g., 'test')")
    # Local: registers a command-line option for the script. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    parser.add_argument("--n_samples", type=int, required=True, help="The number of samples to include in the subset.")
    # Local: registers a command-line option for the script. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    parser.add_argument("--output_dir", type=str, default="./subsets", help="Directory to save the subset files.")
    
    # set a seed for reproducibility of the random sample
    # Local: registers a command-line option for the script. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")

    args = parser.parse_args()  # Local: sets args for later use in this scope. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    random.seed(args.seed)  # Local: seeds Python randomness for reproducibility. Global: keeps ALS ablations reproducible by fixing evaluation subsets.

    # Local: executes this statement in the current code path. Global: keeps ALS ablations reproducible by fixing evaluation subsets.
    create_fixed_subset(args.dataset_name, args.split, args.n_samples, args.output_dir)