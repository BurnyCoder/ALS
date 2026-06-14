"""Create fixed random subset index files for reproducible evaluations.

ALS ablations in the README use fixed subsets so latency/accuracy comparisons
run on identical examples. This script writes the chosen dataset indices to a
text file that `main.py --fixed_subset_path` can later read.
"""

# `os` creates the subset directory and joins output paths.
import os
# `random` shuffles dataset indices under a user-provided seed.
import random
# `argparse` exposes subset creation as a command-line utility.
import argparse
# Hugging Face `load_dataset` retrieves the benchmark split whose indices will be sampled.
from datasets import load_dataset


# This function creates one deterministic shuffled subset file.
def create_fixed_subset(dataset_name, split, n_samples, output_dir):
    """Write a text file containing `n_samples` shuffled dataset indices.

    Locally, the function loads a dataset split, shuffles all row indices, and
    writes the first requested indices. Globally, those files keep ALS and
    baseline ablations on the same examples.
    """
    # The log line records the dataset, split, and requested subset size.
    print(f"Loading dataset {dataset_name} to create a subset of {n_samples} from the {split} split...")

    # Most datasets do not need a Hugging Face config name.
    config_name = None
    # GSM8K uses the `socratic` configuration in this repository.
    if "gsm8k" in dataset_name:
        # Setting the config name lets `load_dataset` find the correct GSM8K variant.
        config_name = "socratic"

    # The actual dataset id defaults to the user-provided name.
    actual_dataset_name = dataset_name
    # The short local name is mapped to the public Hugging Face dataset id.
    if dataset_name == "MATH-500":
        # This mapping keeps CLI usage consistent with the rest of the repo.
        actual_dataset_name = "HuggingFaceH4/MATH-500"

    # Load the requested split so its row count and indices are available.
    dataset = load_dataset(actual_dataset_name, name=config_name, split=split)
    # Dataset length defines the valid index range.
    dataset_size = len(dataset)

    # A subset cannot contain more unique indices than the dataset has rows.
    if n_samples > dataset_size:
        # The error reports both requested and available sizes.
        raise ValueError(f"Requested sample size ({n_samples}) is larger than the dataset size ({dataset_size}).")

    # Build the full index list from zero to dataset_size - 1.
    all_indices = list(range(dataset_size))
    # Shuffle in place using the seed configured by the caller.
    random.shuffle(all_indices)

    # Keep the first `n_samples` shuffled indices as the fixed subset.
    subset_indices = all_indices[:n_samples]

    # Ensure the destination directory exists before opening the file.
    os.makedirs(output_dir, exist_ok=True)
    # The filename encodes dataset, split, and subset size for later identification.
    file_name = f"{dataset_name.replace('/', '-')}_{split}_{n_samples}.txt"
    # Joining directory and filename creates the final save path.
    save_path = os.path.join(output_dir, file_name)

    # Open the subset file for writing one index per line.
    with open(save_path, 'w') as f:
        # Iterate through selected indices in their fixed shuffled order.
        for index in subset_indices:
            # Each line is parsed by `main.py` as one integer row index.
            f.write(f"{index}\n")

    # The success message prints the exact path to pass into `--fixed_subset_path`.
    print(f"Successfully created subset file at: {save_path}")


# Running this file directly starts the subset-file CLI.
if __name__ == "__main__":
    # The parser documents required dataset, split, and size inputs.
    parser = argparse.ArgumentParser(description="Create fixed random subsets of indices for evaluation.")
    # Dataset name can be a Hugging Face id such as `openai/gsm8k`.
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the Hugging Face dataset (e.g., 'openai/gsm8k')")
    # Split selects which dataset partition is indexed.
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (e.g., 'test')")
    # Number of samples controls how many indices are written.
    parser.add_argument("--n_samples", type=int, required=True, help="The number of samples to include in the subset.")
    # Output directory controls where subset text files are stored.
    parser.add_argument("--output_dir", type=str, default="./subsets", help="Directory to save the subset files.")

    # The seed makes the random sample reproducible across runs.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")

    # Parse CLI arguments into a namespace.
    args = parser.parse_args()
    # Seed Python random before shuffling indices.
    random.seed(args.seed)

    # Create the requested subset file.
    create_fixed_subset(args.dataset_name, args.split, args.n_samples, args.output_dir)
