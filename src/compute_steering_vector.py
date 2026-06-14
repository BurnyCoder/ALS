# Local: file src/compute_steering_vector.py provides first-party ALS source context. Global: implements the paper's good-minus-bad latent direction construction.
import torch  # Local: imports torch for this module. Global: implements the paper's good-minus-bad latent direction construction.
import os  # Local: imports os for this module. Global: implements the paper's good-minus-bad latent direction construction.
import argparse  # Local: imports argparse for this module. Global: implements the paper's good-minus-bad latent direction construction.
from tqdm import tqdm  # Local: imports selected helpers from tqdm. Global: implements the paper's good-minus-bad latent direction construction.

def compute_ema(vectors, beta):  # Local: defines the compute_ema function. Global: implements the paper's good-minus-bad latent direction construction.
    # Local: executes this statement in the current code path. Global: implements the paper's good-minus-bad latent direction construction.
    """Computes the Exponential Moving Average of a list of vectors."""
    # Local: opens a condition that selects behavior from current state. Global: implements the paper's good-minus-bad latent direction construction.
    if not vectors:
        return None  # Local: returns the computed result to the caller. Global: implements the paper's good-minus-bad latent direction construction.
    ema = vectors[0]  # Local: sets ema for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
    # Local: iterates through the current collection. Global: implements the paper's good-minus-bad latent direction construction.
    for i in range(1, len(vectors)):
        # Local: sets ema for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
        ema = beta * ema + (1 - beta) * vectors[i]
    return ema  # Local: returns the computed result to the caller. Global: implements the paper's good-minus-bad latent direction construction.

# Local: defines the compute_steering_vector function. Global: implements the paper's good-minus-bad latent direction construction.
def compute_steering_vector(args):
    good_vectors = []  # Local: sets good_vectors for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
    bad_vectors = []  # Local: sets bad_vectors for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.

    # Local: reports progress or diagnostics to the run log. Global: implements the paper's good-minus-bad latent direction construction.
    print(f"Scanning directory: {args.states_dir}")
    # Local: iterates through the current collection. Global: implements the paper's good-minus-bad latent direction construction.
    for filename in tqdm(os.listdir(args.states_dir)):
        # Local: opens a condition that selects behavior from current state. Global: implements the paper's good-minus-bad latent direction construction.
        if filename.endswith(".pt"):
            # Local: sets file_path for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
            file_path = os.path.join(args.states_dir, filename)
            # Local: starts a protected operation that may fail on external or parsed input. Global: implements the paper's good-minus-bad latent direction construction.
            try:
                # Local: sets tensor for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
                tensor = torch.load(file_path)
                # Local: opens a condition that selects behavior from current state. Global: implements the paper's good-minus-bad latent direction construction.
                if "good" in filename:
                    # Local: executes this statement in the current code path. Global: implements the paper's good-minus-bad latent direction construction.
                    good_vectors.append(tensor)
                # Local: checks the next mutually exclusive condition. Global: implements the paper's good-minus-bad latent direction construction.
                elif "bad" in filename:
                    # Local: executes this statement in the current code path. Global: implements the paper's good-minus-bad latent direction construction.
                    bad_vectors.append(tensor)
            # Local: handles a recoverable failure from the protected operation. Global: implements the paper's good-minus-bad latent direction construction.
            except Exception as e:
                # Local: reports progress or diagnostics to the run log. Global: implements the paper's good-minus-bad latent direction construction.
                print(f"Could not load or process file {filename}: {e}")

    # Local: opens a condition that selects behavior from current state. Global: implements the paper's good-minus-bad latent direction construction.
    if not good_vectors or not bad_vectors:
        # Local: stops execution with an explicit error for invalid state. Global: implements the paper's good-minus-bad latent direction construction.
        raise ValueError("Could not find enough good/bad vectors to compute. Please run collect_states.py first.")

    # Local: reports progress or diagnostics to the run log. Global: implements the paper's good-minus-bad latent direction construction.
    print(f"Found {len(good_vectors)} good vectors and {len(bad_vectors)} bad vectors.")

    # Local: opens a condition that selects behavior from current state. Global: implements the paper's good-minus-bad latent direction construction.
    if args.averaging_method == 'mean':
        # Local: reports progress or diagnostics to the run log. Global: implements the paper's good-minus-bad latent direction construction.
        print("Using simple mean averaging.")
        # Local: sets avg_good for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
        avg_good = torch.stack(good_vectors).mean(dim=0)
        # Local: sets avg_bad for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
        avg_bad = torch.stack(bad_vectors).mean(dim=0)
    # Local: checks the next mutually exclusive condition. Global: implements the paper's good-minus-bad latent direction construction.
    elif args.averaging_method == 'ema':
        # Local: reports progress or diagnostics to the run log. Global: implements the paper's good-minus-bad latent direction construction.
        print(f"Using Exponential Moving Average (EMA) with beta={args.ema_beta}.")
        # Local: sets avg_good for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
        avg_good = compute_ema(good_vectors, args.ema_beta)
        # Local: sets avg_bad for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
        avg_bad = compute_ema(bad_vectors, args.ema_beta)
    else:  # Local: handles the remaining branch after earlier checks fail. Global: implements the paper's good-minus-bad latent direction construction.
        # Local: stops execution with an explicit error for invalid state. Global: implements the paper's good-minus-bad latent direction construction.
        raise ValueError(f"Unknown averaging method: {args.averaging_method}")

    # compute steering vector
    # Local: sets steering_vector for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
    steering_vector = avg_good - avg_bad

    # create output directory if it doesn't exist
    # Local: sets output_dir for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
    output_dir = os.path.dirname(args.output_file)
    # Local: opens a condition that selects behavior from current state. Global: implements the paper's good-minus-bad latent direction construction.
    if output_dir:
        # Local: ensures the output directory exists. Global: implements the paper's good-minus-bad latent direction construction.
        os.makedirs(output_dir, exist_ok=True)

    # Local: persists tensor or metric state to disk. Global: implements the paper's good-minus-bad latent direction construction.
    torch.save(steering_vector, args.output_file)
    # Local: reports progress or diagnostics to the run log. Global: implements the paper's good-minus-bad latent direction construction.
    print(f"Steering vector computed and saved to {args.output_file}")
    # Local: reports progress or diagnostics to the run log. Global: implements the paper's good-minus-bad latent direction construction.
    print(f"Vector dimension: {steering_vector.shape}")

# Local: opens a condition that selects behavior from current state. Global: implements the paper's good-minus-bad latent direction construction.
if __name__ == "__main__":
    # Local: sets parser for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.
    parser = argparse.ArgumentParser(description="Compute the steering vector from collected hidden states.")
    # Local: registers a command-line option for the script. Global: implements the paper's good-minus-bad latent direction construction.
    parser.add_argument("--states_dir", type=str, required=True, help="Directory containing the saved hidden states (.pt files).")
    # Local: registers a command-line option for the script. Global: implements the paper's good-minus-bad latent direction construction.
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final steering vector.")
    # Local: registers a command-line option for the script. Global: implements the paper's good-minus-bad latent direction construction.
    parser.add_argument("--averaging_method", type=str, default="mean", choices=["mean", "ema"], help="Averaging method to use.")
    # Local: registers a command-line option for the script. Global: implements the paper's good-minus-bad latent direction construction.
    parser.add_argument("--ema_beta", type=float, default=0.9, help="Beta value for EMA smoothing.")
    args = parser.parse_args()  # Local: sets args for later use in this scope. Global: implements the paper's good-minus-bad latent direction construction.

    # Local: executes this statement in the current code path. Global: implements the paper's good-minus-bad latent direction construction.
    compute_steering_vector(args)