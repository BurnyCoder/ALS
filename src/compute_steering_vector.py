"""Compute ALS steering vectors from labeled hidden-state tensors.

This file implements the paper's offline amortization step: turn saved hidden
states from successful and unsuccessful generations into one reusable direction
`v = E[h_correct] - E[h_incorrect]` that online decoding can add at constant
cost instead of running LatentSeek-style per-query optimization.
"""

# PyTorch is used because hidden states and the final vector are serialized tensors.
import torch
# `os` provides directory scanning and output-path creation without changing the ALS math.
import os
# `argparse` exposes the offline vector computation as a reproducible command-line step.
import argparse
# `tqdm` wraps file iteration so long state-collection directories show progress.
from tqdm import tqdm


# This helper supports an optional smoothed alternative to the paper's simple mean.
def compute_ema(vectors, beta):
    """Compute an exponential moving average over hidden-state vectors.

    Locally, this walks the saved tensors in order and blends each new tensor
    into the running vector. Globally, it provides an alternative estimator for
    the ALS success or failure centroid before the final difference is taken.
    """
    # An empty class of examples cannot define a centroid, so callers receive no vector.
    if not vectors:
        # Returning `None` makes the missing-data state explicit to the caller.
        return None
    # The first tensor initializes the running average with the same shape as every hidden state.
    ema = vectors[0]
    # Each later vector is folded into the running average using the requested decay.
    for i in range(1, len(vectors)):
        # `beta * ema` preserves past evidence while `(1 - beta) * vectors[i]` adds the current state.
        ema = beta * ema + (1 - beta) * vectors[i]
    # The final EMA has the same dimensionality as the hidden states and can be differenced.
    return ema


# This function is the command's main unit of work: load labels, average classes, and save `v`.
def compute_steering_vector(args):
    """Load good/bad hidden states and save the ALS steering direction.

    Locally, filenames produced by `collect_states.py` decide whether a tensor
    belongs to the good or bad pool. Globally, subtracting the bad centroid from
    the good centroid creates the latent direction used by ALS decoding.
    """
    # Good vectors accumulate hidden states from generations that passed enough verifiers.
    good_vectors = []
    # Bad vectors accumulate hidden states from generations that failed that verifier threshold.
    bad_vectors = []

    # The printed directory ties this offline computation back to a specific state-collection run.
    print(f"Scanning directory: {args.states_dir}")
    # `os.listdir` enumerates the saved `.pt` files, and `tqdm` reports progress over them.
    for filename in tqdm(os.listdir(args.states_dir)):
        # Only PyTorch tensor files are part of the steering-vector computation.
        if filename.endswith(".pt"):
            # Joining the directory and filename builds the loadable artifact path.
            file_path = os.path.join(args.states_dir, filename)
            # A corrupt or incompatible tensor should not abort scanning unrelated files.
            try:
                # `torch.load` reconstructs the saved average hidden-state tensor.
                tensor = torch.load(file_path)
                # Filenames tagged `good` represent the empirical success manifold.
                if "good" in filename:
                    # Appending keeps all successful states available for class averaging.
                    good_vectors.append(tensor)
                # Filenames tagged `bad` represent the empirical failure manifold.
                elif "bad" in filename:
                    # Appending keeps all unsuccessful states available for class averaging.
                    bad_vectors.append(tensor)
            # Bad files are reported and skipped so one failed artifact does not hide the rest.
            except Exception as e:
                # The filename and exception give enough context to repair the collection directory.
                print(f"Could not load or process file {filename}: {e}")

    # ALS requires both centroids; without either class, the success-minus-failure vector is undefined.
    if not good_vectors or not bad_vectors:
        # Raising stops before saving a meaningless or partially defined steering vector.
        raise ValueError("Could not find enough good/bad vectors to compute. Please run collect_states.py first.")

    # This status line makes class balance visible because imbalance affects the vector estimate.
    print(f"Found {len(good_vectors)} good vectors and {len(bad_vectors)} bad vectors.")

    # The default method matches the paper: estimate each distribution by its arithmetic mean.
    if args.averaging_method == 'mean':
        # The log line records the estimator used for reproducibility.
        print("Using simple mean averaging.")
        # `torch.stack` creates a batch dimension, and `mean(dim=0)` computes E[h_correct].
        avg_good = torch.stack(good_vectors).mean(dim=0)
        # The same operation over failed states computes E[h_incorrect].
        avg_bad = torch.stack(bad_vectors).mean(dim=0)
    # EMA provides a smoothed alternative when the user selects it explicitly.
    elif args.averaging_method == 'ema':
        # The log line records the smoothing strength that changes the centroid estimate.
        print(f"Using Exponential Moving Average (EMA) with beta={args.ema_beta}.")
        # The good-class EMA acts as a smoothed success centroid.
        avg_good = compute_ema(good_vectors, args.ema_beta)
        # The bad-class EMA acts as a smoothed failure centroid.
        avg_bad = compute_ema(bad_vectors, args.ema_beta)
    # Any other method name would make the saved vector impossible to interpret.
    else:
        # Raising gives the invalid value directly to the command-line user.
        raise ValueError(f"Unknown averaging method: {args.averaging_method}")

    # This subtraction is the central ALS construction: point from failed to successful hidden states.
    steering_vector = avg_good - avg_bad

    # `dirname` extracts the parent directory so the vector file can be created in a new folder.
    output_dir = os.path.dirname(args.output_file)
    # If the output path has a parent directory, ensure it exists before saving.
    if output_dir:
        # `exist_ok=True` makes repeated runs idempotent for the directory creation step.
        os.makedirs(output_dir, exist_ok=True)

    # The tensor is saved once so online ALS can load it without recomputing offline statistics.
    torch.save(steering_vector, args.output_file)
    # This message confirms the exact vector artifact consumed by `main.py`.
    print(f"Steering vector computed and saved to {args.output_file}")
    # Printing the shape catches mismatches with the model hidden size before online decoding.
    print(f"Vector dimension: {steering_vector.shape}")


# Running the file directly turns it into the offline vector-construction CLI.
if __name__ == "__main__":
    # The parser documents and validates the inputs needed to reproduce a steering vector.
    parser = argparse.ArgumentParser(description="Compute the steering vector from collected hidden states.")
    # `states_dir` points at `.pt` hidden states written by `collect_states.py`.
    parser.add_argument("--states_dir", type=str, required=True, help="Directory containing the saved hidden states (.pt files).")
    # `output_file` names the single vector artifact later loaded by ALS generation.
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final steering vector.")
    # The averaging method chooses between the paper's mean estimator and optional EMA smoothing.
    parser.add_argument("--averaging_method", type=str, default="mean", choices=["mean", "ema"], help="Averaging method to use.")
    # `ema_beta` controls how strongly the optional EMA preserves older vectors.
    parser.add_argument("--ema_beta", type=float, default=0.9, help="Beta value for EMA smoothing.")
    # `parse_args` converts CLI strings into typed attributes consumed above.
    args = parser.parse_args()

    # The parsed arguments drive the complete offline steering-vector computation.
    compute_steering_vector(args)
