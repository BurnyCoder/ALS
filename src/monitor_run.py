"""Inspect a saved evaluation logistics file while a run is in progress.

`main.py` saves correctness and latency histories after each example. This
utility loads that checkpoint and reports current accuracy and average time,
which is useful for long ALS, LatentSeek, or baseline runs.
"""

# PyTorch loads the `logistics.pt` checkpoint written by `main.py`.
import torch
# `argparse` exposes the monitor as a small command-line utility.
import argparse
# `os.path.exists` checks that the requested checkpoint path is present.
import os
# NumPy computes mean accuracy and latency from saved Python lists.
import numpy as np


# This function prints current aggregate metrics from a logistics checkpoint.
def monitor_run(log_file):
    """Load `logistics.pt` and print live progress statistics.

    Locally, the function reads saved histories and computes their means.
    Globally, it provides a lightweight way to track ALS-vs-baseline runs
    without waiting for the full benchmark to finish.
    """
    # Missing files usually mean the evaluation has not started or the path is wrong.
    if not os.path.exists(log_file):
        # Print the exact missing path for quick diagnosis.
        print(f"Error: Log file not found at {log_file}")
        # Return early because there is no checkpoint to inspect.
        return

    # Loading can fail if the checkpoint is mid-write or not a PyTorch file.
    try:
        # `torch.load` reconstructs the dictionary written by `main.py`.
        log_data = torch.load(log_file)
        # Correctness history defaults to empty for partial or older checkpoints.
        results_history = log_data.get("results_history", [])
        # Time history defaults to empty for partial or older checkpoints.
        time_history = log_data.get("time_history", [])
    # Any load/parse exception is reported without crashing the monitor.
    except Exception as e:
        # The error message keeps the user informed about checkpoint corruption or timing races.
        print(f"Error loading or parsing log file: {e}")
        # Return early because metrics cannot be computed.
        return

    # The number of completed samples is the length of the correctness history.
    samples_completed = len(results_history)

    # Empty histories mean no scored examples have been saved yet.
    if samples_completed == 0:
        # Print a clear status instead of producing NaN means.
        print("No samples have been completed yet.")
        # Return before computing means over empty lists.
        return

    # Boolean correctness values average to current accuracy.
    current_accuracy = np.mean(results_history)
    # Per-example durations average to current latency.
    current_avg_time = np.mean(time_history)

    # Print a short header for the live progress report.
    print("\n--- Live Run Progress ---")
    # Echo the monitored checkpoint path.
    print(f"Log File: {log_file}")
    # Separator improves readability in terminal output.
    print("-------------------------")
    # Report how many examples have contributed to the current metrics.
    print(f"Samples Completed: {samples_completed}")
    # Report accuracy as both decimal and percentage.
    print(f"Current Accuracy:  {current_accuracy:.4f} ({current_accuracy:.2%})")
    # Report mean wall-clock generation time per completed sample.
    print(f"Current Avg. Time: {current_avg_time:.2f} seconds/sample")
    # Closing separator ends the report block.
    print("-------------------------\
")


# Running this file directly starts the monitor CLI.
if __name__ == "__main__":
    # The parser accepts the path to one `logistics.pt` file.
    parser = argparse.ArgumentParser(description="Monitor the live progress of an evaluation run.")
    # The log file path identifies which evaluation mode/prompt checkpoint to inspect.
    parser.add_argument("--log_file", type=str, required=True, help="Path to the logistics.pt file to monitor.")
    # Parse CLI arguments into a namespace.
    args = parser.parse_args()

    # Print the current progress for the requested checkpoint.
    monitor_run(args.log_file)
