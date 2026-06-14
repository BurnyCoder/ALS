# Local: file src/monitor_run.py provides first-party ALS source context. Global: summarizes saved ALS evaluation progress during long model runs.
import torch  # Local: imports torch for this module. Global: summarizes saved ALS evaluation progress during long model runs.
import argparse  # Local: imports argparse for this module. Global: summarizes saved ALS evaluation progress during long model runs.
import os  # Local: imports os for this module. Global: summarizes saved ALS evaluation progress during long model runs.
import numpy as np  # Local: imports numpy as np for this module. Global: summarizes saved ALS evaluation progress during long model runs.

def monitor_run(log_file):  # Local: defines the monitor_run function. Global: summarizes saved ALS evaluation progress during long model runs.
    # Local: starts a multi-line text literal that Python treats as one value. Global: summarizes saved ALS evaluation progress during long model runs.
    """
    Loads a logistics.pt file and prints the current progress.
    """
    # Local: opens a condition that selects behavior from current state. Global: summarizes saved ALS evaluation progress during long model runs.
    if not os.path.exists(log_file):
        # Local: reports progress or diagnostics to the run log. Global: summarizes saved ALS evaluation progress during long model runs.
        print(f"Error: Log file not found at {log_file}")
        return  # Local: returns the computed result to the caller. Global: summarizes saved ALS evaluation progress during long model runs.

    # Local: starts a protected operation that may fail on external or parsed input. Global: summarizes saved ALS evaluation progress during long model runs.
    try:
        # Local: sets log_data for later use in this scope. Global: summarizes saved ALS evaluation progress during long model runs.
        log_data = torch.load(log_file)
        # Local: sets results_history for later use in this scope. Global: summarizes saved ALS evaluation progress during long model runs.
        results_history = log_data.get("results_history", [])
        # Local: sets time_history for later use in this scope. Global: summarizes saved ALS evaluation progress during long model runs.
        time_history = log_data.get("time_history", [])
    # Local: handles a recoverable failure from the protected operation. Global: summarizes saved ALS evaluation progress during long model runs.
    except Exception as e:
        # Local: reports progress or diagnostics to the run log. Global: summarizes saved ALS evaluation progress during long model runs.
        print(f"Error loading or parsing log file: {e}")
        return  # Local: returns the computed result to the caller. Global: summarizes saved ALS evaluation progress during long model runs.

    # Local: sets samples_completed for later use in this scope. Global: summarizes saved ALS evaluation progress during long model runs.
    samples_completed = len(results_history)

    # Local: opens a condition that selects behavior from current state. Global: summarizes saved ALS evaluation progress during long model runs.
    if samples_completed == 0:
        # Local: reports progress or diagnostics to the run log. Global: summarizes saved ALS evaluation progress during long model runs.
        print("No samples have been completed yet.")
        return  # Local: returns the computed result to the caller. Global: summarizes saved ALS evaluation progress during long model runs.

    # Local: sets current_accuracy for later use in this scope. Global: summarizes saved ALS evaluation progress during long model runs.
    current_accuracy = np.mean(results_history)
    # Local: sets current_avg_time for later use in this scope. Global: summarizes saved ALS evaluation progress during long model runs.
    current_avg_time = np.mean(time_history)

    # Local: reports progress or diagnostics to the run log. Global: summarizes saved ALS evaluation progress during long model runs.
    print("\n--- Live Run Progress ---")
    # Local: reports progress or diagnostics to the run log. Global: summarizes saved ALS evaluation progress during long model runs.
    print(f"Log File: {log_file}")
    # Local: reports progress or diagnostics to the run log. Global: summarizes saved ALS evaluation progress during long model runs.
    print("-------------------------")
    # Local: reports progress or diagnostics to the run log. Global: summarizes saved ALS evaluation progress during long model runs.
    print(f"Samples Completed: {samples_completed}")
    # Local: reports progress or diagnostics to the run log. Global: summarizes saved ALS evaluation progress during long model runs.
    print(f"Current Accuracy:  {current_accuracy:.4f} ({current_accuracy:.2%})")
    # Local: reports progress or diagnostics to the run log. Global: summarizes saved ALS evaluation progress during long model runs.
    print(f"Current Avg. Time: {current_avg_time:.2f} seconds/sample")
    # Local: starts a multi-line text literal that Python treats as one value. Global: summarizes saved ALS evaluation progress during long model runs.
    print("-------------------------\
")  # Local: closes the continued progress-divider print call. Global: summarizes saved ALS evaluation progress during long model runs.

# Local: opens a condition that selects behavior from current state. Global: summarizes saved ALS evaluation progress during long model runs.
if __name__ == "__main__":
    # Local: sets parser for later use in this scope. Global: summarizes saved ALS evaluation progress during long model runs.
    parser = argparse.ArgumentParser(description="Monitor the live progress of an evaluation run.")
    # Local: registers a command-line option for the script. Global: summarizes saved ALS evaluation progress during long model runs.
    parser.add_argument("--log_file", type=str, required=True, help="Path to the logistics.pt file to monitor.")
    args = parser.parse_args()  # Local: sets args for later use in this scope. Global: summarizes saved ALS evaluation progress during long model runs.

    # Local: executes this statement in the current code path. Global: summarizes saved ALS evaluation progress during long model runs.
    monitor_run(args.log_file)
