"""Collect labeled hidden states for the offline ALS steering-vector estimate.

The ALS paper needs hidden states from successful and unsuccessful generations
before it can compute `E[h_correct] - E[h_incorrect]`. This script generates
model answers, labels them with verifier prompts, averages each answer's token
hidden states, and saves one tensor per example with a `good` or `bad` filename.
"""

# `argparse` makes the state-collection run reproducible from command-line flags.
import argparse
# `os` is used for output directories, resume checkpoints, and deterministic hashing.
import os
# Python's random module is seeded so dataset traversal and any helper randomness are repeatable.
import random
# NumPy is seeded alongside PyTorch for consistent experiment setup.
import numpy as np
# PyTorch runs model inference and serializes hidden-state tensors.
import torch
# `tqdm` gives progress feedback because offline state collection can be slow.
from tqdm import tqdm

# The dataset loader formats math problems with the selected solver prompt.
from data import get_dataset
# Ground-truth extraction filters out malformed examples before expensive generation.
from extract_judge_answer import extract_true_answer
# Original generation returns both the answer text and token-level hidden states.
from ori_generation import original_generation
# The reward model supplies the verifier votes that split states into good and bad pools.
from rewards.reward import RewardModel
# Hugging Face loaders instantiate the model/tokenizer pair used for both solving and verification.
from transformers import AutoModelForCausalLM, AutoTokenizer


# Seeding is shared with evaluation so offline vectors and online results are reproducible.
def set_seed(seed):
    """Set all local random seeds used by state collection.

    Locally, this pins PyTorch, CUDA, NumPy, Python hashing, and Python random.
    Globally, reproducible hidden-state pools make the ALS vector comparable
    across reruns and prompt settings.
    """
    # This fixes CPU-side PyTorch random streams.
    torch.manual_seed(seed)
    # This fixes CUDA random streams for all visible GPUs.
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN removes algorithm-level nondeterminism when possible.
    torch.backends.cudnn.deterministic = True
    # Disabling cuDNN benchmarking avoids per-run algorithm selection changes.
    torch.backends.cudnn.benchmark = False
    # Python hash seeding keeps hash-based ordering stable across processes.
    os.environ['PYTHONHASHSEED'] = str(seed)
    # NumPy receives the same seed for helper code that may sample through NumPy.
    np.random.seed(seed)
    # Python's standard RNG receives the same seed for helper code that uses `random`.
    random.seed(seed)


# This parser defines every input that affects the offline hidden-state dataset.
def parse_arguments():
    """Parse command-line arguments for an ALS state-collection run.

    Locally, the parsed values select a model, dataset slice, prompt format, and
    verifier threshold. Globally, those values define the distribution from
    which the eventual steering vector is estimated.
    """
    # The parser description documents this script as the first offline ALS phase.
    parser = argparse.ArgumentParser(description="Collect hidden states for steering vector computation.")
    # The model path identifies the exact LLM whose latent geometry will be steered later.
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    # The dataset name/path determines both examples and answer-extraction rules.
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., 'openai/gsm8k', 'MATH-500')")
    # The output root stores state tensors under a model/dataset-specific folder.
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    # The split defaults to training data because evaluation should remain disjoint.
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    # The prompt index selects free-form boxed output or structured JSON output.
    parser.add_argument("--solver_prompt_idx", type=int, default=0, help="Index of the solver prompt to use")
    # The start index enables collecting a contiguous slice of the dataset.
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to process")
    # The end index caps the slice and stays exclusive like Python ranges.
    parser.add_argument("--end_data_idx", type=int, default=None, help="End index of the data to process (exclusive)")
    # The threshold decides how many verifier approvals make one generation count as successful.
    parser.add_argument("--good_example_threshold", type=int, default=4, choices=[1, 2, 3, 4], help="Number of verifiers that must pass for an example to be 'good'.")
    # The seed controls all reproducibility hooks above.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    # The optional device override lets users force CPU or a specific CUDA setup.
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
    # Resume mode reads the logistics checkpoint and continues after the last saved index.
    parser.add_argument("--resume", action="store_true", help="Resume from a previous run")
    # Returning parsed arguments keeps the command-line interface separate from the work function.
    return parser.parse_args()


# This function performs the complete offline collection pass for one model/dataset/prompt setting.
def main(args):
    """Generate answers, label them, and save averaged hidden-state tensors.

    Locally, every processed example yields one `.pt` file tagged `good` or
    `bad`. Globally, those files become the empirical class distributions used
    by `compute_steering_vector.py`.
    """
    # A nonzero seed value triggers deterministic setup for this collection run.
    if args.seed:
        # The helper centralizes every library-specific seeding call.
        set_seed(args.seed)

    # The explicit device wins; otherwise CUDA is used when available for model inference speed.
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # This status message marks the expensive model-loading phase.
    print("Loading model and tokenizer...")
    # The causal LM supplies both hidden states and verifier generations.
    model = AutoModelForCausalLM.from_pretrained(
        # The path can be local or a Hugging Face identifier.
        args.model_name_or_path,
        # bfloat16 matches common inference settings for the target 7B/8B models.
        torch_dtype=torch.bfloat16,
        # `device_map=device` places the model on the selected compute device.
        device_map=device
    )
    # Evaluation mode disables dropout so hidden states reflect deterministic inference.
    model.eval()
    # The tokenizer must match the model because prompt formatting and decoding use its chat template.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # This status message marks construction of the verifier wrapper.
    print("Loading reward model...")
    # The reward model reuses the same LLM to ask Vera-style verification questions about each answer.
    reward_model = RewardModel(
        # The shared model avoids loading a second verifier model.
        model=model,
        # The tokenizer formats verifier prompts for the shared model.
        tokenizer=tokenizer,
        # The device matches the solver model so verifier generation runs in the same place.
        device=device,
        # The dataset name controls downstream answer/verifier assumptions.
        data_name=args.dataset
    )

    # This log line records the input distribution for the hidden-state pool.
    print(f"Loading dataset: {args.dataset}, split: {args.split}")
    # The loader returns formatted prompts plus raw question/answer fields.
    dataset = get_dataset(
        # The dataset name/path decides which Hugging Face dataset and columns are used.
        args.dataset,
        # The tokenizer applies the model-specific chat template during preprocessing.
        tokenizer=tokenizer,
        # The prompt index keeps state collection aligned with the later evaluation prompt.
        prompt_idx=args.solver_prompt_idx,
        # The split selects train/test or another locally stored dataset split.
        split=args.split,
    )

    # The model basename keeps output paths readable when model paths are absolute.
    model_name = args.model_name_or_path.split("/")[-1]
    # Replacing slashes makes Hugging Face dataset names safe as directory names.
    data_name = args.dataset.replace("/", "-")
    # The base directory groups all artifacts for one model/dataset pair.
    base_output_dir = f"{args.output_dir}/{model_name}-{data_name}"
    # The collection directory includes prompt and threshold because both change labels and vectors.
    output_dir = f"{base_output_dir}/state_collection/prompt{args.solver_prompt_idx}_thresh{args.good_example_threshold}"
    # Hidden-state tensors live in a dedicated subdirectory that `compute_steering_vector.py` scans.
    state_dir = f"{output_dir}/hidden_states/"
    # Creating the directory up front makes every later `torch.save` path valid.
    os.makedirs(state_dir, exist_ok=True)

    # The logistics file tracks counts and resume position separately from the tensor files.
    logistics_path = f"{output_dir}/logistics.pt"
    # Successful-example count starts at zero unless a previous run is resumed.
    good_count = 0
    # Failed-example count starts at zero unless a previous run is resumed.
    bad_count = 0
    # The first index to process defaults to the requested start of the dataset slice.
    start_idx = args.start_data_idx

    # Resume only happens when both the flag and checkpoint file are present.
    if args.resume and os.path.exists(logistics_path):
        # The output directory identifies exactly which run is being resumed.
        print(f"Resuming from {output_dir}")
        # The logistics checkpoint restores counts and the next unprocessed index.
        logistics = torch.load(logistics_path)
        # Missing keys fall back to zero for compatibility with partial checkpoints.
        good_count = logistics.get("good_count", 0)
        # Missing keys fall back to zero for compatibility with partial checkpoints.
        bad_count = logistics.get("bad_count", 0)
        # `start_idx` stores `i + 1`, so the resumed run continues after the last save.
        start_idx = logistics.get("start_idx", 0)

    # If no explicit end was provided, process through the dataset length.
    end_idx = args.end_data_idx if args.end_data_idx is not None else len(dataset)
    # Clamping prevents out-of-range dataset access when a user passes a large end index.
    end_idx = min(end_idx, len(dataset))

    # This line records the exact slice that will contribute tensors to the steering estimate.
    print(f"Starting state collection from index {start_idx} to {end_idx}...")

    # A Python range creates the inclusive/exclusive index sequence used by the main loop.
    data_idx_list = range(start_idx, end_idx)
    # `tqdm` reports which examples have been converted into labeled hidden states.
    for i in tqdm(data_idx_list, desc="Collecting States"):
        # Indexing the mapped dataset yields one formatted prompt and its original answer.
        example = dataset[i]
        # Ground truth extraction normalizes labels and skips unsupported/malformed answers.
        true_answer = extract_true_answer(example["answer"], name=args.dataset)

        # Examples without labels cannot be judged into good or bad pools.
        if true_answer is None:
            # Skipping avoids saving an unlabeled hidden state that would corrupt the vector estimate.
            continue

        # `original_generation` performs greedy decoding and records hidden states at each generated step.
        original_output, hidden_states_list, _ = original_generation(
            # The formatted prompt includes the selected solver instruction and chat template.
            input_text=example["formatted"],
            # The same model that will be steered later supplies the hidden states.
            model=model,
            # The tokenizer encodes the prompt and decodes generated token ids.
            tokenizer=tokenizer,
            # The device matches the loaded model placement.
            device=device,
        )

        # Some generation failures can produce no hidden states, leaving no tensor to save.
        if not hidden_states_list:
            # Skipping keeps the state directory limited to valid tensors.
            continue

        # Verifier prompts judge the generated solution along calculation, answer, completeness, and understanding axes.
        verifications = reward_model.get_verifications(example["question"], original_output)
        # Boolean verifier approvals sum as integers, producing the number of passed checks.
        num_verifiers_passed = sum(verifications.values())
        # The configured threshold maps the verifier count to the binary good/bad label.
        is_good = num_verifiers_passed >= args.good_example_threshold

        # The status string is embedded in the filename so vector computation can classify files cheaply.
        status = "good" if is_good else "bad"
        # Including the dataset index and status makes saved tensors traceable and label-readable.
        file_path = f"{state_dir}/idx_{i}_{status}.pt"
        # Stacking token hidden states and averaging over tokens gives one per-example latent summary.
        avg_hidden_state = torch.stack(hidden_states_list).mean(dim=0)
        # Saving the tensor creates the offline artifact later loaded into the success/failure pools.
        torch.save(avg_hidden_state, file_path)

        # Successful examples increment the good pool size used for final reporting and resume state.
        if is_good:
            # The increment records that one more tensor entered the success distribution.
            good_count += 1
        # Failed examples increment the bad pool size used for final reporting and resume state.
        else:
            # The increment records that one more tensor entered the failure distribution.
            bad_count += 1

        # The logistics checkpoint is rewritten after each example so interruption loses little work.
        torch.save({
            # The good count lets resumed runs preserve progress statistics.
            "good_count": good_count,
            # The bad count lets resumed runs preserve progress statistics.
            "bad_count": bad_count,
            # Total processed examples are derived for convenience in monitoring.
            "total": good_count + bad_count,
            # The next start index is one past the example just saved.
            "start_idx": i + 1,
        # The checkpoint path stays outside `hidden_states/` so tensor scanning remains simple.
        }, logistics_path)

    # Total processed examples determines whether the final percentage report is meaningful.
    total_processed = good_count + bad_count
    # At least one labeled tensor allows reporting class balance.
    if total_processed > 0:
        # This message marks successful completion of the offline collection phase.
        print("\nFinished collecting states.")
        # The good percentage shows how much evidence supports the success centroid.
        print(f"Good examples: {good_count} ({good_count/total_processed:.2%})")
        # The bad percentage shows how much evidence supports the failure centroid.
        print(f"Bad examples: {bad_count} ({bad_count/total_processed:.2%})")
    # No collected tensors means the next offline phase cannot compute a vector.
    else:
        # This message points the user at an empty or fully skipped collection run.
        print("\nNo states were collected.")


# Running this file directly starts the offline collection CLI.
if __name__ == "__main__":
    # CLI parsing happens once before logging and collection.
    args = parse_arguments()
    # This banner makes run metadata easy to find in long terminal logs.
    print("--- Command Line Arguments ---")
    # Iterating over parsed attributes prints every setting that defines the collected distribution.
    for arg, value in vars(args).items():
        # Each key/value pair documents one reproducibility-relevant option.
        print(f"{arg}: {value}")
    # The closing banner separates metadata from collection progress.
    print("----------------------------")
    # The parsed configuration drives the full hidden-state collection pass.
    main(args)
