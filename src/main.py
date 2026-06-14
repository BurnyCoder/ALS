"""Evaluate ALS, ALS-Gated, LatentSeek, and decoding baselines on math datasets.

This file is the online experiment orchestrator for the repository. It loads a
model, formats a benchmark with one or more prompt styles, dispatches to the
selected generation method, judges the answer, records latency, and writes a
`logistics.pt` file that can be resumed or monitored.
"""

# Hugging Face loaders instantiate the causal LM and matching tokenizer for all generation modes.
from transformers import AutoModelForCausalLM, AutoTokenizer
# PyTorch handles model dtype/device placement, vector loading, seeding, and checkpoint saves.
import torch
# The dataset helper applies prompt templates and tokenizer chat formatting.
from data import get_dataset
# `tqdm` shows per-example progress during potentially long evaluations.
from tqdm import tqdm
# The reward model is needed only for the LatentSeek optimization baseline.
from rewards.reward import RewardModel
# Original generation provides LatentSeek's initial answer and hidden-state trajectory.
from ori_generation import original_generation
# Optimized generation implements the online latent optimization baseline.
from opt_generation import optimized_generation
# Standard ALS applies a precomputed steering vector during decoding.
from steered_generation import steered_generation
# ALS-Gated applies the same vector while suppressing nudges around broken JSON.
from gated_generation import gated_steered_generation
# Greedy CoT and Self-Consistency provide token-space decoding baselines.
from baselines import greedy_cot_generation, self_consistency_generation
# `os` builds output paths, checks vector/subset files, and controls hash seeding.
import os
# Answer utilities extract labels, parse model answers, and judge correctness.
from extract_judge_answer import extract_answer, extract_true_answer, judge_answer
# `argparse` exposes evaluation configuration from the command line.
import argparse
# NumPy is seeded for reproducibility alongside PyTorch and Python random.
import numpy as np
# Python random is seeded for helper code that may use standard random sampling.
import random
# `time` measures generation latency per example for the efficiency side of the ALS comparison.
import time


# This function defines every CLI flag that can affect evaluation behavior or artifacts.
def parse_args():
    """Parse command-line options for an evaluation run.

    Locally, this converts strings into typed fields used by `main`. Globally,
    the arguments define the model, dataset, prompt format, generation method,
    ALS vector path, and baseline hyperparameters for a reproducible experiment.
    """
    # The parser description identifies this script as model evaluation rather than offline vector building.
    parser = argparse.ArgumentParser(description="Evaluate the model")
    # The model path determines the hidden-state geometry and tokenizer used by every method.
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    # The dataset controls prompt templates, answer labels, and judge logic.
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., 'openai/gsm8k', 'MATH-500')")
    # The output directory stores per-model, per-dataset logistics files.
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    # The split selects which benchmark partition to evaluate.
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    # This can run both prompt formats or a comma-separated subset.
    parser.add_argument("--prompts_to_run", type=str, default="all", help="Which prompts to run, e.g., '0', '1', or 'all'")
    # The start index supports partial evaluations over contiguous dataset slices.
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    # The end index is exclusive and can shorten a benchmark run.
    parser.add_argument("--end_data_idx", type=int, default=None, help="End index of the data to evaluate (exclusive)")
    # A fixed subset file supports reproducible ablations on selected example ids.
    parser.add_argument("--fixed_subset_path", type=str, default=None, help="Path to a file with a fixed list of indices for evaluation.")
    # The verbose flag is parsed for compatibility with commands that include it.
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose print statements")

    # The seed flag controls all deterministic setup done by `set_seed`.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # Generation mode selects ALS, ALS-Gated, LatentSeek, or baseline decoding.
    parser.add_argument("--generation_mode", type=str, default="latentseek", choices=["latentseek", "als", "greedy_cot", "self_consistency", "als_gated"], help="Generation mode to use")

    # LatentSeek learning rate controls optimizer step size on hidden states.
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    # Gradient clipping optionally limits LatentSeek hidden-state update magnitude.
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    # `k` sets the fraction of original hidden states that LatentSeek optimizes.
    parser.add_argument("--k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    # This bounds the number of LatentSeek reward/gradient update iterations.
    parser.add_argument("--max_num_steps", type=int, default=10, help="Number of optimization iterations")
    # The token cap is shared across ALS and baseline generation paths.
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Number of generated tokens")
    # LatentSeek stops early when verifier reward exceeds this threshold.
    parser.add_argument("--reward_threshold", type=float, default=-0.2, help="Threshold for reward to stop optimization")

    # The vector template resolves to the offline steering vector for each prompt.
    parser.add_argument("--vector_name_template", type=str, default="./vectors/{model_name}_{dataset_name}_p{prompt_idx}.pt", help="Template for steering vector filenames.")
    # Alpha scales the ALS hidden-state nudge `alpha * v`.
    parser.add_argument("--alpha", type=float, default=0.3, help="Strength of the steering intervention")
    # The cosine threshold decides when ALS considers a hidden state off the success direction.
    parser.add_argument("--similarity_threshold", type=float, default=0.1, help="Cosine similarity threshold to trigger steering")

    # `sc_k` sets how many sampled traces vote in Self-Consistency.
    parser.add_argument("--sc_k", type=int, default=5, help="Number of samples for self-consistency")

    # The optional device override lets users force CPU or a specific CUDA mapping.
    parser.add_argument("--device", type=str, default=None)

    # The optional format rule adds a verifier penalty for outputs that miss a required answer shape.
    parser.add_argument("--rule_format_string", type=str, default=None, help="the answer format that should follow")

    # Resume mode reads the existing logistics history and starts after completed examples.
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    # Returning parsed arguments keeps CLI construction separate from evaluation execution.
    return parser.parse_args()


# This helper makes evaluation reproducible across libraries used by generation and judging.
def set_seed(seed):
    """Set random seeds and deterministic backend flags for evaluation.

    Locally, this pins PyTorch, CUDA, cuDNN, NumPy, Python hashing, and Python
    random. Globally, it keeps ALS, LatentSeek, and baseline runs comparable.
    """
    # This fixes CPU-side PyTorch randomness.
    torch.manual_seed(seed)
    # This fixes CUDA randomness across all visible devices.
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN avoids nondeterministic algorithm choices when possible.
    torch.backends.cudnn.deterministic = True
    # Disabling benchmarking prevents cuDNN from choosing algorithms based on runtime heuristics.
    torch.backends.cudnn.benchmark = False
    # Python hash seeding keeps hash-derived ordering stable.
    os.environ['PYTHONHASHSEED'] = str(seed)
    # NumPy receives the same seed for any helper logic that samples through NumPy.
    np.random.seed(seed)
    # Python's standard RNG receives the same seed for sampling helpers.
    random.seed(seed)


# This is the main evaluation loop used by the command-line entrypoint.
def main(args):
    """Evaluate the selected generation mode and write resumable metrics.

    Locally, the function loops over prompt formats and examples, dispatches to
    one generation backend, judges correctness, and records duration. Globally,
    it produces the accuracy/latency evidence used to compare ALS with
    LatentSeek, greedy CoT, and Self-Consistency.
    """
    # The boxed format rule is converted to a regex used by the LatentSeek reward model.
    if args.rule_format_string == "boxed":
        # This regex captures content inside a `\boxed{...}` final-answer form.
        rule_format_string = r'\\boxed{(.*)}'
    # Any other nonempty format string is rejected so the verifier rule is not ambiguous.
    else:
        # A provided but unknown rule would silently change reward semantics, so fail fast.
        if args.rule_format_string:
            # The error makes the accepted format vocabulary explicit.
            raise ValueError("Unknown format")
        # `None` disables format-rule reward checking.
        rule_format_string = None

    # A nonzero seed triggers deterministic setup before model loading and generation.
    if args.seed:
        # The helper centralizes all reproducibility knobs.
        set_seed(args.seed)

    # The explicit device wins; otherwise CUDA is used when available.
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # The causal LM is loaded once and reused across prompts/method calls.
    model = AutoModelForCausalLM.from_pretrained(
            # The model path can point to a local checkpoint or Hugging Face model id.
            args.model_name_or_path,
            # bfloat16 matches the intended efficient inference dtype for the target models.
            torch_dtype=torch.bfloat16,
            # `device_map=device` places the model according to the chosen device string.
            device_map=device
    )
    # Evaluation mode disables dropout and other training-only behavior.
    model.eval()
    # The tokenizer must match the model for chat templates and token ids to align.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Non-LatentSeek modes do not need verifier calls during generation.
    reward_model = None
    # LatentSeek uses verifier reward as the optimization objective.
    if args.generation_mode == "latentseek":
        # The reward wrapper reuses the loaded model to run Vera-style verifier prompts.
        reward_model = RewardModel(
                # The same model scores candidate answers with verifier prompts.
                model=model,
                # The tokenizer formats verifier messages and decodes verifier output.
                tokenizer=tokenizer,
                # The verifier generation runs on the selected device.
                device=device,
                # Dataset name informs reward/extraction assumptions.
                data_name=args.dataset,
                # Optional regex enforces final-answer formatting in the reward.
                rule_format_string=rule_format_string
                )

    # The repository defines two prompt formats per dataset: boxed CoT and JSON.
    num_prompts = 2
    # Running all prompts evaluates both formats.
    if args.prompts_to_run == 'all':
        # `range(2)` yields prompt indices 0 and 1.
        prompts_to_run = range(num_prompts)
    # Otherwise parse a comma-separated list such as `0` or `0,1`.
    else:
        # Each stripped component is converted into an integer prompt index.
        prompts_to_run = [int(p.strip()) for p in args.prompts_to_run.split(',')]

    # Final results collect per-prompt aggregate accuracy and latency for the summary table.
    final_results = {}

    # The outer loop repeats evaluation for every selected prompt format.
    for prompt_idx in prompts_to_run:
        # The banner separates prompt runs in terminal logs.
        print(f"\n{'='*20} EVALUATING PROMPT {prompt_idx+1} {'='*20}")

        # The model basename is used in vector templates and output paths.
        model_name = args.model_name_or_path.split("/")[-1]
        # Slashes in dataset names are replaced so directory names remain valid.
        data_name = args.dataset.replace("/", "-")

        # ALS modes require a precomputed offline vector for the current prompt.
        if args.generation_mode in ["als", "als_gated"]:
            # The template fills in model, dataset, and prompt identifiers to find the vector file.
            vector_path = args.vector_name_template.format(
                # The model name distinguishes vectors for different hidden-state geometries.
                model_name=model_name,
                # The dataset name distinguishes vectors trained on different reasoning distributions.
                dataset_name=data_name,
                # The prompt index distinguishes boxed and JSON state distributions.
                prompt_idx=prompt_idx
            )
            # If the vector is missing, this prompt cannot run ALS correctly.
            if not os.path.exists(vector_path):
                # Warning and continue let other prompts still run if their vectors exist.
                print(f"\nWarning: Could not find steering vector for prompt {prompt_idx} at {vector_path}. Skipping.")
                # Skipping avoids falling back to an incorrect or uninitialized vector.
                continue
            # Loading the tensor gives the online generation loop its fixed ALS direction.
            steering_vector = torch.load(vector_path)

        # The dataset is loaded after prompt selection so each prompt gets its own formatted text.
        dataset = get_dataset(args.dataset,
                              # The tokenizer applies the model-specific chat template.
                              tokenizer=tokenizer,
                              # The prompt index selects boxed CoT or JSON formatting.
                              prompt_idx=prompt_idx,
                              # The split selects the evaluation partition.
                              split=args.split)

        # A fixed subset file overrides contiguous start/end slicing for reproducible ablations.
        if args.fixed_subset_path:
            # Missing subset files should fail before any evaluation work is done.
            if not os.path.exists(args.fixed_subset_path):
                # The exception includes the requested path for diagnosis.
                raise FileNotFoundError(f"Fixed subset file not found at {args.fixed_subset_path}")
            # Opening the subset file reads one dataset index per line.
            with open(args.fixed_subset_path, 'r') as f:
                # Each stripped line is converted into an integer index for dataset selection.
                indices = [int(line.strip()) for line in f]
            # Hugging Face `select` creates a dataset view in the requested fixed order.
            dataset = dataset.select(indices)
            # This log confirms that the run is using a fixed subset rather than full split.
            print(f"Evaluating on a fixed subset of {len(dataset)} samples from {args.fixed_subset_path}")

        # Correctness booleans accumulate across processed examples and are saved for resume.
        results_history = []
        # Per-example durations accumulate for latency reporting and are saved for resume.
        time_history = []

        # The base output directory groups results by model and dataset.
        base_output_dir = f"{args.output_dir}/{model_name}-{data_name}"
        # The final output directory also includes generation mode and prompt index.
        output_dir = f"{base_output_dir}/{args.generation_mode}_eval/prompt{prompt_idx}"
        # Creating the directory ensures the logistics checkpoint path is writable.
        os.makedirs(output_dir, exist_ok=True)

        # By default, evaluation starts at the beginning or user-provided slice below.
        start_data_idx = 0

        # Resume mode restores prior metrics and starts after completed examples.
        if args.resume:
            # This log identifies the directory from which checkpoint data is loaded.
            print(f"Resume from {output_dir}")
            # The logistics checkpoint stores results and timing histories.
            logistics_path = f"{output_dir}/logistics.pt"
            # Resume only loads state if the checkpoint exists.
            if os.path.exists(logistics_path):
                # `torch.load` reconstructs the saved Python lists.
                logistics = torch.load(logistics_path)
                # Results history defaults to empty for backward-compatible checkpoints.
                results_history = logistics.get("results_history", [])
                # Time history defaults to empty for backward-compatible checkpoints.
                time_history = logistics.get("time_history", [])
                # The next index equals the number of already scored examples.
                start_data_idx = len(results_history)

        # Fixed subsets are already selected, so index slicing should cover the whole selected view.
        if args.fixed_subset_path:
            # Start at zero within the subset view.
            start_data_idx = 0
            # End at the subset length, ignoring original dataset indices.
            end_data_idx = len(dataset)
        # Without a fixed subset, combine resume progress and requested contiguous slice.
        else:
            # The larger of resume progress and requested start avoids re-evaluating examples.
            start_data_idx = max(start_data_idx, args.start_data_idx)
            # A missing end index means evaluate through the dataset end.
            end_data_idx = args.end_data_idx if args.end_data_idx is not None else len(dataset)
            # Clamping prevents out-of-range dataset access.
            end_data_idx = min(end_data_idx, len(dataset))

        # This log records the exact dataset slice for the prompt run.
        print(f"Start to evaluate {args.dataset} (Prompt {prompt_idx+1}) from {start_data_idx} to {end_data_idx}...")

        # The Python range drives deterministic example traversal.
        data_idx_list = range(start_data_idx, end_data_idx)
        # `tqdm` shows progress for this prompt's evaluation slice.
        for i in tqdm(data_idx_list, desc=f"Prompt {prompt_idx+1}"):
            # Dataset indexing yields the formatted prompt plus raw fields.
            example = dataset[i]
            # Ground-truth extraction normalizes labels for the selected dataset.
            true_answer = extract_true_answer(example["answer"], name=args.dataset)

            # Examples without a usable ground-truth answer cannot be scored.
            if true_answer is None:
                # Skipping avoids polluting accuracy history with unjudgeable items.
                continue

            # Start the timer immediately before generation so measured latency reflects method cost.
            start_time = time.time()

            # LatentSeek first performs normal generation, then optimizes hidden states for this one problem.
            if args.generation_mode == "latentseek":
                # Original generation returns answer text, hidden states, and token ids for optimization.
                original_output, hidden_states_list, input_ids = original_generation(
                        # The formatted prompt is the model input.
                        input_text=example["formatted"],
                        # The loaded model supplies hidden states.
                        model=model,
                        # The tokenizer encodes and decodes tokens.
                        tokenizer=tokenizer,
                        # The selected device keeps tensors on the model's hardware.
                        device=device,)

                # LatentSeek optimizes the hidden-state slice and decodes a final answer.
                final_output, _, _, _, _ = optimized_generation(
                        # The reward model supplies verifier scores for optimization.
                        reward_model=reward_model,
                        # The same language model projects hidden states and continues decoding.
                        model=model,
                        # The tokenizer decodes optimized token sequences.
                        tokenizer=tokenizer,
                        # Device placement is forwarded to the optimizer path.
                        device=device,
                        # The raw question is inserted into verifier prompts.
                        question=example["question"],
                        # The formatted prompt reconstructs the base input ids.
                        input_text=example["formatted"],
                        # The original answer is the baseline candidate and initial reward target.
                        original_answer=original_output,
                        # The original latent trajectory supplies trainable hidden states.
                        original_hidden_states_list=hidden_states_list,
                        # The original token ids provide prompt and answer context.
                        input_ids=input_ids,
                        # This bounds per-query optimization iterations.
                        max_num_steps=args.max_num_steps,
                        # This controls Adam step size for latent-state updates.
                        lr=args.lr,
                        # Optional clipping stabilizes latent gradients.
                        grad_clip=args.grad_clip,
                        # This selects the fraction of generated hidden states to update.
                        k=args.k,
                        # This enables early stop when verifier reward is sufficient.
                        reward_threshold=args.reward_threshold,
                )
            # Standard ALS uses the precomputed vector with no per-example backward passes.
            elif args.generation_mode == "als":
                # The ALS decoder returns a generated answer string.
                final_output = steered_generation(
                    # The model supplies hidden states and logits.
                    model=model,
                    # The tokenizer handles prompt encoding and answer decoding.
                    tokenizer=tokenizer,
                    # The formatted prompt is the autoregressive context.
                    input_text=example["formatted"],
                    # The loaded vector is the offline success-minus-failure direction.
                    steering_vector=steering_vector,
                    # Alpha scales the additive latent nudge.
                    alpha=args.alpha,
                    # The similarity threshold gates when the nudge is applied.
                    similarity_threshold=args.similarity_threshold,
                    # The token cap keeps generation bounded.
                    max_new_tokens=args.max_new_tokens,
                    # Device placement keeps tensors with the model.
                    device=device,
                )
            # ALS-Gated uses the same vector but zeroes alpha when partial JSON is fragile.
            elif args.generation_mode == "als_gated":
                # The gated decoder returns a generated answer string.
                final_output = gated_steered_generation(
                    # The model supplies hidden states and logits.
                    model=model,
                    # The tokenizer handles prompt encoding, JSON-prefix decoding, and answer decoding.
                    tokenizer=tokenizer,
                    # The formatted prompt usually asks for structured JSON output.
                    input_text=example["formatted"],
                    # The offline vector supplies the success direction.
                    steering_vector=steering_vector,
                    # Alpha is the maximum steering strength before gating.
                    alpha=args.alpha,
                    # The cosine threshold decides when steering would be considered.
                    similarity_threshold=args.similarity_threshold,
                    # The token cap keeps generation bounded.
                    max_new_tokens=args.max_new_tokens,
                    # Device placement keeps tensors with the model.
                    device=device,
                )
            # Greedy CoT uses ordinary model decoding with no activation steering.
            elif args.generation_mode == "greedy_cot":
                # The baseline returns one deterministic answer trace.
                final_output = greedy_cot_generation(
                    # The model generates the answer through Hugging Face `generate`.
                    model=model,
                    # The tokenizer encodes prompt and decodes completion.
                    tokenizer=tokenizer,
                    # The formatted prompt contains the selected CoT instruction.
                    input_text=example["formatted"],
                    # The token cap matches the other generation methods.
                    max_new_tokens=args.max_new_tokens,
                    # Device placement is forwarded to tokenization.
                    device=device,
                )
            # Self-Consistency samples multiple CoT traces and returns the majority-answer trace.
            elif args.generation_mode == "self_consistency":
                # The baseline returns the first sampled generation whose answer wins the vote.
                final_output = self_consistency_generation(
                    # The model generates each sampled trace.
                    model=model,
                    # The tokenizer encodes and decodes each trace.
                    tokenizer=tokenizer,
                    # The same prompt seeds every sample.
                    input_text=example["formatted"],
                    # `sc_k` sets the number of sampled traces.
                    k=args.sc_k,
                    # The token cap applies to each trace.
                    max_new_tokens=args.max_new_tokens,
                    # Device placement is forwarded to generation.
                    device=device,
                    # Dataset name controls answer extraction for voting.
                    data_name=args.dataset,
                    # Prompt index controls boxed versus JSON answer extraction.
                    prompt_idx=prompt_idx,
                    # Model name enables extractor quirks for specific checkpoints.
                    model_name=args.model_name_or_path
                )

            # Duration captures wall-clock generation time for this method/example pair.
            duration = time.time() - start_time

            # Extracting the final answer supports a quick missing-answer check before judging.
            final_answer = extract_answer(final_output,
                                             # Dataset rules determine numerical/symbolic parsing.
                                             data_name=args.dataset,
                                             # Prompt rules determine boxed versus JSON parsing.
                                             prompt_idx=prompt_idx,
                                             # Model-specific rules handle known output quirks.
                                             model_name=args.model_name_or_path)

            # Only nonempty extracted answers are sent to the full judge.
            if final_answer is not None:
                # `judge_answer` may use exact, numeric, symbolic, and dataset-specific equivalence checks.
                is_correct = judge_answer(
                        # The raw final output lets the judge re-extract under its own path.
                        final_output, true_answer, data_name=args.dataset, prompt_idx=prompt_idx)
            # Missing final answers are counted as incorrect.
            else:
                # This keeps accuracy conservative for unparseable generations.
                is_correct = False

            # Append correctness so running accuracy can be resumed and summarized.
            results_history.append(is_correct)
            # Append latency so average generation time can be resumed and summarized.
            time_history.append(duration)

            # The logistics checkpoint is saved after every example for crash-safe resume.
            torch.save({
                # Correctness history is enough to recompute accuracy.
                "results_history": results_history,
                # Time history is enough to recompute average latency.
                "time_history": time_history,
            # The checkpoint path is fixed per mode/prompt output directory.
            }, f"{output_dir}/logistics.pt")

        # Total scored samples is the denominator for final prompt accuracy.
        total_samples = len(results_history)
        # If at least one example was scored, compute aggregate metrics.
        if total_samples > 0:
            # Accuracy is the mean of boolean correctness values.
            final_accuracy = sum(results_history) / total_samples
            # Average time is the mean wall-clock generation duration.
            avg_time = sum(time_history) / total_samples
            # Store aggregates for the final summary table.
            final_results[prompt_idx] = {"accuracy": final_accuracy, "avg_time": avg_time}
            # This line reports final prompt accuracy in the terminal.
            print(f"Prompt {prompt_idx+1} Final accuracy: {final_accuracy:.4f}")
            # This line reports final prompt latency in the terminal.
            print(f"Prompt {prompt_idx+1} Average generation time: {avg_time:.4f} seconds")
        # No scored examples means the prompt produced no metrics.
        else:
            # This message avoids division by zero and explains the missing result row.
            print(f"No samples were evaluated for Prompt {prompt_idx+1}.")

    # The summary header names the generation mode whose prompt rows follow.
    print(f"\n{'='*20} FINAL SUMMARY ({args.generation_mode}) {'='*20}")
    # This row labels prompt number, accuracy, and average latency columns.
    print(f"| {'Prompt':<10} | {'Accuracy':<10} | {'Avg. Time':<15} |")
    # The separator row makes the printed table easier to scan.
    print(f"|{'-'*12}|{'-'*12}|{'-'*17}|")
    # Each prompt result becomes one summary table row.
    for prompt_idx, results in final_results.items():
        # The row converts zero-based prompt index into the one-based label used in paper tables.
        print(f"| {prompt_idx+1:<10} | {results['accuracy']:.4f}   | {results['avg_time']:.4f} sec      |")
    # The closing line visually terminates the summary block.
    print(f"{ '='*55}")


# Running this file directly starts the evaluation CLI.
if __name__ == "__main__":
    # Parse command-line flags before calling the main evaluator.
    args = parse_args()
    # The parsed namespace controls the whole evaluation run.
    main(args)
