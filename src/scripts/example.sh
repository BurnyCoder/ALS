# This example selects the GSM8K dataset for a math-reasoning evaluation run.
PATH_TO_DATA="openai/gsm8k" # The dataset string must contain a supported key such as "AIME_2024", "gsm8k", or "MATH-500".
# This path selects the local or mounted model checkpoint whose hidden states will be evaluated.
PATH_TO_MODEL="/home/plm/Qwen2.5-7B-Instruct" # The model path should point to the same architecture used for any matching ALS vectors.
# `rho` is passed to `--k`, which controls the fraction of hidden states optimized by the LatentSeek-style baseline.
rho=0.2 # Larger values optimize more latent positions and usually increase per-query cost.
# `lr` controls Adam step size for hidden-state optimization in the LatentSeek baseline.
lr=0.05 # Higher learning rates make larger latent updates and can destabilize optimization.
# This selects the structured JSON prompt variant for the solver prompt.
solver_prompt_idx=1 # Prompt 0 is boxed CoT and prompt 1 is JSON, despite the historical "boxex" typo in this note.

# The command is stored as an array so each argument can be commented without breaking shell continuations.
cmd=(
    # The Python executable runs the main evaluator.
    python main.py
    # `--dataset` tells `data.py` which benchmark and answer columns to use.
    --dataset "$PATH_TO_DATA"
    # `--model_name_or_path` loads the causal LM and tokenizer.
    --model_name_or_path "$PATH_TO_MODEL"
    # `--output_dir` stores logistics checkpoints and run outputs.
    --output_dir ./output
    # `--k` forwards the fractional hidden-state update length for LatentSeek.
    --k "$rho"
    # `--lr` forwards the latent optimizer learning rate.
    --lr "$lr"
    # `--solver_prompt_idx` is intended to choose boxed versus JSON prompt formatting.
    --solver_prompt_idx "$solver_prompt_idx"
    # `--device` places model and tensors on CUDA.
    --device "cuda"
)

# Expanding the array executes the exact argument sequence defined above.
"${cmd[@]}"
