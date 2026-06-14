<!-- This badge links the repository documentation to the ALS paper that motivates the code workflow below. -->
[![arXiv](https://img.shields.io/badge/arXiv-2509.18116-blue.svg?style=flat)](https://arxiv.org/abs/2509.18116)

<!-- The title names the method implemented by the source files: amortize latent test-time optimization into reusable steering. -->
# Amortized Latent Steering (ALS) for Efficient LLM Reasoning

<!-- The centered figure gives readers the global offline/online ALS workflow before the command-level instructions. -->
<p align="center">
  <!-- The image is a visual summary of collecting hidden states, computing a vector, and steering generation. -->
  <img src="./img/als.png" alt="ALS" width="320"/>
<!-- Closing the paragraph preserves the existing centered rendering. -->
</p>

<!-- This paragraph explains the repository-level purpose: replace expensive per-query latent optimization with one precomputed vector. -->
This repository contains the implementation of **Amortized Latent Steering (ALS)**, a method for efficient and interpretable test-time guidance of large language models (LLMs). ALS pre-computes a single steering vector from a small training set to guide model reasoning, dramatically reducing test-time compute compared to instance-level optimization methods.

<!-- The horizontal rule separates the project overview from executable setup steps. -->
-----

<!-- Installation describes the local environment needed before running ALS collection/evaluation scripts. -->
## Installation

<!-- This sentence tells users to isolate dependencies so model/evaluation packages do not conflict with other projects. -->
First, create and activate a conda environment.

```bash
# Create and activate the conda environment
conda create -n als python=3.10
conda activate als

# Install core dependencies
pip install torch torchvision torchaudio
pip install transformers datasets tqdm accelerate termcolor matplotlib

# Install dependencies for math evaluation
cd src/extract_judge_answer/latex2sympy
pip install -e .
cd ../../.. 
pip install math-verify word2number
```

<!-- This horizontal rule separates setup from the ALS experiment workflow. -->
-----

<!-- Workflow explains how the paper's offline vector construction and online steering are represented as scripts. -->
## Workflow

<!-- This sentence maps the source files to the ALS algorithm: collect states, compute v, then decode with v. -->
The project is structured around a three-phase workflow: collecting states, computing a steering vector, and running guided generation.

<!-- Step 1 corresponds to estimating the correct/incorrect hidden-state distributions used by ALS. -->
### Step 1: Collect Hidden States (Offline)

<!-- This paragraph tells users that examples are generated and verifier-labeled before vector computation. -->
First, collect hidden states from a small training set and label the model's responses as "good" or "bad."

<!-- This bullet records the GSM8K collection size described in the paper and command examples. -->
  * **GSM8K:** first 1,000 training examples
<!-- This bullet records the MATH-500 collection size described in the paper and command examples. -->
  * **MATH-500:** first 500 training examples

<!-- This paragraph points users to the state-collection script and prompt-format control. -->
Run the `collect_states.py` script for each model and dataset combination. You can switch between prompts by changing `--solver_prompt_idx`.

```bash
# Example for Llama-3.1-8B on GSM8K with prompt 1
python src/collect_states.py \
    --model_name_or_path /path/to/Llama-3.1-8B-Instruct \
    --dataset "openai/gsm8k" \
    --split "train" \
    --solver_prompt_idx 0 \
    --end_data_idx 1000 \
    --resume
```

<!-- Step 2 converts labeled hidden-state tensors into the single vector used by ALS generation. -->
### Step 2: Compute Steering Vector

<!-- This sentence introduces the script that computes E[h_correct] - E[h_incorrect]. -->
Next, compute the steering vector from the collected hidden states.

```bash
# Example for Qwen on MATH-500
python src/compute_steering_vector.py \
    --states_dir "./output/Qwen2.5-7B-Instruct-MATH-500/state_collection/prompt1_thresh3/hidden_states/" \
    --output_file "./vectors/qwen_math500_p1_thresh3.pt"
```

<!-- Step 3 is the online phase where the precomputed vector is used during decoding. -->
### Step 3: Run Guided Generation (Online)

<!-- This sentence transitions from offline artifacts to online evaluation. -->
Finally, use the pre-computed steering vector to guide the LLM's generation at test time.

<!-- The ALS subsection documents the standard ungated hidden-state steering mode. -->
#### **ALS**

<!-- This sentence identifies the main constant-cost steering path implemented by `steered_generation.py`. -->
This is the main, efficient steering mode.

```bash
python src/main.py \
    --generation_mode als \
    --model_name_or_path /path/to/Llama-3.1-8B-Instruct \
    --dataset "openai/gsm8k" \
    --split "test" \
    --output_dir "./output" \
    --vector_name_template "./vectors/llama_gsm8k_p{prompt_idx}_thresh3.pt" \
    --prompts_to_run 0
```

<!-- The ALS-Gated subsection documents the structured-output variant implemented by `gated_generation.py`. -->
#### **ALS-Gated**

<!-- This sentence explains that gating is intended for JSON/schema-constrained prompts. -->
A variant that applies steering selectively, useful for structured outputs.

```bash
python src/main.py \
    --generation_mode als_gated \
    --model_name_or_path /path/to/Qwen2.5-7B-Instruct \
    --dataset "MATH-500" \
    --split "test" \
    --output_dir "./output" \
    --vector_name_template "./vectors/qwen_math500_p{prompt_idx}_thresh3.pt" \
    --prompts_to_run 1 \
    --fixed_subset_path ./subsets/HuggingFaceH4-MATH-500_test200.txt
```

<!-- This horizontal rule separates the main ALS workflow from comparison methods and ablations. -->
-----

<!-- This section documents the baselines used to contextualize ALS accuracy and efficiency. -->
## Baselines & Ablations

<!-- This sentence explains that comparison scripts live in the same `main.py` dispatcher. -->
This repository includes scripts to run baselines and ablation studies for comparison.

<!-- The baselines subsection covers greedy CoT, Self-Consistency, and LatentSeek. -->
### Baselines

<!-- This paragraph introduces the generation modes that do not use the ALS vector. -->
Run standard generation methods like **Greedy CoT**, **Self-Consistency**, or the original **LatentSeek** optimization.

```bash
# Example for LatentSeek baseline
python src/main.py \
    --generation_mode latentseek \
    --model_name_or_path /path/to/Qwen2.5-7B-Instruct \
    --dataset "openai/gsm8k" \
    --split "test" \
    --output_dir "./output" \
    --end_data_idx 500
```

<!-- The ablation subsection documents alpha sweeps that test steering strength sensitivity. -->
### Ablation Studies

<!-- This sentence explains the primary ALS hyperparameter explored by experiments. -->
You can perform sweeps to analyze the effect of steering strength (`--alpha`).

```bash
# Example: Alpha sweep on a 200-example subset of GSM8K
python src/main.py \
    --generation_mode als \
    --model_name_or_path /path/to/Qwen2.5-7B-Instruct \
    --dataset "openai/gsm8k" \
    --fixed_subset_path "./subsets/openai-gsm8k_test_200.txt" \
    --output_dir "./output" \
    --vector_name_template "./vectors/qwen_gsm8k_p{prompt_idx}.pt" \
    --alpha 0.3 # Test values like 0.0, 0.1, 0.3, 0.6
```

<!-- This horizontal rule separates evaluation commands from utility helpers. -->
-----

<!-- Utility commands support reproducible subsets and progress visualization around ALS runs. -->
## Utilities

<!-- This subsection covers fixed index files consumed by `main.py --fixed_subset_path`. -->
### Creating Subsets

<!-- This sentence introduces deterministic subset generation for fair ablation comparisons. -->
Generate fixed subsets for consistent ablation testing.

```bash
# GSM8K, 100 examples
python create_subsets.py \
    --dataset_name "openai/gsm8k" \
    --split "test" --n_samples 100 \
    --output_dir "./subsets"
```

<!-- This subsection documents plotting/monitoring support for saved logistics files. -->
### Visualization

<!-- This sentence describes plotting from saved per-example histories. -->
Generate plots to visualize rolling accuracy from a log file.

```bash
python create_chart.py \
    --log_file "./output/Meta-Llama-3.1-8B-Instruct-openai/gsm8k/als_eval/prompt_0/logistics.pt" \
    --output_dir "./charts"
```

<!-- This horizontal rule separates usage utilities from repository orientation. -->
-----

<!-- This section gives readers a quick source-file map for the ALS implementation. -->
## Key Files

<!-- This bullet points to the evaluation dispatcher that chooses ALS, ALS-Gated, LatentSeek, or baselines. -->
  * `src/main.py`: Main script for running generation and evaluation.
<!-- This bullet points to the offline state collection script. -->
  * `src/collect_states.py`: Collects hidden states for ALS.
<!-- This bullet points to the offline vector computation script. -->
  * `src/compute_steering_vector.py`: Computes the steering vector.
<!-- This bullet points to the standard online ALS implementation. -->
  * `src/steered_generation.py`: Core implementation of ALS.
<!-- This bullet points to the per-query optimization baseline. -->
  * `src/opt_generation.py`: Original LatentSeek implementation.
<!-- This bullet points to the prompt templates that define boxed and JSON formats. -->
  * `src/prompts/`: Contains all prompt templates.

<!-- Notes capture command-line details that affect resuming and prompt selection. -->
## Notes

<!-- This bullet tells users how to continue interrupted collection/evaluation runs. -->
  * Use the `--resume` flag to continue interrupted runs.
<!-- This bullet explains how to select prompt index 0, 1, or both. -->
  * The `--prompts_to_run` argument can be used to select a specific prompt index.
<!-- This bullet points to subset and sweep commands for reproducible experiments. -->
  * Subsets and sweep commands are provided for full reproducibility.

<!-- This horizontal rule separates project notes from citation metadata. -->
-----

<!-- Citation gives the paper reference for the ALS method implemented here. -->
## Citation

```bibtex
@misc{egbuna2025amortizedlatentsteeringlowcost,
      title={Amortized Latent Steering: Low-Cost Alternative to Test-Time Optimization}, 
      author={Nathan Egbuna and Saatvik Gaur and Kevin Zhu and Sunishchal Dev and Ashwinee Panda and Maheep Chaudhary},
      year={2025},
      eprint={2509.18116},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.18116}, 
}
```

<!-- This contact line gives the maintainer email from the original README. -->
For any questions, please email `egbunanathan@gmail.com`.
