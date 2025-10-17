from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from data import get_dataset
from tqdm import tqdm
from rewards.reward import RewardModel
from ori_generation import original_generation
from opt_generation import optimized_generation
from steered_generation import steered_generation
from gated_generation import gated_steered_generation
from baselines import greedy_cot_generation, self_consistency_generation
import os
from extract_judge_answer import extract_answer, extract_true_answer, judge_answer
import argparse
import numpy as np
import random
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., 'openai/gsm8k', 'MATH-500')")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--prompts_to_run", type=str, default="all", help="Which prompts to run, e.g., '0', '1', or 'all'")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=None, help="End index of the data to evaluate (exclusive)")
    parser.add_argument("--fixed_subset_path", type=str, default=None, help="Path to a file with a fixed list of indices for evaluation.")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose print statements")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # generation mode
    parser.add_argument("--generation_mode", type=str, default="latentseek", choices=["latentseek", "steered", "greedy_cot", "self_consistency", "als_gated"], help="Generation mode to use")

    # latentseek args
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--max_num_steps", type=int, default=10, help="Number of optimization iterations")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Number of generated tokens")
    parser.add_argument("--reward_threshold", type=float, default=-0.2, help="Threshold for reward to stop optimization")

    # steered generation args
    parser.add_argument("--vector_name_template", type=str, default="./vectors/{model_name}_{dataset_name}_p{prompt_idx}.pt", help="Template for steering vector filenames.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Strength of the steering intervention")
    parser.add_argument("--similarity_threshold", type=float, default=0.1, help="Cosine similarity threshold to trigger steering")

    # self-consistency args
    parser.add_argument("--sc_k", type=int, default=5, help="Number of samples for self-consistency")

    # device
    parser.add_argument("--device", type=str, default=None)

    # format reward
    parser.add_argument("--rule_format_string", type=str, default=None, help="the answer format that should follow")

    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    return parser.parse_args()


def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


# evaluate function 
def main(args):
    """
    Evaluate model
    """
    if args.rule_format_string == "boxed":
        rule_format_string = r'\\boxed{(.*)}'
    else:
        if args.rule_format_string:
            raise ValueError("Unknown format")
        rule_format_string = None
    
    if args.seed:
        set_seed(args.seed)
    
    # set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load reward model if in latentseek mode
    reward_model = None
    if args.generation_mode == "latentseek":
        reward_model = RewardModel(
                model=model, 
                tokenizer=tokenizer, 
                device=device,
                data_name=args.dataset,
                rule_format_string=rule_format_string
                )

    # --- main loop for prompts ---
    num_prompts = 2
    if args.prompts_to_run == 'all':
        prompts_to_run = range(num_prompts)
    else:
        prompts_to_run = [int(p.strip()) for p in args.prompts_to_run.split(',')]

    final_results = {}

    for prompt_idx in prompts_to_run:
        print(f"\n{'='*20} EVALUATING PROMPT {prompt_idx+1} {'='*20}")

        model_name = args.model_name_or_path.split("/")[-1]
        data_name = args.dataset.replace("/", "-")

        # for steered modes, check for vector and skip prompt if not found
        if args.generation_mode in ["steered", "als_gated"]:
            vector_path = args.vector_name_template.format(
                model_name=model_name, 
                dataset_name=data_name, 
                prompt_idx=prompt_idx
            )
            if not os.path.exists(vector_path):
                print(f"\nWarning: Could not find steering vector for prompt {prompt_idx} at {vector_path}. Skipping.")
                continue
            steering_vector = torch.load(vector_path)

        # load dataset for the specific prompt
        dataset = get_dataset(args.dataset, 
                              tokenizer=tokenizer,
                              prompt_idx=prompt_idx,
                              split=args.split)
        
        # filter dataset if a fixed subset is provided
        if args.fixed_subset_path:
            if not os.path.exists(args.fixed_subset_path):
                raise FileNotFoundError(f"Fixed subset file not found at {args.fixed_subset_path}")
            with open(args.fixed_subset_path, 'r') as f:
                indices = [int(line.strip()) for line in f]
            dataset = dataset.select(indices)
            print(f"Evaluating on a fixed subset of {len(dataset)} samples from {args.fixed_subset_path}")

        # for logging
        results_history = []
        time_history = []
        
        base_output_dir = f"{args.output_dir}/{model_name}-{data_name}"
        output_dir = f"{base_output_dir}/{args.generation_mode}_eval/prompt{prompt_idx}"
        os.makedirs(output_dir, exist_ok=True)

        start_data_idx = 0
        
        if args.resume:
            print(f"Resume from {output_dir}")
            logistics_path = f"{output_dir}/logistics.pt"
            if os.path.exists(logistics_path):
                logistics = torch.load(logistics_path)
                results_history = logistics.get("results_history", [])
                time_history = logistics.get("time_history", [])
                start_data_idx = len(results_history)

        # if using a fixed subset, ignore start/end data_idx from args
        if args.fixed_subset_path:
            start_data_idx = 0
            end_data_idx = len(dataset)
        else:
            start_data_idx = max(start_data_idx, args.start_data_idx)
            end_data_idx = args.end_data_idx if args.end_data_idx is not None else len(dataset)
            end_data_idx = min(end_data_idx, len(dataset))
        
        print(f"Start to evaluate {args.dataset} (Prompt {prompt_idx+1}) from {start_data_idx} to {end_data_idx}...")

        data_idx_list = range(start_data_idx, end_data_idx)
        for i in tqdm(data_idx_list, desc=f"Prompt {prompt_idx+1}"):
            example = dataset[i]
            true_answer = extract_true_answer(example["answer"], name=args.dataset)

            if true_answer is None:
                continue

            start_time = time.time()

            if args.generation_mode == "latentseek":
                original_output, hidden_states_list, input_ids = original_generation(
                        input_text=example["formatted"],
                        model=model,
                        tokenizer=tokenizer,
                        device=device,)
                
                final_output, _, _, _, _ = optimized_generation(
                        reward_model=reward_model,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        question=example["question"],
                        input_text=example["formatted"],
                        original_answer=original_output,
                        original_hidden_states_list=hidden_states_list, 
                        input_ids=input_ids,
                        max_num_steps=args.max_num_steps,
                        lr=args.lr,
                        grad_clip=args.grad_clip,
                        k=args.k,
                        reward_threshold=args.reward_threshold,
                )
            elif args.generation_mode == "steered":
                final_output = steered_generation(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=example["formatted"],
                    steering_vector=steering_vector,
                    alpha=args.alpha,
                    similarity_threshold=args.similarity_threshold,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
            elif args.generation_mode == "als_gated":
                final_output = gated_steered_generation(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=example["formatted"],
                    steering_vector=steering_vector,
                    alpha=args.alpha,
                    similarity_threshold=args.similarity_threshold,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
            elif args.generation_mode == "greedy_cot":
                final_output = greedy_cot_generation(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=example["formatted"],
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
            elif args.generation_mode == "self_consistency":
                final_output = self_consistency_generation(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=example["formatted"],
                    k=args.sc_k,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    data_name=args.dataset,
                    prompt_idx=prompt_idx,
                    model_name=args.model_name_or_path
                )
            
            duration = time.time() - start_time
            
            final_answer = extract_answer(final_output, 
                                             data_name=args.dataset, 
                                             prompt_idx=prompt_idx, 
                                             model_name=args.model_name_or_path)

            if final_answer is not None:
                is_correct = judge_answer(
                        final_output, true_answer, data_name=args.dataset, prompt_idx=prompt_idx)
            else:
                is_correct = False

            results_history.append(is_correct)
            time_history.append(duration)
            
            torch.save({
                "results_history": results_history,
                "time_history": time_history,
            }, f"{output_dir}/logistics.pt")

        total_samples = len(results_history)
        if total_samples > 0:
            final_accuracy = sum(results_history) / total_samples
            avg_time = sum(time_history) / total_samples
            final_results[prompt_idx] = {"accuracy": final_accuracy, "avg_time": avg_time}
            print(f"Prompt {prompt_idx+1} Final accuracy: {final_accuracy:.4f}")
            print(f"Prompt {prompt_idx+1} Average generation time: {avg_time:.4f} seconds")
        else:
            print(f"No samples were evaluated for Prompt {prompt_idx+1}.")

    # --- print final summary table ---
    print(f"\n{'='*20} FINAL SUMMARY ({args.generation_mode}) {'='*20}")
    print(f"| {'Prompt':<10} | {'Accuracy':<10} | {'Avg. Time':<15} |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*17}|")
    for prompt_idx, results in final_results.items():
        print(f"| {prompt_idx+1:<10} | {results['accuracy']:.4f}   | {results['avg_time']:.4f} sec      |")
    print(f"{ '='*55}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
