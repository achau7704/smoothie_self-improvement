"""
This script generates outputs for multi model ensembles.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from src.console import console
from src.constants import HF_MODEL_MAX_LENGTHS, HF_MODELS
from src.utils import (construct_predictions_dir_path, get_generation_output,
                       load_data_config, load_hf_model, load_prompts,
                       make_list_with_shape)

from transformers import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
parser.add_argument(
    "--data_config_path",
    type=str,
    help="Path to config file. This should be a yaml file",
)
parser.add_argument(
    "--hf_cache_dir",
    default="cache",
    type=str,
    help="Directory to cache HF datasets to",
)
parser.add_argument(
    "--results_dir",
    default="generative_ensembles_data/multi_model_results",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--prompts_dir",
    default="multi_model_prompts",
    type=str,
    help="Directory to save prompts to",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo the generation if the results file already exists",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Runs the script in test mode. This will only generate predictions for two samples.",
)
parser.add_argument(
    "--n_generations",
    default=1,
    type=int,
    help="For each model we produce n_generations per sample. Default is 1.",
)
parser.add_argument(
    "--temperature",
    default=0.0,
    type=float,
    help="Temperature for generations. Only used when n_generations > 1.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Random seed if n_generations > 1.",
)

def generate_predictions(args, data_config, prompts, model, tokenizer, output_fpath):
    """
    Generate predictions for a set of prompts.

    Args:
        args (argparse.Namespace): arguments from the command line
        data_config (dict): data config
        prompts (list): list of prompts
        model (transformers.PreTrainedModel): LLM
        tokenizer (transformers.PreTrainedTokenizer): tokenizer
    """

    # Check if the results file already exists
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file {output_fpath} already exists. Skipping.")
        return
    else:
        console.log(f"Will save results to: {output_fpath}")
    
    sequence_texts = []


    if args.n_generations > 1:
        gen_params = {"temperature": args.temperature, "do_sample": True}
    else:
        gen_params = {"do_sample": False}

    for sample_idx in tqdm(range(len(prompts))):
        prompt_encodings = tokenizer(
            prompts[sample_idx],
            return_tensors="pt",
            truncation=True,
            max_length=HF_MODEL_MAX_LENGTHS[args.model],
        ).to(args.device)

        for i in range(args.n_generations):
            output = model.generate(
                **prompt_encodings,
                max_new_tokens=data_config["max_new_tokens"],
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                **gen_params
            )

            # Get the token ids corresponding to the generation. output["sequences"] is a tensor of shape (batch_size, seq_len)
            sequence_texts.append(
                tokenizer.decode(get_generation_output(prompt_encodings, output))
            )
        if args.test and sample_idx == 1:
            break

    # Save to file
    results = {
        "generations": sequence_texts,
    }
    output_fpath.write_text(json.dumps(results, indent=4))


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    train_prompts, test_prompts = load_prompts(data_config, args)

    # For multimodel, there's only one set of prompts
    train_prompts = train_prompts
    test_prompts = test_prompts
    console.log(
        f"Loaded prompts for {data_config['dataset']}. Train: {train_prompts.shape} Test: {test_prompts.shape}"
    )
    predictions_dir = construct_predictions_dir_path(
        data_config, args, multi_model=True
    )

    # Check if the results file already exists
    if args.n_generations == 1:
        train_output_fpath = predictions_dir / f"{args.model}_train.json"
        test_output_fpath = predictions_dir / f"{args.model}_test.json"
    else:
        train_output_fpath = predictions_dir / f"{args.model}_{args.n_generations}_gens_train.json"
        test_output_fpath = predictions_dir / f"{args.model}_{args.n_generations}_gens_test.json"

    if train_output_fpath.exists() and test_output_fpath.exists() and not args.redo:
        console.log(
            f"Results file {train_output_fpath} and {test_output_fpath} already exists. Skipping."
        )
        return
    else:
        console.log(
            f"Will save results to: {train_output_fpath} and {test_output_fpath}"
        )

    # Load model and move onto device
    model, tokenizer = load_hf_model(args)
    console.log(f"Model max length: {HF_MODEL_MAX_LENGTHS[args.model]}")

    if args.n_generations > 1:
        set_seed(args.seed)
        assert args.temperature != 0.0

    # Generate predictions for train and test prompts
    generate_predictions(
        args,
        data_config,
        train_prompts,
        model,
        tokenizer,
        train_output_fpath,
    )
    generate_predictions(
        args,
        data_config,
        test_prompts,
        model,
        tokenizer,
        test_output_fpath,
    )


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
