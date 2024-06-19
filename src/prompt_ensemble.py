"""
This script generates outputs for a fixed set of prompts over a dataset. Outputs are generated according to greedy decoding.

It saves results to individual_prompt_generations.json, which is a dictionary with two keys:
    - generations: a list of lists, where the [i, j] entry corresponds to the LLM's generation for the jth prompt on the ith example
    - generations_token_ids: a list of lists, where the [i, j] entry corresponds to the LLM's generation for the jth prompt on the ith example, but in token ids instead of text
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
    default="generative_ensembles_data/results",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--prompts_dir", default="prompts", type=str, help="Directory to save prompts to"
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

    n_samples, n_prompts = prompts.shape
    sequence_texts = make_list_with_shape(len(prompts), len(prompts[0]))
    with tqdm(total=n_prompts * n_samples) as pbar:
        for sample_idx in range(n_samples):
            for prompt_idx in range(n_prompts):
                prompt_encodings = tokenizer(
                    prompts[sample_idx][prompt_idx],
                    return_tensors="pt",
                    truncation=True,
                    max_length=HF_MODEL_MAX_LENGTHS[args.model],
                ).to(args.device)
                output = model.generate(
                    **prompt_encodings,
                    max_new_tokens=data_config["max_new_tokens"],
                    do_sample=False,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                )

                # Get the token ids corresponding to the generation. output["sequences"] is a tensor of shape (batch_size, seq_len)
                sequence_texts[sample_idx][prompt_idx] = tokenizer.decode(
                    get_generation_output(prompt_encodings, output)
                )
                pbar.update(1)
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
    console.log(
        f"Loaded prompts for {data_config['dataset']}. Train: {train_prompts.shape} Test: {test_prompts.shape}"
    )
    predictions_dir = construct_predictions_dir_path(data_config, args)

    # Load model and move onto device
    model, tokenizer = load_hf_model(args)
    console.log(f"Model max length: {HF_MODEL_MAX_LENGTHS[args.model]}")

    # Generate predictions for train and test prompts
    generate_predictions(
        args,
        data_config,
        train_prompts,
        model,
        tokenizer,
        predictions_dir / "individual_train.json",
    )
    generate_predictions(
        args,
        data_config,
        test_prompts,
        model,
        tokenizer,
        predictions_dir / "individual_test.json",
    )


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
