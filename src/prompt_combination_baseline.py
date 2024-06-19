"""
This script implements the prompt combination baseline, where the outputs of different prompts are combined via a prompt.

python -m src.prompt_combination_baseline \
    --model pythia-1b \
    --data_config_path dataset_configs/e2e_1_shot.yaml \
    --results_dir generative_ensembles_data/benchmark_results
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
from src.constants import HF_MODELS
from src.utils import (check_results_file, construct_predictions_dir_path,
                       get_generation_output, load_data_config, load_hf_model,
                       load_prompts, make_list_with_shape)

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


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    predictions_dir = construct_predictions_dir_path(data_config, args)

    # Check if the results file already exists
    output_fpath = predictions_dir / "prompt_combination_test.json"
    if check_results_file(output_fpath) and not args.redo:
        return

    # Load individual test generations.
    individual_generations_fpath = predictions_dir / "individual_test.json"
    with open(individual_generations_fpath, "r") as f:
        individual_generations = json.load(f)
    individual_generations = np.array(individual_generations["generations"])
    console.log(
        f"Loaded individual generations of shape {individual_generations.shape}"
    )

    # Load model and move onto device
    model, tokenizer = load_hf_model(args)

    sequence_texts = []
    for sample_idx in tqdm(range(len(individual_generations))):
        prompt = "Below are different texts from a prompt. Combine them into a single text which captures information from all of them:\n\n"
        for generation in individual_generations[sample_idx]:
            formatted_generation = generation.strip().split("\n")[0]
            prompt += f"{formatted_generation}\n"
        prompt += "\nThe combined generation is:"

        prompt_encodings = tokenizer(prompt, return_tensors="pt").to(args.device)
        output = model.generate(
            **prompt_encodings,
            max_new_tokens=data_config["max_new_tokens"],
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
        )

        # Get the plain text of the generation.
        sequence_texts.append(
            tokenizer.decode(get_generation_output(prompt_encodings, output))
        )

    # Save to file
    results = {
        "generations": sequence_texts,
    }
    output_fpath.write_text(json.dumps(results, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
