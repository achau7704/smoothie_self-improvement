"""
Computes perplexity score for each prompt and generation.
"""


import argparse
import json
import os

import numpy as np

import evaluate
from src.console import console
from src.constants import HF_MODELS
from src.evaluate.metrics import *
from src.evaluate.scorer import *
from src.utils import (check_results_file, clean_generations,
                       construct_predictions_dir_path, load_data_config,
                       load_hf_dataset, load_prompts)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
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
    default="generative_ensembles_data/benchmark_results",
    type=str,
    help="Results directory",
)
parser.add_argument(
    "--prompts_dir", default="prompts", type=str, help="Directory to save prompts to"
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo evaluation even if results already exist. Otherwise, we only evaluate methods/metrics which aren't already evaluated.",
)

perplexity = evaluate.load("perplexity", module_type="metric")


def compute_perplexity(prompts, generations, dataset_config, args):
    """
    Computes perplexity of generations given prompts.
    """
    n_samples, n_prompts = generations.shape

    # Flatten prompts and generations
    flat_prompts = np.array(prompts).flatten()
    flat_generations = np.array(generations).flatten()
    flat_generations = clean_generations(flat_generations, data_config=dataset_config)

    # Join prompts and generations
    joined = [
        f"{prompt}{generation}"
        for prompt, generation in zip(flat_prompts, flat_generations)
    ]
    joined = np.array(joined)

    # Compute perplexity
    results = perplexity.compute(
        model_id=HF_MODELS[args.model],
        add_start_token=False,
        predictions=joined,
        batch_size=2,
    )
    perplexities = np.array(results["perplexities"])

    # Reshape perplexities
    perplexities = perplexities.reshape(n_samples, n_prompts)
    return perplexities


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    console.log(data_config)
    train_prompts, test_prompts = load_prompts(data_config, args)
    predictions_dir = construct_predictions_dir_path(data_config, args)
    train_output_fpath = predictions_dir / f"perplexity_train.json"
    test_output_fpath = predictions_dir / f"perplexity_test.json"

    # Check if the results file already exists
    if (
        check_results_file(train_output_fpath)
        and check_results_file(test_output_fpath)
        and not args.redo
    ):
        console.log(
            f"Results file already exists at {train_output_fpath}/{test_output_fpath}. Skipping."
        )
        return

    # Load train and test generations
    with open(predictions_dir / "individual_train.json", "r") as f:
        individual_generations_train = json.load(f)
    individual_generations_train = np.array(individual_generations_train["generations"])

    with open(predictions_dir / "individual_test.json", "r") as f:
        individual_generations_test = json.load(f)
    individual_generations_test = np.array(individual_generations_test["generations"])

    # Compute perplexity scores
    train_perplexities = compute_perplexity(
        train_prompts, individual_generations_train, data_config, args
    )
    test_perplexities = compute_perplexity(
        test_prompts, individual_generations_test, data_config, args
    )

    # Extract perplexity minimizing generations
    train_min_perplexity_idx = np.argmin(train_perplexities, axis=1)
    selected_train_generations = individual_generations_train[
        np.arange(len(train_perplexities)), train_min_perplexity_idx
    ]
    test_min_perplexity_idx = np.argmin(test_perplexities, axis=1)
    selected_test_generations = individual_generations_test[
        np.arange(len(test_perplexities)), test_min_perplexity_idx
    ]

    # Save train results to file
    results = {
        "perplexities": train_perplexities.tolist(),
        "generations": selected_train_generations.tolist(),
    }
    console.log(f"Saving results to {train_output_fpath}")
    train_output_fpath.write_text(json.dumps(results, default=int, indent=4))

    # Save test results to file
    results = {
        "perplexities": test_perplexities.tolist(),
        "generations": selected_test_generations.tolist(),
    }
    console.log(f"Saving results to {test_output_fpath}")
    test_output_fpath.write_text(json.dumps(results, default=int, indent=4))


if __name__ == "__main__":
    args = parser.parse_args()
    console.log(args)

    # Set HF cache dir in environment variable
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir
    main(args)
