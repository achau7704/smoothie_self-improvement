"""
This script implements the PairRM baseline, where we use the PairRM model to select the best generation from a set of generations.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json

import llm_blender
import numpy as np
import torch
import yaml
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
from tqdm.auto import tqdm

from src.console import console
from src.constants import HF_MODELS
from src.utils import (check_results_file, clean_generation,
                       construct_predictions_dir_path, load_data_config,
                       load_hf_model, load_prompts, make_list_with_shape)

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
    default="generative_ensembles_data/benchmark_results",
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
    _, test_prompts = load_prompts(data_config, args)
    predictions_dir = construct_predictions_dir_path(data_config, args)

    # Check if the results file already exists
    output_fpath = predictions_dir / "pair_rm_test.json"
    if check_results_file(output_fpath) and not args.redo:
        console.log(f"Results file {output_fpath} already exists. Skipping.")
        return

    # Load individual test generations.
    individual_generations_fpath = predictions_dir / "individual_test.json"
    with open(individual_generations_fpath, "r") as f:
        individual_generations = json.load(f)
    individual_generations = np.array(individual_generations["generations"])
    console.log(
        f"Loaded individual generations of shape {individual_generations.shape}"
    )

    # Clean generations
    for i in range(individual_generations.shape[0]):
        for j in range(individual_generations.shape[1]):
            individual_generations[i][j] = clean_generation(
                individual_generations[i][j], data_config
            )

    # Load blender
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")  # load ranker checkpoint

    instructions = [
        prompts[0] for prompts in test_prompts
    ]  # Use the prompts from prompt_0 as the instructions
    ranks = blender.rank(instructions, individual_generations)
    best_gens = (
        get_topk_candidates_from_ranks(ranks, individual_generations, top_k=1)
        .ravel()
        .tolist()
    )

    # Save to file
    results = {
        "generations": best_gens,
    }
    output_fpath.write_text(json.dumps(results, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
