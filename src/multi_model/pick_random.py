"""
This script implements the pick-random baseline over multiple model generations.
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
from src.multi_model.utils import load_predictions
from src.utils import (check_results_file, construct_predictions_dir_path,
                       get_generation_output, load_data_config, load_hf_model,
                       load_prompts, make_list_with_shape)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_config_path",
    type=str,
    help="Path to config file. This should be a yaml file",
)
parser.add_argument(
    "--results_dir",
    default="generative_ensembles_data/multi_model_results",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo the generation if the results file already exists",
)
parser.add_argument(
    "--model_group",
    help="The models to use for predictions",
)


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    predictions_dir = construct_predictions_dir_path(
        data_config, args, multi_model=True
    )

    # Check if the results file already exists
    output_fpath = predictions_dir / f"pick_random_{args.model_group}_test.json"
    if check_results_file(output_fpath) and not args.redo:
        return

    # Load individual test generations.
    test_generations = load_predictions(predictions_dir, "test", args)

    sequence_texts = []
    for trial in range(10):
        trial_generations = []
        for sample_idx in tqdm(range(len(test_generations))):
            # Select a random generation from the individual generations.
            generation_idx = np.random.randint(test_generations.shape[1])
            generation = test_generations[sample_idx][generation_idx]
            trial_generations.append(generation)
        sequence_texts.append(trial_generations)

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
