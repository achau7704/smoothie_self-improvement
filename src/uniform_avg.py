"""
This script implements our method.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.aggregation import Aggregator
from src.console import console
from src.constants import *
from src.utils import (construct_predictions_dir_path, get_generation_output,
                       load_data_config, load_hf_model, load_prompts,
                       make_list_with_shape)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
parser.add_argument(
    "--data_config_path", type=str, help="Path to the data yaml config."
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
    "--redo", action="store_true", help="Redo the generation if the file already exists"
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Runs the script in test mode. This will only generate predictions for two samples.",
)


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    _, test_prompts = load_prompts(data_config, args)
    console.log(
        f"Loaded prompts for {data_config['dataset']}. Test: {test_prompts.shape}"
    )
    predictions_dir = construct_predictions_dir_path(data_config, args)

    # Check if the results file already exists
    output_fpath = predictions_dir / f"{UNIFORM_AVG}_test.json"
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file {output_fpath} already exists. Skipping.")
        return
    else:
        console.log(f"Will save results to {output_fpath}")

    # Create aggregator
    agg = Aggregator(
        device=args.device,
        model_name=args.model,
        hf_cache_dir=args.hf_cache_dir,
        method=UNIFORM_AVG,
    )

    dataset_texts = []  # Decoded text
    for sample_idx in tqdm(range(len(test_prompts)), ncols=100):
        sample_prompts = test_prompts[sample_idx]
        text = agg.aggregate(
            prompts=sample_prompts,
            max_tokens=data_config["max_new_tokens"],
            on_probabilities=False,
        )
        dataset_texts.append(text)

        if args.test and sample_idx == 1:
            break

    results = {"generations": dataset_texts}
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
