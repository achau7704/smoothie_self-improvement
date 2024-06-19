"""
This script computes the perplexity.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json

import numpy as np
from tqdm.auto import tqdm

import evaluate
from src.console import console
from src.constants import HF_MODELS
from src.evaluate.metrics import *
from src.evaluate.scorer import *
from src.multi_model.utils import MODEL_GROUPS, load_predictions
from src.utils import (check_results_file, construct_predictions_dir_path,
                       load_data_config, load_hf_dataset, load_prompts)

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
    "--prompts_dir", default="prompts", type=str, help="Directory to save prompts to"
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo the generation if the results file already exists",
)
parser.add_argument(
    "--model",
    help="The models to use for predictions",
)
parser.add_argument(
    "--hf_cache_dir",
    default="cache",
    type=str,
    help="Directory to cache HF datasets to",
)

perplexity = evaluate.load("perplexity", module_type="metric")


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    console.log(data_config)
    predictions_dir = construct_predictions_dir_path(
        data_config, args, multi_model=True
    )
    _, test_prompts = load_prompts(data_config, args)
    console.log(f"Prompts shape: {test_prompts.shape}")
    output_fpath = predictions_dir / f"perplexity_{args.model}_test.json"

    # Check if the results file already exists
    if check_results_file(output_fpath) and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    # Load individual  generations.
    fpath = predictions_dir / f"{args.model}_test.json"
    with open(fpath, "r") as f:
        generations = json.load(f)["generations"]

    # Compute perplexity scores

    joined = [
        f"{prompt}{generation}" for prompt, generation in zip(test_prompts, generations)
    ]
    joined = np.array(joined)
    results = perplexity.compute(
        model_id=HF_MODELS[args.model],
        add_start_token=False,
        predictions=joined,
        batch_size=2,
    )
    perplexities = results["perplexities"]

    # Save test results to file
    results = {
        "perplexities": perplexities,
    }
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
