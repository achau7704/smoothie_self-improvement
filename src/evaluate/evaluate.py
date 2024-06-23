"""
This script does evaluation for multi model tasks.

For each method, it saves a file called {method}_{metric}.json, where {method} is the method name and {metric} is the metric name. 

Example command: python -m src.score_summarization --model falcon-1b --config_path configs/cnn_dailymail_0_shot.yaml --n_samples 4
"""

import argparse
import json
from pathlib import Path

import pandas as pd 

from src.console import console
from src.evaluate.metrics import *
from src.evaluate.scorer import *
from src.data_utils import construct_processed_dataset_paths
from src.utils import (construct_predictions_dir_path, check_args, load_data_config)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument(
    "--dataset_config",
    type=str,
    help="Path to config file. This should be a yaml file",
)
parser.add_argument(
    "--data_dir",
    default="smoothie_data/datasets",
    type=str,
    help="Directory with data files",
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="Results directory",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo evaluation even if results already exist. Otherwise, we only evaluate methods/metrics which aren't already evaluated.",
)
parser.add_argument(
    "--model_group",
    help="The models to use for predictions if we are doing multi-model",
)
parser.add_argument(
    "--multi_prompt",
    action="store_true",
)
parser.add_argument(
    "--multi_model",
    action="store_true",
)


def main(args):
    check_args(args)
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    predictions_dir = construct_predictions_dir_path(
        data_config, args, args.model
    )

    # Load scores if they already exist
    scorer = Scorer(predictions_dir=predictions_dir, data_config=data_config, args=args)

    # Load dataset
    _, test_data_path = construct_processed_dataset_paths(args)
    dataset = pd.read_csv(test_data_path)

    references = get_references(dataset)

    predictions_files = list(predictions_dir.glob("*_test.json"))
    for predictions_fpath in predictions_files:
        console.log(predictions_fpath)
        if "gens" in str(predictions_fpath) and "smoothie" not in str(predictions_fpath) or "_test_" in str(predictions_fpath): 
            continue
        for metric in data_config["metrics"]:
            scorer.score_method(
                predictions_fpath=predictions_fpath,
                metric=metric,
                references=references,
            )

    console.log(json.dumps(scorer.summary_scores, indent=4))

    # Save scores
    scorer.save()


if __name__ == "__main__":
    args = parser.parse_args()
    console.log(args)
    main(args)
