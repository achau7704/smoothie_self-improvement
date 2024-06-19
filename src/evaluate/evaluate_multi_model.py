"""
This script does evaluation for multi model tasks.

For each method, it saves a file called {method}_{metric}.json, where {method} is the method name and {metric} is the metric name. 

Example command: python -m src.score_summarization --model falcon-1b --config_path configs/cnn_dailymail_0_shot.yaml --n_samples 4
"""

import argparse
import json
from pathlib import Path

from src.console import console
from src.evaluate.metrics import *
from src.evaluate.scorer import *
from src.utils import (construct_predictions_dir_path, load_data_config,
                       load_hf_dataset)

parser = argparse.ArgumentParser()
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
    help="Results directory",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Redo evaluation even if results already exist. Otherwise, we only evaluate methods/metrics which aren't already evaluated.",
)


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    predictions_dir = construct_predictions_dir_path(
        data_config, args, multi_model=True
    )

    # Load scores if they already exist
    scorer = Scorer(predictions_dir=predictions_dir, data_config=data_config, args=args)

    # Load dataset
    dataset = load_hf_dataset(
        dataset_name=data_config["dataset"],
        is_train=False,
        n_samples=data_config["test_size"],
        hf_cache_dir=args.hf_cache_dir,
    )
    references = get_references(dataset, data_config)
    print(dataset)

    predictions_files = list(predictions_dir.glob("*_test.json"))
    for predictions_fpath in predictions_files:
        console.log(predictions_fpath)
        if "perplexity" in str(predictions_fpath) or ("gens" in str(predictions_fpath) and "smoothie" not in str(predictions_fpath)):
            continue
        for metric in data_config["metrics"]:
            scorer.score_method(
                predictions_fpath=predictions_fpath,
                metric=metric,
                references=references,
            )

    # Compute train individual scores
    # train_dataset = load_hf_dataset(
    #    dataset_name=data_config["dataset"],
    #    is_train=True,
    #    n_samples=data_config["train_size"],
    #    hf_cache_dir=args.hf_cache_dir,
    # )
    # train_references = get_references(train_dataset, data_config)
    # predictions_file = predictions_dir / "individual_train.json"
    # for metric in data_config["metrics"]:
    #    scorer.score_train(predictions_fpath=predictions_file, metric=metric, references=train_references)

    console.log(json.dumps(scorer.summary_scores, indent=4))

    # Save scores
    scorer.save()


if __name__ == "__main__":
    args = parser.parse_args()
    console.log(args)
    main(args)
