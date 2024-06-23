"""
This script implements multimodel and multidataset baselines.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.console import console
from src.constants import *
from src.evaluate.metrics import *
from src.evaluate.scorer import *
from src.data_utils import load_hf_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_dir",
    default="generative_ensembles_data/dataset_combination_results",
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
parser.add_argument(
    "--task_group",
    choices=["acc_group", "rouge2_group"],
    help="The models to use for predictions",
)
parser.add_argument(
    "--hf_cache_dir",
    default="cache",
    type=str,
    help="Directory to cache HF datasets to",
)


acc_tasks = [
    "dataset_configs/squad.yaml",
    "dataset_configs/trivia_qa_knowledge.yaml",
    "dataset_configs/definition_extraction.yaml",
]
rouge2_tasks = [
    "dataset_configs/cnn_dailymail_0_shot.yaml",
    "dataset_configs/xsum_0_shot.yaml",
    "dataset_configs/e2e_nlg_1_shot.yaml",
    "dataset_configs/web_nlg_1_shot.yaml",
]


dataset_to_metrics = {
    "squad": squad_acc,
    "trivia_qa": trivia_qa_acc,
    "definition_extraction": definition_extraction_acc,
    "web_nlg": compute_rouge2_score,
    "e2e_nlg": compute_rouge2_score,
    "xsum": compute_rouge2_score,
    "cnn_dailymail": compute_rouge2_score,
}


def main(args):
    # Load config
    if args.task_group == "acc_group":
        data_configs_fpaths = acc_tasks
    elif args.task_group == "rouge2_group":
        data_configs_fpaths = rouge2_tasks
    else:
        raise ValueError("Invalid task group")

    data_configs = []
    for dataset_config in data_configs_fpaths:
        data_configs.append(
            yaml.load(Path(dataset_config).read_text(), Loader=yaml.FullLoader)
        )

    # We build giant dataframe with rows corresponding to samples and columns corresponding to models + references
    DATA = {}
    # Load test references
    test_dataset_names = []
    test_references = []
    for data_config in data_configs:
        _, test_df = load_hf_dataset(
            config=data_config,
            hf_cache_dir=args.hf_cache_dir,
        )
        references = get_references(test_df, data_config)
        test_references.extend(references)
        test_dataset_names.extend([data_config["dataset"]] * len(references))
    DATA["references"] = test_references
    DATA["dataset_names"] = test_dataset_names

    # Load different model predictions
    results_dir = Path(args.results_dir) / f"{args.model_group}_{args.task_group}"
    for fpath in results_dir.glob("*_test.json"):
        generations = json.loads(fpath.read_text())
        generations = np.array(generations["generations"])
        if "pick_random" in fpath.stem:
            for trial in range(generations.shape[0]):
                DATA[f"{fpath.stem}_{trial}"] = generations[trial].ravel().tolist()
        elif "labeled_oracle" in fpath.stem:
            for trial in range(generations.shape[0]):
                DATA[f"{fpath.stem}_{trial}"] = generations[trial].ravel().tolist()
        else:
            method_name = fpath.stem
            DATA[method_name] = generations.ravel().tolist()

    # Build dataframe
    DATA_DF = pd.DataFrame(DATA)
    RESULTS = {}
    for method_name in DATA_DF.columns:
        if method_name in ["dataset_names", "references"]:
            continue

        scores = []
        for dataset_name in DATA_DF["dataset_names"].unique():
            subset_df = DATA_DF[DATA_DF["dataset_names"] == dataset_name]
            metric_func = dataset_to_metrics[dataset_name]
            scores.extend(
                metric_func(
                    subset_df[method_name].tolist(), subset_df["references"].tolist()
                )
            )

        RESULTS[method_name] = scores

    # print results
    pick_random_scores = []
    labeled_oracle_knn_scores = []
    for method_name, scores in RESULTS.items():
        if "pick_random" in method_name:
            pick_random_scores.append(np.mean(scores))
        elif "labeled_oracle_knn" in method_name:
            labeled_oracle_knn_scores.append(np.mean(scores))
        else:
            console.log(f"{method_name}: {np.mean(scores)}")
    console.log(labeled_oracle_knn_scores)
    console.log(f"Pick random scores: {np.mean(pick_random_scores)}")
    console.log(f"Labeled oracle knn scores: {np.mean(labeled_oracle_knn_scores)}")


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
