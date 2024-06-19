"""
"""

import warnings

import pandas as pd

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from src.console import console
from src.constants import *
from src.evaluate.metrics import *
from src.evaluate.scorer import *
from src.multi_model.utils import MODEL_GROUPS
from src.utils import (get_input_text, load_hf_dataset)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--predictions_dir",
    default="generative_ensembles_data/multi_model_results",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--save_dir",
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
parser.add_argument(
    "--type",
    choices=["knn", "fixed"],
    required=True,
    help="The type of Smoothie to use. See file docstring for more information.",
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
    for data_config_path in data_configs_fpaths:
        data_configs.append(
            yaml.load(Path(data_config_path).read_text(), Loader=yaml.FullLoader)
        )

    # Get test generations and datasets
    test_datasets = []
    test_texts = []
    for data_config in data_configs:
        test_df = load_hf_dataset(
            dataset_name=data_config["dataset"],
            is_train=False,
            n_samples=data_config["test_size"],
            hf_cache_dir=args.hf_cache_dir,
        )
        test_datasets.append(test_df)
        test_texts.extend(get_input_text(test_df, data_config))

    all_test_generations = []
    for model in MODEL_GROUPS[args.model_group]:
        all_test_generations.append([])
        for data_config in data_configs:
            predictions_dir = Path(args.predictions_dir) / data_config["prompt"]
            predictions_fpath = predictions_dir / f"{model}_test.json"
            all_test_generations[-1].extend(
                json.loads(predictions_fpath.read_text())["generations"]
            )
    all_test_generations = np.array(all_test_generations).T
    console.log(f"loaded test generations: {all_test_generations.shape}")

    # We build giant dataframe with rows corresponding to samples and columns corresponding to models + references
    TRAIN_DATA = {}

    # Load test references
    train_dataset_names = []
    train_references = []
    train_texts = []
    for data_config in data_configs:
        train_df = load_hf_dataset(
            dataset_name=data_config["dataset"],
            is_train=True,
            n_samples=data_config["train_size"],
            hf_cache_dir=args.hf_cache_dir,
        )
        references = get_references(train_df, data_config)
        train_references.extend(references)
        train_dataset_names.extend([data_config["dataset"]] * len(references))
        train_texts.extend(get_input_text(train_df, data_config))

    TRAIN_DATA["references"] = train_references
    TRAIN_DATA["dataset_names"] = train_dataset_names
    TRAIN_DATA["texts"] = train_texts

    # Load different train model predictions
    for model in MODEL_GROUPS[args.model_group]:
        model_train_generations = []
        for data_config in data_configs:
            predictions_dir = Path(args.predictions_dir) / data_config["prompt"]
            predictions_fpath = predictions_dir / f"{model}_train.json"
            model_train_generations.extend(
                json.loads(predictions_fpath.read_text())["generations"]
            )
        TRAIN_DATA[model] = model_train_generations

    # Build dataframe
    TRAIN_DATA_DF = pd.DataFrame(TRAIN_DATA)
    console.log(f"Train data shape: {TRAIN_DATA_DF.shape}")

    # Build array of scores. Shape: (n_train_samples, n_models)
    train_scores = []
    for model in MODEL_GROUPS[args.model_group]:
        scores = []
        for dataset_name in TRAIN_DATA_DF["dataset_names"].unique():
            subset_df = TRAIN_DATA_DF[TRAIN_DATA_DF["dataset_names"] == dataset_name]
            metric_func = dataset_to_metrics[dataset_name]
            scores.extend(
                metric_func(subset_df[model].tolist(), subset_df["references"].tolist())
            )
        train_scores.append(scores)
    train_scores = np.array(train_scores).T
    console.log(f"Train scores shape: {train_scores.shape}")

    # Load model
    model_name = "all-mpnet-base-v2"
    embedding_model = SentenceTransformer(model_name)
    embedding_model.max_seq_length = 512
    console.log(f"Loaded embedding model: {embedding_model}")

    # Embed train texts
    train_embeddings = embedding_model.encode(TRAIN_DATA["texts"])
    test_embeddings = embedding_model.encode(test_texts)

    # Compute generations
    all_labeled_baseline_generations = []
    for _ in range(10):
        trial_generations = []
        # Sample indices
        indices = np.random.choice(TRAIN_DATA_DF.shape[0], 50, replace=False)
        console.log(indices)

        if args.type == "knn":
            # Use KNN to further filter down
            sampled_train_embeddings = train_embeddings[indices]
            nbrs = NearestNeighbors(n_neighbors=20, algorithm="auto")
            nbrs.fit(sampled_train_embeddings)
            distances, nn_train_indices = nbrs.kneighbors(test_embeddings)

            for idx in range(len(nn_train_indices)):
                nn_idx = nn_train_indices[idx]
                nn_scores = train_scores[nn_idx].mean(axis=0)
                assert len(nn_scores) == len(MODEL_GROUPS[args.model_group])
                best_model = np.argmax(nn_scores)
                generation = all_test_generations[idx, best_model]
                trial_generations.append(generation)

        else:
            # compute best model from sample
            nn_scores = train_scores[indices].mean(axis=0)
            best_model = np.argmax(nn_scores)
            for idx in range(len(test_texts)):
                generation = all_test_generations[idx, best_model]
                trial_generations.append(generation)
        all_labeled_baseline_generations.append(trial_generations)

    # Save results
    results = {
        "generations": all_labeled_baseline_generations,
    }
    output_dir = Path(args.save_dir) / f"{args.model_group}_{args.task_group}"
    output_dir.mkdir(exist_ok=True, parents=True)
    if args.type == "knn":
        output_fpath = output_dir / f"labeled_oracle_knn_test.json"
    else:
        output_fpath = output_dir / f"labeled_oracle_fixed_test.json"
    console.log(f"Saving to {output_fpath}")
    output_fpath.write_text(json.dumps(results, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
