"""
This script implements the labeled KNN method for the multi-model setting.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from src.console import console
from src.evaluate.metrics import *
from src.evaluate.scorer import *
from src.multi_model.utils import load_predictions
from src.utils import (check_results_file, construct_predictions_dir_path,
                       get_input_text, load_data_config, load_hf_dataset)

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
parser.add_argument(
    "--label_train_n_trials",
    default=10,
    type=int,
    help="Number of trials to run for train oracle sampling method.",
)
parser.add_argument(
    "--label_train_sample_size",
    default=50,
    type=int,
    help="Number of trials to run for train oracle sampling method.",
)
parser.add_argument(
    "--hf_cache_dir",
    default="cache",
    type=str,
    help="Directory to cache HF datasets to",
)


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    console.log(data_config)
    predictions_dir = construct_predictions_dir_path(
        data_config, args, multi_model=True
    )
    output_fpath = predictions_dir / f"labeled_knn_{args.model_group}_test.json"

    # Check if the results file already exists
    if check_results_file(output_fpath) and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    # Load individual train and test generations.
    train_generations = load_predictions(predictions_dir, "train", args)
    test_generations = load_predictions(predictions_dir, "test", args)
    n_models = test_generations.shape[1]

    # Load train and test dataset
    train_dataset = load_hf_dataset(
        dataset_name=data_config["dataset"],
        is_train=True,
        n_samples=data_config["train_size"],
        hf_cache_dir=args.hf_cache_dir,
    )
    train_references = get_references(train_dataset, data_config)

    test_dataset = load_hf_dataset(
        dataset_name=data_config["dataset"],
        is_train=False,
        n_samples=data_config["test_size"],
        hf_cache_dir=args.hf_cache_dir,
    )

    # Embed train and test generations
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    console.log(f"Loaded embedding model: {model_name}")
    train_dataset_embeddings = model.encode(get_input_text(train_dataset, data_config))
    test_dataset_embeddings = model.encode(get_input_text(test_dataset, data_config))

    # Evaluate train generations
    if data_config["dataset"] in [
        "e2e_nlg",
        "cnn_dailymail",
        "xsum",
        "common_gen",
        "web_nlg",
    ]:
        metric_func = METRIC_FUNCS["rouge2"]
    elif data_config["dataset"] in ["squad"]:
        metric_func = METRIC_FUNCS["squad_acc"]
    elif data_config["dataset"] in ["trivia_qa"]:
        metric_func = METRIC_FUNCS["trivia_qa_acc"]
    elif data_config["dataset"] in ["definition_extraction"]:
        metric_func = METRIC_FUNCS["definition_extraction_acc"]
    else:
        raise ValueError(f"Dataset {data_config['dataset']} not supported.")

    generations = []  # Final generations. Shape: (10, n_test)
    for _ in tqdm(range(10)):
        # Sample labeled train set
        sampled_idxs = np.random.choice(
            len(train_generations), args.label_train_sample_size
        )
        sampled_references = [train_references[idx] for idx in sampled_idxs]
        sampled_train_embeddings = train_dataset_embeddings[sampled_idxs]

        # Compute scores for each model
        model_scores = []
        for prompt_idx in range(n_models):
            sampled_generations = train_generations[sampled_idxs, prompt_idx]
            cleaned_generations = clean_generations(sampled_generations, data_config)
            scores = metric_func(cleaned_generations, sampled_references)
            model_scores.append(scores)
        model_scores = np.array(model_scores).T  # shape: (sample_size, n_models)

        # Train KNN
        nbrs = NearestNeighbors(n_neighbors=20, algorithm="auto")
        nbrs.fit(sampled_train_embeddings)

        # Find the k nearest neighbors
        _, sample_train_indices = nbrs.kneighbors(test_dataset_embeddings)

        # Compute best model on average over these indices
        trial_generations = []
        for idx in range(len(sample_train_indices)):
            sample_model_scores = model_scores[sample_train_indices[idx]].mean(axis=0)
            max_idxs = np.where(sample_model_scores == sample_model_scores.max())[0]
            selected_idx = np.random.choice(max_idxs)
            trial_generations.append(test_generations[idx, selected_idx])
        generations.append(trial_generations)

    results = {
        "generations": generations,
    }
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
