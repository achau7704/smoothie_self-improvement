"""
This script implements Smoothie. There are several configurations: 

type
-----
Options: sample_dependent, sample_independent
The type of Smoothie to use. sample_dependent uses a different set of weights for each sample, while sample_independent uses the same set of weights for all samples.

operation
---------
Options: select, merge
The operation to perform. select selects the best voter using computed weights, while merge uses the weights to merge the outputs of all voters.

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
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.console import console
from src.constants import *
from src.model import Smoothie
from src.utils import (check_results_file, clean_generation,
                       construct_predictions_dir_path,
                       embed_individual_generations, get_generation_output,
                       get_input_text, load_data_config, load_hf_dataset,
                       load_hf_model, load_prompts, make_list_with_shape)

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
parser.add_argument(
    "--type",
    choices=["sample_dependent", "sample_independent"],
    required=True,
    help="The type of Smoothie to use. See file docstring for more information.",
)
parser.add_argument(
    "--operation",
    choices=["select", "merge"],
    required=True,
    help="The operation to perform. See file docstring for more information.",
)


def get_top_k_indices(arr, k):
    """
    Returns the indices of the top k elements in an array.
    """
    return np.argpartition(arr, -k)[-k:]


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    _, test_prompts = load_prompts(data_config, args)
    console.log(
        f"Loaded prompts for {data_config['dataset']}. Test: {test_prompts.shape}"
    )
    predictions_dir = construct_predictions_dir_path(data_config, args)

    # Construct the output file based on the parameters
    if args.operation == "select":
        output_fpath = predictions_dir / f"smoothie_select_{args.type}_test.json"
    else:
        output_fpath = (
            predictions_dir / f"smoothie_merge_{args.type}_{args.max_k}_test.json"
        )

    # Check if the results file already exists
    if check_results_file(output_fpath) and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return

    console.log(f"Saving results to {output_fpath}")

    # Create embedding model
    model_name = "all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    console.log(f"Loaded embedding model: {model_name}")

    # Load test dataset and compute embeddings
    test_df = load_hf_dataset(
        dataset_name=data_config["dataset"],
        is_train=False,
        n_samples=data_config["test_size"],
        hf_cache_dir=args.hf_cache_dir,
    )
    test_dataset_embeddings = model.encode(get_input_text(test_df, data_config))

    # Load individual prompt generations from test split
    with open(predictions_dir / "individual_test.json", "r") as f:
        individual_generations_test = json.load(f)
    individual_generations_test = np.array(individual_generations_test["generations"])

    # Compute test embeddings
    test_generation_embeddings = embed_individual_generations(
        individual_generations=individual_generations_test,
        data_config=data_config,
        model=model,
    )

    # Useful constants
    n_samples = len(test_dataset_embeddings)
    n_voters = test_generation_embeddings.shape[1]
    embed_dim = test_generation_embeddings.shape[2]

    # Learn smoothie weights
    if args.type == "sample_dependent":
        # Fit KNN
        nbrs = NearestNeighbors(n_neighbors=20, algorithm="auto")
        nbrs.fit(test_dataset_embeddings)

        # Find the k nearest neighbors
        distances, test_indices = nbrs.kneighbors(test_dataset_embeddings)

        smoothie_dataset_weights = []
        for sample_idx in tqdm(range(len(test_prompts)), ncols=100):
            nn_embeds = test_generation_embeddings[test_indices[sample_idx]]
            smoothie = Smoothie(n_voters=n_voters, dim=embed_dim)
            smoothie.fit(nn_embeds)
            smoothie_dataset_weights.append(smoothie.theta)
        smoothie_dataset_weights = np.array(smoothie_dataset_weights)
    else:
        # Learn a single set of weights for all samples
        smoothie = Smoothie(
            n_voters=test_generation_embeddings.shape[1],
            dim=test_generation_embeddings.shape[2],
        )
        smoothie.fit(test_generation_embeddings)
        console.log(f"Smoothie weights: {smoothie.theta}")
        smoothie_dataset_weights = np.tile(
            smoothie.theta, (test_generation_embeddings.shape[0], 1)
        )

    dataset_texts = []  # Decoded text
    for sample_idx in tqdm(range(len(test_prompts)), ncols=100):
        max_prompt_idx = smoothie_dataset_weights[sample_idx].argmax()
        text = individual_generations_test[sample_idx][max_prompt_idx]

        dataset_texts.append(text)

        if args.test and sample_idx == 1:
            break

    results = {
        "generations": dataset_texts,
        "smoothie_weights": smoothie_dataset_weights.tolist(),
    }
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
