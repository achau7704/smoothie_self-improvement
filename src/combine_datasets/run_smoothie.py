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
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from src.combine_datasets.utils import (acc_tasks, compute_embedding,
                                        rouge2_tasks)
from src.console import console
from src.model import Smoothie
from src.multi_model.utils import MODEL_GROUPS
from src.utils import (embed_individual_generations, get_input_text,
                       load_hf_dataset)

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
    choices=["sample_dependent", "sample_independent"],
    required=True,
    help="The type of Smoothie to use. See file docstring for more information.",
)
parser.add_argument("--k", help="Nearest neighbors size", type=int, default=20)


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

    # Load test dataset
    test_datasets = []
    for data_config in data_configs:
        test_df = load_hf_dataset(
            dataset_name=data_config["dataset"],
            is_train=False,
            n_samples=data_config["test_size"],
            hf_cache_dir=args.hf_cache_dir,
            doc_key=data_config["doc_key"],
        )
        test_datasets.append(test_df)

    # Build text inputs
    text_inputs = []
    for test_df, data_config in zip(test_datasets, data_configs):
        text_inputs.extend(get_input_text(test_df, data_config))
    console.log(f"loaded test data: {len(text_inputs)}")

    # Load test generations
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
    console.log(f"loaded generations: {all_test_generations.shape}")

    ############### SMOOTHIE
    model_name = "all-mpnet-base-v2"

    # Compute  embeddings of the test dataset
    test_dataset_embeddings = embedding_model.encode(text_inputs)

    # Compute embeddings of the generations
    test_generation_embeddings = []
    for llm in MODEL_GROUPS[args.model_group]:
        model_embeddings = []
        for data_config in data_configs:
            predictions_dir = Path(args.predictions_dir) / data_config["prompt"]
            predictions_fpath = predictions_dir / f"{llm}_test.json"
            generations = np.array(
                json.loads(predictions_fpath.read_text())["generations"]
            )
            generations = generations.reshape(-1, 1)

            # Each of the embeddings is an array of shape (n_task_samples, 768)
            embeddings = embed_individual_generations(
                individual_generations=generations,
                model=embedding_model,
            )
            model_embeddings.append(embeddings)
        model_embeddings = np.concatenate(model_embeddings, axis=0)
        test_generation_embeddings.append(model_embeddings)
    test_generation_embeddings = np.concatenate(test_generation_embeddings, axis=1)
    console.log(f"loaded generations: {test_generation_embeddings.shape}")

    n_samples = test_generation_embeddings.shape[0]
    n_voters = test_generation_embeddings.shape[1]
    embed_dim = test_generation_embeddings.shape[2]

    # Learn smoothie weights
    if args.type == "sample_dependent":
        # Fit KNN
        nbrs = NearestNeighbors(n_neighbors=args.k, algorithm="auto")
        nbrs.fit(test_dataset_embeddings)

        # Find the k nearest neighbors
        distances, test_indices = nbrs.kneighbors(test_dataset_embeddings)
        test_indices = test_indices[:, 1:]  # ignore first one

        smoothie_dataset_weights = []
        for sample_idx in range(n_samples):
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

    # Get the best voter for each sample
    dataset_texts = []  # Decoded text
    for sample_idx in range(n_samples):
        max_prompt_idx = smoothie_dataset_weights[sample_idx].argmax()
        text = all_test_generations[sample_idx][max_prompt_idx]
        dataset_texts.append(text)

    # Save to file
    results = {
        "generations": dataset_texts,
        "smoothie_weights": smoothie_dataset_weights.tolist(),
    }

    output_dir = Path(args.save_dir) / f"{args.model_group}_{args.task_group}"
    output_dir.mkdir(exist_ok=True, parents=True)
    if args.type == "sample_dependent":
        output_fpath = output_dir / f"smoothie_{args.type}_k={args.k}_test.json"
    else:
        output_fpath = output_dir / f"smoothie_{args.type}_test.json"
    console.log(f"Saving to {output_fpath}")
    output_fpath.write_text(json.dumps(results, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
