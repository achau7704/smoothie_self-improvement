"""
This script implements Smoothie for multimodels. There are several configurations: 

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
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.aggregation import Aggregator
from src.console import console
from src.constants import *
from src.model import Smoothie
from src.multi_model.utils import load_predictions
from src.utils import (check_results_file, clean_generation,
                       construct_predictions_dir_path,
                       embed_individual_generations, get_generation_output,
                       get_input_text, load_data_config, load_hf_dataset,
                       load_hf_model, load_prompts, make_list_with_shape)

parser = argparse.ArgumentParser()
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
    default="generative_ensembles_data/multi_model_results",
    type=str,
    help="Directory to save results to",
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
    "--model_group",
    help="The models to use for predictions",
)
parser.add_argument(
    "--use_full_text_embeddings",
    action="store_true",
    help="If set to true, Smoothie operates on embeddings of [input text, generation text]. Otherwise, Smoothie uses the embedding of the generation text only.",
)
parser.add_argument("--k", help="Nearest neighbors size", type=int)
parser.add_argument(
    "--n_generations",
    default=1,
    type=int,
    help="If not equal to 1, we replace k-nearest neighbors smoothing with computation over the n_generations per sample",
)


def main(args):
    # Load data config, prompts, and create output directory
    data_config = load_data_config(args)
    predictions_dir = construct_predictions_dir_path(
        data_config, args, multi_model=True
    )

    # Construct the output file based on the parameters

    output_fpath = str(predictions_dir) + f"/smoothie_select_{args.type}_{args.model_group}_"

    no_flags = True 
    if args.type == "sample_dependent" and args.n_generations == 1:
        output_fpath += f"{args.k}_"
        no_flags = False
    elif args.n_generations > 1: 
        output_fpath += f"{args.n_generations}_gens_"
        no_flags=False

    if args.use_full_text_embeddings:
        output_fpath += f"full_embeddings_"
        no_flags = False 

    if no_flags:
        output_fpath += "new_"
    output_fpath += f"test.json"

    output_fpath = Path(output_fpath)

    #if args.use_full_text_embeddings and args.type == "sample_dependent":
    #    output_fpath = (
    #        predictions_dir
    #        / f"smoothie_select_{args.type}_{args.model_group}_k_{args.k}_full_embeddings_test.json"
    #    )
    #elif args.use_full_text_embeddings:
    #    output_fpath = (
    #        predictions_dir
    #        / f"smoothie_select_{args.type}_{args.model_group}_full_embeddings_test.json"
    #    )
    #elif args.type == "sample_dependent":
    #    output_fpath = (
    #        predictions_dir
    #        / f"smoothie_select_{args.type}_{args.model_group}_k_{args.k}_test.json"
    #    )
    #else:
    #    output_fpath = (
    #        predictions_dir
    #        / f"smoothie_select_{args.type}_{args.model_group}_new_test.json"
    #    )

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
    test_generations = load_predictions(predictions_dir, "test", args, final=False)

    # use the temperature=0 version for selection...
    final_test_generations = load_predictions(predictions_dir, "test", args)

    if args.use_full_text_embeddings:
        test_text = []
        console.log(f"Using both input and generation text for embedding.")
        for i, gen_per_sample in enumerate(test_generations):
            test_text.append([test_df.iloc[i][data_config['doc_key']] + gen_per_model for gen_per_model in gen_per_sample])
        test_text = np.array(test_text)
    else:
        test_text = test_generations

    # Compute test embeddings
    test_generation_embeddings = embed_individual_generations(
        individual_generations=test_text,
        data_config=data_config,
        model_name=model_name,
    )

    print(f"test_generation_embeddings.shape: {test_generation_embeddings.shape}")

    # Useful constants
    n_samples = int(len(test_generation_embeddings) / args.n_generations)
    n_voters = test_generation_embeddings.shape[1]
    embed_dim = test_generation_embeddings.shape[2]

    # Learn smoothie weights
    if args.type == "sample_dependent":
        # Fit KNN
        if args.n_generations == 1:
            nbrs = NearestNeighbors(n_neighbors=args.k, algorithm="auto")
            nbrs.fit(test_dataset_embeddings)

            # Find the k nearest neighbors
            distances, test_indices = nbrs.kneighbors(test_dataset_embeddings)
            test_indices = test_indices[:, 1:]  # ignore first one

            smoothie_dataset_weights = []
            for sample_idx in range(n_samples):
                if args.k == 1:
                    nn_embeds = test_generation_embeddings[sample_idx].reshape((1, n_voters, -1))
                else:
                    nn_embeds = test_generation_embeddings[test_indices[sample_idx]]
                smoothie = Smoothie(n_voters=n_voters, dim=embed_dim)
                smoothie.fit(nn_embeds)
                smoothie_dataset_weights.append(smoothie.theta)
            smoothie_dataset_weights = np.array(smoothie_dataset_weights)
        else:
            smoothie_dataset_weights = []
            for sample_idx in range(n_samples):
                # use the multiple generations per sample 
                embs_per_sample = test_generation_embeddings[sample_idx * args.n_generations:(sample_idx+1)*args.n_generations]
                smoothie = Smoothie(n_voters=n_voters, dim=embed_dim)
                smoothie.fit(embs_per_sample)
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
    for sample_idx in range(n_samples):
        max_prompt_idx = smoothie_dataset_weights[sample_idx].argmax()
        text = final_test_generations[sample_idx][max_prompt_idx]

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
