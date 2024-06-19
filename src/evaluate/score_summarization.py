"""
This script does evaluation for summarization tasks.

For each method, it saves a file called {method}_{metric}.json, where {method} is the method name and {metric} is the metric name. 

Example command: python -m src.score_summarization --model falcon-1b --config_path configs/cnn_dailymail_0_shot.yaml --n_samples 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import HF_MODELS
from src.evaluate.metrics import (compute_rouge1_score, compute_rouge2_score,
                                  compute_rougeL_score)
from src.utils import load_hf_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
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
    default="generative_ensembles_data/results",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--prompts_dir", default="prompts", type=str, help="Directory to save prompts to"
)
parser.add_argument(
    "--n_samples", default=100, type=int, help="Number of samples to use"
)

METRICS = {
    "rouge1": compute_rouge1_score,
    "rouge2": compute_rouge2_score,
    "rougeL": compute_rougeL_score,
}


def clean_generations(generations):
    """
    Cleans generations by stripping whitespace and splitting on newlines.
    """
    cleaned_generations = []
    for generation in generations:
        cleaned_generation = generation.strip().split("\n")[0]
        cleaned_generations.append(cleaned_generation)
    return cleaned_generations


def evaluate_individual_generations(results_dir, dataset, config):
    """
    Computes and saves metrics for individual-prompt baseline.
    """

    # We save result to a subdirectory of results_dir called scores
    scores_dir = results_dir / "scores"
    scores_dir.mkdir(exist_ok=True, parents=True)

    predictions_fpath = results_dir / f"individual_prompts_generations.json"
    predictions_dict = json.loads(predictions_fpath.read_text())
    all_generations = np.array(predictions_dict["generations"])
    n_samples, n_prompts = all_generations.shape
    references = dataset[config["reference_key"]].tolist()
    assert (
        len(references) == n_samples
    ), f"Number of samples in dataset ({len(references)}) does not match number of samples in generations ({n_samples})"

    for metric in METRICS:
        dict_to_save = {"means": {}, "scores": []}
        for prompt_idx in range(n_prompts):
            generations = all_generations[:, prompt_idx]
            generations = clean_generations(generations)
            scores = METRICS[metric](generations, references)
            dict_to_save["means"][f"individual_prompt_{prompt_idx}"] = np.mean(scores)
            dict_to_save["scores"].append(scores)

        out_fpath = scores_dir / f"individual_prompts_{metric}.json"
        out_fpath.write_text(json.dumps(dict_to_save, indent=4))
        print(f"Saved scores to {out_fpath}")

        # Score the pick-random baseline. For this, we randomly select one of the prompts for each sample. We run 10 trials of this and report the mean and standard deviation.
        scores_arr = np.array(dict_to_save["scores"])  # shape: (n_prompts, n_samples)
        idxs = np.random.randint(0, n_prompts, size=(10, n_samples))
        sampled_scores = scores_arr[idxs, np.arange(n_samples)]
        means = np.mean(sampled_scores, axis=1)
        std = np.std(means)
        dict_to_save = {
            "mean": np.mean(means),
            "std": std,
        }
        out_fpath = scores_dir / f"pick_random_{metric}.json"
        out_fpath.write_text(json.dumps(dict_to_save, indent=4))


def evaluate_method(results_dir, dataset, config, method_name):
    """
    Evaluates generations from method
    """
    # We save result to a subdirectory of results_dir called scores
    scores_dir = results_dir / "scores"
    scores_dir.mkdir(exist_ok=True, parents=True)

    # Load predictions of method
    predictions_fpath = results_dir / f"{method_name}_generations.json"
    predictions_dict = json.loads(predictions_fpath.read_text())
    generations = predictions_dict["generations"]
    generations = clean_generations(generations)
    references = dataset[config["reference_key"]].tolist()
    assert len(references) == len(
        generations
    ), f"Number of samples in dataset ({len(references)}) does not match number of samples in generations ({len(generations)})"

    for metric in METRICS:
        scores = METRICS[metric](generations, references)
        dict_to_save = {
            "mean": np.mean(scores),
            "scores": scores,
        }

        out_fpath = scores_dir / f"{method_name}_{metric}.json"
        out_fpath.write_text(json.dumps(dict_to_save, indent=4))
        print(f"Saved scores to {out_fpath}")


def main(args):
    # Check that model is valid
    assert args.model in HF_MODELS, f"Model {args.model} not found in HF_MODELS"

    # Load yaml config file
    config = yaml.load(Path(args.data_config_path).read_text(), Loader=yaml.FullLoader)

    # Load prompts from file. This should be a json file with a list of prompts. Each prompt should be a f-string.
    prompts_fpath = (
        Path(args.prompts_dir) / f"{config['prompt']}_size={args.n_samples}.json"
    )
    prompts = json.loads(Path(prompts_fpath).read_text())
    assert isinstance(prompts, list), f"{args.prompts_fpath} must be a list"
    n_prompts = len(prompts[0])
    n_samples = len(prompts)

    # Load dataset–this is mostly to check that the prompts match
    dataset = load_hf_dataset(
        config["dataset"], n_samples=args.n_samples, hf_cache_dir=args.hf_cache_dir
    )
    dataset_name = f"{config['dataset']}_{args.n_samples}"
    assert n_samples == len(
        dataset
    ), f"Number of samples in prompts ({n_samples}) does not match number of samples in dataset ({len(dataset)})"

    # results directory
    results_dir = Path(args.results_dir) / dataset_name / args.model / config["prompt"]
    results_dir.mkdir(exist_ok=True, parents=True)

    # Evaluate individual prompt generations
    evaluate_individual_generations(results_dir, dataset, config)

    # Evaluate shrinkage average
    shrinkage_fpaths = list(results_dir.glob("*generations.json"))
    for shrinkage_fpath in shrinkage_fpaths:
        fname = shrinkage_fpath.stem
        if fname == "individual_prompts_generations":
            continue
        fname = fname.replace("_generations", "")
        evaluate_method(results_dir, dataset, config, fname)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
