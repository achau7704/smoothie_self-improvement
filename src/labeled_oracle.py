"""
    Implements the best-on-val baseline
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import argparse
import json

import numpy as np
from tqdm.auto import tqdm

from src.console import console
from src.evaluate.metrics import *
from src.evaluate.scorer import *
from src.multi_model.utils import load_predictions
from src.utils import (check_results_file, construct_predictions_dir_path,
                       load_data_config, load_hf_dataset,
                       construct_labeled_oracle_predictions_path)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
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
    type=str,
    help="Results directory",
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
    "--redo",
    action="store_true",
    help="Redo evaluation even if results already exist. Otherwise, we only evaluate methods/metrics which aren't already evaluated.",
)
parser.add_argument(
    "--model_group",
    help="The models to use for predictions",
)

TASK2METRIC = {
    "cnn_dailymail": METRIC_FUNCS["rouge2"],
    "definition_extraction": METRIC_FUNCS["definition_extraction_acc"],
    "e2e_nlg": METRIC_FUNCS["rouge2"],
    "squad": METRIC_FUNCS["squad_acc"],
    "trivia_qa": METRIC_FUNCS["trivia_qa_acc"],
    "web_nlg": METRIC_FUNCS["rouge2"],
    "xsum": METRIC_FUNCS["rouge2"]
}

def main(args):
    data_config = load_data_config(args)
    output_fpath = construct_labeled_oracle_predictions_path(data_config, args.model, args)
    predictions_dir = output_fpath.parent 

    if check_results_file(output_fpath) and not args.redo:
        console.log(f"Results file already exists at {output_fpath}. Skipping.")
        return 

    train_generations = load_predictions(predictions_dir, "train", args)
    test_generations = load_predictions(predictions_dir, "test", args)
    n_samples, n_prompts = train_generations.shape

    train_dataset = load_hf_dataset(
        dataset_name=data_config["dataset"],
        is_train=False,
        n_samples=data_config["test_size"],
        hf_cache_dir=args.hf_cache_dir,
        doc_key=data_config["doc_key"]
    )
    train_references = get_references(train_dataset, data_config)
    if data_config['dataset'] not in TASK2METRIC:
        raise ValueError(f"Dataset {data_config['dataset']} not supported.")

    metric_func = TASK2METRIC[data_config['dataset']]

    generations = []
    for _ in range(args.label_train_n_trials):
        sampled_indices = np.random.choice(
            n_samples, args.label_train_sample_size
        )
        sampled_references = [train_references[idx] for idx in sampled_indices]
        prompt_scores = []
        for prompt_idx in range(n_prompts):
            sampled_generations = train_generations[sampled_indices, prompt_idx]
            cleaned_generations = clean_generations(sampled_generations, data_config)
            scores = metric_func(cleaned_generations, sampled_references)
            prompt_scores.append(np.mean(scores))
        best_prompt_idx = np.argmax(prompt_scores)
        generations.append(test_generations[:, best_prompt_idx].tolist())

    results = {
        "generations": generations,
    }
    console.log(f"Saving results to {output_fpath}")
    output_fpath.write_text(json.dumps(results, default=int, indent=4))


if __name__ == "__main__":
    args = parser.parse_args()
    console.log(args)
    main(args)
