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

from src.console import console
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

    # Load test dataset
    test_datasets = []
    for data_config in data_configs:
        test_df = load_hf_dataset(
            dataset_name=data_config["dataset"],
            is_train=False,
            n_samples=data_config["test_size"],
            hf_cache_dir=args.hf_cache_dir,
            doc_key=data_config["doc_key"]
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

    ############### INDIVIDUAL MODELS
    for model_idx in range(all_test_generations.shape[1]):
        model_generations = all_test_generations[:, model_idx]
        results = {
            "generations": model_generations.tolist(),
        }

        output_dir = Path(args.save_dir) / f"{args.model_group}_{args.task_group}"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_fpath = (
            output_dir / f"{MODEL_GROUPS[args.model_group][model_idx]}_test.json"
        )
        console.log(f"Saving to {output_fpath}")
        output_fpath.write_text(json.dumps(results, indent=4))

    ############### PICK RANDOM
    np.random.seed(0)
    sequence_texts = []
    for trial in range(10):
        trial_generations = []
        for sample_idx in range(len(all_test_generations)):
            # Select a random generation from the individual generations.
            generation_idx = np.random.randint(all_test_generations.shape[1])
            generation = all_test_generations[sample_idx][generation_idx]
            trial_generations.append(generation)
        sequence_texts.append(trial_generations)

    # Save to file
    results = {
        "generations": sequence_texts,
    }

    output_dir = Path(args.save_dir) / f"{args.model_group}_{args.task_group}"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_fpath = output_dir / "pick_random_test.json"
    console.log(f"Saving to {output_fpath}")
    output_fpath.write_text(json.dumps(results, indent=4))


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
