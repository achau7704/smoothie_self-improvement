
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from tqdm.auto import tqdm

from transformers import set_seed
import numpy as np


from src.console import console
from src.utils import (
    load_data_config,
    load_hf_dataset, 
    construct_predictions_path, 
    load_hf_model, 
    get_generation_output,
    generate_per_sample_multi_prompt,
    generate_per_sample_single_prompt
)

from src.multi_model.utils import MODEL_GROUPS


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
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--prompts_dir", default="prompts", type=str, help="Directory to save prompts to"
)
parser.add_argument(
    "--redo", action="store_true", help="If set, overwrite existing prompts"
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Runs the script in test mode. This will only generate predictions for two samples.",
)
parser.add_argument(
    "--model_group",
    default=None,
    choices=["7b", "3b", None],
    help="The models to use for predictions. If set to none, we are in the multi-prompt setting.",
)
parser.add_argument(
    "--n_generations",
    default=1,
    type=int,
    help="For each model we produce n_generations per sample. Default is 1.",
)
parser.add_argument(
    "--temperature",
    default=0.0,
    type=float,
    help="Temperature for generations. Only used when n_generations > 1.",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Random seed if n_generations > 1.",
)


def generate_prompts(
    args: argparse.Namespace, data_config: Dict, is_train: bool
) -> None:
    """
    Generates split.
    """
    split = "train" if is_train else "test"

    # Load dataset
    dataset = load_hf_dataset(
        dataset_name=data_config["dataset"],
        is_train=is_train,
        n_samples=data_config[f"{split}_size"],
        hf_cache_dir=args.hf_cache_dir,
        doc_key=data_config["doc_key"],
    )
    console.log(f"Loaded dataset with {len(dataset)} examples (split: {split}).")

    # Check to see if prompts already exist
    if args.test:
        prompts_fpath = Path(args.prompts_dir) / f"{data_config['prompt']}_{split}_test.json"
    else:
        prompts_fpath = Path(args.prompts_dir) / f"{data_config['prompt']}_{split}.json"
    if prompts_fpath.exists() and not args.redo:
        console.log(
            f"Prompts loaded from {prompts_fpath}. Use --redo to overwrite."
        )
        prompts = np.array(json.loads(prompts_fpath.read_text()))
        return prompts
    prompts_fpath.parent.mkdir(parents=True, exist_ok=True)

    prompt_templates_dir = "src/prompts/assets" if args.model_group is None else "src/prompts/multimodel_assets"
    prompt_templates_fpath = (
        Path(prompt_templates_dir) / f"{data_config['prompt']}.json"
    )
    prompt_templates = json.loads(prompt_templates_fpath.read_text())
    console.log(f"Loaded prompt templates from {prompt_templates_fpath}")

    prompts = []
    for row in tqdm(dataset.to_dict(orient="records")):
        sample_prompts = []
        for prompt_template in prompt_templates:
            sample_prompts.append(prompt_template.format(**row))

        # This is a hack to handle the case where there is only one prompt. This occurs
        # for the multi model case where variation comes from different models.
        if len(sample_prompts) == 1:
            sample_prompts = sample_prompts[0]
        prompts.append(sample_prompts)

    if isinstance(prompts[0], list):
        console.log(
            f"Generated {len(prompts[0])} prompts for each of {len(dataset)} samples."
        )
    else:
        console.log(f"Generated {len(prompts)} prompts for {len(dataset)} samples.")

    console.log("#" * 30)
    # Save prompts as json list
    prompts_fpath.write_text(json.dumps(prompts, indent=4))
    console.log(f"Saved prompts to {prompts_fpath}")
    return np.array(prompts)

def generate_predictions(args, data_config, prompts, model_name, model, tokenizer, output_fpath):
    """
    Generate predictions over a dataset using a model.

    Args:
        args (argparse.Namespace): arguments from the command line
        data_config (dict): data config
        prompts (list): list of prompts
        model (transformers.PreTrainedModel): LLM
        tokenizer (transformers.PreTrainedTokenizer): tokenizer
    """
    # Check if the results file already exists
    if output_fpath.exists() and not args.redo:
        console.log(f"Results file {output_fpath} already exists. Skipping.")
        return
    else:
        console.log(f"Will save results to: {output_fpath}")
    
    if args.n_generations > 1:
        gen_params = {"temperature": args.temperature, "do_sample": True}
    else:
        gen_params = {"do_sample": False}

    sequence_texts = []
    for sample_idx in tqdm(range(len(prompts))):
        if isinstance(prompts[0], np.ndarray):
            texts = generate_per_sample_multi_prompt(data_config, args, model_name, model, tokenizer, prompts[sample_idx], gen_params)
            sequence_texts.append(texts) # ensures that we have a list of lists
        else:
            texts = generate_per_sample_single_prompt(data_config, args, model_name, model, tokenizer, prompts[sample_idx], gen_params)
            sequence_texts.extend(texts) # one big list
        if args.test and sample_idx == 1:
            break

    results = {"generations": sequence_texts,}
    output_fpath.write_text(json.dumps(results, indent=4))


def main(args):
    # Load yaml configs file
    data_config = load_data_config(args)
    train_prompts = generate_prompts(args=args, data_config=data_config, is_train=True)
    test_prompts = generate_prompts(args, data_config, is_train=False)

    # TODO make flag for args.multi_model args.multi_prompt 
    # TODO remove generate_prompts 

    model_names = [args.model] if args.model_group is None else MODEL_GROUPS[args.model_group]

    for model_name in model_names:
        train_output_fpath, test_output_fpath = construct_predictions_path(data_config, model_name, args)
        if train_output_fpath.exists() and test_output_fpath.exists() and not args.redo:
            console.log(
                f"Results file {train_output_fpath} and {test_output_fpath} already exists. Skipping."
            )
            return
        else:
            console.log(
                f"Will save results to: {train_output_fpath} and {test_output_fpath}"
            )

        model, tokenizer = load_hf_model(model_name, args)

        if args.n_generations > 1:
            set_seed(args.seed)
            assert args.temperature != 0

        generate_predictions(
            args,
            data_config,
            train_prompts,
            model_name,
            model,
            tokenizer,
            train_output_fpath
        )
        generate_predictions(
            args,
            data_config,
            test_prompts,
            model_name,
            model,
            tokenizer,
            test_output_fpath
        )

if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
