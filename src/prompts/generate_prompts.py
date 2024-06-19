"""
This script generates the prompts to feed into an LLM for each dataset. Each prompt template is defined as a function.

Example command: python -m src.generate_prompts --hf_cache_dir cache --prompts_dir prompts --config_path configs/gsm8k_1_shot_cot_simple.yaml  --n_samples 4
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from tqdm.auto import tqdm

from src.console import console
from src.utils import load_hf_dataset, prompt_openai

parser = argparse.ArgumentParser()
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
    "--prompts_dir", default="prompts", type=str, help="Directory to save prompts to"
)
parser.add_argument(
    "--prompt_templates_dir",
    default="src/prompts/assets",
    type=str,
    help="Directory with prompt templates",
)
parser.add_argument(
    "--n_samples", default=100, type=int, help="Number of samples to use"
)
parser.add_argument(
    "--redo", action="store_true", help="If set, overwrite existing prompts"
)


def generate_web_nlg_prompt(prompt_template, row):
    """
    Generate a prompt for the web_nlg dataset.
    """
    triples = ""
    for triple in row["modified_triple_sets"]["mtriple_set"][0]:
        triples += f"{triple}\n"
    return prompt_template.format(triples=triples)


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
    )
    console.log(f"Loaded dataset with {len(dataset)} examples (split: {split}).")

    # Check to see if prompts already exist
    prompts_fpath = Path(args.prompts_dir) / f"{data_config['prompt']}_{split}.json"
    if prompts_fpath.exists() and not args.redo:
        console.log(
            f"Prompts already exist at {prompts_fpath}. Use --redo to overwrite."
        )
        return
    prompts_fpath.parent.mkdir(parents=True, exist_ok=True)

    # Load prompt template
    prompt_templates_fpath = (
        Path(args.prompt_templates_dir) / f"{data_config['prompt']}.json"
    )
    prompt_templates = json.loads(prompt_templates_fpath.read_text())
    console.log(f"Loaded prompt templates from {prompt_templates_fpath}")

    prompts = []
    for row in tqdm(dataset.to_dict(orient="records")):
        sample_prompts = []
        for prompt_template in prompt_templates:
            # The web_nlg data has a different format, so we need to handle it differently
            if data_config["dataset"] == "web_nlg":
                sample_prompts.append(generate_web_nlg_prompt(prompt_template, row))
            else:
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
    console.log(f"Example prompt")
    console.log(prompts[0][0])
    console.log(dataset.to_dict(orient="records")[0])
    # Save prompts as json list
    prompts_fpath.write_text(json.dumps(prompts, indent=4))
    console.log(f"Saved prompts to {prompts_fpath}")


def main(args):
    # Load yaml configs file
    data_config = yaml.load(
        Path(args.data_config_path).read_text(), Loader=yaml.FullLoader
    )
    console.log(data_config)
    generate_prompts(args=args, data_config=data_config, is_train=True)
    generate_prompts(args, data_config, is_train=False)


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
