"""


Currently supported datasets

    - squad
    - trivia_qa
    - xsum
    - cnn_dailymail
    - definition_extraction
    - e2e_nlg
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml

from src.console import console
from src.data_utils import (generate_prompt, get_embedding_inputs,
                            load_hf_dataset)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config",
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
    "--data_dir",
    default="smoothie_data/datasets",
    type=str,
    help="Directory with data files",
)
parser.add_argument(
    "--prompt_templates_dir",
    default="prompt_templates",
    type=str,
    help="Directory with prompt templates",
)

def main(args):
    # Load yaml configs file
    data_config = yaml.load(
        Path(args.dataset_config).read_text(), Loader=yaml.FullLoader
    )
    console.log(data_config)

    # Extract config name from file name
    config_name = Path(args.dataset_config).stem

    # Load dataset
    train_df, test_df = load_hf_dataset(
        config=data_config,
        hf_cache_dir=args.hf_cache_dir,
    )
    console.log(f"Loaded dataset with {len(train_df)} training examples.")
    console.log(f"Loaded dataset with {len(test_df)} testing examples.")

    # Generate multi-prompt prompts
    fpath = (
        Path(args.prompt_templates_dir) / f"{data_config['multi_prompt_template']}.json"
    )
    multi_prompt_templates = json.loads(fpath.read_text())
    console.log(f"Multi prompt templates:", multi_prompt_templates)

    for idx, prompt_template in enumerate(multi_prompt_templates):
        train_df[f"multi_prompt_{idx}"] = train_df.apply(
            lambda row: generate_prompt(config=data_config, prompt_template=prompt_template, row=row), axis=1
        )
        test_df[f"multi_prompt_{idx}"] = test_df.apply(
            lambda row: generate_prompt(config=data_config, prompt_template=prompt_template, row=row), axis=1
        )
        

    # Generate multi-model prompts
    fpath = (
        Path(args.prompt_templates_dir) / f"{data_config['multi_model_prompt_template']}.txt"
    )
    multi_model_prompt_template = fpath.read_text()
    console.log("Multi model prompt template:", multi_model_prompt_template)
    train_df["multi_model_prompt"] = train_df.apply(
        lambda row: generate_prompt(config=data_config, prompt_template=multi_model_prompt_template, row=row), axis=1
    )
    test_df["multi_model_prompt"] = test_df.apply(
        lambda row: generate_prompt(config=data_config, prompt_template=multi_model_prompt_template, row=row), axis=1
    )
    
    # Add embedding input text
    train_df["embedding_input"] = train_df.apply(
        lambda row: get_embedding_inputs(config=data_config, row=row), axis=1
    )
    test_df["embedding_input"] = test_df.apply(
        lambda row: get_embedding_inputs(config=data_config, row=row), axis=1
    )

    # Rename reference column
    train_df = train_df.rename(columns={data_config["reference_key"]: "reference"})
    test_df = test_df.rename(columns={data_config["reference_key"]: "reference"})

    # Add index to each row, corresponding to the number of the row
    train_df["idx"] = train_df.reset_index().index
    test_df["idx"] = test_df.reset_index().index
    
    # Remove columns that are not needed
    columns_to_keep = [
        "idx",
        "reference",
        "embedding_input",
        "multi_model_prompt",
    ]
    columns_to_keep.extend([f"multi_prompt_{idx}" for idx in range(len(multi_prompt_templates))])
    train_df = train_df[columns_to_keep]
    test_df = test_df[columns_to_keep]

    # Save dataframes
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    train_fpath = Path(args.data_dir) / f"{config_name}_train.csv"
    train_df.to_csv(train_fpath, index=False)
    console.log(f"Saved training data to {train_fpath}")
    test_fpath = Path(args.data_dir) / f"{config_name}_test.csv"
    test_df.to_csv(test_fpath, index=False)
    console.log(f"Saved testing data to {test_fpath}")


if __name__ == "__main__":
    console.log("#" * 30)
    args = parser.parse_args()
    console.log(args)
    main(args)
