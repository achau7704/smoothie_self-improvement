"""
This file contains utility functions for loading datasets and generating prompts.
"""

from datasets import load_dataset
from src.constants import HF_TEST_DATASETS, HF_TRAIN_DATASETS
import pandas as pd
from typing import Dict

# TODO: This is not the cleanest logic. We might want to refactor this to be more modular.
def load_hf_dataset(
    config: Dict, hf_cache_dir: str
) -> pd.DataFrame:
    """
    Load a dataset from the HuggingFace datasets library. Returns dataset as pandas dataframe.

    Args:
        config (str): dataset config
        cache_dir (str): cache directory

    Returns:
        train_df (pd.DataFrame): train split
        test_df (pd.DataFrame): test split

    """

    if config["dataset"] in ["squad"]:
        # These datasets don't have a train split, so we treat the "train" as the first half of the validation split
        hf_url, subset, split = HF_TEST_DATASETS[config["dataset"]]

        if subset is not None:
            dataset = load_dataset(
                hf_url,
                subset,
                split=split,
                cache_dir=hf_cache_dir,
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset(
                hf_url, split=split, cache_dir=hf_cache_dir, trust_remote_code=True
            )
        # Convert the dataset to a pandas dataframe
        data_df = dataset.to_pandas()

        train_df = data_df.iloc[: len(data_df) // 2]
        test_df = data_df.iloc[len(data_df) // 2 :]
    elif config["dataset"] in ["definition_extraction"]:
        # For this, we take the first 100 as the train
        hf_url, subset, split = HF_TEST_DATASETS[config["dataset"]]

        if subset is not None:
            dataset = load_dataset(
                hf_url,
                subset,
                split=split,
                cache_dir=hf_cache_dir,
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset(
                hf_url, split=split, cache_dir=hf_cache_dir, trust_remote_code=True
            )
        data_df = dataset.to_pandas()

        # Get train and test splits
        train_df = data_df.iloc[:100]
        test_df = data_df.iloc[100:]

    else:
        
        # Get train split
        hf_url, subset, split = HF_TRAIN_DATASETS[config["dataset"]]
        if subset is not None:
            dataset = load_dataset(
                hf_url,
                subset,
                split=split,
                cache_dir=hf_cache_dir,
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset(
                hf_url, split=split, cache_dir=hf_cache_dir, trust_remote_code=True
            )
        train_df = dataset.to_pandas()
        
        # Get test split
        hf_url, subset, split = HF_TEST_DATASETS[config["dataset"]]

        if subset is not None:
            dataset = load_dataset(
                hf_url,
                subset,
                split=split,
                cache_dir=hf_cache_dir,
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset(
                hf_url, split=split, cache_dir=hf_cache_dir, trust_remote_code=True
            )
        test_df = dataset.to_pandas()

    # If n_samples is greater than 0, only load a random number of n_samples examples.
    if config["train_size"] > 0:
        n_samples = min(config["train_size"], len(train_df))
        train_df = data_df.sample(n=config["train_size"], random_state=42)

    if config["test_size"] > 0:
        n_samples = min(config["test_size"], len(test_df))
        test_df = data_df.sample(n=config["test_size"], random_state=42)

    return train_df, test_df



def generate_prompt(config: Dict, prompt_template: str, row: Dict) -> str:
    """
    Generates a prompt for a given dataset.

    Args:
        config (Dict): dataset config
        template (str): prompt template
        row (Dict): row from dataset
    
    Returns:
        prompt (str): generated prompt
    """
    if config["dataset"] == "web_nlg":
        triples = ""
        for triple in row["modified_triple_sets"]["mtriple_set"][0]:
            triples += f"{triple}\n"
        return prompt_template.format(triples=triples)
    else:
        return prompt_template.format(**row)

def get_embedding_inputs(config: Dict, row: Dict) -> str:
    """
    Constructs the text sequence used for generating embeddings.

    Args:
        config (Dict): dataset config
        row (Dict): row from dataset
    
    Returns:
        embedding_input (str): text sequence for generating embeddings
    """
    if config["dataset"] == "web_nlg":
        embedding_input = ""
        for triple in row["modified_triple_sets"]["mtriple_set"][0]:
            embedding_input += f"{triple}\n"
    elif config["dataset"] == "squad":
        embedding_input = row["context"]
    else:
        raise NotImplementedError(f"Embedding inputs not implemented for {config['dataset']}")
    return embedding_input