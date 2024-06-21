import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from datasets import load_dataset
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastembed import TextEmbedding

from src.console import console
from src.constants import HF_MODELS, HF_TEST_DATASETS, HF_TRAIN_DATASETS

transformers.logging.set_verbosity_error()


def load_data_config(args: argparse.Namespace):
    """
    Load a data config yaml file.

    Args:
        args (argparse.Namespace): arguments from the command line
    """
    return yaml.load(Path(args.dataset_config).read_text(), Loader=yaml.FullLoader)


def load_prompts(data_config: Dict, args: argparse.Namespace):
    """
    Load prompts from a json file.

    Args:
        data_config (dict): data config
        args (argparse.Namespace): arguments from the command line

    Returns:
        train_prompts (list): list of prompts
        test_prompts (list): list of prompts
    """

    train_prompts_fpath = Path(args.prompts_dir) / f"{data_config['prompt']}_train.json"
    train_prompts = json.loads(train_prompts_fpath.read_text())

    test_prompts_fpath = Path(args.prompts_dir) / f"{data_config['prompt']}_test.json"
    test_prompts = json.loads(test_prompts_fpath.read_text())
    return np.array(train_prompts), np.array(test_prompts)


def construct_predictions_dir_path(
    data_config: Dict, args: argparse.Namespace, multi_model: bool = False
):
    """
    Construct the path to the directory where predictions will be saved.

    Args:
        args (argparse.Namespace): arguments from the command line
        data_config (dict): data config
    """
    if multi_model:
        results_dir = Path(args.results_dir) / data_config["prompt"]
        results_dir.mkdir(exist_ok=True, parents=True)
    else:
        results_dir = Path(args.results_dir) / data_config["prompt"] / args.model
        results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir


def load_hf_model(args: argparse.Namespace):
    """
    Load a HuggingFace model and tokenizer.

    Args:
        args (argparse.Namespace): arguments from the command line
    """
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODELS[args.model],
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
    )
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODELS[args.model],
        cache_dir=args.hf_cache_dir,
        truncation_side="left",
        trust_remote_code=True,
    )
    return model, tokenizer


def make_list_with_shape(d1, d2):
    """
    Make a list with shape (d1, d2)
    """
    return [[None for _ in range(d2)] for _ in range(d1)]


def prompt_openai(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: str,
):
    """

    Args:
        api_key (str): OpenAI api key
        model (str): model to use
        prompt (str): prompt to be completed
        max_tokens (int): Number of tokens to generate.
        temperature (float): What sampling temperature to use.
        stop (str): Sequence at which to stop generation.
    """

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    return completion.choices[0].message.content


def get_latent_state(
    prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str
):
    """
    Get the embedding of the last token of the prompt.

    Args:
        prompt (str): prompt to be completed
        model (AutoModelForCausalLM): model to generate from
        tokenizer (AutoTokenizer): tokenizer to use
        device (str): device to use
    """
    input = tokenizer(prompt, return_tensors="pt")
    input = {k: v.to(device) for k, v in input.items()}

    output = model.generate(
        **input,
        output_hidden_states=True,  # Makes sure we can get hidden states
    )
    # Get last hidden states
    last_hidden_state = outputs["hidden_states"][-1].detach()[
        :, -1, :
    ]  # shape is (batch_size, seq_len, hidden_size)

    if "cuda" in device:
        embedding = last_hidden_state.cpu().numpy()
    else:
        embedding = last_hidden_state.numpy()
    return embedding


def get_generation_output(input, output):
    """
    By default, Huggingface returns the prompt + the generated text. This function
    returns only the generated text.
    """
    input_len = input["input_ids"].shape[1]
    return output["sequences"][0, input_len:].detach().to("cpu").tolist()


def clean_generation(generation: str, data_config: Dict):
    """
    Extracts a generation from the full output of the model. This function is dataset specific. For instance, GSM8K answers span multiple lines, while most others only span one line.
    """
    if data_config["dataset"] == "gsm8k":
        # Check for "Question:"
        return generation.split("Question")[0].strip()
    else:
        return generation.strip().split("\n")[0]


def clean_generations(generations: list, data_config: Dict):
    """
    Cleans generations from the model output. This function is dataset specific. For instance, GSM8K answers span multiple lines, while most others only span one line.
    """
    return [clean_generation(generation, data_config) for generation in generations]


def check_results_file(output_fpath: Path) -> bool:
    """
    Create the output file path and check if the file already exists.

    Args:
        output_fpath (Path): path to the output file

    Returns:
        True if the file exists, False otherwise
    """
    if output_fpath.exists():
        return True
    else:
        return False


def compute_embedding(embedding_model_name, text_inputs):
    if embedding_model_name in ["all-mpnet-base-v2"]:
        embedding_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5", 
            providers=["CUDAExecutionProvider"]
        )
        embeddings_generator = embedding_model.embed(text_inputs)  # reminder this is a generator
        embeddings_list = list(embedding_model.embed(text_inputs))
        return np.array(embeddings_list)
    else:
        raise ValueError("Invalid model name")



def embed_individual_generations(
    individual_generations: np.ndarray, data_config: dict, model_name: str
):
    """
    This function returns embeddings of a matrix of individual generations. It applies a dataset
    specific preprocessing step.
    """
    n_samples, n_prompts = individual_generations.shape

    # Post process the individual generations
    processed_generations = []
    for sample_idx in range(n_samples):
        processed_generations.append([])
        for prompt_idx in range(n_prompts):
            generation = individual_generations[sample_idx, prompt_idx]
            cleaned_generation = clean_generation(generation, data_config)
            processed_generations[-1].append(cleaned_generation)
    processed_generations = np.array(processed_generations)

    # Construct the embeddings
    flattened_generations = processed_generations.flatten()
    embeddings = compute_embedding(model_name, flattened_generations)
    embeddings = embeddings.reshape(n_samples, n_prompts, -1)
    return embeddings
