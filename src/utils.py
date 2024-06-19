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
    return yaml.load(Path(args.data_config_path).read_text(), Loader=yaml.FullLoader)


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


def load_hf_dataset(
    dataset_name: str, is_train: bool, n_samples: int, hf_cache_dir: str
) -> pd.DataFrame:
    """
    Load a dataset from the HuggingFace datasets library. Returns dataset as pandas dataframe.

    Args:
        dataset_name (str): dataset name
        is_train (bool): whether to load the train or test split
        n_samples (int): number of samples to load
        cache_dir (str): cache directory
    """

    if dataset_name in ["squad"]:
        # These datasets don't have a train split, so we treat the "train" as the first half of the validation split
        hf_url, subset, split = HF_TEST_DATASETS[dataset_name]

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

        if is_train:
            # Take the first half
            data_df = data_df.iloc[: len(data_df) // 2]
        else:
            # Take the second half
            data_df = data_df.iloc[len(data_df) // 2 :]
    elif dataset_name in ["definition_extraction"]:
        # For this, we take the first 100 as the train
        hf_url, subset, split = HF_TEST_DATASETS[dataset_name]

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

        if is_train:
            data_df = data_df.iloc[:100]
        else:
            data_df = data_df.iloc[100:]

    else:
        if is_train:
            hf_url, subset, split = HF_TRAIN_DATASETS[dataset_name]
        else:
            hf_url, subset, split = HF_TEST_DATASETS[dataset_name]

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

    if dataset_name == "sentence_compression":
        # The sentence_compression dataset has a different structure. We have to create a data frame
        # from the examples.
        lst = data_df["set"].tolist()
        sentence = [x[0] for x in lst]
        summary = [x[1] for x in lst]
        data_df = pd.DataFrame.from_dict({"sentence": sentence, "summary": summary})

    # If n_samples is greater than 0, only load a random number of n_samples examples.
    if n_samples > 0:
        n_samples = min(n_samples, len(data_df))
        data_df = data_df.sample(n=n_samples, random_state=42)

    return data_df


def move_tensors_to_cpu(tuple_of_tuples):
    result_tuple = tuple(
        tuple(tensor.detach().to("cpu") for tensor in inner_tuple)
        for inner_tuple in tuple_of_tuples
    )
    torch.cuda.empty_cache()  # Free up GPU memory
    return result_tuple


def move_tensors_to_gpu(tuple_of_tuples):
    result_tuple = tuple(
        tuple(tensor.detach().to("cuda") for tensor in inner_tuple)
        for inner_tuple in tuple_of_tuples
    )
    torch.cuda.empty_cache()  # Free up GPU memory
    return result_tuple


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


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


def get_input_text(data_df, data_config):
    """
    Returns input text for dataset.
    """
    if data_config["dataset"] == "web_nlg":
        texts = []
        for row in data_df.to_dict(orient="records"):
            triples = ""
            for triple in row["modified_triple_sets"]["mtriple_set"][0]:
                triples += f"{triple}\n"
            texts.append(triples)
        return texts
    else:
        return data_df[data_config["doc_key"]].tolist()
