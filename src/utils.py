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
from src.constants import HF_MODEL_MAX_LENGTHS



transformers.logging.set_verbosity_error()


GROUP_TO_DATASET_CONFIGS = {
    "acc_group": [
        "dataset_configs/squad.yaml",
        "dataset_configs/trivia_qa_knowledge.yaml",
        "dataset_configs/definition_extraction.yaml"
    ],
    "rouge2_group": [
        "dataset_configs/cnn_dailymail_0_shot.yaml",
        "dataset_configs/xsum_0_shot.yaml",
        "dataset_configs/e2e_nlg_1_shot.yaml",
        "dataset_configs/web_nlg_1_shot.yaml",
    ]
}

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

def construct_predictions_dir_path(data_config, args, model):
    if args.model_group is not None:
        results_dir = Path(args.results_dir) / data_config["prompt"]
        results_dir.mkdir(exist_ok=True, parents=True)
    else:
        results_dir = Path(args.results_dir) / data_config["prompt"] / model
        results_dir.mkdir(exist_ok=True, parents=True)

    return results_dir 
def construct_predictions_path(data_config: Dict, model: str, args: argparse.Namespace):
    """
    Construct the paths where train and test predictions will be saved.

    Args:
        args (argparse.Namespace): arguments from the command line
        data_config (dict): data config
    """
    results_dir = construct_predictions_dir_path(data_config, args, model)

    if args.model_group is None:
        file_name = "individual_"
    else:
        file_name = f"{model}_"

    if args.n_generations > 1:
        file_name += f"{args.n_generations}_gens_"

    if args.test:
        file_name += "test_"

    train_output_fpath = results_dir / f"{file_name}train.json"
    test_output_fpath = results_dir / f"{file_name}test.json"
    return train_output_fpath, test_output_fpath

def construct_smoothie_predictions_path(data_config: Dict, model: str, args: argparse.Namespace):
    """
    Construct the paths where train and test predictions will be saved.

    Args:
        args (argparse.Namespace): arguments from the command line
        data_config (dict): data config
    """
    results_dir = construct_predictions_dir_path(data_config, args, model)
    if args.model_group is not None:
        output_fpath = str(results_dir) + f"/smoothie_{args.type}_{args.model_group}_"
    else:
        output_fpath = str(results_dir) + f"/smoothie_{args.type}_"
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
    return output_fpath

def construct_pick_random_predictions_path(data_config: Dict, model: str, args: argparse.Namespace):
    results_dir = construct_predictions_dir_path(data_config, args, model)
    if args.model_group is not None:
        output_fpath = results_dir / f"pick_random_{args.model_group}_test.json"
    else:
        output_fpath = results_dir / "pick_random_test.json"
    output_fpath = Path(output_fpath)
    return output_fpath 

def construct_labeled_oracle_predictions_path(data_config: Dict, model: str, args: argparse.Namespace):
    results_dir = construct_predictions_dir_path(data_config, args, model)
    if args.model_group is not None:
        output_fpath = results_dir / f"labeled_oracle_{args.model_group}_test.json"
    else:
        output_fpath = results_dir / "labeled_oracle_test.json"
    output_fpath = Path(output_fpath)
    return output_fpath 


def load_hf_model(model_name, args: argparse.Namespace):
    """
    Load a HuggingFace model and tokenizer.

    Args:
        args (argparse.Namespace): arguments from the command line
    """
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODELS[model_name],
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
    )
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODELS[model_name],
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
    dataset_name: str, is_train: bool, n_samples: int, hf_cache_dir: str, doc_key: str,
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

    if dataset_name == "web_nlg":
        data_df[doc_key] = data_df.apply(lambda x: "\n".join(x['modified_triple_sets']['mtriple_set'][0]), axis=1)

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


def clean_generation(generation: str):
    """
    Extracts a generation from the full output of the model. This function is dataset specific. For instance, GSM8K answers span multiple lines, while most others only span one line.
    """
    return generation.strip().split("\n")[0]


def clean_generations(generations: list, data_config: Dict):
    """
    Cleans generations from the model output. This function is dataset specific. For instance, GSM8K answers span multiple lines, while most others only span one line.
    """
    return [clean_generation(generation) for generation in generations]


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
    individual_generations: np.ndarray, model_name: str
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
            cleaned_generation = clean_generation(generation)
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
    return data_df[data_config["doc_key"]].tolist()

def generate_per_sample_single_prompt(data_config, args, model_name, model, tokenizer, prompt, gen_params):
    sequence_texts = []
    #print(f"PROMPT IS {prompt}")
    prompt_encodings = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=HF_MODEL_MAX_LENGTHS[model_name],
    ).to(args.device)

    for i in range(args.n_generations):
        output = model.generate(
            **prompt_encodings,
            max_new_tokens=data_config["max_new_tokens"],
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            **gen_params
        )

        # Get the token ids corresponding to the generation. output["sequences"] is a tensor of shape (batch_size, seq_len)
        sequence_texts.append(
            tokenizer.decode(get_generation_output(prompt_encodings, output))
        )

    # returns a list of args.n_generations outputs

    return sequence_texts

def generate_per_sample_multi_prompt(data_config, args, model_name, model, tokenizer, prompts, gen_params):
    sequence_texts = [] # will be a list: p1 output1, p1 output2, ..., p2 output1, p2 output2, ...
    for prompt_idx in range(len(prompts)):
        texts = generate_per_sample_single_prompt(
            data_config,
            args,
            model_name,
            model, 
            tokenizer,
            prompts[prompt_idx],
            gen_params
        )
        sequence_texts.extend(texts)
    # returns a list of n_prompts * n_generations outputs
    return sequence_texts

