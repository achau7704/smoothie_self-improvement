import argparse
from pathlib import Path
from typing import Dict

import json
import numpy as np
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastembed import TextEmbedding

from src.constants import HF_MODELS
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


MODEL_GROUPS = {
    "7b": ["mistral-7b", "llama-2-7b", "vicuna-7b", "gemma-7b", "nous-capybara"],
    "3b": ["phi-2", "pythia-2.8b", "gemma-2b", "incite-3b", "dolly-3b"],
}



def check_args(args):
    if not args.multi_model and not args.multi_prompt:
        raise ValueError('Either --multi_model or --multi_prompt must be set.')
    if args.multi_model and args.model_group is None:
        raise ValueError('--multi_model is set but the --model_group is not specified.')

def load_data_config(args: argparse.Namespace):
    """
    Load a data config yaml file.

    Args:
        args (argparse.Namespace): arguments from the command line
    """
    return yaml.load(Path(args.dataset_config).read_text(), Loader=yaml.FullLoader)

def construct_predictions_dir_path(data_config, args, model):
    if args.multi_model:
        results_dir = Path(args.results_dir) / data_config["dataset"]
        results_dir.mkdir(exist_ok=True, parents=True)
    else:
        results_dir = Path(args.results_dir) / data_config["dataset"] / model
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

    if args.multi_prompt:
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
    if args.multi_model:
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
    if args.test:
        output_fpath += "test_"
        no_flags = False
    if no_flags:
        output_fpath += "new_"
    output_fpath += f"test.json"
    output_fpath = Path(output_fpath)
    return output_fpath

def construct_pick_random_predictions_path(data_config: Dict, model: str, args: argparse.Namespace):
    results_dir = construct_predictions_dir_path(data_config, args, model)
    if args.multi_model:
        output_fpath = results_dir / f"pick_random_{args.model_group}_test.json"
    else:
        output_fpath = results_dir / "pick_random_test.json"
    output_fpath = Path(output_fpath)
    return output_fpath 

def construct_labeled_oracle_predictions_path(data_config: Dict, model: str, args: argparse.Namespace):
    results_dir = construct_predictions_dir_path(data_config, args, model)
    if args.multi_model:
        output_fpath = results_dir / f"labeled_oracle_{args.model_group}_test.json"
    else:
        output_fpath = results_dir / "labeled_oracle_test.json"
    output_fpath = Path(output_fpath)
    return output_fpath 

def construct_labeled_knn_predictions_path(data_config: Dict, model: str, args: argparse.Namespace):
    results_dir = construct_predictions_dir_path(data_config, args, model)
    output_fpath = results_dir / f"labeled_knn_{args.model_group}_test.json"
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

def generate_per_sample_single_prompt(data_config, args, model_name, model, tokenizer, prompt, gen_params):
    sequence_texts = []
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

def load_predictions(predictions_dir, split, args, for_selection=True):
    """
    Load predictions from a given split.

    Args:
    - predictions_dir (Path): The directory containing the predictions.
    - split (str): The split to load predictions for.

    Returns:
    - list: The predictions for the split.
    """
    models = [args.model] if args.multi_prompt else MODEL_GROUPS[args.model_group]

    predictions = []
    for model in models:

        if args.multi_prompt:
            file_name = "individual_"
        else:
            file_name = f"{model}_"
            
        if args.n_generations > 1 and not for_selection:
            file_name += f"{args.n_generations}_gens_"

        fpath = predictions_dir / f"{file_name}{split}.json"
        with open(fpath, "r") as f:
            predictions.append(json.load(f)["generations"])


    predictions = np.array(predictions)
    if len(predictions.shape) == 3:
        predictions = predictions.reshape((predictions.shape[1], predictions.shape[2]))
    else:
        predictions = predictions.T 
    # shape should be (n_samples * n_generations, n_prompts or n_models)
    return predictions
