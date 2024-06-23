
import argparse
import json

import pandas as pd
from tqdm.auto import tqdm

from transformers import set_seed


from src.console import console
from src.data_utils import construct_processed_dataset_paths
from src.utils import (
    load_data_config,
    construct_predictions_path, 
    load_hf_model, 
    check_args,
    generate_per_sample_multi_prompt,
    generate_per_sample_single_prompt
)

from src.utils import MODEL_GROUPS


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
parser.add_argument(
    "--dataset_config",
    type=str,
    help="Path to config file. This should be a yaml file",
)
parser.add_argument(
    "--data_dir",
    default="smoothie_data/datasets",
    type=str,
    help="Directory with data files",
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
parser.add_argument(
    "--multi_prompt",
    action="store_true",
)
parser.add_argument(
    "--multi_model",
    action="store_true",
)

def generate_predictions(args, data_config, data_path, model_name, model, tokenizer, output_fpath):
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

    # load the data 
    dataset = pd.read_csv(data_path)

    if args.n_generations > 1:
        gen_params = {"temperature": args.temperature, "do_sample": True}
    else:
        gen_params = {"do_sample": False}

    sequence_texts = []
    progress_bar = tqdm(range(len(dataset)))
    for sample_idx, row in dataset.iterrows():
        if args.multi_model:
            prompt = row['multi_model_prompt']
            texts = generate_per_sample_single_prompt(data_config, args, model_name, model, tokenizer, prompt, gen_params)
            sequence_texts.extend(texts) # one big list

        else:
            prompts = row[4:].values
            texts = generate_per_sample_multi_prompt(data_config, args, model_name, model, tokenizer, prompts, gen_params)
            sequence_texts.append(texts) # ensures that we have a list of lists


        progress_bar.update(1)
        if args.test and sample_idx == 1:
            break

    results = {"generations": sequence_texts,}
    output_fpath.write_text(json.dumps(results, indent=4))


def main(args):
    check_args(args)
    # Load yaml configs file
    data_config = load_data_config(args)

    # TODO make flag for args.multi_model args.multi_prompt 

    model_names = [args.model] if args.multi_prompt else MODEL_GROUPS[args.model_group]


    train_data_fpath, test_data_fpath = construct_processed_dataset_paths(args)

    for model_name in model_names:
        # each model has its own predictions file 
        train_output_fpath, test_output_fpath = construct_predictions_path(data_config, model_name, args)
        if train_output_fpath.exists() and test_output_fpath.exists() and not args.redo:
            console.log(
                f"Results file {train_output_fpath} and {test_output_fpath} already exists. Skipping."
            )
            continue
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
            train_data_fpath,
            model_name,
            model,
            tokenizer,
            train_output_fpath
        )
        generate_predictions(
            args,
            data_config,
            test_data_fpath,
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
