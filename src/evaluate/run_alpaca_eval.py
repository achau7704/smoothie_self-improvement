"""
This script executes the ALPACA evaluation for the generative ensembles project.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.console import console
from src.constants import HF_MODELS
from src.utils import initialize_alpaca_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="LLM to use")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
parser.add_argument(
    "--data_config_path",
    type=str,
    help="Path to config file. This should be a yaml file",
)
parser.add_argument(
    "--method_config_path",
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
    default="generative_ensembles_data/results",
    type=str,
    help="Directory to save results to",
)
parser.add_argument(
    "--prompts_dir", default="prompts", type=str, help="Directory to save prompts to"
)
parser.add_argument(
    "--n_samples", default=100, type=int, help="Number of samples to use"
)
parser.add_argument(
    "--redo", action="store_true", help="Redo the generation if the file already exists"
)


def main(args):
    # Initialize experiment
    (
        data_config,
        method_config,
        prompts,
        dataset,
        results_dir,
    ) = initialize_alpaca_experiment(args=args, return_results_dir=True)
    n_samples, n_prompts = len(prompts), len(prompts[0])

    # Load generations
    generation_files = list(results_dir.glob("*_generations.json"))
    method_to_gens = {}
    for gen_file in generation_files:
        method_name = gen_file.stem.replace("_generations", "")
        gens = json.loads(gen_file.read_text())["generations"]
        method_to_gens[method_name] = gens

        # Check that generations is a dimension-1 list of length n_samples
        assert isinstance(gens, list), f"Generations must be a list"
        assert n_samples == len(
            gens
        ), f"Number of samples in generations ({n_samples}) does not match number of samples in dataset ({len(dataset)})"

    instructions = dataset[data_config["instruction_key"]].tolist()
    base_outputs = dataset[data_config["reference_key"]].tolist()

    # reference_outputs the path to the outputs of the reference model. Each dictionary should contain the keys (instruction and output) that are formatted in the prompts.
    reference_outputs = []
    for i in range(n_samples):
        reference_outputs.append(
            {"instruction": instructions[i], "output": base_outputs[i]}
        )
    reference_fpath = results_dir / f"reference_outputs.json"
    reference_fpath.write_text(json.dumps(reference_outputs, indent=4))
    print(f"Saving reference outputs to {reference_fpath}")

    # all_model_outputs : The json path to the outputs of all models to add to the leaderboard (as a single file or by globbing multiple files). Each dictionary should contain the keys (instruction and output) that are formatted in the prompts and a column generator with the name of the current model.

    command_script = "#!/bin/bash\n\n"
    command_script += """export OPENAI_API_KEY=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\n\n"""
    leaderboard_path = results_dir / "leaderboard.csv"
    for method in method_to_gens:
        results = []
        for i in range(n_samples):
            results.append(
                {
                    "instruction": instructions[i],
                    "output": method_to_gens[method][i],
                    "generator": method,
                }
            )
        out_fpath = results_dir / f"{method}_outputs.json"
        out_fpath.write_text(json.dumps(results, indent=4))
        print(f"Saving {method} outputs to", out_fpath)

        # Save command to script
        command = f"alpaca_eval make_leaderboard \
            --leaderboard_path {leaderboard_path} \
            --all_model_outputs {out_fpath} \
            --reference_outputs {reference_fpath} \
            --annotators_config weighted_alpaca_eval_gpt4_turbo"
        command_script += command + "\n"

    # Save command script
    command_script_fpath = Path("alpaca_eval_commands.sh")
    command_script_fpath.write_text(command_script)
    console.print(f"Saved command script to", command_script_fpath)


if __name__ == "__main__":
    console.print("#" * 30)
    args = parser.parse_args()
    console.print(args)
    main(args)
