import json
import os
import numpy as np

MODEL_GROUPS = {
    "7b": ["mistral-7b", "llama-2-7b", "vicuna-7b", "gemma-7b", "nous-capybara"],
    "3b": ["phi-2", "pythia-2.8b", "gemma-2b", "incite-3b", "dolly-3b"],
}


def load_predictions(predictions_dir, split, args, for_selection=True):
    """
    Load predictions from a given split.

    Args:
    - predictions_dir (Path): The directory containing the predictions.
    - split (str): The split to load predictions for.

    Returns:
    - list: The predictions for the split.
    """
    models = [args.model] if args.model_group is None else MODEL_GROUPS[args.model_group]

    predictions = []
    for model in models:

        if args.model_group is None:
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
