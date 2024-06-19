import json
import os
import numpy as np

MODEL_GROUPS = {
    "7b": ["mistral-7b", "llama-2-7b", "vicuna-7b", "gemma-7b", "nous-capybara"],
    "3b": ["phi-2", "pythia-2.8b", "gemma-2b", "incite-3b", "dolly-3b"],
}


def load_predictions(predictions_dir, split, args, final=True):
    """
    Load predictions from a given split.

    Args:
    - predictions_dir (Path): The directory containing the predictions.
    - split (str): The split to load predictions for.

    Returns:
    - list: The predictions for the split.
    """
    models = MODEL_GROUPS[args.model_group]
    predictions = []
    for model in models:
        if args.n_generations > 1 and not final:
            fpath = predictions_dir / f"{model}_{args.n_generations}_gens_{split}.json"
        else:
            fpath = predictions_dir / f"{model}_{split}.json"
        with open(fpath, "r") as f:
            predictions.append(json.load(f)["generations"])

    predictions = np.array(predictions).T
    assert predictions.shape[1] == len(models)
    return predictions
