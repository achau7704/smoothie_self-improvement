import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.console import console
from src.evaluate.metrics import METRIC_FUNCS, gsm8k_majority_vote
from src.utils import clean_generations


class Scorer:
    """
    A simple class to store scores. Handles loading scores from file, updating scores, and saving scores to file.

    Saves two levels of scores: one containing per-sample scores, and the other containing dataset-level scores.
    """

    def __init__(
        self, predictions_dir: Path, data_config: Dict, args: argparse.Namespace
    ):
        """

        Args:
            predictions_dir: Path to the directory containing predictions.
            data_config: The dataset config.
            args: The argparse args.

        """
        self.args = args
        self.data_config = data_config
        self.summary_scores_fpath = predictions_dir / "summary_scores.json"
        self.sample_scores_fpath = predictions_dir / "sample_scores.json"

        # Load summary scores if they exist
        self.summary_scores = {}
        if self.summary_scores_fpath.exists() and not self.args.redo:
            console.log(
                f"Summary scores already exist at {self.summary_scores_fpath}. Will add to them."
            )
            self.summary_scores = json.loads(self.summary_scores_fpath.read_text())

        # Load sample scores if they exist
        self.sample_scores = {}
        if self.sample_scores_fpath.exists() and not self.args.redo:
            console.log(
                f"Sample scores already exist at {self.sample_scores_fpath}. Will add to them."
            )
            self.sample_scores = json.loads(self.sample_scores_fpath.read_text())

        for metric in data_config["metrics"]:
            if metric not in self.summary_scores:
                self.summary_scores[metric] = {}
            if metric not in self.sample_scores:
                self.sample_scores[metric] = {}

    def score_method(self, predictions_fpath: Path, metric: str, references: List[str]):
        """
        Scores a method for a given metric.

        Args:
            predictions_fpath: Path to the predictions file.
            metric: The metric to use.
            references: The references.

        """
        assert (
            "_test" in predictions_fpath.stem
        ), f"Expected predictions file to contain '_test'. Got {predictions_fpath.stem}"
        method_name = predictions_fpath.stem.replace("_test", "")

        # Check if the method has already been scored
        if (
            method_name in self.summary_scores[metric]
            and method_name in self.sample_scores[metric]
        ) and not self.args.redo:
            console.log(f"Skipping {method_name} ({metric}) because it already exists.")
            return

        metric_func = METRIC_FUNCS[metric]
        predictions_dict = json.loads(predictions_fpath.read_text())
        if not "generations" in predictions_dict:
            console.log(
                f"Skipping {method_name} ({metric}) because it has no generations."
            )
            return
        generations = predictions_dict["generations"]

        if "individual" in method_name:
            for prompt_idx in range(len(generations[0])):
                if (
                    f"individual_{prompt_idx}" in self.summary_scores[metric]
                    and f"individual_{prompt_idx}" in self.sample_scores[metric]
                    and not self.args.redo
                ):
                    console.log(
                        f"Skipping individual_{prompt_idx} ({metric}) because it already exists."
                    )
                    continue

                console.log(f"Evaluating {method_name}_{prompt_idx} ({metric})")
                prompt_generations = [sample[prompt_idx] for sample in generations]
                cleaned_generations = clean_generations(
                    prompt_generations, self.data_config
                )
                scores = metric_func(cleaned_generations, references)
                self.sample_scores[metric][f"{method_name}_{prompt_idx}"] = list(scores)
                self.summary_scores[metric][f"{method_name}_{prompt_idx}"] = np.mean(
                    scores
                )

        elif method_name.startswith("labeled_oracle"):
            scores = []
            for trial_generations in generations:
                cleaned_generations = clean_generations(
                    trial_generations, self.data_config
                )
                scores = metric_func(cleaned_generations, references)
                scores.append(np.mean(scores))
            self.summary_scores[metric][method_name] = np.mean(scores)

        elif method_name.startswith("labeled_knn"):
            scores = []
            for trial_generations in generations:
                cleaned_generations = clean_generations(
                    trial_generations, self.data_config
                )
                scores = metric_func(cleaned_generations, references)
                scores.append(np.mean(scores))
            self.summary_scores[metric][method_name] = np.mean(scores)

        elif method_name.startswith("pick_random"):
            console.log(f"Evaluating {method_name} ({metric})")
            trial_scores = []
            for trial_generations in generations:
                cleaned_generations = clean_generations(
                    trial_generations, self.data_config
                )
                trial_scores.append(metric_func(cleaned_generations, references))
            trial_scores = np.array(trial_scores)
            self.sample_scores[metric][method_name] = list(trial_scores.mean(axis=0))
            self.summary_scores[metric][method_name] = trial_scores.mean()
        else:
            console.log(f"Evaluating {method_name} ({metric})")
            cleaned_generations = clean_generations(generations, self.data_config)
            scores = metric_func(cleaned_generations, references)
            self.sample_scores[metric][method_name] = list(scores)
            self.summary_scores[metric][method_name] = np.mean(scores)

    def score_train(self, predictions_fpath: Path, metric: str, references: List[str]):
        """
        This is an oracle where we imagine we have access to a small amount of labeled training data.
        We use this training data to select the best generator, and then evaluate that generator on the test set.
        """

        if "oracle_best_prompt" in self.summary_scores and not self.args.redo:
            console.log(f"Skipping oracle_best_prompt because it already exists.")
            return

        metric_func = METRIC_FUNCS[metric]
        predictions_dict = json.loads(predictions_fpath.read_text())
        generations = predictions_dict["generations"]
        prompt_scores = []
        for prompt_idx in range(len(generations[0])):
            console.log(f"Evaluating train_best_prompt_{prompt_idx} ({metric})")
            prompt_generations = [sample[prompt_idx] for sample in generations]
            cleaned_generations = clean_generations(
                prompt_generations, self.data_config
            )
            scores = metric_func(cleaned_generations, references)
            prompt_scores.append(np.mean(scores))

        best_prompt_idx = np.argmax(prompt_scores)
        self.summary_scores[metric]["train_best_prompt"] = self.summary_scores[metric][
            f"individual_{best_prompt_idx}"
        ]

    def score_gsm8k_majority_vote(self, predictions_fpath: Path, references: List[str]):
        """
        Special function to compute majority vote for GSM8K.
        """
        assert (
            "_test" in predictions_fpath.stem
        ), f"Expected predictions file to contain '_test'. Got {predictions_fpath.stem}"
        metric = "gsm8k_acc"
        method_name = "majority_vote"

        # Check if the method has already been scored
        if (
            method_name in self.summary_scores[metric]
            and method_name in self.sample_scores[metric]
        ) and not self.args.redo:
            console.log(f"Skipping {method_name} ({metric}) because it already exists.")
            return

        predictions_dict = json.loads(predictions_fpath.read_text())
        generations = predictions_dict["generations"]

        # Construct clean generations
        cleaned_generations = []
        for sample_generations in generations:
            cleaned_generations.append(
                clean_generations(sample_generations, self.data_config)
            )
        cleaned_generations = np.array(cleaned_generations)
        print(cleaned_generations.shape)
        scores = gsm8k_majority_vote(cleaned_generations, references)
        self.sample_scores[metric][method_name] = list(scores)
        self.summary_scores[metric][method_name] = np.mean(scores)

    def save(self):
        """
        Saves the scores to file.
        """
        self.summary_scores_fpath.write_text(json.dumps(self.summary_scores, indent=4))
        self.sample_scores_fpath.write_text(json.dumps(self.sample_scores, indent=4))


def get_web_nlg_references(reference_list):
    """
    The webnlg dataset has references in a different format than the other datasets. This function converts the references to the format used by the other datasets.
    """
    new_references = []
    for reference in reference_list:
        new_references.append([])
        for text, comment in zip(reference["text"], reference["comment"]):
            if comment == "good":
                new_references[-1].append(text)
    return new_references


def get_gsm8k_references(reference_list):
    """
    The GSM8K dataset has references in a different format than the other datasets. This function converts the references to the format used by the other datasets.
    """
    new_references = []
    for reference in reference_list:
        reference = reference.split("####")[-1].strip()
        new_references.append(reference)
    return new_references


def get_trivia_qa_references(reference_list):
    """
    The TriviaQA dataset has references in a different format than the other datasets. This function converts the references to the format used by the other datasets.
    """
    new_references = []
    for reference in reference_list:
        new_references.append(reference["normalized_aliases"])
    return new_references


def get_references(dataset, config):
    if config["dataset"] == "web_nlg":
        return get_web_nlg_references(dataset[config["reference_key"]].tolist())
    elif config["dataset"] == "gsm8k":
        return get_gsm8k_references(dataset[config["reference_key"]].tolist())
    elif config["dataset"] == "trivia_qa":
        return get_trivia_qa_references(dataset[config["reference_key"]].tolist())
    else:
        return dataset[config["reference_key"]].tolist()
