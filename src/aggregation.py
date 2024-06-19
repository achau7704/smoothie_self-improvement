"""
Contains implementations of different aggregation methods
"""

from typing import List

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import *
from src.model import Generator


class Aggregator:
    """
    An aggregator object handles combining the outputs of multiple prompts using some aggregation algorithm.
    """

    def __init__(
        self,
        device: str,
        model_name: str,
        method: str,
        hf_cache_dir: str,
    ):
        """
        Args:
            - device (str): Device to use. Must be one of "cpu" or "cuda".
            - model_name (str): Model to use. Must be one of the models in HF_MODELS
            - method (str): Aggregation method to use. Must be one of ["uniform"]
            - hf_cache_dir (str): Directory to cache HF datasets to


        """
        self.device = device
        self.model_name = model_name
        self.hf_cache_dir = hf_cache_dir

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODELS[model_name], cache_dir=hf_cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            HF_MODELS[model_name], cache_dir=hf_cache_dir
        )
        self.model.to(device)

        self.method = method
        self.max_length = HF_MODEL_MAX_LENGTHS[model_name]

    def aggregate(
        self,
        prompts: List,
        max_tokens: int,
        on_probabilities: bool,
        weights: np.ndarray = None,
    ):
        """
        Aggregates the results of the model on the given prompts. Performs aggregation in probability space.

        Args:
            prompts (list): List of prompts to aggregate
            max_tokens (int): Maximum number of tokens to generate
            on_probabilities (bool): Whether to aggregate on probabilities or logits
            weights (np.ndarray): Weights to use for aggregation. Only used for some methods.

        Returns:
            generated_token_idxs (list): List of token indices generated at each time step
            text (str): Generated text
            selected_idxs (list): List of indices of selected prompts at each time step
        """

        # Create list of generators
        generators = []
        for prompt in prompts:
            generators.append(
                Generator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_length=self.max_length,
                )
            )

        generated_token_idxs = []  # Track generated tokens
        for step in range(max_tokens):
            outputs = []  # Track output from each generator

            # Randomly shuffle the generators. At some point in the future
            # we might implement an early stopping algorithm based on whether the
            # first generators agree with each other. In that case, looking at the generators in
            # a different order each time is helpful. But right now it doesn't matter.
            generator_idx_order = np.random.permutation(len(generators))

            for generator_idx in generator_idx_order:
                generator = generators[generator_idx]
                outputs.append(
                    generator.get_output(return_probabilities=on_probabilities)
                )

            outputs = np.array(outputs)
            aggregated_outputs = combine_outputs(
                outputs=outputs,
                method=self.method,
                weights=weights,
            )

            # Pick the token with the highest probability
            selected_idx = aggregated_outputs.argmax()
            generated_token_idxs.append(selected_idx)

            # Update the generators
            for generator in generators:
                generator.update(generated_token_idxs[-1])

            if generated_token_idxs[-1] == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated_token_idxs, skip_special_tokens=True)
        return text


def combine_outputs(outputs: np.ndarray, method: str, weights: np.ndarray = None):
    """
    Given a set of outputs (either logits or probabilities) over the next token (corresponding to each prompt), combines them using the given method.

    Args:
        outputs (np.ndarray): Array of either probabilities or logits. Shape: (n_prompts, vocab_size)
        method (str): Method to use. Must be one of ["uniform"]
        weights (np.ndarray): Weights to use for aggregation. Only used for some methods.

    Returns
        combined_probs (np.ndarray): Combined probabilities
    """

    if method == UNIFORM_AVG:
        return outputs.mean(axis=0)
    elif method == SMOOTHIE:
        # Compute the weighted average of the probabilities
        if weights is None:
            raise ValueError("Weights must be provided for Smoothie aggregation")
        return (outputs.T * weights).T.sum(axis=0)
    else:
        raise ValueError(f"Strategy {method} not found")
