"""
This script implements evaluation metrics for different tasks.
"""

import time
from collections import Counter
from typing import List, Union

import numpy as np
from tqdm.auto import tqdm

from evaluate import load


def compute_rouge2_score(
    generations: List[str], references: Union[List[List[str]], List[str]]
) -> List[float]:
    """
    Computes ROUGE-2 score over a list of generations and references.

    Args:
        generations (List[str]): List of generated texts
        references (Union[List[List[str]], List[str]]): List of reference texts

    Returns:
        List[float]: List of ROUGE-2 scores
    """
    rouge = load("rouge")
    results = rouge.compute(
        predictions=generations, references=references, use_aggregator=False
    )
    return results["rouge2"]


def squad_acc(generations, references):
    correct = []
    for gen, ref in zip(generations, references):
        if ref.lower() in gen.lower():
            correct.append(1)
        else:
            correct.append(0)
    return correct


def trivia_qa_acc(generations, references):
    correct = []
    for gen, refs in zip(generations, references):
        gen_lower = gen.lower()
        if any(ref.lower() in gen_lower for ref in refs):
            correct.append(1)
        else:
            correct.append(0)
    return correct


# TODO: UPDATE THIS TO INCORPORATE LIST OF REFERENCES
def definition_extraction_acc(generations, references):
    correct = []
    for gen, ref in zip(generations, references):
        gen_lower = gen.lower()
        if ref.lower() in gen_lower:
            correct.append(1)
        else:
            correct.append(0)
    return correct


METRIC_FUNCS = {
    "rouge2": compute_rouge2_score,
    "squad_acc": squad_acc,
    "trivia_qa_acc": trivia_qa_acc,
    "definition_extraction_acc": definition_extraction_acc,
}

MULTI_MODEL_TASK2METRIC = {
    "cnn_dailymail": "rouge2",
    "definition_extraction": "definition_extraction_acc",
    "e2e_nlg": "rouge2",
    "squad": "squad_acc",
    "trivia_qa": "trivia_qa_acc",
    "web_nlg": "rouge2",
    "xsum": "rouge2",
}
