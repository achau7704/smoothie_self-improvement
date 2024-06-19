"""
This script implements evaluation metrics for different tasks.
"""

import time
from collections import Counter

import numpy as np
from nltk.translate import meteor
from nltk.translate.bleu_score import sentence_bleu
from openai import OpenAI
from tqdm.auto import tqdm

from evaluate import load

client = OpenAI(api_key="sk-proj-aPPKdGdf1VmZP679chveT3BlbkFJp8SXIGhLuqeSel0BlXmi")


def compute_bert_score(
    generations, references, model_name="microsoft/deberta-xlarge-mnli"
):
    """
    Compute BERTScore between generations and references.

    The original bertscore function returns an object of the form:
        {
            'precision': [1.0, 1.0],
            'recall': [1.0, 1.0],
            'f1': [1.0, 1.0],
            'hashcode': 'distilbert-base-uncased_L5_no-idf_version=0.3.10(hug_trans=4.10.3)'
        }

    The best model is microsoft/deberta-xlarge-mnli.

    Args:
        generations (list): List of generated sentences
        references (list): List of reference sentences
        model_name (str): Name of the model used to generate the sentences
    """
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=generations, references=references, model_type=model_name
    )

    # Return f1
    return results["f1"]


def compute_rouge1_score(generations, references):
    """
    Compute ROUGE-1 score between generations and references.

    Args:
        generations (list): List of generated sentences
        references (list): List of reference sentences
    """
    rouge = load("rouge")
    results = rouge.compute(
        predictions=generations, references=references, use_aggregator=False
    )
    return results["rouge1"]


def compute_rouge2_score(generations, references):
    """
    Compute ROUGE score between generations and references.

    Args:
        generations (list): List of generated sentences
        references (list): List of reference sentences
    """
    rouge = load("rouge")
    results = rouge.compute(
        predictions=generations, references=references, use_aggregator=False
    )
    return results["rouge2"]


def compute_rougeL_score(generations, references):
    """
    Compute ROUGE score between generations and references.

    Args:
        generations (list): List of generated sentences
        references (list): List of reference sentences
    """
    rouge = load("rouge")
    results = rouge.compute(
        predictions=generations, references=references, use_aggregator=False
    )
    return results["rougeL"]


def compute_bleu_score(generations, references):
    """
    Compute BLEU score between generations and references.

    Args:
        generations (list): List of generated sentences
        references (list): List of reference sentences
    """
    scores = []
    for i in range(len(generations)):
        reference_words = [references[i].split()]
        generation_words = generations[i].split()
        score = sentence_bleu(reference_words, generation_words)
        scores.append(score)
    return {"bleu": scores}


def compute_meteor_score(generations, references):
    """
    Compute METEOR score between generations and references.

    Args:
        generations (list): List of generated sentences
        references (list): List of reference sentences
    """
    scores = []
    for i in range(len(generations)):
        reference = [references[i].split()]
        generation = generations[i].split()
        score = meteor(reference, generation)
        scores.append(score)
    return scores


def compute_gpt3_similarity_score(generations, references):
    """
    Compute GPT3.5 score between generations and references.

    Args:
        generations (list): List of generated sentences
        references (list): List of reference sentences
    """
    PROMPT = """You will given human generated reference text and a machine-generated text. Your task is to score the semantic and lexical similarity between the two texts. The score should be between 0 and 5, where 0 means the two texts are completely dissimilar and 5 communicate effectively the same information.

Reference: {reference}
Candidate: {generation}

Return the score as a JSON object with the key "score" and the value as the score"""
    scores = []
    for generation, reference in tqdm(
        zip(generations, references), total=len(generations)
    ):
        prompt = PROMPT.format(reference=reference, generation=generation)
        print(prompt)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        eval_gen = completion.choices[0].message.content
        score_dict = eval(eval_gen)
        score = score_dict["score"]
        scores.append(score)

        # Wait for 15 seconds every 100 requests
        if len(scores) % 100 == 0:
            time.sleep(15)

    return scores


def compute_gpt3_comparison(document, reference, summary1, summary2):
    """
    Given a document, a reference, and two candidate summaries, prompts GPT-3.5 to return which summary is better.

    Returns True if summary1 is better, and False if summary2 is.

    """
    EVAL_TEMPLATE = """Select the output (a) or (b) that provides the best summary for the document with respect to the ground-truth summary. Your answer should ONLY contain: Output (a) or Output (b). Here's an example:

### Document:
{document}

### Ground-truth summary:
{reference}

### Output (a):
{summary1}

### Output (b):
{summary2}

### Which is best, Output (a) or Output (b)?"""
    prompt = EVAL_TEMPLATE.format(
        document=document, reference=reference, summary1=summary1, summary2=summary2
    )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    eval_gen = completion.choices[0].message.content
    if "Output (a)" in eval_gen:
        return True
    else:
        return False


def extract_gsm8k_answer(generation):
    """
    Extract the answer from a GSM-8K generation.

    Args:
        generation (str): Generated answer
    """
    generation = generation.split("\n\n")[0]
    lines = generation.split("\n")
    answer = None
    for line in lines:
        prefix = "The answer is"
        line = line.strip()
        if line.startswith(prefix):
            answer = line[len(prefix) :].strip()
            if len(answer.split()) > 1:
                # In case the answer has units appended
                answer = answer.split()[0]
            if answer.startswith("$"):
                # In case the answer is a monetary value
                answer = answer[1:]
    return str(answer)


def gsm8k_acc(generations, references):
    """
    Compute GSM-8K score between generations and references.

    Args:
        generations (list): List of generated answers
        references (list): List of reference sentences
    """
    answers = []
    for generation in generations:
        answer = extract_gsm8k_answer(generation)
        answers.append(str(answer))

    correct = []
    for answer, reference in zip(answers, references):
        if answer == str(reference):
            correct.append(1)
        else:
            correct.append(0)
    return correct


def gsm8k_majority_vote(generations, references):
    """
    Compute GSM-8K majority vote between generations and references.
    """
    n_samples, n_prompts = generations.shape
    majority_vote_predictions = []
    for sample_idx in range(n_samples):
        predictions = []
        for prompt_idx in range(n_prompts):
            generation = generations[sample_idx, prompt_idx]
            answer = extract_gsm8k_answer(generation)
            predictions.append(answer)
        majority_vote = Counter(predictions).most_common(1)[0][0]
        majority_vote_predictions.append(majority_vote)

    correct = []
    for answer, reference in zip(majority_vote_predictions, references):
        if answer == str(reference):
            correct.append(1)
        else:
            correct.append(0)
    return correct


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
    "rouge1": compute_rouge1_score,
    "rouge2": compute_rouge2_score,
    "rougeL": compute_rougeL_score,
    "bert_score": compute_bert_score,
    "meteor": compute_meteor_score,
    "gsm8k_acc": gsm8k_acc,
    "gsm8k_majority_vote": gsm8k_majority_vote,
    "squad_acc": squad_acc,
    "trivia_qa_acc": trivia_qa_acc,
    "definition_extraction_acc": definition_extraction_acc,
}
