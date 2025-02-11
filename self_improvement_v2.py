"""
This script explores a self-improvement mechanism of LLMs using 3 different approaches:
1. Smoothie
2. Pick-Random
3. Single Model

It can apply a filtering method (Smoothie or Pick-Random) to generated data.
It then prepares the fine-tuning dataset and fine-tunes the target model using LoRA.
Finally, it evaluates the fine-tuned model on the test dataset by calculating accuracy.
"""

import argparse
import subprocess
import gc
import os
import numpy as np
import evaluate
import string
import json
import jsonlines
from pathlib import Path
from tqdm import tqdm
import torch
import re
import unicodedata
import wandb
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer

from utils import (MODEL_GROUPS, check_args, construct_predictions_path,
                       generate_per_sample_multi_prompt,
                       generate_per_sample_single_prompt, load_data_config,
                       load_hf_model, clean_generations)

from constants import HF_MODEL_MAX_LENGTHS, HF_MODELS

from self_improvement_v2_utils import normalize_pythia


# Define constants (for single model)
DATASET = "squad"
MODEL_GROUP = "3b"

# Define paths
TRAIN_FILE = "../smoothie_data/datasets/squad_train.jsonl"
TEST_FILE = "../smoothie_data/datasets/squad_test.jsonl"
DATASET_CONFIG_PATH = "../dataset_configs/squad.yaml"

# For generating data
DATA_DIR = "../smoothie_data/datasets"

RESULTS_DIR = Path("../smoothie_data/multi_model_results_old")  
FINE_TUNED_MODEL_DIR = Path(RESULTS_DIR) / "fine_tuned"


# Check that directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FINE_TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

wandb.init(
    project = "smoothie_self_improvement",

    config = {
        
    }
)



def apply_smoothie(teacher_models):
    """
    Applies the Smoothie algorithm to filter the generated data using a set of teacher models.

    Args:
        teacher_models (str): Identifier for the teacher model(s) or group.

    Returns:
        str: Path to the filtered data file.
    """
    print("Applying Smoothie algorithm...")

    # Define Smoothie parameters
    smoothie_type = "sample_dependent"  # or sample_independent
    use_full_text_embeddings = False      # Set to True if required
    k = 10                                 # Number of nearest neighbors
    n_generations = 1                     # Number of generations per sample
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct the command to run Smoothie
    cmd = [
        "python", "run_smoothie_train.py",
        "--model_group", teacher_models,
        "--dataset_config", DATASET_CONFIG_PATH,
        "--data_dir", DATA_DIR,
        "--results_dir", str(RESULTS_DIR),
        "--type", smoothie_type,
        "--k", str(k),
        "--n_generations", str(n_generations),
        "--device", device,
        "--multi_model"
    ]

    if use_full_text_embeddings:
        cmd.append("--use_full_text_embeddings")

    # Execute the command
    try:
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(result.stdout)
        print("Smoothie filtering completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Smoothie filtering failed with error:\n{e.stderr}")
        raise

    # Construct the path to the filtered data file
    # filtered_data_path = RESULTS_DIR / f"filtered_{teacher_models}.json"
    # return str(filtered_data_path)
    return



def apply_pick_random_baseline(teacher_models):
    """
    Applies the Pick Random Baseline algorithm to filter the generated data using a set of teacher models.

    Args:
        teacher_models (str): Identifier for the teacher model(s) or group.

    Returns:
        str: Path to the filtered data file.
    """
    print("Applying Pick Random Baseline algorithm...")

    # Define Pick-Random parameters
    n_generations = 1  # Number of generations per sample
    seed = 42          
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct the command to run Pick-Random
    cmd = [
        "python", "pick_random_baseline_train.py",
        "--model_group", teacher_models,
        "--dataset_config", DATASET_CONFIG_PATH,
        "--data_dir", DATA_DIR,
        "--results_dir", str(RESULTS_DIR),
        "--n_generations", str(n_generations),
        "--seed", str(seed),
        "--multi_model"
    ]

    # Execute the command
    try:
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(result.stdout)
        print("Pick-Random filtering completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Pick-Random filtering failed with error:\n{e.stderr}")
        raise

    # Construct the path to the filtered data file
    # filtered_data_path = RESULTS_DIR / f"filtered_pick_random_{teacher_models}.json"
    # return str(filtered_data_path)
    return



def apply_single_model(teacher_model):
    """
    Applies the Single-Model filter to use generations from a single teacher model.

    Args:
        teacher_model (str): Identifier for the single teacher model.

    Returns:
        str: Path to the generated data file.
    """
    print("Applying Single-Model filter...")

    generations_file = RESULTS_DIR / DATASET / MODEL_GROUP /f"{teacher_model}_train.json"

    # Check if the generations file exists
    if not generations_file.exists():
        print(f"Error: Generations file not found at {generations_file}")
        raise FileNotFoundError(f"Generations file not found: {generations_file}")

    print("Single-Model filtering completed successfully.")
    return str(generations_file)



def prepare_finetuning_data(input_file, generations_file, output_file):
    """
    Prepares the fine-tuning dataset by aligning generated answers with corresponding prompts.

    Args:
        input_file (str or Path): Path to the JSON Lines (.jsonl) file containing input data (e.g., training dataset).
        generations_file (str or Path): Path to the JSON file containing generated responses.
        output_file (str or Path): Path to save the prepared fine-tuning data in JSON format.

    Returns:
        None
    """
    print("Preparing fine-tuning data...")

    # Ensure paths are Path objects
    input_file = Path(input_file)
    generations_file = Path(generations_file)
    output_file = Path(output_file)

    # Load the input dataset
    try:
        with jsonlines.open(input_file, mode='r') as reader:
            input_data = list(reader)
        print(f"Loaded {len(input_data)} samples from input file: {input_file}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        raise

    # Load the generations
    try:
        with open(generations_file, 'r', encoding='utf-8') as f:
            generations_data = json.load(f)
            generations = generations_data.get("generations", [])
        print(f"Loaded {len(generations)} generations from file: {generations_file}")
    except FileNotFoundError:
        print(f"Error: Generations file not found at {generations_file}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {generations_file}: {e}")
        raise

    # Validate alignment
    if len(input_data) != len(generations):
        print(f"Error: Number of input samples ({len(input_data)}) does not match number of generations ({len(generations)}).")
        raise ValueError("Mismatched lengths between input data and generations.")

    # Prepare the fine-tuning dataset
    finetuning_data = {
        "responses": []
    }

    for idx, (sample, generation) in enumerate(zip(input_data, generations)):
        # Extract the input prompt
        input_prompt = sample.get("multi_model_prompt", "").strip()
        if not input_prompt:
            print(f"Warning: Missing 'multi_model_prompt' in sample index {idx}. Skipping.")
            continue

        # Extract the generated response
        if isinstance(generation, str):
            response_text = generation.strip()
        elif isinstance(generation, list) and generation:
            response_text = generation[0].strip()  # Use the first response in the list
        else:
            print(f"Warning: Invalid generation format in index {idx}. Skipping.")
            continue

        # Remove "Question:" and anything following it
        if "Question" in response_text:
            response_text = response_text.split("Question:")[0].strip()
        if "\n\n" in response_text:
            response_text = response_text.split("\n\n")[0].strip()

        # Append to the fine-tuning dataset
        finetuning_data["responses"].append({
            "input": input_prompt,
            "output": response_text
        })

    print(f"Prepared {len(finetuning_data['responses'])} fine-tuning samples.")

    # Save the fine-tuning dataset
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(finetuning_data, f, indent=4)
        print(f"Fine-tuning data saved to {output_file}")
    except Exception as e:
        print(f"Error saving fine-tuning data to {output_file}: {e}")
        raise



def normalize_text(text):
    """
    Normalize text by performing the following operations:
    1. Convert to lowercase.
    2. Remove periods.
    3. Replace dashes with spaces.
    4. Collapse multiple spaces into a single space.
    5. Remove "question:" and everything after it.
    6. Strip leading and trailing whitespace.
    
    Args:
        text (str): The text to normalize.
        
    Returns:
        str: The normalized text.
    """
    # Step 1: Convert to lowercase
    normalized = text.lower()
    
    # Step 2: Remove "question:" and everything after it, if present
    question_marker = "question:"
    answer_marker = "a:"

    if question_marker in normalized:
        normalized = normalized.split(question_marker, 1)[0]
    elif answer_marker in normalized:
        # If "a:" is present, take everything after it.
        normalized = normalized.split(answer_marker, 1)[1]
    
    # Step 3: Remove periods
    normalized = normalized.replace(".", "")
    
    # Step 5: Collapse multiple spaces into a single space
    normalized = ' '.join(normalized.split())

    normalized = normalized.split("\n", 1)[0].strip()
    
    # Step 6: Strip leading and trailing whitespace
    normalized = normalized.strip()
    
    return normalized

def evaluate_model(
    model,
    tokenizer,
    dataset_path,
    device,
    data_config,
    model_name,
    batch_size=16,
    rouge_threshold=0.5
):
    """
    Evaluate the model on a given dataset using batching and ROUGE-L F-measure
    as a metric to determine correctness.
    
    If the ROUGE-L F-measure >= `rouge_threshold`, we count the sample as correct.

    Args:
        model (AutoModelForCausalLM): The model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        dataset_path (str): Path to the evaluation dataset in JSON lines format.
        device (str): Device to run the model on.
        data_config (dict): Configuration including max_new_tokens.
        model_name (str): Name of the model being evaluated.
        batch_size (int): Number of samples to process in a single batch.
        rouge_threshold (float): Minimum ROUGE-L F-measure required to count the answer as correct.

    Returns:
        float: Accuracy of the model (percentage of answers above `rouge_threshold`).
    """
    print(f"Evaluating model on {dataset_path} with ROUGE-L threshold={rouge_threshold}...")

    gc.collect()
    torch.cuda.empty_cache()

    # Save memory by not storing all intermediate activations
    model.gradient_checkpointing_enable()

    # Load the ROUGE metric
    rouge_metric = evaluate.load("rouge")

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f.readlines()]

    correct = 0
    total = 0

    # Process dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i : i + batch_size]

        # Prepare inputs for the batch
        input_prompts = [sample["multi_model_prompt"] for sample in batch]
        reference_answers = [sample["reference"] for sample in batch]

        # Need this for Pythia models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize
        tokenized_batch = tokenizer(
            input_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)

        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_batch,
                max_new_tokens=60,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                temperature=0.0
            )

        generated_sequences = outputs.sequences
        generated_answers = []

        # Decode outputs
        for seq in generated_sequences:
            text = tokenizer.decode(seq, skip_special_tokens=True)

            # If there's an "Answer:" in the text, keep only what's after that
            if "Answer:" in text:
                text = text.split("Answer:", 1)[1].strip()
       

            generated_answers.append(text)

        # Compare each generated answer with the reference answer
        for gen_answer, ref_answer, prompt in zip(generated_answers, reference_answers, input_prompts):
            # 1) Basic normalization
            norm_gen = gen_answer.lower()
            norm_ref = ref_answer.lower()
            if norm_gen in norm_ref or norm_ref in norm_gen:
                correct += 1
                total += 1
                # print("---------------------------------------------")
                # print(f"Prompt: {prompt}")
                # print(f"Reference: {ref_answer}")
                # print(f"Generated: {gen_answer}")
                # print("---------------------------------------------")
                continue

            # Compute ROUGE for this single reference and prediction
            results = rouge_metric.compute(
                predictions=[norm_gen],
                references=[ref_answer]
            )
    
            rouge_1_score = results["rouge1"]

            # Decide correctness based on threshold
            if rouge_1_score >= rouge_threshold:
                correct += 1
                total += 1
                # print("---------------------------------------------")
                # print(f"Prompt: {prompt}")
                # print(f"Reference: {norm_gen}")
                # print(f"Generated: {ref_answer}")
                # print(f"ROUGE-1: {rouge_1_fmeasure:.3f} (> {rouge_threshold})")
                # print("---------------------------------------------")
                continue
            else:
                # Debug info for incorrect cases
                print("---------------------------------------------")
                print(f"Prompt: {prompt}")
                print(f"Reference: {norm_ref}")
                print(f"Generated: {norm_gen}")
                print(f"ROUGE-1 Score: {rouge_1_score:.3f} (< {rouge_threshold})")
                print("---------------------------------------------")
                total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy (ROUGE-based): {accuracy:.2f}% ({correct}/{total})")
    return accuracy



def fine_tune_model_lora(fine_tuning_data_path, model_name, output_dir, num_epochs=10):
    """
    Fine-tune the model on the provided dataset with LoRA.

    Args:
        fine_tuning_data_path (str): Path to the fine-tuning dataset.
        model_name (str): Name or path of the base model.
        output_dir (str): Directory to save the fine-tuned model.
        num_epochs (int): Number of training epochs.
    """

    print("Starting LoRA fine-tuning process...")

    gc.collect()
    torch.cuda.empty_cache()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Save memory by not storing all intermediate activations
    model.gradient_checkpointing_enable()

    # Need this for Pythia models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(r=32, lora_alpha=64, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)

    # Load and preprocess fine-tuning dataset
    print(f"Loading dataset from {fine_tuning_data_path}...")
    with open(fine_tuning_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        responses = data.get("responses", [])

    if not responses:
        raise ValueError(f"No responses found in dataset: {fine_tuning_data_path}")

    dataset = Dataset.from_list(responses)

    # Tokenize the dataset
    def tokenize_function(examples):
        inputs = tokenizer(examples["input"], truncation=True, max_length=64, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["output"], truncation=True, max_length=64, padding="max_length")
        labels["input_ids"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels["input_ids"]}

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=num_epochs,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",   # log once per epoch
        logging_steps=1,           # ensures logging each epoch
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("LoRA fine-tuning completed. Saving the model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def fine_tune_model(fine_tuning_data_path, model_name, output_dir, num_epochs=1):
    """
    Fine-tune the model on the provided dataset.
    Args:
        fine_tuning_data_path (str): Path to the fine-tuning dataset.
        model_name (str): Name or path of the base model.
        output_dir (str): Directory to save the fine-tuned model.
        num_epochs (int): Number of training epochs.
    """
    print("Staring fine-tuning process...")

    gc.collect()
    torch.cuda.empty_cache()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name)

    # Save memory by not storing all intermediate activations
    model.gradient_checkpointing_enable()

    # Need this for Pythia models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess fine-tuning dataset
    with open(fine_tuning_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        responses = data.get("responses", [])

    if not responses:
        raise ValueError(f"No responses found in dataset: {fine_tuning_data_path}")

    dataset = Dataset.from_list(responses)

    # Tokenize the dataset
    def tokenize_function(examples):
        inputs = tokenizer(examples["input"], truncation=True, max_length=64, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["output"], truncation=True, max_length=64, padding="max_length")
        labels["input_ids"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels["input_ids"]}

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=num_epochs,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",
        logging_steps=1,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    print("Fine-tuning completed. Saving the model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)




def main():
    teacher_models = "3b"

    # Model Filtering
    # Apply Smoothie filtering
    # try:
    #     filtered_path = apply_smoothie(teacher_models)
    #     print(f"Filtered data saved to: poo")
    # except Exception as e:
    #     print(f"Error during Smoothie application: {e}")

    # Apply Pick-Random filtering
    # try:
    #     filtered_path = apply_pick_random_baseline(teacher_models)
    #     print(f"Filtered data saved to: poo_2")
    # except Exception as e:
    #     print(f"Error during Pick-Random application: {e}")

    # Apply Single-Model filtering
    # try:
    #     generations_path = apply_single_model(teacher_models)
    #     print(f"Generations file used: {generations_path}")
    # except Exception as e:
    #     print(f"Error during Single-Model application: {e}")

    # parser = argparse.ArgumentParser(description="Training Loop for LLM Improvement")
    # parser.add_argument("--filter", choices=["smoothie", "pick_random", "single_model"], required=True, help="Filtering method to use.")
    # parser.add_argument("--teacher", required=True, help="Model(s) used to generate synthetic data")
    # parser.add_argument("--target", required=True, help="Model to be fine-tuned on")


    # Prepare fine-tuning data

    # input_file = "../smoothie_data/datasets/squad_train.jsonl"  # Replace with your input file path
    # generations_file = "../smoothie_data/multi_model_results_old/squad/7b/llama-2-7b_train.json"  # Replace with your generations file path

    # generations_file_path = Path(generations_file)
    # output_file = str(generations_file_path.with_name(f"{generations_file_path.stem}_finetuning_v4_{generations_file_path.suffix}"))


    # try:
    #     prepare_finetuning_data(input_file, generations_file, output_file)
    # except Exception as e:
    #     print(f"Error during fine-tuning data preparation: {e}")



    #model_name = "google/gemma-2b"
    #model_name = "EleutherAI/pythia-2.8b"
    model_name = "EleutherAI/pythia-1b"
    train_dataset_path = "../smoothie_data/datasets/squad_train.jsonl"
    test_dataset_path = "../smoothie_data/datasets/squad_test.jsonl"
    fine_tuning_data_path = "../smoothie_data/multi_model_results_old/squad/3b/gemma-2b_train_finetuning_v4_.json"
    output_dir = "../smoothie_data/multi_model_results_old/fine_tuned" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_config = load_data_config(argparse.Namespace(dataset_config=DATASET_CONFIG_PATH))

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.to(device)

    # Evaluate on train dataset
    print("Evaluating on train dataset before fine-tuning...")
    # Evaluate with ROUGE
    train_accuracy = evaluate_model(
    model=base_model,
    tokenizer=tokenizer,
    dataset_path=train_dataset_path,
    device=device,
    data_config=data_config,
    model_name=model_name,
)
  

    #Fine-tune the model
    fine_tune_model(fine_tuning_data_path, model_name, output_dir)


    # Load the fine-tuned model
    print("Loading fine-tuned model...")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(output_dir)
    fine_tuned_model.to(device)

    # Evaluate on test dataset
    print("Evaluating on test dataset after fine-tuning...")
    test_accuracy = evaluate_model(fine_tuned_model, tokenizer, test_dataset_path, device, data_config, model_name=model_name)

    print(f"Train Accuracy Before Fine-tuning: {train_accuracy:.2f}%") 
    print(f"Test Accuracy After Fine-tuning: {test_accuracy:.2f}%")



if __name__ == "__main__":
    main()
