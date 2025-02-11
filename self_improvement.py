"""
This script performs the following steps:
1. Generates synthetic data using teacher models.
2. Applies a filtering method (Smoothie or Pick-Random) to the generated data.
3. Prepares the fine-tuning dataset.
4. Fine-tunes the target model using LoRA.
5. Evaluates the fine-tuned model on the test dataset by calculating accuracy.
"""
import subprocess
import os
import argparse
import json
import time
import re
import unicodedata
from pathlib import Path
from typing import Dict, Union, List

import jsonlines
from tqdm.auto import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)

import yaml  # For loading YAML config files

from utils import (
    construct_smoothie_train_time_predictions_path,
    load_data_config,
    load_predictions,
    clean_generations,
    construct_smoothie_predictions_path,
    construct_pick_random_predictions_path,
    get_references,
    MODEL_GROUPS  # Imported for handling model groups
)

from data_utils import get_reference
from datasets import load_dataset


# Define paths
TRAIN_FILE = "../smoothie_data/datasets/squad_train.jsonl"
TEST_FILE = "../smoothie_data/datasets/squad_test.jsonl"
DATASET_CONFIG_PATH = "../dataset_configs/squad.yaml"  # Update accordingly
DATA_DIR = "../smoothie_data/datasets"  # Update accordingly
RESULTS_DIR = Path("../smoothie_data/multi_model_results_old")  # Update accordingly
GENERATED_DATA_DIR = Path(RESULTS_DIR) / "data"
FILTERED_DIR = Path(RESULTS_DIR) / "filtered"
FINE_TUNED_MODEL_DIR = Path(RESULTS_DIR) / "fine_tuned"

# Check that directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FILTERED_DIR.mkdir(parents=True, exist_ok=True)
FINE_TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def generate_data(teacher):
    """
    Generates synthetic data using teacher models.

    Args:
        teacher (str): The model(s) used to generate the synthetic data.
    """
    cmd = [
        "python", "get_generations.py",
        "--dataset_config", DATASET_CONFIG_PATH,
        "--data_dir", DATA_DIR,
        "--results_dir", str(GENERATED_DATA_DIR),
        "--n_generations", "1",  # Adjust as needed
        "--seed", "42",
    ]

    if teacher in MODEL_GROUPS:
        cmd.append([
            "--model_group, teacher",
            "--multi_model"
        ])
    else:
        cmd.extend([
            "--model", teacher
        ])
    subprocess.run(cmd, check=True)
    print("Data generation completed.")

    
def apply_smoothie(data_config, teacher_models):
    """
    Applies the Smoothie algorithm to filter the generated data.
    """
    print("Applying Smoothie algorithm...")

    # Define parameters for Smoothie
    smoothie_type = "sample_independent" 
    use_full_text_embeddings = False  # Set to True if needed
    k = 5  # Number of nearest neighbors
    n_generations = 1  # Based on data generation setup
    model_group = teacher_models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = str(RESULTS_DIR)

    # Create a minimal args Namespace required by utils.py
    class Args:
        def __init__(self):
            self.model_group = model_group
            self.multi_model = True
            self.type = smoothie_type
            self.k = k
            self.n_generations = n_generations
            self.use_full_text_embeddings = use_full_text_embeddings
            self.device = device
            self.test = False  # Assuming test is False; adjust if needed
            self.results_dir = results_dir

    args = Args()

    # Define the command to run run_smoothie_train.py
    cmd = [
        "python", "run_smoothie_train.py",
        "--model_group", args.model_group,  
        "--device", args.device,  # e.g., "cuda"
        "--dataset_config", DATASET_CONFIG_PATH,
        "--data_dir", str(DATA_DIR),  # Ensure this points to the correct directory
        "--results_dir", str(RESULTS_DIR),
        "--type", args.type,
        "--k", str(args.k),
        "--n_generations", str(args.n_generations),
    ]

    if args.use_full_text_embeddings:
        cmd.append("--use_full_text_embeddings")
    if args.multi_model:
        cmd.append("--multi_model")

    # Execute the Smoothie filtering command with a progress bar
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600  # 10 minutes timeout; adjust as needed
        )
        print(result.stdout)
        print("Smoothie filtering completed successfully.")
    except subprocess.TimeoutExpired:
        print("Smoothie filtering timed out.")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Smoothie filtering failed with error:\n{e.stderr}")
        exit(1)

    # Update this return statement if the path is different
    return construct_smoothie_predictions_path(data_config, teacher_models, args)


def apply_pick_random_baseline(data_config, teacher_models):
    """
    Applies the pick-random baseline to filter the generated data by invoking pick_random_baseline.py.
    
    Args:
        data_config (dict): The data configuration loaded from the dataset config.
        teacher_models (str): Identifier for the teacher model(s).

    
    Returns:
        Path: The path to the filtered generations file.
    """
    print("Applying pick-random baseline...")

    # Define parameters for pick-random baseline
    # These can be adjusted as needed
    model_group = teacher_models
    multi_model = True
    n_generations = 1
    seed = 42

    # Create a minimal args Namespace required by utils.py
    class Args:
        def __init__(self):
            self.model_group = model_group
            self.multi_model = multi_model
            self.n_generations = n_generations
            self.seed = seed
            self.test = False  # Assuming test is False; adjust if needed
            self.results_dir = str(RESULTS_DIR)
            self.pick_random_run_id = 0  # Index of generations we want to use

    args = Args()

    # Construct the predictions path using utility functions

    # Define the command to run pick_random_baseline.py
    cmd = [
        "python", "pick_random_baseline_train.py",
        "--dataset_config", DATASET_CONFIG_PATH,
        "--data_dir", str(DATA_DIR),
        "--results_dir", str(RESULTS_DIR),
        "--model_group", args.model_group,  # Adjust if using model groups
        "--multi_model",
        "--n_generations", str(args.n_generations),
        "--seed", str(args.seed),
        "--pick_random_run_id", str(args.pick_random_run_id)
    ]

    # Execute the pick-random baseline filtering command
    try:
        subprocess.run(cmd, check=True)
        print("Pick-random baseline filtering completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running pick_random_baseline.py: {e}")
        exit(1)

    return construct_pick_random_predictions_path(data_config, teacher_models, args)


def apply_single_model(data_config, teacher_model):
    """
    Applies the Single-Model filter to use generations from a single teacher model.

    Args:
        data_config (dict): The data configuration loaded from the dataset config.
        teacher_model (str): Identifier for the single teacher model.

    Returns:
        Path: The path to the generated generations file.
    """
    print("Applying Single-Model filter...")

    # Define the path to the generated data
    # Assuming that generate_data with a single model saves the generations in a specific path
    # Modify this according to how your generate_data function names the output files
    generations_file = GENERATED_DATA_DIR / f"{teacher_model}_train.json"

    # Check if the generations file exists
    if not generations_file.exists():
        print(f"Error: Generations file not found at {generations_file}")
        exit(1)

    print("Single-Model filtering completed successfully.")
    return str(generations_file)



def prepare_finetuning_data(generations_file, input_file, output_file):
    """
    Prepares the fine-tuning dataset by aligning generated answers with corresponding prompts.
    
    This function processes the generations and squad test samples to create a dataset suitable 
    for fine-tuning a language model. It ensures that each generated answer corresponds to the 
    correct input prompt and saves the aligned data in a JSON Lines (.jsonl) format.
    
    Args:
        generations_file (str or Path): 
            Path to the JSON file containing generated answers under the "generations" key.
            
        input_file (str or Path): 
            Path to the JSON Lines (.jsonl) file containing  test samples
            
        output_file (str or Path): 
            Path where the prepared fine-tuning data will be saved in JSON Lines (.jsonl) format.
        
    Returns:
        None
    """
    print(f"Preparing fine-tuning data...")
    # Ensure Path objects
    generations_file = Path(generations_file)
    input_file = Path(input_file)
    output_file = Path(output_file)
    
    # 1. Load Generations
    try:
        with open(generations_file, 'r', encoding='utf-8') as f:
            generations_data = json.load(f)
            generations = generations_data.get("generations", [])
        print(f"Loaded {len(generations)} generations from {generations_file}")
    except FileNotFoundError:
        print(f"Error: Generations file not found at {generations_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {generations_file}: {e}")
        return
    
    # 2. Load Test Samples
    samples = []
    try:
        with jsonlines.open(input_file, mode='r') as reader:
            for obj in reader:
                samples.append(obj)
        print(f"Loaded {len(samples)} samples from {input_file}")
    except FileNotFoundError:
        print(f"Error: Squad test file not found at {input_file}")
        return
    except Exception as e:
        print(f"Error reading squad test file: {e}")
        return
    
    # 3. Validate Alignment
    if len(generations) != len(samples):
        print(f"Error: Number of generations ({len(generations)}) does not match number of samples ({len(samples)}).")
        return
    else:
        print("Validation Passed: Number of generations matches number of samples.")
    
    # 4. Prepare Fine-Tuning Data
    finetuning_data = []
    for idx, (sample, generation) in enumerate(zip(samples, generations)):
        # Extract input prompt
        input_prompt = sample.get('multi_model_prompt', '').strip()
        if not input_prompt:
            print(f"Warning: Missing 'multi_model_prompt' for sample idx {idx}. Skipping.")
            continue

        # Extract reference answer
        reference = sample.get('reference', '').strip()
        if not reference:
            print(f"Warning: Missing 'reference' in sample index {idx}. Skipping.")
            continue
        
        # Extract answer from generation
        # Assuming the answer is before the '\n\nQuestion:' delimiter
        if '\n\nQuestion:' in generation:
            answer = generation.split('\n\nQuestion:')[0].strip()
        elif '\n\nAnswer:' in generation:
            answer = generation.split('\n\nAnswer:')[0].strip()
        else:
            # If no delimiter, assume entire generation is the answer
            answer = generation.strip()
        
        if not answer:
            print(f"Warning: Empty answer for sample idx {idx}. Skipping.")
            continue
        
        # # Debugging: Inspect a sample and its generation
        # if idx < 5:  # Inspect first 5 samples
        #     print(f"---\nSample {idx + 1}:")
        #     print(f"Prompt:\n{input_prompt}")
        #     print(f"Reference Answer: {reference}")
        #     print(f"Generated Answer: {answer}\n---")

        # Append to fine-tuning data
        finetuning_data.append({
            "input": input_prompt,
            "output": answer
        })
    
    print(f"Prepared {len(finetuning_data)} fine-tuning samples.")
    
    # 5. Save Fine-Tuning Data
    try:
        with jsonlines.open(output_file, mode='w') as writer:
            writer.write_all(finetuning_data)
        print(f"Fine-tuning data saved to {output_file}")
    except Exception as e:
        print(f"Error saving fine-tuning data to {output_file}: {e}")
        return



def fine_tune(
    fine_tuning_data_path: str,
    model_name: str,
    output_dir: str
):
    """
    Fine-tunes a language model using LoRA on the provided dataset with predefined hyperparameters.
    
    This function handles the entire fine-tuning process, including data loading, 
    tokenization, model configuration with LoRA, training, and saving the fine-tuned model.
    
    Args:
        fine_tuning_data_path (str):
            Path to the fine-tuning dataset (.jsonl file) containing 'input' and 'output' keys.
        
        model_name (str, optional):
            Identifier for the pre-trained model to be fine-tuned.
        
        output_dir (str, optional):
            Directory where the fine-tuned model and related files will be saved (default is "./results/fine_tuned_model").
    
    Returns:
        None
    """
    # 1. Define Fine-Tuning Parameters
    # Device Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 8
    num_train_epochs = 3
    learning_rate = 5e-5
    max_input_length = 512
    max_output_length = 512
    save_steps = 500
    eval_steps = 500
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.2
    target_modules = None #['query_key_value']
    use_8bit = True
    fp16 = True

    # 2. Load the Fine-Tuning Dataset
    print("Loading fine-tuning dataset...")
    dataset = load_dataset("json", data_files=fine_tuning_data_path, split="train")
    print(f"Number of samples in fine-tuning dataset: {len(dataset)}")

    # 3. Load the Tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added [PAD] token to the tokenizer.")

    # 4. Define the Tokenization Function
    def tokenize_function(examples):
        # Tokenize the input
        inputs = tokenizer(
            examples["input"],
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt"
        )
        
        # Tokenize the output (labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["output"],
                padding="max_length",
                truncation=True,
                max_length=max_output_length,
                return_tensors="pt"
            )
        
        # Replace padding token id's in labels by -100 so they are ignored by the loss
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100

        # Print tensor shapes before returning
        print(f"Batch size: {inputs['input_ids'].shape[0]}")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Attention Mask shape: {inputs['attention_mask'].shape}")
        print(f"Labels shape: {labels['input_ids'].shape}")
        
        # Combine inputs and labels
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }

    # 5. Apply Tokenization
    print("Tokenizing the dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input", "output"]
    )
    print("Tokenization completed.")

    # 6. Load the Pre-trained Model
    print(f"Loading pre-trained model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        load_in_8bit=use_8bit,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        trust_remote_code=True  # Necessary for some models like LLaMA
    )
    print("Pre-trained model loaded.")

    # 7. Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Optional: Print LoRA parameters
    print("LoRA configuration completed.")

    # 8. Define Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine", # for learning rate decay
        weight_decay=0.01,  # Add regularization
        fp16=fp16,
        save_steps=save_steps,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="loss", 
        remove_unused_columns=True,
        push_to_hub=False,
        disable_tqdm=False,
        logging_strategy='epoch'
    )
    print("Training arguments set.")

    # 9. Initialize the Trainer
    print("Initializing the Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    print("Trainer initialized.")

    # 10. Start Fine-Tuning
    print("Commencing fine-tuning...")
    trainer.train()
    print("Fine-tuning completed.")

    # 11. Save the Fine-Tuned Model
    print(f"Saving the fine-tuned model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuned model saved.")



# ============================
# Helper Functions 
# ============================


def extract_answer(
    generated_text: str,
    answer_markers: List[str] = None
) -> str:
    """
    Extracts the answer from the generated text based on predefined answer markers.
    Additionally, cuts off the answer at the first newline and removes a trailing period if present.

    Args:
        generated_text (str): The complete text generated by the model.
        answer_markers (List[str], optional): List of markers indicating where the answer starts.

    Returns:
        str: The processed extracted answer.
    """
    if answer_markers is None:
        answer_markers = ["Answer:", "answer:", "ANSWER:", "Answer -", "Answer –"]

    answer = ""
    for marker in answer_markers:
        # Use regex for case-insensitive search and handle possible separators
        pattern = re.escape(marker) + r"\s*(.*)"
        match = re.search(pattern, generated_text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            break  # Stop after the first matching marker

    if not answer:
        # If no marker is found, assume the entire text is the answer
        answer = generated_text.strip()

    # Cut off at the first newline
    answer = answer.split('\n')[0]

    # Remove trailing period if present
    if answer.endswith('.'):
        answer = answer[:-1]

    return answer.strip()

def normalize_text(text: str) -> str:
    """
    Normalizes the text by converting to lowercase, stripping whitespace, and removing extra spaces.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    if not isinstance(text, str):
        return ""
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def evaluate_generations(
    dataset: List[dict],
    generations: List[str],
    normalize: bool = True,
) -> float:
    """
    Evaluates the accuracy of model generations against dataset references.

    Args:
        dataset (List[dict]): The loaded dataset with prompts and references.
        generations (List[str]): The list of generated answers.
        normalize (bool, optional): Whether to normalize texts before comparison. Defaults to True.

    Returns:
        float: The accuracy percentage.
    """
    if len(generations) != len(dataset):
        print(f"Warning: Number of generations ({len(generations)}) does not match number of dataset samples ({len(dataset)}).")
        min_length = min(len(generations), len(dataset))
        print(f"Proceeding with the first {min_length} samples.\n")
    else:
        min_length = len(generations)

    correct = 0
    total = 0

    for idx in tqdm(range(min_length), desc="Evaluating"):
        reference = dataset[idx]['reference']
        generated_text = generations[idx]

        # Extract answer from generated text
        extracted_answer = extract_answer(generated_text)

        # Normalize if required
        if normalize:
            normalized_reference = normalize_text(reference)
            normalized_generated = normalize_text(extracted_answer)
        else:
            normalized_reference = reference.strip()
            normalized_generated = extracted_answer.strip()

        # Compare answers
        if normalized_reference in normalized_generated or normalized_generated in normalized_reference:
            is_correct = True
        else:
            is_correct = False
        if is_correct:
            correct += 1

        total += 1

        # Optional: Print details for correct matches
        if is_correct:
            print(f"---\nSample {total}:")
            print(f"Prompt:\n{dataset[idx]['prompt']}")
            print(f"Reference: {reference}")
            print(f"Generated Answer: {extracted_answer}")
            print(f"Status: Correct\n---\n")

        # Optional: Print details for incorrect matches (Uncomment if needed)
        # else:
        #     print(f"---\nSample {total}:")
        #     print(f"Prompt:\n{dataset[idx]['prompt']}")
        #     print(f"Reference: {reference}")
        #     print(f"Generated Answer: {extracted_answer}")
        #     print(f"Status: Incorrect\n---\n")

    # Calculate metrics
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Total Correct: {correct}")
    print(f"Total Evaluated: {total}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy



def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    dataset_path: str,
    dataset_config: Dict,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 512,
    num_beams: int = 1,
) -> float:
    """
    Evaluates the fine-tuned model on the given dataset by calculating accuracy.

    Args:
        model_path (str): Path to the fine-tuned model directory.
        tokenizer_path (str): Path to the tokenizer directory.
        dataset_path (str): Path to the evaluation dataset (.jsonl file).
        dataset_config (dict): Configuration dictionary containing dataset name.
        batch_size (int, optional): Number of samples to process at once. Defaults to 8.
        device (str, optional): Device to run the model on. Defaults to 'cuda' if available.
        max_length (int, optional): Maximum length of generated answers. Defaults to 512.
        num_beams (int, optional): Number of beams for beam search. Defaults to 1.
        no_repeat_ngram_size (int, optional): Prevent repetition of n-grams. Defaults to 2.

    Returns:
        float: Accuracy of the model on the dataset.
    """
    # Load the tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load the dataset
    print("Loading dataset...")
    with jsonlines.open(dataset_path, mode='r') as reader:
        dataset = list(reader)
    print(f"Number of samples in test dataset: {len(dataset)}")

    correct = 0
    incorrect = 0
    total = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        inputs = []
        references = []
        for row in batch:
            # Extract the input prompt
            input_prompt = row.get("multi_model_prompt", "")  # Adjust key if needed
            inputs.append(input_prompt)

        # Get the references for the current batch
        batch_references = get_references(batch)
        references.extend(batch_references)
        
        tokenizer.pad_token = tokenizer.eos_token

        # Tokenize the inputs
        encoding = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Define desired maximum tokens for generation
        desired_max_new_tokens = 100  # Adjust as needed

        with torch.no_grad():
            # Generate answers
            outputs = model.generate(
                #temperature=0.0,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=desired_max_new_tokens,
                #pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                # return_dict_in_generate=True,
                #do_sample=False  # Set to True if using sampling
            )

        # Decode the generated answers
        generated_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract only the answer part from the generated text
        extracted_answers = [extract_answer(gen) for gen in generated_answers]

        # Compare generated answers with references
        for idx, (gen, ref) in enumerate(zip(extracted_answers, batch_references)):
            total += 1
            # Comparison Logic
            # Using case-insensitive substring matching
            if ref.lower() in gen.lower() or gen.lower() in ref.lower():
                correct += 1

                # # **Print the details of correct matches**
                # print(f"---\nSample {total}:")
                # print(f"Prompt:\n{inputs[idx]}")
                # print(f"Reference: {ref}")
                # print(f"Generated Answer: {gen}")
                # print(f"Status: Correct\n---\n")
            else:
                if incorrect < 20:
                    # **Print the details of incorrect matches**
                    print(f"---\nSample {total}:")
                    print(f"Prompt:\n{inputs[idx]}")
                    print(f"Reference: {ref}")
                    print(f"Generated Answer: {gen}")
                    print(f"Status: Incorrect\n---\n")
                    incorrect += 1

    # Calculate accuracy
    print(f"Total Correct: {correct}")
    print(f"Total Evaluated: {total}")
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy



def main():

    # Smoothie: python self_improvement.py --filter smoothie --teacher 3b --target google/gemma-2b
    # Pick-Random: python self_improvement.py --filter pick_random --teacher 3b --target google/gemma-2b
    # Single Model: python self_improvement.py --filter single_model --teacher gemma-2b --target google/gemma-2b

    # EleutherAI/pythia-2.8b

    parser = argparse.ArgumentParser(description="Training Loop for LLM Improvement")
    parser.add_argument("--filter", choices=["smoothie", "pick_random", "single_model"], required=True, help="Filtering method to use.")
    parser.add_argument("--teacher", required=True, help="Model(s) used to generate synthetic data")
    parser.add_argument("--target", required=True, help="Model to be fine-tuned on")
    #parser.add_argument("--eval_mode", choices=["live", "pre_generated"], default="live", help="Evaluation mode: 'live' to generate answers using the model, 'pre_generated' to use existing generations.")
    args = parser.parse_args()

    # Step #1: Load data config
    data_config = load_data_config(argparse.Namespace(dataset_config=DATASET_CONFIG_PATH))

    # Add necessary fields to data_config for path derivation
    data_config['teacher'] = args.teacher
    data_config['filter'] = args.filter
    data_config['dataset'] = "squad"

    # Step #2: Pre-Fine-Tuning Evaluation on Train Set
    print("Starting pre-fine-tuning evaluation on the training set...")
    pre_finetune_accuracy = evaluate_model(
        model_path=args.target,
        tokenizer_path=args.target,
        dataset_path=TRAIN_FILE,
        dataset_config=data_config,
        batch_size=8,            # Adjust as needed
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=1000,         # Adjust as needed
    )
    print(f"Pre-Fine-Tuning Model Accuracy on Train Set: {pre_finetune_accuracy:.2f}%\n")

    

    # Step 3: Generate Synthetic Data (this step works)
    #generate_data(teacher=args.teacher)


    # Step 4: Apply Filtering (this step works)
    if args.filter == "smoothie":
        generations_file = apply_smoothie(data_config, args.teacher)
    elif args.filter == "pick_random":
        generations_file = apply_pick_random_baseline(data_config, args.teacher)
    # Single model
    elif args.filter == "single_model":
        generations_file = apply_single_model(data_config, args.teacher)
    else:
        print(f"Unsupported filter: {args.filter}")
        exit(1)


    # Step 5: Fine-Tune the Target Model
    # Define the output file
    generations_file_path = Path(generations_file)
    output_file = str(generations_file_path.with_name(f"{generations_file_path.stem}_finetuning{generations_file_path.suffix}"))

    # Prepare fine-tuning data
    prepare_finetuning_data(generations_file, TRAIN_FILE, output_file)

    
    # Step 6: Fine-Tune the Target Model
    # Determine the path to fine-tuning data
    fine_tuning_data_path = output_file
    fine_tuned_model_dir = (Path(FINE_TUNED_MODEL_DIR) / f"fine_tuned_{args.target}_with_{args.teacher}_{args.filter}")
    fine_tuned_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Call the fine_tune function
    fine_tune(
        fine_tuning_data_path=fine_tuning_data_path,
        model_name=args.target,
        output_dir=str(fine_tuned_model_dir)
    )
    
    print("Self-improvement loop completed successfully.")


    # Step #7: Evaluate the Fine-Tuned Model
    print("Starting evaluation of the fine-tuned model...")

    # Load dataset config (assuming it's a YAML or JSON file with at least 'dataset' key)
    config_path = Path(DATASET_CONFIG_PATH)
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Unsupported config file format. Use JSON or YAML.")

    # Ensure 'dataset' key exists
    if 'dataset' not in config:
        raise KeyError("Config must include a 'dataset' key indicating the dataset name.")

    print(f"Evaluating {fine_tuned_model_dir} on {config['dataset']} dataset...")

    # Evaluate the model
    accuracy = evaluate_model(
        model_path=str(fine_tuned_model_dir),
        tokenizer_path=str(fine_tuned_model_dir),
        dataset_path=TEST_FILE,
        dataset_config=DATASET_CONFIG_PATH,
        batch_size=8,            # Adjust as needed
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=1000,           # Adjust as needed
    )
    
    print(f"Evaluation completed. Model Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()