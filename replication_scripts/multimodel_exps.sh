#!/bin/bash

set -x  # Enable command tracing

# Replication script for multiple model experiments
# To run: ./replication_scripts/multimodel_exps.sh

RESULTS_DIR="smoothie_data/multi_model_results"

# Array of dataset configs
dataset_configs=(
    "dataset_configs/squad.yaml"
    "dataset_configs/trivia_qa.yaml"
    "dataset_configs/definition_extraction.yaml"
    "dataset_configs/cnn_dailymail.yaml"
    "dataset_configs/e2e_nlg.yaml"
    "dataset_configs/xsum.yaml"
    "dataset_configs/web_nlg.yaml"
    "dataset_configs/acc_group.yaml"
    "dataset_configs/rouge2_group.yaml"
    "dataset_configs/gsm8k.yaml"
)

# Select model group - uncomment one
#model_group="1b"
model_group="3b"
#model_group="7b"

for dataset_config in "${dataset_configs[@]}"; do
    echo "Processing dataset config: $dataset_config"

    python -m src.make_dataset \
        --dataset_config $dataset_config

    python -m src.get_generations \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --results_dir $RESULTS_DIR \
        --multi_model

    # Pick random baseline
    python -m src.pick_random_baseline \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --results_dir $RESULTS_DIR \
        --multi_model

    # Labeled oracle
    python -m src.labeled_oracle \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --results_dir $RESULTS_DIR \
        --multi_model 

    python -m src.pair_rm \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --results_dir $RESULTS_DIR \
        --multi_model 

    # Smoothie sample independent
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_independent

    # Smoothie sample dependent
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_dependent \
        --k 1
    
    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_dependent \
        --k 10

    # Smoothie train time sample independent
    python -m src.run_smoothie_train_time \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_independent --redo

    # Smoothie train time sample dependent
    python -m src.run_smoothie_train_time \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_dependent \
        --k 20 --redo

    # Evaluate
    python -m src.evaluate.evaluate \
        --dataset_config $dataset_config \
        --model_group $model_group \
        --multi_model \
        --results_dir $RESULTS_DIR --redo

    echo "Finished processing $dataset_config"
    echo "----------------------------------------"
done

set +x  # Disable command tracing