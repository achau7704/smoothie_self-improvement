#!/bin/bash

## SUMMARIZATION
RESULTS_DIR="smoothie_data/multi_model_results"
DATA_CONFIGS=("dataset_configs/cnn_dailymail.yaml")
models=("pythia-410m")

# Iterate through models and tasks
for dataset_config in "${DATA_CONFIGS[@]}"; do
    # Generate prompts
    python -m src.make_dataset \
        --dataset_config $dataset_config

    python -m src.get_generations \
        --dataset_config $dataset_config \
        --model_group 3b \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --test

    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --model_group 3b \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_independent \

    python -m src.run_smoothie \
        --dataset_config $dataset_config \
        --model_group 3b \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --type sample_dependent \
        --k 20


    python -m src.pick_random_baseline \
        --dataset_config $dataset_config \
        --model_group 3b \
        --results_dir $RESULTS_DIR \
        --multi_model \

    python -m src.labeled_oracle \
        --dataset_config $dataset_config \
        --model_group 3b \
        --results_dir $RESULTS_DIR \
        --multi_model \
        --redo 


    # include labeled knn
    python -m src.labeled_knn \
        --dataset_config $dataset_config \
        --model_group 3b \
        --results_dir $RESULTS_DIR \
        --multi_model \


    python -m src.evaluate.evaluate \
        --dataset_config $dataset_config \
        --model_group 3b \
        --multi_model \
        --results_dir $RESULTS_DIR \

done
