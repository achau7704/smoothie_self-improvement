#!/bin/bash

# Replication script for multiple model experiments
# To run: ./replication_scripts/multimodel_exps.sh

RESULTS_DIR="smoothie_data/multi_model_results"

# Dataset to run - uncomment one
#dataset_config="dataset_configs/squad.yaml"
#dataset_config="dataset_configs/trivia_qa.yaml"
#dataset_config="dataset_configs/definition_extraction.yaml"
#dataset_config="dataset_configs/cnn_dailymail.yaml"
#dataset_config="dataset_configs/e2e_nlg.yaml"
#dataset_config="dataset_configs/xsum.yaml"
#dataset_config="dataset_configs/web_nlg.yaml"
#dataset_config="dataset_configs/acc_group.yaml"
#dataset_config="dataset_configs/rouge2_group.yaml"

# Select model group - uncomment one
#model_group="3b"
model_group="7b"

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
    --multi_model --redo

# Labeled oracle
python -m src.labeled_oracle \
    --dataset_config $dataset_config \
    --model_group $model_group \
    --results_dir $RESULTS_DIR \
    --multi_model \
    --redo

# Smoothie sample independent
python -m src.run_smoothie \
    --dataset_config $dataset_config \
    --model_group $model_group \
    --results_dir $RESULTS_DIR \
    --multi_model \
    --type sample_independent --redo

# Smoothie sample dependent
python -m src.run_smoothie \
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