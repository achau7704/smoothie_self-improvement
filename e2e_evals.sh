#!/bin/bash

# BENCHMARK RESULTS
RESULTS_DIR="generative_ensembles_data/benchmark_results"
DATA_CONFIGS=("dataset_configs/e2e_nlg_1_shot_10_prompts.yaml" "dataset_configs/e2e_nlg_1_shot_7_prompts.yaml" "dataset_configs/e2e_nlg_1_shot_3_prompts.yaml" "dataset_configs/e2e_nlg_1_shot.yaml")
models=("falcon-1b")

# Iterate through models and tasks
for model in "${models[@]}"; do
    for data_config_path in "${DATA_CONFIGS[@]}"; do

        python -m src.evaluate.evaluate_text \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR

    done
done



RESULTS_DIR="generative_ensembles_data/benchmark_results"
DATA_CONFIGS=("dataset_configs/e2e_nlg_1_shot_7_prompts.yaml")
models=("pythia-410m" "pythia-1b" "pythia-2.8b" "pythia-6.9b")

# Iterate through models and tasks
for model in "${models[@]}"; do
    for data_config_path in "${DATA_CONFIGS[@]}"; do

        python -m src.evaluate.evaluate_text \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR

    done
done


