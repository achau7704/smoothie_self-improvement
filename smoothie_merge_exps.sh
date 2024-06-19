#!/bin/bash

# BENCHMARK RESULTS
RESULTS_DIR="generative_ensembles_data/benchmark_results"
DATA_CONFIGS=("dataset_configs/squad_1_shot.yaml")
models=("falcon-1b" "phi-2" "llama-2-7b")

# Iterate through models and tasks
for model in "${models[@]}"; do
    for data_config_path in "${DATA_CONFIGS[@]}"; do

        # Generate prompts
        python -m src.prompts.generate_prompts \
            --data_config_path $data_config_path
            
        # Generate prompt predictions
        python -m src.prompt_ensemble \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR

        # Uniform-avg
        python -m src.uniform_avg \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR

        # Evaluate
        python -m src.evaluate.evaluate_text \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR

    done
done



