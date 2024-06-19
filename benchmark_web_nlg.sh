#!/bin/bash

## SUMMARIZATION
RESULTS_DIR="generative_ensembles_data/benchmark_results"
DATA_CONFIGS=("dataset_configs/web_nlg_1_shot.yaml")
models=("pythia-1b")

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

        # Generate prompt-combination predictions
        python -m src.prompt_combination_baseline \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR

        # Generate uniform-avg predictions
        python -m src.uniform_avg \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR

        # Generate smoothie predictions
        python -m src.run_smoothie \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR --redo

    done
done



