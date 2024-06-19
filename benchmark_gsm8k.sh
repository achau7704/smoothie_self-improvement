#!/bin/bash

## SUMMARIZATION
RESULTS_DIR="generative_ensembles_data/benchmark_results"
DATA_CONFIGS=("dataset_configs/gsm8k_1_shot_cot_simple.yaml")
models=("llema-7b")

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

        # TODO: IMPLEMENT PROMPT COMBINATION BASELINE

        # Generate uniform-avg predictions
        python -m src.uniform_avg \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR
        
        python -m src.smoothie_sample_dependent \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR --use_max

    done
done



