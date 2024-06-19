#!/bin/bash

## CODE-LLAMA
RESULTS_DIR="generative_ensembles_data/benchmark_results"

# Generate prompts
python -m src.prompts.generate_prompts \
    --data_config_path dataset_configs/humaneval_0_shot.yaml

# Iterate through models
models=("codellama-7b", "codellama-7b-py", "codellama-7b-instruct")
for model in "${models[@]}"; do

    # Generate prompt predictions
    python -m src.prompt_ensemble \
        --model $model \
        --data_config_path dataset_configs/humaneval_0_shot.yaml \
        --results_dir $RESULTS_DIR

    python -m src.smoothie_sample_dependent \
        --data_config_path dataset_configs/humaneval_0_shot.yaml \
        --model $model \
        --results_dir $RESULTS_DIR --use_max
    
    python -m src.smoothie_sample_dependent \
        --data_config_path dataset_configs/humaneval_0_shot.yaml \
        --model $model \
        --results_dir $RESULTS_DIR
    
    # Eval command goes here
done


