#!/bin/bash

## SQUAD
RESULTS_DIR="generative_ensembles_data/multi_model_results"



# Generate prompts
python -m src.prompts.generate_prompts \
    --data_config_path dataset_configs/squad.yaml

# Iterate through models
models=("nous-hermes-llama-2-7b" "together-llama-2-7b" "vicuna-7b")
for model in "${models[@]}"; do

    # Generate prompt predictions
    python -m src.multi_model_ensemble \
        --model $model \
        --data_config_path dataset_configs/squad.yaml \
        --results_dir $RESULTS_DIR

    python -m src.smoothie_sample_dependent \
        --data_config_path dataset_configs/squad.yaml \
        --results_dir $RESULTS_DIR --use_max
done



