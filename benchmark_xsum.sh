#!/bin/bash

## Scaling parameter
RESULTS_DIR="generative_ensembles_data/benchmark_results"
DATA_CONFIGS=("dataset_configs/xsum_1_shot_5_prompts.yaml")
models=("pythia-410m" "pythia-1b" "pythia-2.8b" "pythia-6.9b")

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

        # Pick random baseline
        python -m src.pick_random_baseline \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR

        # Generate smoothie predictions
        python -m src.smoothie_sample_dependent \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR --use_max

        python -m src.evaluate.evaluate_text \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR

    done
done