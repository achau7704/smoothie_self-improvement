#!/bin/bash

## SQUAD
RESULTS_DIR="generative_ensembles_data/benchmark_results"



# Generate prompts
python -m src.prompts.generate_prompts \
    --data_config_path dataset_configs/squad_1_shot.yaml

# Iterate through models
models=("pythia-410m" "pythia-1b" "pythia-2.8b" "pythia-6.9b"  "falcon-1b")
for model in "${models[@]}"; do

    # Generate prompt predictions
    python -m src.prompt_ensemble \
        --model $model \
        --data_config_path dataset_configs/squad_1_shot.yaml \
        --results_dir $RESULTS_DIR

    python -m src.pick_random_baseline \
        --model $model \
        --data_config_path dataset_configs/squad_1_shot.yaml \
        --results_dir $RESULTS_DIR

    python -m src.smoothie_sample_dependent \
        --data_config_path dataset_configs/squad_1_shot.yaml \
        --model $model \
        --results_dir $RESULTS_DIR --use_max
    
    python -m src.smoothie_sample_dependent \
        --data_config_path dataset_configs/squad_1_shot.yaml \
        --model $model \
        --results_dir $RESULTS_DIR
    
    python -m src.evaluate.evaluate_text \
            --model $model \
            --data_config_path dataset_configs/squad_1_shot.yaml  --redo
done



