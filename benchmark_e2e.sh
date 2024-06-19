#!/bin/bash

## SUMMARIZATION
RESULTS_DIR="generative_ensembles_data/benchmark_results"
DATA_CONFIGS=("dataset_configs/e2e_nlg_1_shot.yaml")
models=("falcon-1b")

# Iterate through models and tasks
for model in "${models[@]}"; do
    for data_config_path in "${DATA_CONFIGS[@]}"; do

        # Generate prompts
        #python -m src.prompts.generate_prompts \
        #    --data_config_path $data_config_path
            
        # Generate prompt predictions
        #python -m src.prompt_ensemble \
        #    --model $model \
        #    --data_config_path $data_config_path \
        #    --results_dir $RESULTS_DIR

        # Pick random baseline
        #python -m src.pick_random_baseline \
        #    --model $model \
        #    --data_config_path $data_config_path \
        #    --results_dir $RESULTS_DIR

        python -m src.run_smoothie \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR \
            --type sample_dependent \
            --operation select

        #python -m src.smoothie_sample_dependent \
        #    --model $model \
        #    --data_config_path $data_config_path \
        #    --results_dir $RESULTS_DIR

        #python -m src.smoothie_sample_dependent \
        #    --model $model \
        #    --data_config_path $data_config_path \
        #    --results_dir $RESULTS_DIR --use_max --redo

        python -m src.evaluate.evaluate_text \
            --model $model \
            --data_config_path $data_config_path --redo

    done
done



