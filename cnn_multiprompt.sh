#!/bin/bash

## SUMMARIZATION
RESULTS_DIR="smoothie_data/multi_prompt_results"
DATA_CONFIGS=("dataset_configs/cnn_dailymail.yaml")
models=("pythia-410m")

# Iterate through models and tasks
for model in "${models[@]}"; do
    for dataset_config in "${DATA_CONFIGS[@]}"; do
        # Generate prompts
        #python -m src.make_dataset \
        #    --dataset_config $dataset_config

        #python -m src.get_generations \
        #    --dataset_config $dataset_config \
        #    --model $model \
        #    --results_dir $RESULTS_DIR \
        #    --multi_prompt \
        #    --test

        # python -m src.run_smoothie \
        #   --dataset_config $dataset_config \
        #    --model $model \
        #    --results_dir $RESULTS_DIR \
        #    --multi_prompt \
        #    --type sample_independent \

        python -m src.run_smoothie \
            --dataset_config $dataset_config \
            --model $model \
            --results_dir $RESULTS_DIR \
            --multi_prompt \
            --type sample_dependent \
            --k 20 


        # python -m src.pick_random_baseline \
        #    --dataset_config $dataset_config \
        #    --model $model \
        #    --results_dir $RESULTS_DIR \
        #    --multi_prompt \

        #python -m src.labeled_oracle \
        #    --dataset_config $dataset_config \
        #    --model $model \
        #    --results_dir $RESULTS_DIR \
        #    --multi_prompt \
        #    --redo


        python -m src.evaluate.evaluate \
            --dataset_config $dataset_config \
            --model $model \
            --multi_prompt \
            --results_dir $RESULTS_DIR \

    done
done



