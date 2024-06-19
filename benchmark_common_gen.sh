#!/bin/bash

RESULTS_DIR="generative_ensembles_data/benchmark_results"
DATA_CONFIGS=("dataset_configs/common_gen_1_shot_2_prompts.yaml")
models=("falcon-1b")

for model in ${models[@]}; do
    for data_config_path in ${DATA_CONFIGS[@]}; do
        python3 -m src.prompts.generate_prompts --data_config_path $data_config_path
        python3 -m src.prompt_ensemble --model $model --data_config_path $data_config_path --results_dir $RESULTS_DIR
        python -m src.prompt_combination_baseline --model $model --data_config_path $data_config_path --results_dir $RESULTS_DIR
        python -m src.uniform_avg --model $model --data_config_path $data_config_path --results_dir $RESULTS_DIR
        python -m src.run_smoothie --model $model --data_config_path $data_config_path --results_dir $RESULTS_DIR
        python -m src.smoothie_top_3 --model $model --data_config_path $data_config_path --results_dir $RESULTS_DIR
        python -m src.smoothie_sample_dependent --model $model --data_config_path $data_config_path --results_dir $RESULTS_DIR
        python -m src.pick_random_baseline --model $model --data_config_path $data_config_path --results_dir $RESULTS_DIR
        python -m src.smoothie_sample_dependent --model $model --data_config_path $data_config_path --results_dir $RESULTS_DIR --use_max
        python -m src.smoothie_sample_dependent --model $model --data_config_path $data_config_path --results_dir $RESULTS_DIR --temperature 0.3

    done
done

