#!/bin/bash
# This script runs the experiments for Table 1 in the paper. This covers the following datasets: CNN/DailyMail, XSum, Squad, E2E, WebNLG, and CommonGen.


RESULTS_DIR="generative_ensembles_data/multi_model_results"
PROMPTS_DIR="multi_model_prompts"
MODELS=("mistral-7b"  "llama-2-7b"  "vicuna-7b"  "gemma-7b"  "nous-capybara")
DATA_CONFIGS=("dataset_configs/cnn_dailymail_0_shot.yaml" "dataset_configs/xsum_0_shot.yaml" "dataset_configs/squad.yaml" "dataset_configs/e2e_nlg_1_shot.yaml" "dataset_configs/definition_extraction.yaml" "dataset_configs/web_nlg_1_shot.yaml" "dataset_configs/trivia_qa_knowledge.yaml")
TYPES=("3b")

for type in "${TYPES[@]}"; do
    for data_config_path in "${DATA_CONFIGS[@]}"; do
        python -m src.prompts.generate_prompts \
            --data_config_path $data_config_path \
            --prompt_templates_dir src/prompts/multimodel_assets \
            --prompts_dir $PROMPTS_DIR 

        for model in "${MODELS[@]}"; do
            python -m src.multi_model.ensemble \
                --model $model \
                --data_config_path $data_config_path \
                --prompts_dir $PROMPTS_DIR \
                --results_dir $RESULTS_DIR 
        done

        python -m src.multi_model.labeled_knn \
            --data_config_path $data_config_path\
            --results_dir $RESULTS_DIR --model_group $type

        python -m src.multi_model.pick_random \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR --model_group $type

        python -m src.multi_model.run_smoothie \
            --data_config_path $data_config_path \
            --type sample_dependent --model_group $type --k 20

        python -m src.multi_model.run_smoothie \
            --data_config_path $data_config_path \
            --type sample_independent --model_group $type
    done
done

for data_config_path in "${DATA_CONFIGS[@]}"; do
    python -m src.evaluate.evaluate_multi_model \
            --data_config_path $data_config_path
done