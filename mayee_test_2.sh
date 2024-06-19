#!/bin/bash
# This script runs the experiments for Table 1 in the paper. This covers the following datasets: CNN/DailyMail, XSum, Squad, E2E, WebNLG, and CommonGen.

RESULTS_DIR="generative_ensembles_data/multi_model_results"
PROMPTS_DIR="multi_model_prompts"
MODELS=("llama-2-7b" "mistral-7b" "vicuna-7b" "gemma-7b" "nous-capybara")
DATA_CONFIGS=("dataset_configs/cnn_dailymail_0_shot.yaml" "dataset_configs/definition_extraction.yaml" "dataset_configs/xsum_0_shot.yaml" "dataset_configs/squad.yaml" "dataset_configs/e2e_nlg_1_shot.yaml" "dataset_configs/trivia_qa_knowledge.yaml")
# DATA_CONFIGS=("dataset_configs/cnn_dailymail_0_shot.yaml")

for data_config_path in "${DATA_CONFIGS[@]}"; do
    #python -m src.prompts.generate_prompts \
    #    --data_config_path $data_config_path \
    #    --prompt_templates_dir src/prompts/multimodel_assets \
    #    --prompts_dir $PROMPTS_DIR

    #python -m src.multi_model.pick_random \
    #    --data_config_path $data_config_path \
    #    --results_dir $RESULTS_DIR --model_group 7b --redo

    #python -m src.multi_model.run_smoothie \
    #    --data_config_path $data_config_path \
    #    --use_full_text_embeddings \
    #    --type sample_independent \
    #    --model_group 7b --redo

    python -m src.multi_model.run_smoothie \
        --data_config_path $data_config_path \
        --type sample_dependent \
        --n_generations 1 \
        --k 1 \
        --model_group 7b


    #python -m src.multi_model.run_smoothie \
    #    --data_config_path $data_config_path \
    #    --type sample_independent \
    #    --model_group 7b --redo

    python -m src.evaluate.evaluate_multi_model \
        --data_config_path $data_config_path \
        --results_dir $RESULTS_DIR

done


