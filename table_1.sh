#!/bin/bash
# This script runs the experiments for Table 1 in the paper. This covers the following datasets: CNN/DailyMail, XSum, Squad, E2E, WebNLG, and CommonGen.


RESULTS_DIR="generative_ensembles_data/benchmark_results"
MODELS=("falcon-1b" "phi-2" "llama-2-7b")
DATA_CONFIGS=("dataset_configs/definition_extraction.yaml" "dataset_configs/web_nlg_1_shot.yaml")

for model in "${MODELS[@]}"; do
    for data_config_path in "${DATA_CONFIGS[@]}"; do
        python -m src.prompts.generate_prompts \
            --data_config_path $data_config_path --redo

        python -m src.prompt_ensemble \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR --redo

        python -m src.pick_random_baseline \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR --redo

        python -m src.labeled_oracle \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR --redo

        python -m src.run_smoothie \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR \
            --type sample_dependent \
            --operation select --redo

        python -m src.perplexity_baseline \
            --data_config_path $data_config_path \
            --model $model --redo

        python -m src.evaluate.evaluate_text \
            --model $model \
            --data_config_path $data_config_path \
            --results_dir $RESULTS_DIR --redo
    done
done


RESULTS_DIR="generative_ensembles_data/multi_model_results"
PROMPTS_DIR="multi_model_prompts"
MODELS=("llama-2-7b" "mistral-7b" "vicuna-7b" "gemma-7b" "nous-capybara")
DATA_CONFIGS=("dataset_configs/definition_extraction.yaml" "dataset_configs/web_nlg_1_shot.yaml")

for data_config_path in "${DATA_CONFIGS[@]}"; do
    python -m src.prompts.generate_prompts \
        --data_config_path $data_config_path \
        --prompt_templates_dir src/prompts/multimodel_assets \
        --prompts_dir $PROMPTS_DIR --redo

    for model in "${MODELS[@]}"; do
        python -m src.multi_model.ensemble \
            --model $model \
            --data_config_path $data_config_path \
            --prompts_dir $PROMPTS_DIR \
            --results_dir $RESULTS_DIR --redo
    done

    python -m src.multi_model.labeled_oracle \
        --data_config_path $data_config_path\
        --results_dir $RESULTS_DIR --model_group 7b --redo

    python -m src.multi_model.pick_random \
        --data_config_path $data_config_path \
        --results_dir $RESULTS_DIR --model_group 7b --redo

    python -m src.multi_model.run_smoothie \
        --data_config_path $data_config_path \
        --type sample_dependent --model_group 7b --redo

    for model in "${MODELS[@]}"; do
        python -m src.multi_model.compute_perplexity \
            --model $model \
            --data_config_path $data_config_path \
            --prompts_dir $PROMPTS_DIR \
            --results_dir $RESULTS_DIR --redo
    done

    python -m src.multi_model.perplexity_baseline \
        --data_config_path $data_config_path \
        --results_dir $RESULTS_DIR --redo --model_group 7b

    python -m src.evaluate.evaluate_multi_model \
        --data_config_path $data_config_path  --redo
done


