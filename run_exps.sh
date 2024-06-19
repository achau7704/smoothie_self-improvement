#!/bin/bash

# Pick random 
#python -m src.combine_datasets.baselines --task_group acc_group --model_group 7b
#python -m src.combine_datasets.baselines --task_group acc_group --model_group 3b
#python -m src.combine_datasets.baselines --task_group rouge2_group --model_group 7b
#python -m src.combine_datasets.baselines --task_group rouge2_group --model_group 3b


# Smoothie
python -m src.combine_datasets.run_smoothie --task_group rouge2_group --model_group 7b --type sample_dependent --k 400
python -m src.combine_datasets.run_smoothie --task_group rouge2_group --model_group 7b --type sample_dependent --k 300
python -m src.combine_datasets.run_smoothie --task_group rouge2_group --model_group 7b --type sample_dependent --k 200
python -m src.combine_datasets.run_smoothie --task_group rouge2_group --model_group 7b --type sample_dependent --k 100