#!/bin/bash

# Directory containing the JSON files
output_dir="algorithm_outputs"
# Path to the leaderboard CSV
leaderboard_path="leaderboard.csv"
# Reference outputs file
reference_outputs="gpt_4_reference_outputs.json"
# Annotators config
annotators_config="alpaca_eval_gpt4_turbo_fn"

# Loop through each JSON file in the directory
for json_file in "$output_dir"/*.json; do
  # Run the command with the current JSON file
  alpaca_eval make_leaderboard \
    --leaderboard_path "$leaderboard_path" \
    --all_model_outputs "$json_file" \
    --reference_outputs "$reference_outputs" \
    --annotators_config "$annotators_config"
done


# Run evaluation on base models
output_dir="downloaded_outputs"
for json_file in "$output_dir"/*.json; do
  # Run the command with the current JSON file
  alpaca_eval make_leaderboard \
    --leaderboard_path "$leaderboard_path" \
    --all_model_outputs "$json_file" \
    --reference_outputs "$reference_outputs" \
    --annotators_config "$annotators_config"
done