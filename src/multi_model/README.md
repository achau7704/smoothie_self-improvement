# multi-model experiments 

This directory contains code for running the multi-model experiments. Results are saved to `generative_ensembles_data/multi_model_results`.

Generate prompts by running:

```bash
> python -m src.prompts.generate_prompts --data_config_path dataset_configs/squad.yaml --prompt_templates_dir src/prompts/multimodel_assets --prompts_dir multi_model_prompts
```

Generate predictions for a model by running:
```bash
> python -m src.multi_model.ensemble \
    --model $model \
    --data_config_path $data_config_path \
    --prompts_dir $PROMPTS_DIR \
    --results_dir $RESULTS_DIR
```

Generate pick_random predictions 
```bash
> python -m src.multi_model.pick_random --data_config_path dataset_configs/squad.yaml --models llama-2-7b mistral-7b snorkel-7b
```


Generate smoothie predictions by running:
```bash
> python -m src.multi_model.run_smoothie --data_config_path dataset_configs/squad.yaml --type sample_dependent --models llama-2-7b mistral-7b snorkel-7b
```