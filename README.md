# smoothie

This is the codebase for Smoothie. It allows you to both use Smoothie, and reproduce the experiments in the paper. 

We store all datasets, predictions, and results from the paper in a Hugging Face dataset. You can download the dataset from HuggingFace by running the following command:

```bash
> huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
> git clone https://huggingface.co/datasets/hazyresearch/smoothie_data
```

## Reproducing the paper

### Datasets
`dataset_configs` contains the configuration files for the datasets used in the paper. 

For single task datasets, each configuration file contains the following fields:

```yaml 
# Name of the dataset
dataset: e2e_nlg
# Name of the prompt template to use for the multi-prompt setting
multi_prompt_template: e2e_nlg_1_shot
# Name of the prompt template to use for the multi-model setting
multi_model_prompt_template: e2e_nlg_1_shot
# Maximum number of new tokens to generate
max_new_tokens: 50
# Preprocessing function to use
preprocess: preprocess_e2e_nlg
# Train and test sizes
train_size: 250
test_size: 1000
# Metrics to compute for evaluation
metrics:
  - rouge1
  - rouge2
  - rougeL
```

For multi-task datasets, each configuration file contains the following fields:

```yaml
# Configuration for each task in the dataset
tasks:
  - squad.yaml # Configuration file for the first task
  - trivia_qa.yaml # Configuration file for the second task
  - definition_extraction.yaml # Configuration file for the third task
```

Train and test splits for datasets are saved to `$HF_DATASETS_DIR/datasets/` as `$config_filename_train.tsv` and `$config_filename_test.tsv`. Each tsv file contains the following columns:
- `idx`: Unique identifier for the sample
- `input_text`: Text input used when computing lookup embeddings.
- `multi_model_prompt_template`: The prompt used for the multi-model setting.
- `multi_prompt_template_{i}`: The prompt used for the i-th prompt in the multi-prompt setting.
- `reference_text`: The reference text for the sample.

To produce the train and test splits, run the following command:

```bash
> python -m src.make_dataset --config $config_filename
```


# OLD DOCUMENTATION

This is the repository for the generative-ensembles project.

## Folder structure

- `src`: Contains method implementations and scripts for running experiments.
- `notebooks`: Contains a random assortment of jupyter notebooks.
- `dataset_configs`: Contains configuration files for datasets. See below for more information.
- `method_configs`: Contains configuration files for methods. See below for more information.

## Adding a new dataset

If you want to add a new dataset, you need to:

1. Define a dataset configuration file in `dataset_configs`.
2. Add the dataset to `src/constants.py`.
3. Create a prompt template.
4. Write an evaluation function.

### Dataset configuration files

Every experiment operates on a *dataset*. Datasets are defined by a configuration file. For instance:

```yaml
dataset: e2e_nlg
prompt: e2e_nlg_1_shot
max_new_tokens: 50
doc_key: meaning_representation
reference_key: human_reference
train_size: 250
test_size: 1000
metrics:
  - rouge1
  - rouge2
  - rougeL
```

where:

- `dataset` is the name of the dataset.
- `prompt` is the name of the prompt template to use. See below for more information on prompts.
- `max_new_tokens` is the maximum number of new tokens to generate.
- `doc_key` is the key for the text input in the dataset. 
- `reference_key` is the key for the gold/ground-truth reference in the dataset.
- `train_size` is the number of samples in the training set.
- `test_size` is the number of samples in the test set.
- `metrics` is a list of metrics to compute. See below for more information on evaluation.

### Adding dataset to `src/constants.py`

`src/constants.py` contains important dataset specific constants. If you add a new dataset, be sure to update `HF_TRAIN_DATASETS` and `HF_TEST_DATASETS` in this file.

### Creating a prompt template

The `src/prompts` directory contains code for generating prompts. Generating prompts over a dataset has two steps. 

First, we write save templates for the prompts (i.e., f-strings) to `src/prompts/assets/`. The name of the saved file is the name of the prompt template. For example, here is the prompt template for `e2e_nlg_1_shot.json`:
```json
[
    "Transform the meaning representation into a sentence.\n\nMeaning representation: name[Alimentum], food[Chinese], priceRange[less than \u00a320], area[riverside], familyFriendly[yes]\nNatural language: Alimentum is a family-friendly Chinese food restaurant in the Riverside area where you can eat for low prices.\n\nMeaning representation: name[Strada], food[Japanese], priceRange[less than \u00a320], customer rating[average], familyFriendly[yes], near[Rainbow Vegetarian Caf\u00e9]\nNatural language: Near the Rainbow Vegetarian Caf\u00e9 is the Strada, which has a price range less then 20 pounds, is family friendly, serves Japanese, and has an average customer rating.\n\nMeaning representation: {meaning_representation}\nNatural language:",
    "Transform the meaning representation into a sentence.\n\nMeaning representation: name[Green Man], food[Italian], priceRange[moderate], area[city centre], familyFriendly[yes], near[All Bar One]\nNatural language: Green Man is a moderately priced Italian restaurant in the city centre, near to All Bar One. It is kid friendly.\n\nMeaning representation: name[The Waterman], food[Indian], priceRange[cheap], customer rating[average], area[riverside], familyFriendly[no]\nNatural language: The Waterman it is an adult Indian food restaurant. Its food price range is cheap, customer rating on average near to riverside area.\n\nMeaning representation: {meaning_representation}\nNatural language:",
    "Transform the meaning representation into a sentence.\n\nMeaning representation: name[The Cambridge Blue], eatType[pub], food[Indian], priceRange[cheap], near[Caf\u00e9 Brazil]\nNatural language: The Cambridge Blue is a cheap pub that offers Indian food. It is located near Caf\u00e9 Brazil.\n\nMeaning representation: name[The Eagle], eatType[coffee shop], food[Indian], priceRange[\u00a320-25], customer rating[high], area[city centre], familyFriendly[yes], near[Burger King]\nNatural language: The Eagle is a coffee shop providing Indian food in the \u00a320-25 price range. It is located in the city centre. It is near Burger King. Its customer rating is high.\n\nMeaning representation: {meaning_representation}\nNatural language:",
    "Transform the meaning representation into a sentence.\n\nMeaning representation: name[The Cambridge Blue], eatType[pub], food[Japanese], priceRange[more than \u00a330], near[Caf\u00e9 Brazil]\nNatural language: The Cambridge Blue Pub serves Japanese food at \u00a330 plus. You can find it near the Caf\u00e9 Brazil.\n\nMeaning representation: name[Fitzbillies], eatType[coffee shop], food[Fast food], priceRange[more than \u00a330], customer rating[high], area[city centre], familyFriendly[no]\nNatural language: In city centre Fitzbillies coffee shop offers a high customer rating. Fast food is offered with a price range of more than \u00a330. We are not children friendly.\n\nMeaning representation: {meaning_representation}\nNatural language:",
    "Transform the meaning representation into a sentence.\n\nMeaning representation: name[Browns Cambridge], priceRange[high], customer rating[3 out of 5]\nNatural language: Browns Cambridge is an expensive venue with a customer rating 3 out of 5\n\nMeaning representation: name[The Wrestlers], food[French], priceRange[less than \u00a320], customer rating[average], familyFriendly[no]\nNatural language: There is a restaurant The Wrestlers they serve French food and price rang is less than \u00a320. Although it isn't a family-friendly restaurant and the customer rating is only average\n\nMeaning representation: {meaning_representation}\nNatural language:"
]
```
This contains 5 prompt templates, where each template has a different in-context demonstration. You can also see that each template is an f-string, where the variable names (i.e., `{meaning_representation}`) correspond to the column name in the original dataset dataframe.


### Evaluation 

`src/evaluate/evaluate_text.py` is a python function which performs evaluation for summarization and data2text tasks. 

## Setup

Create a virtual environment and install the required packages.

```bash
> conda env create -n "generative-ensembles" -f environment.yml # Create virtual environment
> conda activate generative-ensembles # Activate virtual environment
> chmod -R a+w . # Give write permissions to all users, if working on a cluster
> pip install -r requirements.txt # Install requirements–NOT SURE IF THIS WORKS
```

Clone the huggingface datasets repository. Note: the datasets repository is big so it may take up space.

```bash
> huggingface-cli login
> git clone https://huggingface.co/datasets/hazyresearch/generative_ensembles_data
```

## Common commands

Launch notebook

```bash
> jupyter-lab --port 9999 --allow-root --no-browser # launch notebook
```
