import time
from typing import List

import pandas as pd
from tqdm.auto import tqdm

from src.utils import prompt_openai


def alpaca_eval_original(dataset: pd.DataFrame) -> List[List[str]]:
    """
    The alpaca-eval dataset evaluates instruction following. We use the original instructions here, formatted into the mistral-style.
    """
    prompts = []
    for j, row in tqdm(dataset.iterrows(), total=len(dataset)):
        instruction = row["instruction"]
        sample_prompts = [f"<s>[INST] {instruction} [/INST]"]
        prompts.append(sample_prompts)
    return prompts


def alpaca_eval_rephrased_5(dataset: pd.DataFrame) -> List[List[str]]:
    """
    The alpaca-eval dataset evaluates instruction following. For each instruction in the evaluation set,
    we use GPT-3.5 to generate four additional rephrasings (5 total prompts)
    """
    openai_key = "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"

    gpt3_template = """Here is an instruction for a language model: 
    

\"{instruction}\"

Generate a rephrasing of this prompt, with additional details that should elicit a better response from a model."""

    prompts = []

    for j, row in tqdm(dataset.iterrows(), total=len(dataset)):
        instruction = row["instruction"]
        rephrasing_prompt = gpt3_template.format(instruction=instruction)
        sample_prompts = [f"<s>[INST] {instruction} [/INST]"]
        for _ in range(4):
            output = prompt_openai(
                api_key=openai_key,
                model="gpt-3.5-turbo",
                prompt=rephrasing_prompt,
                max_tokens=1000,
                temperature=0.5,
                stop="",
            )
            if output.startswith('"'):
                output = output[1:]
            if output.endswith('"'):
                output = output[:-1]
            sample_prompts.append(f"<s>[INST] {output} [/INST]")
        prompts.append(sample_prompts)

        if j % 10 == 0:
            time.sleep(30)
    return prompts


def alpaca_eval_rephrased_7(dataset: pd.DataFrame) -> List[List[str]]:
    """
    The alpaca-eval dataset evaluates instruction following. For each instruction in the evaluation set,
    we use GPT-3.5 to generate six additional rephrasings (7 total prompts)
    """
    openai_key = "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"

    gpt3_template = """Here is an instruction for a language model: 
    

\"{instruction}\"

Generate a rephrasing of this prompt, with additional details that should elicit a better response from a model."""

    prompts = []

    for j, row in tqdm(dataset.iterrows(), total=len(dataset)):
        instruction = row["instruction"]
        rephrasing_prompt = gpt3_template.format(instruction=instruction)
        sample_prompts = [f"<s>[INST] {instruction} [/INST]"]
        for _ in range(6):
            output = prompt_openai(
                api_key=openai_key,
                model="gpt-3.5-turbo",
                prompt=rephrasing_prompt,
                max_tokens=1000,
                temperature=0.5,
                stop="",
            )
            if output.startswith('"'):
                output = output[1:]
            if output.endswith('"'):
                output = output[:-1]
            sample_prompts.append(f"<s>[INST] {output} [/INST]")
        prompts.append(sample_prompts)

        if j % 10 == 0:
            time.sleep(30)
    return prompts


def alpaca_eval_with_guidance(dataset: pd.DataFrame) -> List[List[str]]:
    """
    The alpaca-eval dataset evaluates instruction following. For each instruction in the evaluation set,
    we use GPT-3.5 to generate four versions of the instruction which contain additional guidance.
    """
    openai_key = "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"

    gpt3_template = """For the instruction, produce a "modified" version which explicitly directs the model's response to have a certain quality or contain certain content. The quality provided should be appropriate for the instruction.

####
Original: I want to get better at networking at work
Modified: Tell me how I can get better at networking. List 10 steps I can take.

####
Original: You are given a description that provides a set of facts or a scenario. It is up to you to craft a story from these facts and scenarios. The missing pieces must be filled in with imaginative but logical information.\n\nTen European football teams \u2013 the Netherlands, England, Belgium, Denmark, France, Germany, Norway, Sweden, Switzerland and Wales \u2013 will participate in a season-long \u201cOneLove\u201d campaign promoting inclusion and opposing discrimination.
Modified: You are given a description that provides a set of facts or a scenario. It is up to you to craft a story from these facts and scenarios. The missing pieces must be filled in with imaginative but logical information. Make sure to include a title. \n\nTen European football teams \u2013 the Netherlands, England, Belgium, Denmark, France, Germany, Norway, Sweden, Switzerland and Wales \u2013 will participate in a season-long \u201cOneLove\u201d campaign promoting inclusion and opposing discrimination.

####
Original: 63-year-old male with diabetes for seven to eight years (BbA1c consistently between 5.9-6.5, fasting blood sugar around 7, other times high between 8-9-9.5, no low blood sugar). CKD for five years (starting with 24-hour urine protein within 200, GFR around 100, but in the last 1-2 years urine protein between 300-400mg, GFR between 60-70, most recent one being 56). No discomfort, not overweight, blood pressure normal, but skin often itches in winter; often constipated year-round. <br><br>Current medication: 1. Allisartan Tablets 240mg/day 2. Dapagliflozin Tablets 10mg/day 3. Metformin HCL 500mg*3/day 4. Pancreatic kininogenase enteric-coated Tablets.<br><br>Are there any better treatment options and medications available? Avoid or delay dialysis if possible. Are there any other exams that need to be done? What are the recommendations for exercise and diet in addition to medication? When should the above medication be switched to insulin due to CKD?<br>
Modified: 63-year-old male with diabetes for seven to eight years (BbA1c consistently between 5.9-6.5, fasting blood sugar around 7, other times high between 8-9-9.5, no low blood sugar). CKD for five years (starting with 24-hour urine protein within 200, GFR around 100, but in the last 1-2 years urine protein between 300-400mg, GFR between 60-70, most recent one being 56). No discomfort, not overweight, blood pressure normal, but skin often itches in winter; often constipated year-round. <br><br>Current medication: 1. Allisartan Tablets 240mg/day 2. Dapagliflozin Tablets 10mg/day 3. Metformin HCL 500mg*3/day 4. Pancreatic kininogenase enteric-coated Tablets.<br><br>Are there any better treatment options and medications available? Avoid or delay dialysis if possible. Are there any other exams that need to be done? What are the recommendations for exercise and diet in addition to medication? When should the above medication be switched to insulin due to CKD? List factors that I should pay attention to<br>

###

###
Instruction: {instruction}
Modified:"""

    prompts = []

    for j, row in tqdm(dataset.iterrows(), total=len(dataset)):
        instruction = row["instruction"]
        rephrasing_prompt = gpt3_template.format(instruction=instruction)
        sample_prompts = [f"<s>[INST] {instruction} [/INST]"]
        for _ in range(4):
            output = prompt_openai(
                api_key=openai_key,
                model="gpt-3.5-turbo",
                prompt=rephrasing_prompt,
                max_tokens=1000,
                temperature=1.0,
                stop="",
            )
            if output.startswith('"'):
                output = output[1:]
            if output.endswith('"'):
                output = output[:-1]
            sample_prompts.append(f"<s>[INST] {output} [/INST]")
        prompts.append(sample_prompts)

        if j % 10 == 0:
            time.sleep(30)
    return prompts
