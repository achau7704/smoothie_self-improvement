"""
Prompt functions for web_nlg dataset.
"""

from typing import List

import pandas as pd
from numpy import array


def web_nlg_1_shot(dataset: pd.DataFrame) -> List[List[str]]:
    """
    e2e_nlg asks a LLM to generate a restaurant review given metadata.
    """
    TEMPLATES = []
    instructions = [
        "Following is a set of knowledge graph triples. Generate a coherent piece of text that contains all of the information in the triples. Only use information from the provided triples.",
        "Generate a coherent paragraph incorporating the information from the given set of knowledge graph triples.",
        "Formulate a cohesive passage using the details provided in the set of knowledge graph triples.",
        "Create a unified text that encompasses all the data presented in the knowledge graph triples.",
        "Compose a seamless narrative integrating the content found within the knowledge graph triples.",
    ]
    icl_samples = [
        {
            "triples": [
                "14th_New_Jersey_Volunteer_Infantry_Monument | category | Historic_districts_in_the_United_States"
            ],
            "text": "The 14th New Jersey Volunteer Infantry Monument is categorised as a historic district in the United States.",
        },
        {
            "triples": [
                "Bacon_Explosion | country | United_States",
                "United_States | leader | Joe_Biden",
                "United_States | ethnicGroup | White_Americans",
                "United_States | capital | Washington,_D.C.",
            ],
            "text": "Bacon Explosion comes from the United States where Joe Biden was once a leader and Washington D.C. is the capital. White Americans are one ethnic group there.",
        },
        {
            "triples": [
                "Batagor | dishVariation | Siomay",
                "Batagor | country | Indonesia",
            ],
            "text": "Batagor is a variation of the Siomay dish and is found in Indonesia.",
        },
        {
            "triples": [
                "Elliot_See | almaMater | University_of_Texas_at_Austin",
                "Elliot_See | deathPlace | St._Louis",
                "Elliot_See | birthPlace | Dallas",
                "Elliot_See | selectedByNasa | 1962",
            ],
            "text": "Elliot See who was born in Dallas and died in St Louis, graduated from the University of Texas at Austin and was selected by NASA in 1962.",
        },
        {
            "triples": [
                "Ashgabat_International_Airport | location | Ashgabat",
                "Ashgabat_International_Airport | elevationAboveTheSeaLevelInMetres | 211",
                "Ashgabat_International_Airport | runwayLength | 900.0",
            ],
            "text": "Ashgabat International Airport is located in Ashgabat 211 metres above sea level with a 0.9 km long runway.",
        },
    ]

    for j in range(len(instructions)):
        prompt = f"{instructions[j]}\n\n"
        prompt += "### Triples\n"
        for triple in icl_samples[j]["triples"]:
            prompt += f"{triple}\n"
        prompt += "\n### Text\n" + icl_samples[j]["text"] + "\n"
        prompt += "\n### Triples\n{triples}\n### Text"
        TEMPLATES.append(prompt)

    prompts = []
    for sample_idx in range(len(dataset)):
        sample_prompts = []
        triples = dataset.iloc[sample_idx]["modified_triple_sets"]
        # Contstruct triple text
        triple_text = ""
        for triple in triples["mtriple_set"][0]:
            triple_text += f"{triple}\n"
        for prompt_idx in range(len(instructions)):
            sample_prompts.append(TEMPLATES[prompt_idx].format(triples=triple_text))
        prompts.append(sample_prompts)
    return prompts
