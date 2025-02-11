import json

from prompting.metadata_tools_v1 import (
    gen_mols_acquired_drug_resistance_metadata, gen_mols_alzheimer_metadata,
    gen_mols_dyslipidemia_metadata, gen_mols_lung_cancer_metadata,
    gen_mols_multiple_sclerosis_metadata, gen_mols_parkinson_metadata,
    make_answer_chat_model_metadata, mols_gen_metadata,
    mols_gen_total_metadata)

INSTRUCT_TOOLS_LARGE = json.dumps(
    [
        gen_mols_lung_cancer_metadata,
        gen_mols_acquired_drug_resistance_metadata,
        gen_mols_dyslipidemia_metadata,
        gen_mols_parkinson_metadata,
        gen_mols_multiple_sclerosis_metadata,
        gen_mols_alzheimer_metadata,
        make_answer_chat_model_metadata,
        mols_gen_metadata,
    ],
    indent=4,
    separators=(",", ": "),
)
INSTRUCT_TOOLS_LITTLE = json.dumps(
    [
        mols_gen_total_metadata,
        gen_mols_acquired_drug_resistance_metadata,
        make_answer_chat_model_metadata,
        mols_gen_metadata,
    ],
    indent=4,
    separators=(",", ": "),
)

system_validator_prompt_large = (
    """
You have to determine if the function is correct based on the user's request and history. 
You must answer in the form of a dictionary, the function is chosen correctly - repeat it, if not - write your answer.

Avalible functions:"""
    + str(INSTRUCT_TOOLS_LARGE)
    + """ 
Example 1.
History:
User: What is it cocrystal?
Function: {"name": "make_answer_chat_model", "parameters": {"msg": "What is it cocrystal?"}}
You: {"name": "make_answer_chat_model", "parameters": {"msg": "What is it cocrystal?"}}

Never edit the dict with 'make_answer_chat_model' function!
"""
)
system_validator_prompt_little = (
    """
You have to determine if the function is correct based on the user's request and history. 
You must answer in the form of a dictionary, the function is chosen correctly - repeat it, if not - write your answer.

Avalible functions:"""
    + str(INSTRUCT_TOOLS_LITTLE)
    + """ 
Example 1.
History:
User: What is it cocrystal?
Function: {"name": "make_answer_chat_model", "parameters": {"msg": "What is it cocrystal?"}}
You: {"name": "make_answer_chat_model", "parameters": {"msg": "What is it cocrystal?"}}

Never edit the dict with 'make_answer_chat_model' function!
"""
)

query_pattern_valid = f"""
History: {{history}}
User: {{msg}}
Function: {{func}}
You: """

self_reflection = """

Your task:
If the answer from the tool does not answer the user's question at all (you're sure!), 
then suggest the function to be generated again. If you are satisfied with the answer, 
simply return it to the user.
"""
