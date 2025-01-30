import json

from multi_agents_system.case_scalable_system.prompting.metadata_tools import gen_mols_metadata, make_answer_chat_model_metadata, compute_by_rdkit_metadata, \
    train_gen_models_metadata, automl_predictive_models_metadata, inference_predictive_models_metadata, compute_docking_score_metadata

INSTRUCT_TOOLS_LARGE = json.dumps(
    [gen_mols_metadata, make_answer_chat_model_metadata, compute_by_rdkit_metadata, \
    train_gen_models_metadata, automl_predictive_models_metadata, inference_predictive_models_metadata, compute_docking_score_metadata],
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

query_pattern_valid = f"""
History: {{history}}
User: {{msg}}
Function: {{func}}
You: """
