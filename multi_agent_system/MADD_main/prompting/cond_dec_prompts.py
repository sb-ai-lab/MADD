import json

from prompting.metadata_tools_v2 import (
    gen_mols_acquired_drug_resistance_metadata,
    gen_mols_alzheimer_metadata,
    gen_mols_dyslipidemia_metadata,
    gen_mols_lung_cancer_metadata,
    gen_mols_multiple_sclerosis_metadata,
    gen_mols_parkinson_metadata,
    make_answer_chat_model_metadata,
    mols_gen_metadata,
    mols_gen_total_metadata,
)

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
        make_answer_chat_model_metadata,
    ],
    indent=4,
    separators=(",", ": "),
)

prompt_many_funcs = (
    """
You are a conductor with tool-calling capabilities.

When using tools, respond in the format [{"name": function name, "parameters": dictionary of function arguments}, ...].  
Do not use variables. Your answer must consist only of a JSON list of dictionaries.  
If there is not enough information for the arguments from the request, take them from the message history provided to you.

If the arguments are in the wrong form or missing (and not found in history),  
call 'make_answer_chat_model' with 'msg' containing the user's query.  
If the user asks what you can do, call 'make_answer_chat_model' too (using the query from the user in "msg").  
If you are asked to generate something, use 'make_answer_chat_model' only as a last resort!

Decompose complex queries into multiple tool calls when applicable.

For example:  
User: What can you do?  
You: [{"name": "make_answer_chat_model", "parameters": {"msg": "What can you do?"}}]  

User: Give me active molecules against GSK-3beta protein.  
You: [{"name": "gen_mols_alzheimer", "parameters": {"num": 1}}]  

User: Suggest several molecules that have high docking affinity with KRAS G12C protein.  
You: [{"name": "gen_mols_lung_cancer", "parameters": {"num": 2}}]  

User: Generate highly potent non-covalent BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential to affect B cells as a therapeutic target for the treatment of multiple sclerosis.  
You: [{"name": "gen_mols_multiple_sclerosis", "parameters": {"num": 1}}]  

User: Give me active molecules against GSK-3beta protein. I want them to be not toxic. Suggest some small molecules that inhibit KRAS G12C - a target responsible for non-small cell lung cancer.  
You: [{"name": "gen_mols_alzheimer", "parameters": {"num": 1}}, {"name": "gen_mols_lung_cancer", "parameters": {"num": 1}}] 

"""
    + INSTRUCT_TOOLS_LARGE
)
