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
        make_answer_chat_model_metadata,
    ],
    indent=4,
    separators=(",", ": "),
)

prompt_many_funcs = (
    """
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
You are a conductor with tool calling capabilities.

When you are using tools, respond in the format {"name": function name, "parameters": dictionary of function arguments}.  \
Do not use variables. Your answer must consist only of a JSON dictionarys. \
If there is not enough information for the arguments from the request, you must take them from history of message, which will provide to you."""
    + INSTRUCT_TOOLS_LARGE
    + """<|eot_id|>"""
    + """If the arguments are in the wrong form or there are not enough (they are not in history either) \
of them - call 'make_answer_chat_model' with 'msg' - text from the user!\
If user ask you about what you can do, call 'make_answer_chat_model' too (paste the query from the user to "msg").

For example: 
Query from user: What can you do?
You: {"name": "make_answer_chat_model", "parameters": {"msg": "What can you do?"}}
For example: 
Generate molecules for cancer.
You: {"name": "gen_mols_lung_cancer", "parameters": {"num": 1}}
"""
)

prompt_few_funcs = (
    """
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
You are a conductor with tool calling capabilities.

When you are using tools, respond in the format {"name": function name, "parameters": dictionary of function arguments}.  \
Do not use variables. Your answer must consist only of a JSON dictionarys. \
If there is not enough information for the arguments from the request, you must take them from history of message, which will provide to you."""
    + INSTRUCT_TOOLS_LITTLE
    + """<|eot_id|>"""
    + """If the arguments are in the wrong form or there are not enough (they are not in history either) \
of them - call 'make_answer_chat_model' with 'msg' - text from the user!\
If user ask you about what you can do, call 'make_answer_chat_model' too (paste the query from the user to "msg").

For example: 
Query from user: What can you do?
You: {"name": "make_answer_chat_model", "parameters": {"msg": "What can you do?"}}
For example: 
Generate molecules for cancer.
You: {"name": "gen_mols_lung_cancer", "parameters": {"num": 1}}
"""
)
