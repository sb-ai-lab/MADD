import json

from multi_agents_system.case_scalable_system.prompting.metadata_tools import (
    gen_mols_metadata, make_answer_chat_model_metadata,
    request_fine_tuning_metadata)

INSTRUCT_TOOLS_LARGE = json.dumps(
    [gen_mols_metadata, request_fine_tuning_metadata, make_answer_chat_model_metadata],
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
    + """<|eot_id|>

For example: 
Query from user: What can you do?
You: {"name": "make_answer_chat_model", "parameters": {"msg": "What can you do?"}}
For example: 
Generate molecules for cancer.
You: {"name": "gen_mols", "parameters": {"num": 1, "case": "Cnsr", model: "CVAE"}}
For example: 
Generate molecule for cirrhosis
You: {"name": "gen_mols", "parameters": {"num": 1, "case": "ANOTHER", model: "RL"}}
"""
)
