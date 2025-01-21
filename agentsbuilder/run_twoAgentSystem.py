"""
Example of multi-agents chemical pipeline with tools for drug generation (by API).
There are 2 agents.

Process:
- Reading from a file with example queries. 
- Pipeline starts. 
- Decomposer define tasks.
- Conductor-executor agent define and run tools, reflects on everyone tool response and return answer.
"""

import json
import re

import pandas as pd
from protollm.agents.llama31_agents.llama31_agent import Llama31ChatModel
from agentsbuilder.tools.two_agent_tools import (gen_mols_acquired_drug_resistance, gen_mols_alzheimer,
                         gen_mols_dyslipidemia, gen_mols_lung_cancer,
                         gen_mols_multiple_sclerosis, gen_mols_parkinson,
                         request_mols_generation)

from agentsbuilder.compute_metrics import (add_answers, check_total_answer,
                                           exctrac_mols_and_props,
                                           validate_conductor,
                                           validate_decompose)
from agentsbuilder.system_architecture_zoo import TwinsChain

# run pipeline on val data
if __name__ == "__main__":
    llm = Llama31ChatModel(
        api_key="KEY",
        base_url="URL",
        model="meta-llama/llama-3.1-70b-instruct",
        temperature=0.0,
        max_tokens=5000,
    )
    path = "LLMagentsBuilder/agentsbuilder/experiment3_clear.xlsx"
    questions = pd.read_excel(path).values.tolist()
    tools = [
        gen_mols_parkinson,
        gen_mols_lung_cancer,
        gen_mols_acquired_drug_resistance,
        gen_mols_dyslipidemia,
        gen_mols_multiple_sclerosis,
        gen_mols_alzheimer,
        request_mols_generation,
    ]

    chain = TwinsChain(llm=llm, tools=tools)
    answers_store = []
    total_success = []
    store_tools_answers = []

    for i, q in enumerate(questions):
        print("Task № ", i)
        is_match_full = True
        try:
            tasks, funcs, store, true_mols = chain.run_chain(q[1])
        except:
            continue

        store_tools_answers.append(true_mols)
        answers_store.append(store)
        succ_dec = validate_decompose(i, tasks, path)

        try:
            # if the agent's work was completed correctly
            if funcs != []:
                subtask_n = 0
                for func, ans, true_mol in zip(funcs, store, true_mols):
                    match = re.search(r"```({.*?})```", func, re.DOTALL)
                    func_dict = json.loads(match.group(1))
                    succ_cond = validate_conductor(
                        i,
                        func_dict,
                        subtask_n,
                        path,
                        func_field="action",
                        parm_field="action_input",
                    )
                    subtask_n += 1
                    # mols_list - molecules and names of props
                    mols_list = exctrac_mols_and_props(true_mol)

                    # check: all molecules from func in finally result
                    # if all([mols_list[i] in ans for i in range(len(mols_list))]) and is_match_full:
                    if check_total_answer(mols_list, ans) and is_match_full:
                        continue
                    else:
                        is_match_full = False
            else:
                subtask_n += 1
                is_match_full = False
        except:
            is_match_full = False

        # add info: are all molecules present in the answer? (True/False)
        total_success.append(is_match_full)

        add_answers(
            [answers_store, store_tools_answers, total_success], "./answers__3exp.xlsx"
        )
