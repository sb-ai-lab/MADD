"""
Example of multi-agents chemical pipeline with tools for drug generation (by API).
There are 2 agents.

Process:
- Reading from a file with example queries. 
- Pipeline starts. 
- Decomposer define tasks.
- Conductor-executor agent define and run tools, reflects on everyone tool response and return answer.
"""
from langchain.agents import (
    create_structured_chat_agent,
    AgentExecutor
)
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
import pandas as pd
from protollm.agents.llama31_agents.llama31_agent import Llama31ChatModel
from examples.chemical_pipeline.tools import gen_mols_parkinson, gen_mols_lung_cancer, gen_mols_acquired_drug_resistance, \
         gen_mols_dyslipidemia, gen_mols_multiple_sclerosis, gen_mols_alzheimer, request_mols_generation
from prompting.prompting_twins import system_prompt_conductor_no_reflection, system_prompt_decomposer, human_prompt
from examples.chemical_pipeline.testcase import validate_decompose, validate_conductor, add_answers, check_total_answer, exctrac_mols_and_props


class TwinsChain:
    def __init__(self, tools: list, llm: BaseChatModel):
        self.tools = tools
        self.llm = llm
        self.agents_meta = {
            'conductor': {
                'prompt': system_prompt_conductor_no_reflection,
                'variables': ["tools", "tool_names"]},
            'decomposer': {
                'prompt': system_prompt_decomposer,
                'variables': []
            }}
        self.conductor = self._create_agent_executor('conductor')
        self.decomposer = self.llm
    
    def _complite_prompt_from_template(self, text: str, input_variables: list = ["tools", "tool_names"]):
        system_message = SystemMessagePromptTemplate.from_template(
            text,
            input_variables=input_variables
        )
        human_message = HumanMessagePromptTemplate.from_template(
            human_prompt,
            input_variables=["input", "agent_scratchpad"]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                human_message
            ]
        )
        return prompt
    
    def _complite_prompt_no_template(self, system_txt: str, user_txt: str):
        messages = [
            SystemMessage(
                content=system_txt
            ),
            HumanMessage(
                content=user_txt
            )
        ]
        return messages
    
    def _create_agent_executor(self, agent_name: str):
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._complite_prompt_from_template(
                self.agents_meta[agent_name]['prompt'], 
                self.agents_meta[agent_name]['variables']
            ),
            stop_sequence=True
        )
        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            output_keys=["output"],
            early_stopping_method="generate"
        )
        return executor
    
    def run_chain(self, user_msg: str):
        input = self._complite_prompt_no_template(self.agents_meta['decomposer']['prompt'], user_msg)
        tasks = eval(eval(self.decomposer.invoke(input).content)['action_input'])
        store = []
        funcs = []
        res_funcs = []
        for n, task in enumerate(tasks):
            
            answer = self.conductor.invoke({
                "input": task
            })
            output = answer["output"]
            store.append(output)
            funcs.append(answer['intermediate_steps'][0][0].log)
            res_funcs.append(answer['intermediate_steps'][0][1])
            print('Answer, part №: ', n + 1)
            print(answer['intermediate_steps'][0][0].log)
            print(output)
        
        return tasks, funcs, store, res_funcs

# run pipeline on test data
if __name__ == "__main__":
    llm = Llama31ChatModel(
            api_key='', 
            base_url="https://api.vsegpt.ru/v1",
            model="meta-llama/llama-3.1-70b-instruct",
            temperature=0.5, max_tokens=5000
        )
    path = './agentsbuilder/experiment1.xlsx'
    questions = pd.read_excel(path).values.tolist()
    tools = [gen_mols_parkinson, gen_mols_lung_cancer, gen_mols_acquired_drug_resistance,
         gen_mols_dyslipidemia, gen_mols_multiple_sclerosis, gen_mols_alzheimer, request_mols_generation]

    chain = TwinsChain(llm=llm, tools=tools)
    answers_store = []
    total_success = []
    
    for i, q in enumerate(questions):
        print('Task № ', i)
        is_match_full = True
        tasks, funcs, store, true_mols = chain.run_chain(q[1])
        answers_store.append(store)
            
        succ_dec = validate_decompose(i, tasks, path)
        subtask_n = 0
        for func, ans, true_mol in zip(funcs, store, true_mols):
            match = re.search(r"```({.*?})```", func, re.DOTALL)
            func_dict = json.loads(match.group(1))
            succ_cond = validate_conductor(i, func_dict, subtask_n, path, func_field='action', parm_field='action_input')
            subtask_n += 1
            # mols_list - molecules and names of props
            mols_list = exctrac_mols_and_props(true_mol)
            # check: all molecules from func in finally result
            # if all([mols_list[i] in ans for i in range(len(mols_list))]) and is_match_full:
            if check_total_answer(mols_list, ans) and is_match_full:
                continue
            else:
                is_match_full = False
        
        # add info: are all molecules present in the answer? (True/False)
        total_success.append(is_match_full)
        
        add_answers([answers_store, total_success], './answers.xlsx')
            
            