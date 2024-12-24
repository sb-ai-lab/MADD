"""
Example of multi-agents chemical pipeline with tools for drug generation (by API).
There are 2 agents.

Process:
- Reading from a file with example queries. 
- Pipeline starts. 
- Decomposer define tasks.
- Conductor-executor agent define and run tools, reflects on everyone tool response and return answer.
"""
from langchain_ollama import ChatOllama
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
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate


decompose_prompt = """You are an agent decomposing the input message into subparagraphs. Answer \
    should only save objects from the input message. You must split human message into a lot of subtasks!
    
    example_human: Generate a lot of molecules for multple scleroses!
    example_ai: output=Scheduler(task1='Generate a lot of molecules for multple scleroses!') 

    example_human: Generate a potential molecule to combat insecticide resistance
    example_ai: output=Scheduler(task1='Generate a potential molecule to combat insecticide resistance')

    example_human: Generate a molecule, define properties for it
    example_ai: output=Scheduler(task1='Generate a molecule', task2='Define properties for it')

    example_human: Generate a molecule, define properties. Generate 4 molecules
    example_ai: output=Scheduler(task1='Generate a molecule', task2='Define properties', task3='Generate 4 molecules')

    example_human: What should I do for you to generate molecules?
    example_ai: output=Scheduler(task1='What should I do for you to generate molecules?')
    
    example_human: Generate 5 molecules to treat Alzheimer's, 1 to fight insects, 2 to treat sclerosis. 
    And tell me what you can do in general?
    example_ai: output=Scheduler(task1='Generate 5 molecules to treat Alzheimer's', task2='Generate 1 molecule to to fight insects', 
    task3='Generate 2 molecules treat sclerosis', task4='tell me what you can do in general?')
    
    Your answer must consist only of the user's message, there should not be any new words in it!

    Human: {question}
    AI: """


class Scheduler(BaseModel):
    "Decompose message into a lot of action"

    task1: str = Field(
        description="1 of the tasks to fulfill a user request.", required=True
    )
    task2: str = Field(
        None, description="2 of the tasks to fulfill a user request.", required=False
    )
    task3: str = Field(
        None, description="3 of the tasks to fulfill a user request.", required=False
    )
    task4: str = Field(
        None, description="4 of the tasks to fulfill a user request.", required=False
    )
    task5: str = Field(
        None, description="5 of the tasks to fulfill a user request.", required=False
    )

class DecomposeAgent():
    """
    Agent for decompose input message into subtasks (in range from 1 to 5).
    """

    def __init__(self):
        decompose_model = ChatOllama(
            model="llama3.1",
            keep_alive=-1,
            temperature=0,
            max_new_tokens=512,
        )
        # decompose_model = Llama31ChatModel(
        #     api_key='sk-or-vv-7fcc4ab944ca013feb7608fb7c0f001e5c12c32abf66233aad414183b4191a79', 
        #     base_url="https://api.vsegpt.ru/v1",
        #     model="meta-llama/llama-3.1-70b-instruct",
        #     temperature=0.5, max_tokens=5000
        # )
        prompt_decompose = PromptTemplate.from_template(decompose_prompt)
        structured_llm = decompose_model.with_structured_output(Scheduler)
        self.agent = prompt_decompose | structured_llm

    def invoke(self, input: str) -> list:
        # if local Llama 3.1-8b (Ollama service)
        tasks = self.agent.invoke(input)
        return [i[1] for i in tasks if i[1] != None]


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
        # self.decomposer = self.llm
        self.decomposer = DecomposeAgent()
    
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
        tasks = self.decomposer.invoke(user_msg)
        store = []
        funcs = []
        res_funcs = []
        for n, task in enumerate(tasks):
            # TODO: rm if tasks more then 1
            if n > 0:
                continue
            try:
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
            except:
                pass

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
    store_tools_answers = []     
    
    for i, q in enumerate(questions):
        print('Task № ', i)
        is_match_full = True
        tasks, funcs, store, true_mols = chain.run_chain(q[1])
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
            else:
                subtask_n += 1
                is_match_full = False
        except:
            is_match_full = False
        
        # add info: are all molecules present in the answer? (True/False)
        total_success.append(is_match_full)
        
        add_answers([answers_store, store_tools_answers, total_success], './answers.xlsx')
            
            