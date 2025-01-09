from langchain.agents import (
    create_structured_chat_agent,
    AgentExecutor
)
from pathlib import Path
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.chat_models.base import BaseChatModel

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from prompting.prompting_twins import system_prompt_conductor_no_reflection, system_prompt_decomposer, human_prompt
from agents import RAGAgent, DecomposeAgent
        
        
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
            # # TODO: rm if tasks more then 1
            # if n > 0:
            #     continue
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

        
class TripleChain(TwinsChain):
    def __init__(self, tools: list, llm: BaseChatModel, path_to_docs: Path):
        super().__init__(tools, llm)
        self.chemical_agent = RAGAgent(path_to_docs, llm)
        
    def run_chain(self, user_msg: str):
        tasks = self.decomposer.invoke(user_msg)
        store = []
        funcs = []
        res_funcs = []
        for n, task in enumerate(tasks):
            # # TODO: rm if tasks more then 1
            # if n > 0:
            #     continue
            try:
                full_question = self.chemical_agent(task)
                answer = self.conductor.invoke({
                    "input": full_question
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
        
            
            