from typing import List, Optional, Union
import operator
import os
from typing import Annotated
import pandas as pd

from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from prompting.automl_and_data_process_prompts import (
    automl_prompt,
    ds_builder_prompt,
    dataset_processing_prompt,
)
from multi_agent_system.MADD_main.tools.automl_tools import automl_tools
from multi_agent_system.MADD_main.tools.data_gathering import fetch_BindingDB_data, fetch_chembl_data
from multi_agent_system.MADD_main.tools.dataset_tools import filter_columns

import yaml
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from openai import OpenAI
from prompting.chat_prompts import (HELP_FOR_TOOLS, INSTRUCT_DESCP_FOR_CHAT,
                                    INSTRUCT_INTRO)
from prompting.decompose_prompts import decompose_prompt
from prompting.summary_prompts import prompt_finally
from prompting.validator_prompts import (query_pattern_valid,
                                         system_validator_prompt_large,
                                         system_validator_prompt_little)
from pydantic.v1 import BaseModel, Field

with open("multi_agent_system/MADD_main/config.yaml", "r") as file:
    config = yaml.safe_load(file)

if config["prompt_type"] == 1:
    from prompting.conductor_prompts_v1 import (INSTRUCT_TOOLS_LARGE,
                                                INSTRUCT_TOOLS_LITTLE,
                                                prompt_few_funcs,
                                                prompt_many_funcs)
elif config["prompt_type"] == "r1":
    from prompting.conductor_prompts_r1 import (INSTRUCT_TOOLS_LARGE,
                                                INSTRUCT_TOOLS_LITTLE,
                                                prompt_few_funcs,
                                                prompt_many_funcs)
else:
    from prompting.conductor_prompts_v2 import (INSTRUCT_TOOLS_LARGE,
                                                INSTRUCT_TOOLS_LITTLE,
                                                prompt_few_funcs,
                                                prompt_many_funcs)


class BaseAgent:
    available_models = ["meta-llama/llama-3.1-70b-instruct", "deepseek/deepseek-r1", "deepseek/deepseek-r1-alt", "deepseek/deepseek-r1-distill-llama-70b"]

    def __init__(
        self,
        model_name: str = "llama3.1",
        max_tokens: int = 10000,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.url = url
        self.client = self._initialize_client()

    def _initialize_client(self):
        if self.model_name in self.available_models:
            return OpenAI(api_key=self.api_key, base_url=self.url)
        else:
            return ChatOllama(
                model=self.model_name,
                keep_alive=-1,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
            )

    def request(self, prompt: Union[str, List[dict]]) -> str:
        if isinstance(self.client, OpenAI):  # OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=["<|eom_id|><|start_header_id|>", "<|start_header_id|>"],
                extra_headers={"X-Title": "DrugDesign"},
            )
            return response.choices[0].message.content
        else:  # ChatOllama
            return self.client.invoke(prompt)


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


class DecomposeAgent(BaseAgent):
    """
    Agent for decompose input message into subtasks (in range from 1 to 5).
    """

    def __init__(
        self, model_name: str = "llama3.1", api_key: str = None, url: str = None
    ):
        self.model_name = model_name
        if model_name == "gigachat":
            super().__init__(model_name=model_name, api_key=api_key, url=url)
        else:
            decompose_model = ChatOllama(
                model="llama3.1",
                keep_alive=-1,
                temperature=0,
                max_new_tokens=512,
            )
            prompt_decompose = PromptTemplate.from_template(decompose_prompt)
            structured_llm = decompose_model.with_structured_output(Scheduler)
            self.decompose_agent = prompt_decompose | structured_llm

    def call(self, input: str) -> list:
        tasks = self.decompose_agent.invoke(input)
        try:
            return [i[1] for i in tasks if i[1] != None]
        except:
            return [input]


class ConductorAgent(BaseAgent):
    """Agent to define tools based on user tasks."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        is_many_funcs: bool = True,
    ):
        super().__init__(model_name=model_name, api_key=api_key, url=url)
        self.prompt = prompt_many_funcs if is_many_funcs else prompt_few_funcs
        self.tools = INSTRUCT_TOOLS_LARGE if is_many_funcs else INSTRUCT_TOOLS_LITTLE

    def call(self, history: List[str]) -> Union[dict, bool]:
        formatted_history = [eval(item) for item in history]
        system_content = self.prompt + self.tools

        prompt = [{"role": "system", "content": system_content}] + [formatted_history[-1]]
        response = self.request(prompt).replace("<|python_tag|>", "")
        try:
            return eval(response.replace("json", "").replace("```", ""))
        except:
            print(
                "Warning: Conductor returned unexpected format (not JSON). Retrying..."
            )
            return False


class SummaryAgent(BaseAgent):
    """Agent for compiling a final answer from answers to subtasks."""

    def __init__(
        self,
        api_key: str,
        url: str,
        model_name: str = "meta-llama/llama-3.1-70b-instruct",
    ):
        super().__init__(model_name=model_name, api_key=api_key, url=url)

    def call(self, questions: List[str], answers: str) -> str:
        prompt = [
            {
                "role": "user",
                "content": f"{prompt_finally}\nQuestions: {questions}\nAnswers: {answers}\nYour answer:",
            }
        ]
        return self.request(prompt)


class ChatAgent(BaseAgent):
    """Agent for answering free-topic questions."""

    def __init__(
        self,
        api_key: str,
        url: str,
        model_name: str = "meta-llama/llama-3.1-70b-instruct",
    ):
        super().__init__(model_name=model_name, api_key=api_key, url=url)

    def call(self, msg: str) -> str:
        prompt_content = f"<FUNCTIONS>{INSTRUCT_INTRO}{INSTRUCT_DESCP_FOR_CHAT}</FUNCTIONS>\n\n{HELP_FOR_TOOLS}\nQuery: {msg}"
        prompt = [{"role": "user", "content": prompt_content}]
        return self.request(prompt)


class ValidateAgent(BaseAgent):
    """Class for validating answers from the Conductor."""

    def __init__(
        self, api_key: str, model_name: str, url: str, is_many_funcs: bool = False
    ):
        super().__init__(model_name=model_name, api_key=api_key, url=url)
        self.prompt = (
            system_validator_prompt_large
            if is_many_funcs
            else system_validator_prompt_little
        )

    def _parsing(s: str):
        start = s.find("{")
        end = s.rfind("}")

        if start != -1 and end != -1 and start < end:
            return s[start : end + 1]
        else:
            return None

    def validate_tool(self, msg: str, tool: dict, history: List[str]) -> bool:
        formatted_history = history
        prompt_content = query_pattern_valid.format(
            history=formatted_history, msg=msg, func=tool
        )
        prompt = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": prompt_content},
        ]
        response = self.request(prompt)

        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and start < end:
            response_formatted = response[start : end + 1]
        else:
            response_formatted = None

        try:
            valid_func = eval(response_formatted)
        except:
            valid_func = tool
            print(
                "PROCESS: Validator return text answer, return not validated tool for calling..."
            )
        return valid_func == tool, valid_func


def data_gathering_agent(state: dict, config: dict) -> Command:
    """
    Agent JUST for DOWNLOAD data from external sources.
    
    Fetches data from BindingDB and ChEMBL databases according to the provided task,
    using a React agent for decision making. Saves paths to the created datasets
    and maintains execution history.

    Args:
        state: State dictionary containing a 'task' key with task description
        config: Configuration with customizable parameters (including LLM)

    Returns:
        Command: Command object with updates for:
            - past_steps: History of executed steps
            - nodes_calls: Record of node calls
            - metadata: Paths to created datasets
    """
    print("--------------------------------")
    print("DataGathering agent called")
    print(state["task"])
    print("--------------------------------")
    task = state["task"]

    agent = create_react_agent(
        config["configurable"]["llm"],
        [fetch_BindingDB_data, fetch_chembl_data],
        state_modifier=ds_builder_prompt,
        debug=True,
    )
    task_formatted = f"""\nYou are tasked with executing: {task}."""

    response = agent.invoke({"messages": [("user", task_formatted)]})
    
    ds_paths = [i for i in [os.environ.get('DS_FROM_CHEMBL', ''), os.environ.get('DS_FROM_BINDINGDB', '')] if i != '']


    return Command(
        update={
            "past_steps": Annotated[set, operator.or_](set([(task, response["messages"][-1].content)])),
            "nodes_calls": Annotated[set, operator.or_](
                set([("dataset_builder_agent", (("text", response["messages"][-1].content),))])
            ),
            "metadata": Annotated[dict, operator.or_]({"dataset_builder_agent": ds_paths}),
        }
    )


def automl_agent(state: dict, config: dict) -> Command:
    """
    Agent for machine learning, deep learning tasks.
    
    Handles automated ML/DL training, generation, and prediction tasks. Determines
    the appropriate dataset source (BindingDB, ChEMBL, or user-provided) and
    configures the agent accordingly. Manages training constraints and server case checks.

    Args:
        state: State dictionary containing a 'task' key with task description
        config: Configuration with customizable parameters (including LLM)

    Returns:
        Command: Command object with updates for:
            - past_steps: History of executed steps
            - nodes_calls: Record of node calls
    """
    print("--------------------------------")
    print("ml_dl agent called")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    dataset = [os.environ['DS_FROM_BINDINGDB'] if not [os.environ['DS_FROM_CHEMBL'] 
                                                       if not os.environ.get('DS_FROM_USER', False) 
                                                       else os.environ.get('DS_FROM_USER', False)] 
               else os.environ['DS_FROM_USER']]
    if dataset != ['False']:
        if dataset[0][-3:] == 'csv':
            dataset_columns = pd.read_csv(dataset[0]).columns
        elif dataset[0][-3:] == 'csv':
            dataset_columns = pd.read_excel(dataset[0]).columns
        
    if dataset != ['False']:
        agent = create_react_agent(
            config["configurable"]["llm"],
            automl_tools,
            state_modifier=automl_prompt,
            debug=True,
        )
        task_formatted = f"""\nYou are tasked with executing: {task}. Attention: Check if there is a case on the server, if there is one, do not start the training. If there no such case, you MUST use 'run_ml_dl_training_by_daemon'!!! Columns in users dataset: """ + str(list(dataset_columns)) + f"You must pass target_column one of these column names. Feature column should be it must be something related to the SMILES string (find the correct name). Pass this path ({dataset[0]}) to 'path'! YOU MUST ALWAYS CALL TOOLS!"
    else:
        agent = create_react_agent(
            config["configurable"]["llm"],
            automl_tools[1:],
            # no ml_dl_training without datadet existing
            state_modifier=automl_prompt,
            debug=True,
        )
        task_formatted = f"""\n{task}. Attention: Check if there is a case on the server, if there no such case, do not start generation or prediction. If it is in the learning process, tell the user that it cannot be launched until the learning process is complete. Don't do a check for cases on Alzheimer's, sclerosis, parkinsonism, dyslipidemia, drug resistance, lung cancer (they are always there! start generation immediately). You should run tool!!!"""
    
    for _ in range(3):
        response = agent.invoke({"messages": [("user", task_formatted)]})
        if len(response['messages']) > 2:
            break

    return Command(
        update={
            "past_steps": Annotated[set, operator.or_](
                set([(task, response["messages"][-1].content)])
            ),
            "nodes_calls": Annotated[set, operator.or_](
                set([("ml_dl_agent", (("text", response["messages"][-1].content),))])
            ),
        }
    )

def dataset_processing_agent(state: dict, config: dict) -> Command:
    """
    Agent for dataset preprocessing (e.g. filtering columns).

    This agent modifies datasets according to user requests before they
    are passed into ML/DL training or prediction stages.

    Args:
        state (dict): Contains keys like 'task' that describe the user request.
        config (dict): Configuration object with a Language Model (LLM).

    Returns:
        Command: Update object including past steps, node calls, and responses.
    """

    print("--------------------------------")
    print("Dataset processing agent called")
    print(state["task"])
    print("--------------------------------")

    agent = create_react_agent(
        config["configurable"]["llm"],
        [filter_columns],
        state_modifier=dataset_processing_prompt,
        debug=True,
    )

    task_formatted = f"\nYou are tasked with processing dataset: {state['task']}"

    response = agent.invoke({"messages": [("user", task_formatted)]})

    return Command(
        update={
            "past_steps": Annotated[set, operator.or_](
                set([(state["task"], response["messages"][-1].content)])
            ),
            "nodes_calls": Annotated[set, operator.or_](
                set([("dataset_processing_agent", (("text", response["messages"][-1].content),))])
            ),
        }
    )
