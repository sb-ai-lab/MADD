from typing import List, Optional, Union
import operator
import os
from typing import Annotated
import pandas as pd
import yaml
with open("multi_agent_system/MADD_main/config.yaml", "r") as file:
    config = yaml.safe_load(file)
    os.environ["URL_PRED"] = config["URL_PRED"]
    os.environ["URL_GEN"] = config["URL_GEN"]
    os.environ["url"] = config["url"]
    os.environ["conductor_model"] = config["conductor_model"]
    os.environ["OPENAI_API_KEY"] = config["llama_api_key"]
    os.environ["DS_FROM_USER"] = str(config["DS_FROM_USER"])

from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from multi_agent_system.MADD_main.prompting.automl_and_data_process_prompts import (
    automl_prompt,
    ds_builder_prompt,
    dataset_processing_prompt,
)
from multi_agent_system.MADD_main.tools.automl_tools import automl_tools
from multi_agent_system.MADD_main.tools.data_gathering import fetch_BindingDB_data, fetch_chembl_data
from multi_agent_system.MADD_main.tools.dataset_tools import filter_columns


from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from openai import OpenAI
from protollm.connectors import create_llm_connector

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
    dataset = [os.environ["DS_FROM_USER"]]
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

def run_pipeline(ds_input = "data.csv"):

    config = {
        "configurable": {
            "llm": create_llm_connector(
                f"{os.environ['url']};{os.environ['conductor_model']}",
                temperature=0.0
            )
        }
    }

    state_processing = {
        "task": f"""Remove columns ['Unnamed: 0', 'Unnamed: 0.2', 'Unnamed: 0.1'] from dataset {ds_input}.
        Return path to cleaned dataset."""
    }

    processing_result: Command = dataset_processing_agent(state_processing, config)

    response_text = list(processing_result.update["past_steps"])[0][1]
    print("\nAgent output:", response_text)

    cleaned_path = 'ds/data_columns_filtered.csv'

    os.environ["DS_FROM_USER"] = cleaned_path

    state_automl = {
        "task": f"Train predictive model on {cleaned_path} with IC50 as target and Smiles as feature column."
    }

    automl_result: Command = automl_agent(state_automl, config)

    print(automl_result.update)

if __name__ == "__main__":
    run_pipeline()