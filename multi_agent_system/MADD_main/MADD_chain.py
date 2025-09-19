"""
Multi-Agent LLM System chain.

Launch example:
    chain = Chain(
        conductor_model=MODEL_NAME,
        validate_model=MODEL_NAME,
        llama_api_key=KEY,
    )
    while True:
        response = chain.run(str(input()))
"""

import copy
import os
import time
from os import listdir
from os.path import isfile, join
from typing import Union

import pandas as pd
import yaml
from agents import (ChatAgent, ConductorAgent, DecomposeAgent, SummaryAgent,
                    ValidateAgent)
from memory import ChatMemory
from prompting.props import props_descp_dict
from prompting.props import enter, props_descp_dict, props_name
from multi_agent_system.MADD_main.tools.tools import (gen_mols_acquired_drug_resistance, gen_mols_all_case,
                   gen_mols_alzheimer, gen_mols_dyslipidemia,
                   gen_mols_lung_cancer, gen_mols_multiple_sclerosis,
                   gen_mols_parkinson, make_answer_chat_model,
                   request_mols_generation)

TOTAL_QUERYS = 0


class Chain:
    """
    A multi-agent system chain for processing tasks using large language models (LLMs)
    and generating molecular structures for various medical conditions. The system
    uses multiple agents, including decomposition, conductor, validation, and summarization
    agents, to achieve the desired outputs.

    Attributes
    ----------
    attempt : int
        Maximum number of attempts to fix a function after failed validation.
    msg_for_store : int
        Number of messages to store in the chat memory.
    chat_history : ChatMemory
        Stores the chat history for task processing.
    decompose_agent : DecomposeAgent
        Agent responsible for breaking down the input into subtasks.
    conductor_agent : ConductorAgent
        Agent responsible for managing the sequence of operations and calling tools.
    chat_agent : ChatAgent
        Agent that manages direct communication with the user when task execution fails.
    summary_agent : SummaryAgent
        Agent that summarizes results when multiple subtasks are processed.
    validate_agent : ValidateAgent
        Agent responsible for validating tools and outputs from the conductor agent.
    tools_map : dict
        Mapping of tool names to their corresponding Python functions for molecule generation.
    conf : dict
        Configuration loaded from the YAML file.

    Methods
    -------
    __init__(conductor_model: str, llama_api_key: str, url: str, attempt: int, is_many_funcs: bool, msg_for_store: int) -> None:
        Initializes the chain with the given parameters and agents.
    rm_last_saved_file(dir: str) -> None:
        Deletes all files in the specified visualization directory.
    call_tool(tool: dict) -> Union[list, int, str]:
        Executes the specified tool with the given parameters.
    task_handler() -> str:
        Handles the execution of tools, validates their outputs, and manages retries.
    run(human_message: str) -> str:
        Processes the input message through decomposition, task execution, and summarization.

    Usage
    -----
    To initialize and run the chain:

        chain = Chain(
            conductor_model=MODEL_NAME,
            llama_api_key=API_KEY,
            url="https://api.example.com",
            attempt=3,
            is_many_funcs=True,
            msg_for_store=5
        )
        response = chain.run("Generate molecules for Alzheimer's disease")
    """

    def __init__(
        self,
        conductor_model: str,
        llama_api_key: str = None,
        url: str = "",
        attempt: int = 1,
        is_many_funcs: bool = True,
        msg_for_store: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        conductor_model : str
            Name of model
        llama_api_key : str, optional
            Api-key from personal account
        url: str
            Base url for OpenAI client
        attempt : int, optional
            Number for attempts to fix function after faild validation
        is_many_funcs : bool, optional
            If True -> a separate generation function will be used for each case;
            if not -> one function will be used but with different arguments
        msg_for_store : int, optional
            Number of message for store
        """
        if len(llama_api_key) < 1:
            raise ValueError("API key for LLAMA API is missing.")
        self.attempt = attempt
        self.msg_for_store = msg_for_store
        self.chat_history = ChatMemory(
            msg_limit=msg_for_store, model_type=conductor_model
        )
        self.decompose_agent = DecomposeAgent(conductor_model)
        self.conductor_agent = ConductorAgent(
            model_name=conductor_model,
            api_key=llama_api_key,
            url=url,
            is_many_funcs=is_many_funcs,
        )
        self.chat_agent, self.summary_agent, self.validate_agent = (
            ChatAgent(model_name=conductor_model, api_key=llama_api_key, url=url),
            SummaryAgent(model_name=conductor_model, api_key=llama_api_key, url=url),
            ValidateAgent(
                api_key=llama_api_key,
                model_name=conductor_model,
                url=url,
                is_many_funcs=is_many_funcs,
            ),
        )
        self.tools_map = {
            "request_mols_generation": request_mols_generation,
            "gen_mols_alzheimer": gen_mols_alzheimer,
            "gen_mols_multiple_sclerosis": gen_mols_multiple_sclerosis,
            "gen_mols_acquired_drug_resistance": gen_mols_acquired_drug_resistance,
            "gen_mols_dyslipidemia": gen_mols_dyslipidemia,
            "gen_mols_parkinson": gen_mols_parkinson,
            "gen_mols_lung_cancer": gen_mols_lung_cancer,
            "make_answer_chat_model": make_answer_chat_model,
            "gen_mols_all_case": gen_mols_all_case,
        }
        with open("multi_agent_system/MADD_main/config.yaml", "r") as file:
            self.conf = yaml.safe_load(file)

    def rm_last_saved_file(self, dir: str = "multi_agent_system/MADD_main/vizualization/"):
        onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

        if onlyfiles != []:
            for file in onlyfiles:
                os.remove(dir + file)
            print("PROCESS: All files for vizualization deleted successfully")

    def call_tool(self, tool: dict) -> Union[list, int, str]:
        """
        Call tool like python function

        Parameters
        ----------
        tool : dict
            Name of function and parameters

        Returns
        answer : Union[list, int, str]
            Answer from current function
        """
        answer = self.tools_map[tool["name"].replace(" ", "")](tool["parameters"])
        return answer

    def task_handler(self) -> str:
        """Define tool for call and call it.
        Validate answer by self-reflection.

        Returns
        -------
            Output from tools
        """
        global TOTAL_QUERYS
        attempt, total_attempt = 0, 0
        success, tool = False, None

        # in case the format of the conductors answer is not correct
        while not (success):
            try:
                total_attempt += 1
                if total_attempt > self.attempt:
                    # if something went wrong - run chat agent
                    return (
                        self.chat_agent.call(
                            eval(self.chat_history.store[-1])["content"]
                        ),
                        "make_answer_chat_model",
                        "",
                    )
                temp_chat_history = copy.deepcopy(self.chat_history)

                if not (tool):
                    tool = self.conductor_agent.call(self.chat_history.store)

                if isinstance(tool, dict):
                    pass
                elif len(tool) > 1:
                    tool = tool[0]

                print(f'TOOL: {tool["name"]} {tool["parameters"]}')

                # validate by instruction
                success, valid_tool = self.validate_agent.validate_tool(
                    self.chat_history.store[-1],
                    tool,
                    self.chat_history.store[:-1],
                )

                if not (success):
                    if attempt >= self.attempt:
                        attempt = 0
                    else:
                        attempt += 1
                        self.chat_history = temp_chat_history
                        tool = valid_tool
                        print("PROCESS: Validation for function to call not passed")
                        continue
                # False - cos yet not passed function calling
                success = False

                start_time = time.time()
                res, mol = self.call_tool(tool)
                tools_time = time.time() - start_time
                print("Time for mols generation: ", tools_time)

                print("PROCESS: getted response from tool")
                success = True

                self.chat_history.add(str(tool), "assistant")
                self.chat_history.add(str(res), "ipython")

                return res, tool["name"], mol
            except:
                pass

    def run(self, human_message: str) -> str:
        """Run chain"""
        start_time = time.time()

        def collect_table_answer(finally_ans: str, tables: list) -> list:
            temp_prop = ""
            used = []
            for table in tables:
                for prop in props_name:
                    if prop in table and not (prop in used):
                        temp_prop += props_descp_dict[prop]
                        used.append(prop)
                finally_ans += table
                finally_ans += enter
            return [finally_ans, temp_prop]

        global TOTAL_QUERYS

        self.rm_last_saved_file()

        if human_message == "" or human_message == " ":
            response = "You didn't write anything - can I help you with something?"
            return response

        tasks = self.decompose_agent.call(human_message)
        decomposer_time = time.time() - start_time
        print("Decomposer time: ", decomposer_time)
        print("PROCESS: Define tasks: ", tasks)

        start_conductor_time = time.time()
        free_response_store, tables_store = [], []

        # limit for message store varies depending on the number of subtasks
        self.chat_history.msg_limit = 1 * len(tasks)
        mols = []

        for i, task in enumerate(tasks):
            self.chat_history.add(task, "user")
            res, tool, mol = self.task_handler()
            try:
                mols.append(mol["Molecules"])
            except:
                mols.append(mol)

            end_cond = time.time() - start_conductor_time
            print(f"Conductor time for task № {i + 1}: ", end_cond)
            start_conductor_time = time.time()

            if tool == "make_answer_chat_model":
                free_response_store.append(res)
            else:
                # if answer with generated molecules
                tables_store.append(res)

        finally_ans = ""

        # if there are more then 1 task
        if i != 0:
            finally_ans, descp_props = collect_table_answer(finally_ans, tables_store)
            for free_ans in free_response_store:
                finally_ans += free_ans

            # another format of input
            if self.conf["conductor_model"] == "gigachat":
                finally_ans = self.summary_agent.call(
                    tasks, str(free_response_store) + "\n" + str(tables_store)
                )
            else:
                finally_ans = self.summary_agent.call(tasks, finally_ans)

        # if just 1 answer
        else:
            if free_response_store != []:
                finally_ans = free_response_store[0]
            else:
                finally_ans, descp_props = collect_table_answer(
                    finally_ans, tables_store
                )
        TOTAL_QUERYS += 1

        if tables_store != [] and descp_props != "":
            finally_ans += (
                enter + "Description of properties in table: \n" + descp_props
            )

        return str(finally_ans).replace("[N+]", "[N+\]")


if __name__ == "__main__":
    with open("agentsbuilder/six_case_multi_agents_proto/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    chain = Chain(
        conductor_model=config["conductor_model"],
        llama_api_key=config["llama_api_key"],
        is_many_funcs=bool(config["is_many_funcs"]),
        attempt=int(config["attemps"]),
        url=config["url"],
    )

    questions = pd.read_excel(config["validation_path"]).values.tolist()

    for i, q in enumerate(questions):
        response = chain.run(q[1])
        print(response)
