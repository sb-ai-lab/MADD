"""
Multi-Agent LLM System Chain for Scalable Case Solutions.

This pipeline is designed to solve an unlimited number of cases by fine-tuning existing models for any specific disease. 
It integrates decomposition, molecule generation, and summary capabilities to provide insights and solutions based on input queries.


Launch example:
    chain = Chain(
        conductor_model=MODEL_NAME,
        url=BASE_URL,
        llama_api_key=KEY
    )
    while True:
        response = chain.run(str(input()))
"""

import os
import time
from os import listdir
from os.path import isfile, join
from typing import Union

import yaml
from agents import ChatAgent, ConductorAgent, DecomposeAgent, SummaryAgent
from memory import ChatMemory
from case_scalable_main_system.prompting.props import props_descp_dict
from prompting.props import enter, props_descp_dict, props_name
from tools import gen_mols, make_answer_chat_model, run_fine_tuning


class Chain:
    """
    A class implementing a multi-agent system pipeline for solving case-specific problems with LLM-based components.
    The pipeline can dynamically fine-tune models for scalability to address an unlimited number of cases,
    particularly for generating molecular structures tailored to diseases.

    Attributes
    ----------
    conductor_model : str
        Name of the language model used as the conductor agent.
    llama_api_key : str, optional
        API key for authenticating with the Llama-based services.
    url : str
        The base URL for the API.
    msg_for_store : int, optional
        Number of messages to store in the chat history for processing context (default is 1).

    Methods
    -------
    rm_last_saved_file(dir: str = "multi_agents_system/vizualization/")
        Deletes all saved visualization files from the specified directory.
    call_tool(tool: dict) -> Union[list, int, str]
        Executes the tool function specified in the tool dictionary and returns the result.
    task_handler() -> str
        Defines the tool to call, validates its output, and manages the interaction between agents.
    run(human_message: str) -> str
        Main method to process an input query, decompose it into tasks, generate solutions, and return the output.
    """

    ...

    def __init__(
        self,
        conductor_model: str,
        url: str,
        llama_api_key: str = None,
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
            Base url
        msg_for_store : int, optional
            Number of message for store
        """
        if len(llama_api_key) < 1:
            raise ValueError("API key for Llama is missing.")
        self.attempt = 1
        self.msg_for_store = msg_for_store
        self.chat_history = ChatMemory(
            msg_limit=msg_for_store, model_type=conductor_model
        )
        self.decompose_agent = DecomposeAgent(conductor_model)
        self.conductor_agent = ConductorAgent(
            model_name=conductor_model, api_key=llama_api_key, url=url
        )
        self.chat_agent, self.summary_agent = (
            ChatAgent(model_name=conductor_model, api_key=llama_api_key, url=url),
            SummaryAgent(model_name=conductor_model, api_key=llama_api_key, url=url),
        )
        self.tools_map = {
            "gen_mols": gen_mols,
            "run_fine_tuning": run_fine_tuning,
            "make_answer_chat_model": make_answer_chat_model,
        }
        with open("case_scalable_main_system/config.yaml", "r") as file:
            self.conf = yaml.safe_load(file)

    def rm_last_saved_file(self, dir: str = "multi_agents_system/vizualization/"):
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
        total_attempt = 0
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

                if not (tool):
                    tool = self.conductor_agent.call(self.chat_history.store)

                if isinstance(tool, dict):
                    pass
                elif len(tool) > 1:
                    tool = tool[0]

                print(f'TOOL: {tool["name"]} {tool["parameters"]}')

                success = False

                res, mol = self.call_tool(tool)
                print("PROCESS: getted response from tool")
                success = True

                self.chat_history.add(str(tool), "assistant")
                self.chat_history.add(str(res), "ipython")

                return res, tool["name"], mol
            except:
                pass

    def run(self, human_message: str) -> str:
        """Run chain"""

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

        self.rm_last_saved_file()

        if human_message == "" or human_message == " ":
            response = "You didn't write anything - can I help you with something?"
            return response

        tasks = self.decompose_agent.call(human_message)
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

            finally_ans = self.summary_agent.call(tasks, finally_ans)

        # if just 1 answer
        else:
            if free_response_store != []:
                finally_ans = free_response_store[0]
            else:
                finally_ans, descp_props = collect_table_answer(
                    finally_ans, tables_store
                )

        if tables_store != [] and descp_props != "":
            finally_ans += (
                enter + "Description of properties in table: \n" + descp_props
            )

        return str(finally_ans).replace("[N+]", "[N+\]")


if __name__ == "__main__":
    with open("case_scalable_main_system/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    chain = Chain(
        conductor_model=config["conductor_model"],
        llama_api_key=config["llama_api_key"],
        url=config["url"],
    )

    while True:
        response = chain.run(input())
        print(response)
