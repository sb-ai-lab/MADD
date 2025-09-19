"""A script for running MADD on benchmarks (S, M or L) and calculating metrics"""

import os

import yaml

with open("multi_agent_system/MADD_main/config.yaml", "r") as file:
    config = yaml.safe_load(file)
    os.environ["URL_PRED"] = config["URL_PRED"]
    os.environ["URL_GEN"] = config["URL_GEN"]

from os import listdir
from os.path import isfile, join
from typing import Union

import pandas as pd
from agents import (
    ChatAgent,
    ConductorAgent,
    DecomposeAgent,
    SummaryAgent,
)
from memory import ChatMemory
from prompting.props import enter, props_descp_dict, props_name
from testcase.validate_pipeline import (
    add_answers,
    check_total_answer,
    exctrac_mols_and_props,
    validate_conductor,
    validate_decompose,
)

from multi_agent_system.MADD_main.tools.tools import (
    gen_mols_acquired_drug_resistance,
    gen_mols_all_case,
    gen_mols_alzheimer,
    gen_mols_dyslipidemia,
    gen_mols_lung_cancer,
    gen_mols_multiple_sclerosis,
    gen_mols_parkinson,
    make_answer_chat_model,
    request_mols_generation,
)

TOTAL_QUERYS = 0


class ValidationChain:
    """
    A pipeline for solving case-specific problems using a multi-agent system with LLM-based components.
    This chain supports task decomposition, molecular generation, tool validation, and summarization.

    Attributes
    ----------
    conductor_model : str
        The name of the language model used as the conductor agent.
    llama_api_key : str
        API key for accessing the Llama-based services.
    url : str
        Base URL for the API used.
    attempt : int
        Number of attempts to fix a tool call after a failed validation.
    is_many_funcs : bool
        Determines if separate generation functions are used for each case or a single function is used.
    msg_for_store : int
        Number of messages to store in the chat history for context processing.
    validation_path : str
        Path to the file containing validation data.
    tools_map : dict
        A mapping of tool names to their corresponding function implementations.

    Methods
    -------
    rm_last_saved_file(dir="multi_agents_system/vizualization/")
        Deletes all visualization files from the specified directory.

    call_tool(tool: dict) -> Union[list, int, str]
        Executes the specified tool function with the provided parameters.

    task_handler(sub_task_number: int) -> str
        Handles individual tasks by selecting and validating tools, then executing them.

    run(human_message: str) -> str
        Main method to process a human query, decompose it into tasks, generate solutions,
        validate outputs, and summarize the results.
    """

    def __init__(
        self,
        conductor_model: str,
        llama_api_key: str = None,
        url: str = "",
        attempt: int = 1,
        is_many_funcs: bool = True,
        msg_for_store: int = 1,
        validation_path: str = "./experiment1.xlsx",
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
            raise ValueError("API key for VSE GPT is missing.")
        self.attempt = attempt
        self.msg_for_store = msg_for_store
        self.chat_history = ChatMemory(
            msg_limit=msg_for_store, model_type=conductor_model
        )
        self.validation_path = validation_path
        self.decompose_agent = DecomposeAgent(conductor_model)
        self.conductor_agent = ConductorAgent(
            model_name=conductor_model,
            api_key=llama_api_key,
            url=url,
            is_many_funcs=is_many_funcs,
        )
        self.chat_agent, self.summary_agent = (
            ChatAgent(model_name=conductor_model, api_key=llama_api_key, url=url),
            SummaryAgent(model_name=conductor_model, api_key=llama_api_key, url=url)
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

    def rm_last_saved_file(
        self, dir: str = "multi_agent_system/MADD_main/vizualization/"
    ):
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

    def task_handler(self, sub_task_number) -> str:
        """Define tool for call and call it.
        Validate answer by self-reflection.

        Returns
        -------
            Output from tools
        """
        global TOTAL_QUERYS
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
                    tool = self.conductor_agent.call([self.chat_history.store[-1]])

                    try:
                        # это валидирует выбор тулза агентом Кондуктором (==Оркестратором), записывает функцию в файл исходный
                        is_valid = validate_conductor(
                            TOTAL_QUERYS, tool, sub_task_number, self.validation_path
                        )
                        print(is_valid)
                        if not (is_valid):
                            print(is_valid)
                    except Exception as e:
                        print("VALIDATION ERROR: ", e)

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

                return res, tool["name"], mol, is_valid
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

        global TOTAL_QUERYS
        self.chat_history.store = []

        # self.rm_last_saved_file()

        if human_message == "" or human_message == " ":
            response = "You didn't write anything - can I help you with something?"
            return response

        # temporary!!!
        tasks = self.decompose_agent.call(human_message)
        # tasks = [human_message]

        try:
            # validate decomposer agent
            validate_decompose(TOTAL_QUERYS, tasks, self.validation_path)
        except:
            pass

        free_response_store, tables_store = [], []

        # limit for message store varies depending on the number of subtasks
        self.chat_history.msg_limit = 20
        mols, answers_store = [], []
        full_success_tool_selection = []

        for i, task in enumerate(tasks):
            self.chat_history.add(task, "user")
            res, tool, mol, success_tool_selection = self.task_handler(i)
            full_success_tool_selection.append(success_tool_selection)
            try:
                mols.append(mol["Molecules"])
            except:
                mols.append(mol)

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
        TOTAL_QUERYS += 1

        if tables_store != [] and descp_props != "":
            finally_ans += (
                enter + "Description of properties in table: \n" + descp_props
            )

        is_match_full = True
        true_mols, total_success = [], []

        if tables_store == []:
            return [
                answers_store,
                tables_store,
                total_success,
                full_success_tool_selection,
            ]

        for j in range(len(tables_store)):
            true_mol = tables_store[j]
            mols_list = exctrac_mols_and_props(true_mol)
            true_mols.append(true_mol)
            if check_total_answer(mols_list, finally_ans) and is_match_full:
                continue
            else:
                is_match_full = False

        total_success.append(is_match_full)

        for i in range(len(tables_store)):
            answers_store.append(finally_ans)

        return [
            answers_store,
            tables_store,
            total_success[0],
            full_success_tool_selection,
        ]


if __name__ == "__main__":
    (
        answers_store,
        tables_store,
        total_success,
        all_success_tool_selection,
        bool_success_tool_selection,
    ) = ([], [], [], [], [])

    chain = ValidationChain(
        conductor_model=config["conductor_model"],
        llama_api_key=config["llama_api_key"],
        is_many_funcs=bool(config["is_many_funcs"]),
        attempt=int(config["attemps"]),
        url=config["url"],
        validation_path=config["validation_path"],
    )

    questions = pd.read_excel(config["validation_path"]).values.tolist()

    for i, q in enumerate(questions):
        try:
            answers, tables, success, success_tool_selection = chain.run(q[1])
        except:
            print("Error in chain!")
            answers, tables, success, success_tool_selection = [""], [""], True, [False]
        answers_store.append(answers), tables_store.append(
            tables
        ), total_success.append(success), all_success_tool_selection.append(
            success_tool_selection
        ),
        bool_success_tool_selection.append(all(success_tool_selection))
        # this creates a file with the final response and responses from the called tools added,
        # and the 3rd column in it indicates whether all the molecules from the tools are present in the response or not
        add_answers(
            [
                answers_store,
                tables_store,
                total_success,
                all_success_tool_selection,
                bool_success_tool_selection,
            ],
            "./answers_ds2_17_09_v2.xlsx",
        )

    ssa_metrict = 100 / len(total_success) * total_success.count(True)
    ts_metrict = (
        100 / len(bool_success_tool_selection) * bool_success_tool_selection.count(True)
    )
    fa_metric = ssa_metrict * ts_metrict / 100

    print(" METRICS ")
    print("SSA: ", ssa_metrict)
    print("TS: ", ts_metrict)
    print("FA: ", fa_metric)
