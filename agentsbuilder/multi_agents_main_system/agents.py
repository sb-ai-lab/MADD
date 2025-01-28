from typing import List, Optional, Union

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

with open("agentsbuilder/multi_agents_main_system/config.yaml", "r") as file:
    config = yaml.safe_load(file)

if config["prompt_type"] == 1:
    from prompting.conductor_prompts_v1 import (INSTRUCT_TOOLS_LARGE,
                                                INSTRUCT_TOOLS_LITTLE,
                                                prompt_few_funcs,
                                                prompt_many_funcs)
else:
    from prompting.conductor_prompts_v2 import (INSTRUCT_TOOLS_LARGE,
                                                INSTRUCT_TOOLS_LITTLE,
                                                prompt_few_funcs,
                                                prompt_many_funcs)


class BaseAgent:
    available_models = ["meta-llama/llama-3.1-70b-instruct"]

    def __init__(
        self,
        model_name: str = "llama3.1",
        max_tokens: int = 10000,
        temperature: float = 1.0,
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
        return [i[1] for i in tasks if i[1] != None]


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

        prompt = [{"role": "system", "content": system_content}] + formatted_history
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
        formatted_history = [{"role": "user", "content": item} for item in history]
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
