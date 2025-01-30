import uuid
from typing import List, Optional, Union

import yaml
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from openai import OpenAI
from prompting.chat_prompts import (HELP_FOR_TOOLS, INSTRUCT_DESCP_FOR_CHAT,
                                    INSTRUCT_INTRO)
from prompting.conductor_prompts_v1 import (INSTRUCT_TOOLS_LARGE,
                                            prompt_many_funcs)
from prompting.decompose_prompts import decompose_prompt
from prompting.summary_prompts import prompt_finally
from pydantic.v1 import BaseModel, Field

with open("case_scalable_main_system/config.yaml", "r") as file:
    config = yaml.safe_load(file)


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
        # Local model in ITMO network (custom API)
        elif self.model_name == "local":
            return {
                "url": config["url"],
                "job_id": str(uuid.uuid4()),
                "temperature": 0.1,
                "token_limit": 64000,
            }
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

    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
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
        # if local Llama 3.1-8b (Ollama service)
        tasks = self.decompose_agent.invoke(input)
        return [i[1] for i in tasks if i[1] != None]


class Orchestrator(BaseAgent):
    """Agent to define tools based on user tasks."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, api_key=api_key, url=url)
        self.prompt = prompt_many_funcs
        self.tools = INSTRUCT_TOOLS_LARGE

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
