from typing import List, Optional, Union

import yaml
from agents import BaseAgent

with open(
    "multi_agent_system/MADD_main/config.yaml",
    "r",
) as file:
    config = yaml.safe_load(file)

from prompting.cond_dec_prompts import (
    INSTRUCT_TOOLS_LARGE,
    INSTRUCT_TOOLS_LITTLE,
    prompt_many_funcs,
)
from prompting.cond_summ_prompts import (
    INSTRUCT_TOOLS_LARGE,
    INSTRUCT_TOOLS_LITTLE,
    prompt_many_funcs,
)


class ConductorSummarizerAgent(BaseAgent):
    """Agent for define tools and summarizing finally answer."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        is_many_funcs: bool = True,
    ):
        super().__init__(model_name=model_name, api_key=api_key, url=url)
        self.prompt = prompt_many_funcs
        self.tools = INSTRUCT_TOOLS_LARGE if is_many_funcs else INSTRUCT_TOOLS_LITTLE

    def call(self, history: List[str]) -> Union[dict, bool]:
        formatted_history = [eval(item) for item in history]
        system_content = self.prompt + self.tools

        prompt = [{"role": "system", "content": system_content}] + formatted_history

        response = self.request(prompt).replace("<|python_tag|>", "")
        try:
            return eval(response.replace("json", "").replace("```", ""))
        except:
            return response


class ConductorDecomposerAgent(BaseAgent):
    """Agent for decomposing tasks and define tools."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        is_many_funcs: bool = True,
    ):
        super().__init__(model_name=model_name, api_key=api_key, url=url)
        self.prompt = prompt_many_funcs
        self.tools = INSTRUCT_TOOLS_LARGE if is_many_funcs else INSTRUCT_TOOLS_LITTLE

    def call(self, history: List[str]) -> Union[dict, bool]:
        formatted_history = [eval(item) for item in history]
        system_content = self.prompt + self.tools

        prompt = [{"role": "system", "content": system_content}] + [
            formatted_history[-1]
        ]
        response = self.request(prompt).replace("<|python_tag|>", "")
        try:
            return eval(response.replace("json", "").replace("```", ""))
        except:
            print("Warning: Conductor returned unexpected format.")
            return [
                {
                    "name": "make_answer_chat_model",
                    "parameters": {"msg": "Conductor returned unexpected format."},
                }
            ]
