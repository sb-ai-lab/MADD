import yaml
import os 
from protollm.connectors import create_llm_connector

with open("multi_agent_system/MADD_main/config.yaml", "r") as file:
    config = yaml.safe_load(file)
    os.environ["URL_PRED"] = config["URL_PRED"]
    os.environ["URL_GEN"] = config["URL_GEN"]
    os.environ["url"] = config["url"]
    os.environ["conductor_model"] = config["conductor_model"]
    os.environ["OPENAI_API_KEY"] = config["llama_api_key"]
    os.environ["DS_FROM_USER"] = str(config["DS_FROM_USER"])
    agent_conf = {'llm': create_llm_connector(
                f"{os.environ['url']};{os.environ['conductor_model']}", temperature=0.0
            )}
    
from multi_agent_system.MADD_main.agents import data_gathering_agent
    
if __name__ =="__main__":
    state = {}
    state["task"] = "Download data from BindigDB for KRAS with IC50 values."
    res = data_gathering_agent(state, agent_conf)
    print('-----RESULT-----')
    print(list(res.update['past_steps'])[-1][-1])