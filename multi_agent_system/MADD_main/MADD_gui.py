
import re
from datetime import datetime
from os import listdir
import os
from os.path import isfile, join

import gradio as gr
import yaml

# Initialize the chain with the proper key
with open("multi_agent_system/MADD_main/config.yaml", "r")as file:
    config = yaml.safe_load(file)
    os.environ['URL_PRED'] = config["URL_PRED"]
    os.environ['URL_GEN'] = config["URL_GEN"]
    
# it must be here!!! 
from multi_agent_system.MADD_main.MADD_chain import Chain


chain = Chain(
    conductor_model=config["conductor_model"],
    llama_api_key=config["llama_api_key"],
    is_many_funcs=bool(config["is_many_funcs"]),
    attempt=int(config["attemps"]),
    url=config["url"],
)


def extract_time(file_name):
    time_str = re.search(r"(\d{2}:\d{2}:\d{2}\.\d+)", file_name).group(1)
    return datetime.strptime(time_str, "%H:%M:%S.%f")


def add_message(
    history: list[list], message: dict
) -> (list[list], gr.MultimodalTextbox):
    """Add message to chat history"""
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history: list[list]) -> list[list]:
    """Make answer on users message by agents system.
    Checks whether visualization files have been generated - if so,
    transfers images to the chat history.
    """
    try:
        if "/tmp/gradio/" in history[-2][0][0]:
            print("PROCESS: Received file from user")
            answer = chain.run(history[-1][0], docking_loaded=True)
        else:
            answer = chain.run(history[-1][0])
        onlyfiles = [
            f
            for f in listdir("multi_agent_system/MADD_main/vizualization")
            if isfile(join("multi_agent_system/MADD_main/vizualization", f))
        ]
    except:
        answer = chain.run(history[-1][0])
        onlyfiles = [
            f
            for f in listdir("multi_agent_system/MADD_main/vizualization")
            if isfile(join("multi_agent_system/MADD_main/vizualization", f))
        ]

    # if answer consist of molecules and images or just text answer
    try:
        if onlyfiles != []:
            history.append((None, answer))  # add text
            sorted_files = sorted(onlyfiles, key=extract_time)
            for file in sorted_files[::-1]:
                image = gr.Image("multi_agent_system/MADD_main/vizualization/" + file)
                history.append((None, image))  # add image
        else:
            history.append((None, answer))
    # if answer without molecules
    except:
        history[-1][1] = answer
        if onlyfiles != []:
            for file in onlyfiles:
                image = gr.Image("multi_agent_system/MADD_main/vizualization/" + file)
                history.append((None, image))
    return history


with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Введите сообщение",
        show_label=False,
    )

    # procces user input
    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )

    # answer from bot
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")

    # reload text field for next users question
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])


demo.launch(
    share=True,
    server_name=config["frontend_address"],
    server_port=config["frontend_port"],
)
