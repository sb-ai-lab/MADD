"""
Example of using a single agent in a chemical pipeline with tools for drug generation (by API).

Process:
- Reading from a file with example queries. 
- Pipeline starts. 
- Results are written to the same file.
"""
from langchain.agents import (
    create_structured_chat_agent,
    AgentExecutor
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
import pandas as pd
from protollm.agents.llama31_agents.llama31_agent import Llama31ChatModel
from multi_agent_system.MADD_main.testcase.validate_pipeline import \
    exctrac_mols_and_props, check_total_answer, validate_decompose, validate_conductor, add_answers
from tools.single_agent_tools import tools



# Create the system and human prompts
system_prompt = '''
Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a JSON blob to specify a tool by providing an "action" key (tool name) and an "action_input" key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per JSON blob, as shown:

{{ "action": $TOOL_NAME, "action_input": $INPUT }}

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action: $JSON_BLOB

Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action: {{ "action": "Final Answer", "action_input": "Final response to human" }}


Begin! Reminder to ALWAYS respond with a valid JSON blob of a single action. Use tools if necessary. 
Respond directly if appropriate. Format is Action:```$JSON_BLOB``` then Observation
In the "Final Answer" you must ALWAYS display all generated molecules!!!
For example answer must consist table (!):
| Molecules | QED | Synthetic Accessibility | PAINS | SureChEMBL | Glaxo | Brenk | BBB | IC50 |
\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| Fc1ccc2c(c1)CCc1ccccc1-2 | 0.6064732613170888 
| 1.721973678244476 | 0 | 0 | 0 | 0 | 1 | 0 |\n| O=C(Nc1ccc(C(=O)c2ccccc2)cc1)c1ccc(F)cc1 | 0.728441789442482 
| 1.4782662488060723 | 0 | 0 | 0 | 0 | 1 | 1 |\n| O=C(Nc1ccccc1)c1ccc(NS(=O)(=O)c2ccc3c(c2)CCC3=O)cc1 | 
0.6727786031171711 | 1.9616124655434675 | 0 | 0 | 0 | 0 | 0 | 0 |\n| Cc1ccc(C)c(-n2c(=O)c3ccccc3n(Cc3ccccc3)c2=O)c1 
| 0.5601042919484651 | 1.920664623176684 | 0 | 0 | 0 | 0 | 1 | 1 |\n| Cc1ccc2c(c1)N(C(=O)CN1C(=O)NC3(CCCc4ccccc43)C1=O)CC2 
| 0.8031696199670261 | 3.3073398307371438 | 0 | 0 | 0 | 1 | 1 | 0 |"
'''

human_prompt = '''{input}
{agent_scratchpad}
(Reminder to respond in a JSON blob no matter what)'''

system_message = SystemMessagePromptTemplate.from_template(
    system_prompt,
    input_variables=["tools", "tool_names"],
)
human_message = HumanMessagePromptTemplate.from_template(
    human_prompt,
    input_variables=["input", "agent_scratchpad"],
)

# Create the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        system_message,
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        human_message,
    ]
)

# Initialize the custom LLM
llm = Llama31ChatModel(
    api_key='API_KEY',
    base_url="BASE_URL",
    model="meta-llama/llama-3.1-70b-instruct",
    temperature=0.0,
    max_tokens=5000
)

# Create the structured chat agent
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

# Create the AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    output_keys=["output"],
    early_stopping_method="generate"
)

# Example usage of the agent
if __name__ == "__main__":
    req_error_cnt = 0
    answers_store, store_tools_answers, total_success = [], [], []
    path = './exp3_clean.xlsx'
    questions = pd.read_excel(path).values.tolist()
    
    answers_store, store_tools_answers, total_success
    
    for i, q in enumerate(questions):
        print('Task № ', i)
        try:
            response = agent_executor.invoke({
                "input": q[1]
            })
            
            validate_decompose(i, response["intermediate_steps"], path)
            for n, tools_pred in enumerate(response["intermediate_steps"]):
                name_tool = tools_pred[0].tool
                func = {'name': name_tool}
                validate_conductor(i, func, n, path)
                
            # Access the output
            final_answer = response["output"]
            # Print the final answer
            print(f"Agent's Response: \n {final_answer}")
            
            true_mols, answers = [], []
            is_match_full = True   
            
            for i in range(len(response["intermediate_steps"])):
                true_mol = response["intermediate_steps"][i][1]
                mols_list = exctrac_mols_and_props(true_mol)
                true_mols.append(true_mol)
                answers.append(final_answer)
                if check_total_answer(mols_list, final_answer) and is_match_full:
                        continue
                else:
                    is_match_full = False
                
            total_success.append(is_match_full)
            answers_store.append(answers)
            store_tools_answers.append(true_mols)
            
            add_answers([answers_store, store_tools_answers, total_success], './answers_exp3_1agent.xlsx')    
        except:
            req_error_cnt += 1
            print('Someting went wrong (request), number of errors:', req_error_cnt)
            total_success.append([])
            answers_store.append([])
            store_tools_answers.append([])
            
        