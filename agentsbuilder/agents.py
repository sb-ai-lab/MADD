from langchain_ollama import ChatOllama
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

from langchain.chat_models.base import BaseChatModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_nomic import NomicEmbeddings

from utils import prepare_documents

from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from pathlib import Path


class RAGAgent():
    def __init__(self, path_to_docs: Path, llm: BaseChatModel):
        self.vector_store = self._bd_initialization(path_to_docs)
        self.llm = llm
        self.system_txt = 'You are an agent who supplements the user’s question \
            based on the context so that it is more complete and accurate. Return only the extended question with clarifications (no water!), start with the phrase “I want you to...”. \
            You have the context: '
        
    def _bd_initialization(self, path_to_docs: str):
        docs = prepare_documents(path=path_to_docs)

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=100
        )
        doc_splits = splitter.split_documents(docs)
        embd = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

        vectore_store = Chroma.from_documents(
            documents=doc_splits, collection_name="rag-chem", embedding=embd
        )

        return vectore_store
    
    def retrieve(self, question):
        retrieved_docs = self.vector_store.similarity_search(question, k=1)
        return {"context": retrieved_docs}
    
    def __call__(self, question: str):
        rag_context = self.retrieve(question)
        input = [SystemMessage(self.system_txt + str(rag_context['context'])), HumanMessage(question)]
        result = self.llm(input)
        return result.content
        
        
decompose_prompt = """You are an agent decomposing the input message into subparagraphs. Answer \
    should only save objects from the input message. You must split human message into a lot of subtasks!
    
    example_human: Generate a lot of molecules for multple scleroses!
    example_ai: output=Scheduler(task1='Generate a lot of molecules for multple scleroses!') 

    example_human: Generate a potential molecule to combat insecticide resistance
    example_ai: output=Scheduler(task1='Generate a potential molecule to combat insecticide resistance')

    example_human: Generate a molecule, define properties for it
    example_ai: output=Scheduler(task1='Generate a molecule', task2='Define properties for it')

    example_human: Generate a molecule, define properties. Generate 4 molecules
    example_ai: output=Scheduler(task1='Generate a molecule', task2='Define properties', task3='Generate 4 molecules')

    example_human: What should I do for you to generate molecules?
    example_ai: output=Scheduler(task1='What should I do for you to generate molecules?')
    
    example_human: Generate 5 molecules to treat Alzheimer's, 1 to fight insects, 2 to treat sclerosis. 
    And tell me what you can do in general?
    example_ai: output=Scheduler(task1='Generate 5 molecules to treat Alzheimer's', task2='Generate 1 molecule to to fight insects', 
    task3='Generate 2 molecules treat sclerosis', task4='tell me what you can do in general?')
    
    Your answer must consist only of the user's message, there should not be any new words in it!

    Human: {question}
    AI: """


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

class DecomposeAgent():
    """
    Agent for decompose input message into subtasks (in range from 1 to 5).
    """

    def __init__(self):
        decompose_model = ChatOllama(
            model="llama3.1",
            keep_alive=-1,
            temperature=0,
            max_new_tokens=512,
        )
        prompt_decompose = PromptTemplate.from_template(decompose_prompt)
        structured_llm = decompose_model.with_structured_output(Scheduler)
        self.agent = prompt_decompose | structured_llm

    def invoke(self, input: str) -> list:
        # if local Llama 3.1-8b (Ollama service)
        tasks = self.agent.invoke(input)
        return [i[1] for i in tasks if i[1] != None]
        