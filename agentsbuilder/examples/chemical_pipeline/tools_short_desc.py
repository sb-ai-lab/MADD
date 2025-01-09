from langchain.agents import tool

import requests
import json

def make_markdown_table(props: dict) -> str:
    """Create a table in Markdown format dynamically based on dict keys.

    Args:
        props (dict): properties of molecules

    Returns:
        str: table with properties
    """
    # get all the keys for column headers
    headers = list(props.keys())

    # prepare the header row
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # get the number of rows (assuming all lists in the dictionary are the same length)
    num_rows = len(next(iter(props.values())))

    # fill the table rows dynamically based on the keys
    for i in range(num_rows):
        row = [
            str(props[key][i]) for key in headers
        ]
        markdown_table += "| " + " | ".join(row) + " |\n"

    return markdown_table

# Define tools using the @tool decorator
@tool
def request_mols_generation(num: int) -> list:
    """Generates molecules without case.

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "RNDM"
    }
    resp = requests.post('http://10.32.2.2:81/case_generator', data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans

@tool
def gen_mols_alzheimer(num: int) -> list:
    """Generation of drug molecules for the treatment of Alzheimer's disease. 

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Alzhmr"
    }
    resp = requests.post('http://10.32.2.2:81/case_generator', data=json.dumps(params))

    ans = make_markdown_table(json.loads(resp.json()))

    return ans

@tool
def gen_mols_multiple_sclerosis(num: int) -> list:
    """
    Generation of molecules for the treatment of multiple sclerosis.

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Sklrz"
    }
    resp = requests.post('http://10.32.2.2:81/case_generator', data=json.dumps(params))

    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_dyslipidemia(num: int) -> list:
    """
    Generation of molecules for the treatment of dyslipidemia.
    
    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Dslpdm"
    }
    resp = requests.post('http://10.32.2.2:81/case_generator', data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans

@tool
def gen_mols_acquired_drug_resistance(num: int) -> list:
    """
    Generation of molecules for acquired drug resistance.
    
    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "TBLET"
    }
    resp = requests.post('http://10.32.2.2:81/case_generator', data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans

@tool
def gen_mols_lung_cancer(num: int) -> list:
    """
    Generation of molecules for the treatment of lung cancer.
    
    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Cnsr"
    }
    resp = requests.post('http://10.32.2.2:81/case_generator', data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans

@tool
def gen_mols_parkinson(num: int) -> list:
    """
    Generation of molecules for the treatment of Parkinson's disease.
    
    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Prkns"
    }
    resp = requests.post('http://10.32.2.2:81/case_generator', data=json.dumps(params))

    ans = make_markdown_table(json.loads(resp.json()))

    return ans
