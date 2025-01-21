import time
from datetime import datetime
from random import randrange
from typing import Dict

import pandas as pd
import yaml
from agents import ChatAgent
from multi_agents_main_system.prompting.props import props_descp_dict
from rdkit import Chem
from rdkit.Chem import Draw
from utils.API import call_for_generation


def get_props_description(props: list) -> str:
    descp = """
    """
    for prop in props:
        try:
            descp += props_descp_dict[prop]
        except:
            continue

    return descp


def mols_vizualization(mols: list):
    """Create vizualization for molecules in SMILES format.
    Save png-images in directory './vizualization'.

    Args:
        mols (list): SMILES string

    Returns:
        None
    """
    # randrange gives you an integral value
    time = [str(datetime.now().time()) for i in range(len(mols))]

    for i, mol in enumerate(mols):
        img = Draw.MolToImage(Chem.MolFromSmiles(mol))
        img.save(f"multi_agents_system/main_system/vizualization/mol{time[i]}_{i}.png")

    print(f"PROCESS: Saved: {i + 1} vizualizations of SMILES")


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
        row = [str(props[key][i]) for key in headers]
        markdown_table += "| " + " | ".join(row) + " |\n"

    return markdown_table


def make_smiles_str(smiles: list) -> str:
    """Convert list with molecules into string

    Args:
        smiles (list): molecules in SMILES

    Returns:
        str: molecules in SMILES
    """
    res = ""
    for s in smiles:
        res += s + " "
    return res


def make_answer_chat_model(msg: str) -> Dict:
    """Answers a question using a chat model (chemical assistant).
    Suitable for free questions that do not require calling other tools.

    Args:
        msg (str): message from user

    Returns:
        answer (str): answer for human message from agent.
    """
    with open("multi_agents_system/main_system/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    answer = ChatAgent(
        model_name=config["conductor_model"],
        api_key=config["api_vse_gpt"],
        url=config["url"],
    ).call(msg["msg"])
    return [answer, ""]


def gen_mols_all_case(num: int, case: str = "RNDM") -> list:
    """Generates molecules for specific disease.

    Args:
        num (int): number of molecules to generate
        case (str): name of specific case, can be:
                    'Alzhmr' - Alzheimer,
                    'Sklrz' - Skleroz,
                    'Prkns' - Parkinson,
                    'Cnsr' - Canser,
                    'Dslpdm' - Dyslipidemia,
                    'TBLET' - Drug resistance,
                    'RNDM' - random generation.
                    Defaults is RNDM.

    Returns:
        list: list of generated molecules
    """
    params = {"numb_mol": int(num["num"]), "cuda": True, "case_": case}
    _, mol_dict = call_for_generation(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])

    return [res, mol_dict]


def request_mols_generation(num: int) -> list:
    """Generates molecules.

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {"numb_mol": int(num["num"]), "cuda": True, "case_": "RNDM"}
    _, mol_dict = call_for_generation(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])

    return [res, mol_dict]


def gen_mols_alzheimer(num: int) -> list:
    """Generation of drug molecules for the treatment of Alzheimer's disease.

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {"numb_mol": int(num["num"]), "cuda": True, "case_": "Alzhmr"}
    _, mol_dict = call_for_generation(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])

    return [res, mol_dict]


def gen_mols_multiple_sclerosis(num: int) -> list:
    """Generation of molecules for the treatment of multiple sclerosis.

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {"numb_mol": int(num["num"]), "cuda": True, "case_": "Sklrz"}
    _, mol_dict = call_for_generation(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])

    return [res, mol_dict]


def gen_mols_dyslipidemia(num: int) -> list:
    """
    Generation of molecules for the treatment of dyslipidemia
    """
    params = {"numb_mol": int(num["num"]), "cuda": True, "case_": "Dslpdm"}
    _, mol_dict = call_for_generation(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])

    return [res, mol_dict]


def gen_mols_acquired_drug_resistance(num: int) -> list:
    """
    Generation of molecules for acquired drug resistance.
    """
    params = {"numb_mol": int(num["num"]), "cuda": True, "case_": "TBLET"}
    _, mol_dict = call_for_generation(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])

    return [res, mol_dict]


def gen_mols_lung_cancer(num: int) -> list:
    """
    Generation of molecules for the treatment of lung cancer.
    """
    params = {"numb_mol": int(num["num"]), "cuda": True, "case_": "Cnsr"}
    _, mol_dict = call_for_generation(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])

    return [res, mol_dict]


def gen_mols_parkinson(num: int) -> list:
    """
    Generation of molecules for parkinson.
    """
    params = {"numb_mol": int(num["num"]), "cuda": True, "case_": "Prkns"}
    _, mol_dict = call_for_generation(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])

    return [res, mol_dict]


def create_table(properties: dict) -> pd.DataFrame:
    data = pd.DataFrame(properties)
    data.to_csv("./example_props.csv")
    return data


def compute_docking_score_metadata():
    NotImplemented
