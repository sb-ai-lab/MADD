import json

import requests
from langchain.agents import tool


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


# Define tools using the @tool decorator
@tool
def request_mols_generation(num: int) -> list:
    """Generates random molecules.

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    params = {"numb_mol": num, "cuda": True, "mean_": 0.0, "std_": 1.0, "case_": "RNDM"}
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_alzheimer(num: int) -> list:
    """Generation of drug molecules for the treatment of Alzheimer's disease. GSK-3beta inhibitors with high activity. \
    These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability

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
        "case_": "Alzhmr",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))

    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_multiple_sclerosis(num: int) -> list:
    """Generation of molecules for the treatment of multiple sclerosis.\
            There are high activity tyrosine-protein kinase BTK inhibitors or highly potent non-covalent \
            BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential \
            to affect B cells as a therapeutic target for the treatment of multiple sclerosis.

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
        "case_": "Sklrz",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))

    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_dyslipidemia(num: int) -> list:
    """
    Generation of molecules for the treatment of dyslipidemia.
    Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and
    the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances
    reverse cholesterol transport via ABCA1 upregulation
    , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be
    PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.

    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Dslpdm",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_acquired_drug_resistance(num: int) -> list:
    """
    Generation of molecules for acquired drug resistance.
    Molecules that selectively induce apoptosis in drug-resistant tumor cells.
    It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "TBLET",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_lung_cancer(num: int) -> list:
    """
    Generation of molecules for the treatment of lung cancer.
    Molecules are inhibitors of KRAS protein with G12C mutation.
    The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.
    Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D,
    V14I, L19F, Q22K, D33E, Q61H, K117N and A146V/T.
    """
    params = {"numb_mol": num, "cuda": True, "mean_": 0.0, "std_": 1.0, "case_": "Cnsr"}
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))
    ans = make_markdown_table(json.loads(resp.json()))

    return ans


@tool
def gen_mols_parkinson(num: int) -> list:
    """
    Generation of molecules for parkinson.
    """
    params = {
        "numb_mol": num,
        "cuda": True,
        "mean_": 0.0,
        "std_": 1.0,
        "case_": "Prkns",
    }
    resp = requests.post("http://10.32.2.2:81/case_generator", data=json.dumps(params))

    ans = make_markdown_table(json.loads(resp.json()))

    return ans


# List of tools
tools = [
    gen_mols_parkinson,
    gen_mols_lung_cancer,
    gen_mols_acquired_drug_resistance,
    gen_mols_dyslipidemia,
    gen_mols_multiple_sclerosis,
    gen_mols_alzheimer,
    request_mols_generation,
]
