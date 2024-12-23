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
    """Generates molecules. It need when users ask to give a molecule without indicating why and what kind of molecule it is.

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
    """Generation of drug molecules for the treatment of Alzheimer's disease. \
        GSK-3beta inhibitors with high activity. \
        These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability.\
        Compounds that contain heterocyclic moieties to enhance binding affinity for amyloid-beta aggregates.\
        Hot keys: inhibitory activity against glycogen synthase kinase 3 beta (GSK-3β); compound should \
        demonstrate a permeability coefficient of at least 1.5 to ensure effective crossing of the blood-brain barrier; \
        tau protein kinases with an IC50 value lower than 50 nM.


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
    Generation of molecules for the treatment of multiple sclerosis.\
    There are high activity tyrosine-protein kinase BTK inhibitors or highly potent non-covalent \
    BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential \
    to affect B cells as a therapeutic target for the treatment of multiple sclerosis.\
    Hot keys: high activity tyrosine-protein kinase BTK inhibitors;  inhibit Bruton's tyrosine kinase \
    through non-covalent interaction; non-covalent BTK inhibitors with enhanced permeability across the blood-brain \
    barrier and high selectivity for Cytoplasmic tyrosine-protein kinase BMX;  immune signaling pathways \
    to treat multiple sclerosis.

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
    Generation of molecules for the treatment of dyslipidemia.\
    Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and \
    the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances reverse cholesterol transport via ABCA1 upregulation\
    , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be  PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.\
    Hot keys: molecules that disrupt the interaction between CD36 and oxidized LDL; ligands for SREBP-1c \
    inhibition to regulate fatty acid and triglyceride synthesis; dual-action agents that modulate both HDL \
    and LDL for improved cardiovascular outcomes; AMPK pathway to promote fatty acid oxidation;\
    IC50 value lower than 50 µM for inhibiting CETP;  Tmax of less than 2 hours for rapid action in lipid regulation;\
    a negative logD at pH 7.4 for improved selectivity in tissues; ANGPTL3 inhibitor to reduce plasma triglycerides;\
    PPARα agonist with reduced side;  ApoC-III antisense oligonucleotide with enhanced cellular uptake.
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
    Generation of molecules for acquired drug resistance. \
    Molecules that selectively induce apoptosis in drug-resistant tumor cells.\
    It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.\
    Hot keys: molecular structures targeting drug resistance mechanisms in cancer cells;\
    ABC transporters involved in multidrug resistance; molecules that selectively induce apoptosis in drug-resistant tumor cells;\
    counteracting drug resistance in oncological therapies; treatment sensitivity in resistant tumors; compounds that enhance the \
    efficacy of existing anti-resistance treatments; synergistic compound that significantly enhances the activity of existing therapeutic \
    agents against drug-resistant pathogens; selectively target the Ras-Raf-MEK-ERK signaling pathway\
    molecules targeting the efflux pumps responsible for drug resistance.\
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
    Generation of molecules for the treatment of lung cancer. \
    Molecules are inhibitors of KRAS protein with G12C mutation. \
    The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.\
    Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D, \
    V14I, L19F, Q22K, D33E, Q61H, K117N, G12C and A146V/T.\
    Hot keys: HRAS and NRAS proteins; KRAS G12C protein mutation, which drives cancer growth in lung cancer;\
    avoiding binding to HRAS and NRAS; low cross-reactivity with other RAS isoforms;  molecules to specifically bind and inhibit KRAS G12C
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
    Generation of molecules for the treatment of Parkinson's disease.\
        These compounds should possess high bioavailability, cross the blood-brain barrier efficiently, and show \
        minimal metabolic degradation.\
        Hot keys: derivatives from amino acid groups with modifications in side chains to enhance bioactivity; \
        heterocyclic compounds featuring nitrogen and sulfur to improve pharmacokinetics; molecules using a \
        fragment-based approach, combining elements of natural alkaloids; molecules with properties of glutamate \
        receptor antagonists for neuroprotection; compounds that inhibit monoamine oxidase B (MAO-B);\
        dopamine agonist with improved blood-brain barrier penetration; dual-action molecule combining D2 \
        receptor agonism and A2A antagonism;  PDE10A inhibitor with enhanced selectivity and potency.
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

@tool
def make_answer_chat_model(num: int) -> list:
    """Answers a question using a chat model (chemical assistant)). \
    Suitable for free questions that do not require calling other tools or insufficient/incorrect arguments to call the function. \
    Use it if you need to tell a person how to use what function (if there is doubt in his question)! \
    Use this function if the user asks to define properties for molecules but does not provide them in Smiles format. \
    For example, if user ask: 'What needs to be done for you to define the properties?';\n \
    'What is needed to generate molecules?;

    Args:
        msg (str): Message from user

    Returns:
        str: text answer
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