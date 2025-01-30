from datetime import datetime
from typing import Dict

import pandas as pd
import yaml
from agents import ChatAgent
from case_scalable_main_system.prompting.props import props_descp_dict
from rdkit import Chem
from rdkit.Chem import Draw
from utils.API import call_for_generation, call_for_ic50, call_for_docking

from typing import List
from rdkit.Chem import  AllChem

from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def eval_qed(smiles: list):
    store = []
    for smile in smiles:
        store.apprnd(Chem.QED.qed(Chem.MolFromSmiles(smile)))
    return store
 
def check_brenk(smiles: list):
    params_brenk = FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_brenk = FilterCatalog(params_brenk)
    mol = Chem.MolFromSmiles(smiles)
    
    return 1*catalog_brenk.HasMatch(mol)

def eval_sa(smiles: list):
    store = []
    for smile in smiles:
        store.apprnd(sascorer.calculateScore(Chem.MolFromSmiles(smile)))
        
    return store

def PatternFilter(patterns, smiles: list):
    structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))
    molecule = Chem.MolFromSmiles(smiles)
    return int(any(molecule.HasSubstructMatch(struct) for struct in structures))

def eval_P_S_G(smiles: list, type_n: str = 'all'):
    """Evalute PAINS, SureChEMBL and Glaxo metrics for smile.

    Args:
        smile (list): SMILES molecule describe
        type_n (str): What function need to evalute, if "all" - all functions will be evalute and return
    Returns:
        List: List with metrics of PAINS,SureChEMBL and Glaxo metrics
    """
    store = []
    for smile in smiles:
        alert_table = pd.read_csv('utils/alert_collections.csv')
        patterns = dict()
        if type_n == 'all':
            for name in ['PAINS', 'SureChEMBL', 'Glaxo']:
                patterns[name] = alert_table[alert_table['rule_set_name'] == name]['smarts']
            store.append([PatternFilter(patterns[i], smile) for i in ['PAINS','Glaxo','SureChEMBL']]  )
        else:
            patterns[type_n] = alert_table[alert_table['rule_set_name'] == type_n]['smarts']
            store.append(PatternFilter(patterns[type_n], smile))
      
def check_diversity(smiles: List[str], train_mols:List[str]):
    fpgen = AllChem.GetRDKitFPGenerator()
    self_scores = []
    gen_fp = [fpgen.GetFingerprint(mol) for mol in [Chem.MolFromSmiles(i) for i in smiles]]
    train_fp = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in train_mols]
    for i,mol in enumerate(gen_fp):
        self_scores.append(max(DataStructs.BulkTanimotoSimilarity(mol, gen_fp.pop(i))))

    train_scores = DataStructs.BulkTanimotoSimilarity(gen_fp, train_fp)
    return self_scores, max(train_scores)


def compute_by_rdkit(molecules: list, property: str):
    mapping_tools = {
        'Brenk': check_brenk, 'Diversity': check_diversity, 'PAINS': eval_P_S_G, 'SureChEMBL': eval_P_S_G, 'Glaxo': eval_P_S_G, 'SA': eval_sa, 'QED': eval_qed
     }
    if property in mapping_tools.keys():
        answer = mapping_tools[property](molecules)
    else:
        answer = False
        print('PROCESS: There is no such property in RDKIT: ', property)
    
    return answer
 
def get_props_description(props: list) -> str:
    descp = """"""
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
        img.save(f"multi_agents_system/vizualization/mol{time[i]}_{i}.png")

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
    with open("case_scalable_main_system/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    answer = ChatAgent(
        model_name=config["conductor_model"],
        api_key=config["llama_api_key"],
        url=config["url"],
    ).call(msg["msg"])
    return [answer, ""]


def train_gen_models(model: str, epoch: int, case_name: str) -> bool:
    """
    Start training the generative model
    """
    params = {"model": model, "epoch": epoch, "case_name": case_name}
    is_succes = call_for_generation(**params)
    
    return eval(is_succes)


def automl_predictive_models(property: str) -> bool:
    """
    Start training the predictive model by AUTOML
    """
    params = {"model": "AUTOML", "epoch": None, "case_name": "", "property": property}
    _, is_succes = call_for_generation(**params)
    
    return eval(is_succes)

    
def inference_predictive_models(property: str, molecules: list) -> dict:
    """Launch of inference predictive model"""
    if property == "IC50":
        _, answer = call_for_ic50(molecules)
    
    else:
        answer = False
        print('PROCESS: There is no predictive model for this property. Property: ', property)
        
    return answer

def compute_docking_score(molecules: list) -> dict:
     _, answer = call_for_docking(molecules)
     return answer
 

def gen_mols(num: int, case: str = "RNDM", model: str = "CVAE") -> list:
    """Generates molecules for specific disease.

    Args:
        num (int): number of molecules to generate
        case (str): name of specific case, for example:
                    'Alzhmr' - Alzheimer,
                    'Sklrz' - Skleroz,
                    'Prkns' - Parkinson,
                    'Cnsr' - Canser,
                    'Dslpdm' - Dyslipidemia,
                    'TBLET' - Drug resistance,
                    'RNDM' - random generation,
                    'HERE_CAN_BE_ANOTER_CASE_NAME' - another case.
                    Defaults is RNDM.
        model (str): model name, can be: 'CVAE', 'TVAE', 'LSTM', 'GraphGA', 'Japon agent system', 'RL'.

    Returns:
        list: list of generated molecules
    """

    params = {"numb_mol": int(num["num"]), "cuda": True, "case_": case, "model": model}
    _, mol_dict = call_for_generation(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])

    return [res, mol_dict]


def create_table(properties: dict) -> pd.DataFrame:
    data = pd.DataFrame(properties)
    data.to_csv("./example_props.csv")
    return data
