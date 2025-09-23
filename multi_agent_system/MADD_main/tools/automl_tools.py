import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Process
from typing import List, Tuple, Union

import pandas as pd
import requests
from langchain.tools import tool
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from multi_agent_system.MADD_main.prompting.props import props_descp_dict

conf = {"url_pred": os.environ["URL_PRED"], "url_gen": os.environ["URL_GEN"]}


def convert_to_base64(image_file_path):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    pil_image = Image.open(image_file_path)
    pil_image.save("tmp.png", format="png")

    with open("tmp.png", "rb") as image_file:
        result = base64.b64encode(image_file.read()).decode("utf-8")
        os.remove("tmp.png")
        return result


def convert_to_html(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = (
        f'<img src="data:image/jpeg;base64,{img_base64}" style="max-width: 100%;"/>'
    )
    return image_html


def get_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def generate_for_base_case(
    numb_mol: int = 1,
    cuda: bool = True,
    mean_: float = 0.0,
    std_: float = 1.0,
    url: str = f"{os.environ.get('URL_GEN')}/case_generator",
    case_: str = "RNDM",
    **kwargs,
) -> Tuple[requests.models.Response, dict]:
    """Function that call Chem server API for generate molecules with properties by choosen case. By default it call random generation case.

    Args:
        numb_mol (int, optional): Number of moluecules that need to generate. Defaults to 1.
        cuda (bool, optional): Cuda usage mode. Defaults to True.
        mean_ (float, optional): mean of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 0.0.
        std_ (float, optional): std of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 1.0.
        url (_type_, optional): URL to API srver. Defaults to 'http://10.32.2.4:80/case_generator'.
        case_ (str, optional): Key for api, that define what case you choose for. Can be choose from: 'Alzhmr','Sklrz','Prkns','Cnsr','Dslpdm','TBLET', 'RNDM'.
                               Where: 'Alzhmr' - Alzheimer,
                                        'Sklrz' - Skleroz,
                                        'Prkns' - Parkinson,
                                        'Cnsr' - Canser,
                                        'Dslpdm' - Dyslipidemia,
                                        'TBLET' - Drug resistance,
                                        'RNDM' - random generation.
                                        Defaults to RNDM.
    Returns:
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.

    Example:
        numbs = 4
        params = {'numb_mol': numbs, 'cuda': False, 'mean_': 0, case_ = 'RNDM
                'std_': 1}
        resp_mol, mols = call_for_generation(**params,hello='world')
        print(mols)
        >> {'Molecules': ['Cc1cc(C(=O)OCC(=O)NCC2CCCO2)nn1C', 'CSC1=CC=C(C(=O)O)C(C(=O)c2ccc(C(F)(F)F)cc2)S1', 'CSc1cc(-c2ccc(-c3ccccc3)cc2)nc(C(C)=O)c1O', 'CC(C)N(CC(=O)NCc1cn[nH]c1)Cc1ccccc1'],
          'Docking score': [-6.707, -7.517, -8.541, -7.47],
            'QED': [0.7785404162969669, 0.8150693008303525, 0.5355361484098266, 0.8174264075095671],
              'SA': [2.731063371805302, 3.558887012627684, 2.2174895913203354, 2.2083851588937087],
                'PAINS': [0, 0, 0, 0],
                  'SureChEMBL': [0, 0, 0, 0],
                    'Glaxo': [0, 0, 0, 0]}
    """

    params = {
        "numb_mol": numb_mol,
        "cuda": cuda,
        "mean_": mean_,
        "std_": std_,
        "case_": case_,
        **kwargs,
    }
    try:
        resp = requests.post(url, data=json.dumps(params))
        if resp.status_code != 200:
            print(
                "ERROR: response status code from requests to generative model: ",
                resp.status_code,
            )

    except requests.exceptions.RequestException as e:
        print(e)

    return resp, json.loads(resp.json())


def filter_valid_strings(
    df: pd.DataFrame, column_name: str, max_length: int = 200
) -> pd.DataFrame:
    """
    Removes molecules longer than 200 characters.

    Example:
    -------
    >>> df = pd.DataFrame({'text': ['abc', 'def'*100, 123]})
    >>> filtered_df = filter_valid_strings(df, 'text')
    >>> print(filtered_df)
    """
    try:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        is_string = df[column_name].apply(lambda x: isinstance(x, str))
        valid_length = df[column_name].str.len() <= max_length

        filtered_df = df[is_string & valid_length].copy()

        return filtered_df

    except Exception as e:
        raise ValueError(e)


temp = [
    "| Molecules | QED | Synthetic Accessibility | PAINS | SureChEMBL | Glaxo | Brenk | BBB | IC50 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| Cc1ccc(C2=C(c3ccc(N4CCN(C)CC4)cc(=O)n3)C(=O)N(C)C2=O)c(C(=O)N2CCCC2)c1 | 0.59 | 2.81 | 0 | 0 | 0 | 1 | 1 | 1 |\n",
    {
        "Molecules": [
            "Cc1ccc(C2=C(c3ccc(N4CCN(C)CC4)cc(=O)n3)C(=O)N(C)C2=O)c(C(=O)N2CCCC2)c1"
        ],
        "QED": [0.59],
        "Synthetic Accessibility": [2.81],
        "PAINS": [0],
        "SureChEMBL": [0],
        "Glaxo": [0],
        "Brenk": [1],
        "BBB": [1],
        "IC50": [1],
    },
]


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
    Save png-images in directory 'vizualization'.

    Args:
        mols (list): SMILES string

    Returns:
        None
    """
    # randrange gives you an integral value
    time = [str(datetime.now().time()) for i in range(len(mols))]

    for i, mol in enumerate(mols):
        img = Draw.MolToImage(Chem.MolFromSmiles(mol))
        img.save(f"MADD/imgs/mol{time[i]}_{i}.png")

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


def gen_mols_alzheimer(num: int) -> list:
    """
    Generation of drug molecules for the treatment of Alzheimer's disease. GSK-3beta inhibitors with high activity. \
    These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability

    Args:
        num (int): number of molecules to generate

    Returns:
        list: list of generated molecules
    """
    print("TOOL: alzheimer")
    params = {"numb_mol": num, "cuda": True, "case_": "Alzhmr"}
    _, mol_dict = generate_for_base_case(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])
    print("GEN_MOLECULES: ")
    print(mol_dict)
    print("END_GEN_MOLECULES")

    return [res, mol_dict]


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
    print("TOOL: sclerosis")
    params = {"numb_mol": num, "cuda": True, "case_": "Sklrz"}
    _, mol_dict = generate_for_base_case(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])
    print("GEN_MOLECULES: ")
    print(mol_dict)
    print("END_GEN_MOLECULES")
    return [res, mol_dict]


def gen_mols_dyslipidemia(num: int) -> list:
    """
    Generation of molecules for the treatment of dyslipidemia.\
    Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and \
    the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances reverse cholesterol transport via ABCA1 upregulation\
    , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be  PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.",
        
    Args:
    num (int): number of molecules to generate
    """
    print("TOOL: dyslipidemia")
    params = {"numb_mol": num, "cuda": True, "case_": "Dslpdm"}
    _, mol_dict = generate_for_base_case(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])
    print("GEN_MOLECULES: ")
    print(mol_dict)
    print("END_GEN_MOLECULES")
    return [res, mol_dict]


def gen_mols_acquired_drug_resistance(num: int) -> list:
    """
    Generation of molecules for acquired drug resistance. \
    Molecules that selectively induce apoptosis in drug-resistant tumor cells.\
    It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.
        
    Args:
    num (int): number of molecules to generate
    """
    print("TOOL: Drug_Resistance")
    params = {"numb_mol": num, "cuda": True, "case_": "TBLET"}
    _, mol_dict = generate_for_base_case(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])
    print("GEN_MOLECULES: ")
    print(mol_dict)
    print("END_GEN_MOLECULES")
    return [res, mol_dict]


def gen_mols_lung_cancer(num: int) -> list:
    """
    Generation of molecules for the treatment of lung cancer. \
    Molecules are inhibitors of KRAS protein with G12C mutation. \
    The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.\
    Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D, \
    V14I, L19F, Q22K, D33E, Q61H, K117N and A146V/T.
    
    Args:
    num (int): number of molecules to generate
    """
    print("TOOL: lung cancer")
    params = {"numb_mol": num, "cuda": True, "case_": "Cnsr"}
    _, mol_dict = generate_for_base_case(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])
    print("GEN_MOLECULES: ")
    print(mol_dict)
    print("END_GEN_MOLECULES")
    return [res, mol_dict]


def gen_mols_parkinson(num: int) -> list:
    """
    Generation of molecules for the treatment of Parkinson's disease.

    Args:
    num (int): number of molecules to generate
    """
    print("TOOL: Parkinson")
    params = {"numb_mol": num, "cuda": True, "case_": "Prkns"}
    _, mol_dict = generate_for_base_case(**params)

    res = make_markdown_table(mol_dict)

    mols_vizualization(mol_dict["Molecules"])
    print("GEN_MOLECULES: ")
    print(mol_dict)
    print("END_GEN_MOLECULES")
    return [res, mol_dict]


@tool
def get_state_from_server(url: str = "pred") -> Union[dict, str]:
    """Get information about all available models (cases),
    their status (training, trained), description, metrics.

    Important note: if the returned dictionary has the status key not Training, Trained, None, but text content.
    Then an error occurred. And this is its description. Notify the user about it.

    Args:
        url (str): Flag for server, for predictive is 'pred', for generative is 'gen'
    """
    if url == "pred":
        url = conf["url_pred"]
    else:
        url = conf["url_gen"]

    url_ = url.split("http://")[1]
    resp = requests.get("http://" + url_.split("/")[0] + "/check_state")
    if resp.status_code == 500:
        print(f"Server error:{resp.status_code}")
        return "Server error"
    state = json.loads(resp.content)
    return state["state"]


@tool
def get_case_state_from_server(case: str, url: str = "pred") -> Union[dict, str]:
    """Get information about a specific case/model (if found),
    its status (in training, trained), metrics, etc.
    Example of user query: 'Check status for 'Ki_predictor''.

    Important note: if the returned dictionary has the status key not Training, Trained, None, but text content.
    Then an error occurred. And this is its description. Notify the user about it.

    Args:
        case (str): Name of case
        url (str): Flag for server, for predictive is 'pred', for generative is 'gen'
    """
    if url == "pred":
        url = conf["url_pred"]
    else:
        url = conf["url_gen"]

    if (
        case.strip().lower()
        in """- Alzheimer's disease: 'alzheimer', 'alzheimer's', 'alzheimers'
    - Parkinson's disease: 'parkinson', 'parkinson's', 'parkinsons'
    - Multiple sclerosis: 'multiple sclerosis', 'sclerosis'
    - Dyslipidemia: 'dyslipidemia'
    - Acquired drug resistance: 'acquired drug resistance', 'drug resistance'
    - Lung cancer: 'lung cancer', 'lung'"""
    ):
        return {case: "generative model is exist!"}

    url_ = url.split("http://")[1]
    resp = requests.get("http://" + url_.split("/")[0] + "/check_state")
    if resp.status_code == 500:
        print(f"Server error:{resp.status_code}")
        return "Server error"
    state = json.loads(resp.content)
    try:
        return state["state"][case]
    except:
        return f"Case with name: {case} not found"


@tool
def predict_prop_by_smiles(
    smiles_list: List[str], case: str = "no_name_case", timeout: int = 20
) -> Tuple[requests.Response, dict]:
    """
    Runs property prediction using inference-ready (previously trained) ML models. And RDKIT funcs
    (They need to be calculated using this function, passing feature_column (if the user asks!)is:
    'Validity', 'Brenk', 'QED', 'Synthetic Accessibility', 'LogP', 'Polar Surface Area',
    'H-bond Donors', 'H-bond Acceptors', 'Rotatable Bonds', 'Aromatic Rings',
    'Glaxo', 'SureChEMBL', 'PAINS'.
    They are calculated automatically (by simply func) if they fall into arg: 'feature_column'.

    Args:
        smiles_list (List[str]): A list of molecules in SMILES format.
        case (str, optional): Name of model (model names can be obtained by calling 'get_state_from_server').
        timeout (int, optional): The timeout duration (in minutes).

    Returns:
        Tuple[requests.Response, dict]: A tuple containing the HTTP response object and the parsed JSON response.
    """
    url = conf["url_pred"] + "/predict_ml"
    params = {"case": case, "smiles_list": smiles_list, "timeout": timeout}
    resp = requests.post(url, json.dumps(params))
    return resp, resp.json()


def train_gen_with_data(
    case="no_name",
    data_path="./data_dir_for_coder/kras_g12c_affinity_data.xlsx",  # path to client data folder
    feature_column=["smiles"],
    target_column=[
        "docking_score",
        "QED",
        "Synthetic Accessibility",
        "PAINS",
        "SureChEMBL",
        "Glaxo",
        "Brenk",
        "IC50",
    ],  # All propreties from dataframe you want to calculate in the end
    regression_props=[
        "docking_score"
    ],  # Column name with data for regression tasks (That not include in calculcateble propreties)
    classification_props=[],  # Column name with data for classification tasks (That not include in calculcateble propreties)
    description="Descrption not provided",
    timeout=2,  # min
    url: str = conf["url_gen"] + "/train_gen_models",
    fine_tune: bool = True,
    n_samples=10,
    **kwargs,
):
    """
    Trains a generative deep learning model using user-provided or prepared by a special agent dataset.

    Args:
        case (str): A name of case.
        data_path (str): Path to data for training (in csv or excel format). Must consist SMILES!
        feature_column (list): Names of columns with features (input data) for training. Default is ['smiles'].
        target_column (list): Names of columns (properties) with target data for training. All propreties from dataframe you want to calculate in the end
        regression_props (list): Names of columns with data for regression tasks. Skip if you dont need regression!
        classification_props (list): Names of columns with data for classification tasks. Skip if you dont need classification!
        description (str): Description of model/case.
        timeout (int): Timeout for training in minutes.
        url (str): URL of the server to send the training request to.
        fine_tune (bool): Set alvays to False.
        samples (int): Number of samples for validation. Default is 10.
    """
    start_time = time.time()
    try:
        df = pd.read_csv(
            data_path
        ).to_dict()  # Transfer df to dict for server data transfer
    except:
        df = pd.read_excel(data_path).to_dict()

    params = {
        "case": case,
        "data": df,
        "target_column": target_column,
        "feature_column": feature_column,
        "timeout": timeout,
        "description": description,
        "regression_props": regression_props,
        "classification_props": classification_props,
        "fine_tune": fine_tune,
        "n_samples": n_samples,
        **kwargs,
    }

    p = Process(target=requests.post, args=[url, json.dumps(params)])
    p.start()

    time.sleep(4)
    print("--- %s seconds ---" % (time.time() - start_time))


def train_ml_with_data(
    case="No_name",
    data_path="automl/data/data_4j1r.csv",  # path to client data folder
    feature_column=["Smiles"],
    target_column=[
        "Docking score"
    ],  # All propreties from dataframe you want to calculate in the end,
    regression_props=["Docking score"],
    classification_props=[],
    description="",
    timeout=2,
) -> Union[bool, str]:
    """
    Trains a predictive machine learning model using user-provided or prepared by a special agent dataset.

    This function reads a dataset from a specified file, processes it into a dictionary,
    and sends it to a remote server for training. The training process runs asynchronously
    using a separate process.

    Args:
        case (str, optional): Name of model.
        data_path (str, optional): Path to the CSV file containing the dataset.
        feature_column (list, optional): The name of the column containing the input features. Default is "Smiles".
        target_column (list, optional): All propreties from dataframe you want to calculate in the end.
        regression_props (list, optional): Column names with data for regression tasks. Skip if you dont need regression!
        classification_props (list, optional): Column name with data for classification tasks. Skip if you dont need classification!
        timeout (int, optional): The timeout duration (in minutes) for the request.
        description (str): Description of model/case

    Returns:
        bool (succces or not)"""
    start_time = time.time()
    try:
        df = pd.read_csv(
            data_path
        ).to_dict()  # Transfer df to dict for server data transfer
    except:
        df = pd.read_excel(data_path).to_dict()
    start_time = time.time()
    params = {
        "case": case,
        "data": df,
        "target_column": target_column,
        "feature_column": feature_column,
        "timeout": timeout,
        "description": description,
        "regression_props": regression_props,
        "classification_props": classification_props,
    }

    p = Process(
        target=requests.post,
        args=[f"{conf['url_pred']}/train_ml", json.dumps(params)],
    )
    p.start()

    time.sleep(10)
    p.terminate()

    print("--- %s seconds ---" % (time.time() - start_time))

    return True


def ml_dl_training(
    case: str,
    path: str,
    feature_column=["smiles"],
    target_column=["docking_score"],
    regression_props=["docking_score"],
    classification_props=[],
):
    def get_case_state_from_server(case: str, url: str = "pred") -> Union[dict, str]:
        if url == "pred":
            url = conf["url_pred"]
        else:
            url = conf["url_gen"]

        url_ = url.split("http://")[1]
        resp = requests.get("http://" + url_.split("/")[0] + "/check_state")
        if resp.status_code == 500:
            print(f"Server error:{resp.status_code}")
            return "Server error"
        state = json.loads(resp.content)
        try:
            return state["state"][case]
        except:
            return f"Case with name: {case} not found"

    ml_ready = False
    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    while not ml_ready:
        time.sleep(7)
        print("Training ml-model in progress for case: ", case)
        try:
            st = get_case_state_from_server(case, "pred")
            if isinstance(st, dict):
                if st["ml_models"]["status"] == "Trained":
                    ml_ready = True
            time.sleep(7)
        except:
            print("Something went wrong!!!")

    if ml_ready:
        print("ML-model is ready.")
    try:
        train_gen_with_data(
            case=case,
            data_path=path,
            feature_column=feature_column,
            target_column=target_column,
            regression_props=regression_props,
            classification_props=classification_props,
            # TODO: rm after testing automl pipeline
            # epoch=1,
        )
        print("Start training gen model for case: ", case)
    except:
        print("Something went wrong!!!")


@tool
def just_ml_training(
    case: str,
    path: str,
    feature_column: list = ["canonical_smiles"],
    target_column: list = ["docking_score"],
    regression_props: list = ["docking_score"],
    classification_props: list = [],
) -> bool:
    """
    Launch training of ONLY ML-model (predictive).

    Use only as a last resort!

    Args:
        case (str): Name of model.
        path (str): Path to the CSV file containing the dataset.
        feature_column (list): The name of the column containing the input features. You must be sure that such a column exists!
        target_column (list): All propreties from dataframe you want to calculate in the end. This field cannot be left blank (no empty list)!
        regression_props (list, optional): Column names with data for regression tasks. Fill in the list! It should duplicate feature_column.
        classification_props (list, optional): Column name with data for classification tasks. Set '[]' if you dont need classification!
    """

    if regression_props == [] and classification_props == []:
        regression_props = target_column
    if len(target_column) < 1:
        raise ValueError(
            "target_column is empty! You must set value. For example = ['IC50']"
        )
    if len(feature_column) < 1:
        raise ValueError(
            "feature_column is empty! You must set value. For example = ['smiles']"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_excel(path)

    if len(df.values.tolist()) < 300:
        raise ValueError(
            "Training on this data is impossible. The dataset is too small!"
        )

    for column in feature_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )
    for column in target_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )

    # delete molecules eith len more 200
    clear_df = filter_valid_strings(df, feature_column[0])
    if path.split(".")[-1] == "csv":
        clear_df.to_csv(path)
    else:
        clear_df.to_excel(path)

    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    return True


base_case_dict = {
    "parkinson": gen_mols_parkinson,
    "parkinson's": gen_mols_parkinson,
    "parkinsons": gen_mols_parkinson,
    "болезнь паркинсона": gen_mols_parkinson,
    "lung cancer": gen_mols_lung_cancer,
    "lung": gen_mols_lung_cancer,
    "рак легких": gen_mols_lung_cancer,
    "acquired drug resistance": gen_mols_acquired_drug_resistance,
    "drug resistance": gen_mols_acquired_drug_resistance,
    "resistance": gen_mols_acquired_drug_resistance,
    "лекарственная устойчивость": gen_mols_acquired_drug_resistance,
    "dyslipidemia": gen_mols_dyslipidemia,
    "дислипидемия": gen_mols_dyslipidemia,
    "multiple sclerosis": gen_mols_multiple_sclerosis,
    "sclerosis": gen_mols_multiple_sclerosis,
    "рассеянный склероз": gen_mols_multiple_sclerosis,
    "alzheimer": gen_mols_alzheimer,
    "alzheimer's": gen_mols_alzheimer,
    "alzheimers": gen_mols_alzheimer,
    "болезнь альцгеймера": gen_mols_alzheimer,
}


@tool
def generate_mol_by_case(
    case: str,
    n_samples: int = 1,
) -> dict:
    """Runs molecules generation using inference-ready (previously trained) generative models.

    IMPORTANT:
    1. For pre-defined disease cases, use the complete and precise disease name as shown below.
    These cases have fixed names that cannot be changed or written differently.

    2. Additionally, user-trained custom models may be available. These custom cases must be
    referenced exactly by the name used during training (case-sensitive).

    Available pre-defined disease cases (fixed names - must be used exactly as shown):
    - Alzheimer's disease: 'alzheimer', 'alzheimer's', 'alzheimers'
    - Parkinson's disease: 'parkinson', 'parkinson's', 'parkinsons'
    - Multiple sclerosis: 'multiple sclerosis', 'sclerosis'
    - Dyslipidemia: 'dyslipidemia'
    - Acquired drug resistance: 'acquired drug resistance', 'drug resistance'
    - Lung cancer: 'lung cancer', 'lung'

    Examples of correct usage for pre-defined cases:
    - "Generate molecules for Alzheimer's disease"
    - "Create compounds for multiple sclerosis"
    - "Produce molecules targeting dyslipidemia"

    For user-trained custom models:
    - Use the exact case name specified during training
    - Names are case-sensitive and must match exactly

    Args:
        case (str, optional): Name of disease case (use complete disease name for pre-defined cases,
                            or exact training name for custom models).
        n_samples (int, optional): Number of molecules to generate. Default is 1
    """
    case_low = case.strip().lower()

    base_case_func = base_case_dict.get(case_low, None)
    if base_case_func:
        res = base_case_func(n_samples)
        return res

    url = conf["url_gen"] + "/generate_gen_models_by_case"

    params = {
        "case": case,
        "n_samples": n_samples,
    }
    start_time = time.time()
    resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))
    try:
        return json.loads(resp.json())
    except:
        resp.json()


@tool
def run_ml_dl_training_by_daemon(
    case: str,
    path: str,
    feature_column: list = ["smiles"],
    target_column: list[str] = ["docking_score"],
    regression_props: list[str] = ["docking_score"],
    classification_props: list = [],
) -> Union[bool, str]:
    """
    Starts training the predictive and generative models (this is normal
    if the user asks for only one thing).
    The status can be checked with "get_state_case_from_server".
    Use it to train models!!!

    Args:
        case (str): Name of model.
        path (str): Path to the CSV file containing the dataset.
        feature_column (list): The name of the column containing the input features. You must be sure that such a column exists!
        target_column (list): All propreties from dataframe you want to calculate in the end. This field cannot be left blank (no empty list)!
        regression_props (list, optional): Column names with data for regression tasks. Fill in the list! It should duplicate feature_column.
        classification_props (list, optional): Column name with data for classification tasks. Set '[]' if you dont need classification!

    note: Either regression_props or classification_props must be filled in.
    """
    if isinstance(feature_column, str):
        feature_column = [feature_column]
    if isinstance(target_column, str):
        target_column = [target_column]

    if regression_props == [] and classification_props == []:
        regression_props = target_column
    if len(target_column) < 1:
        raise ValueError(
            "target_column is empty! You must set value. For example = ['IC50']"
        )
    if len(feature_column) < 1:
        raise ValueError(
            "feature_column is empty! You must set value. For example = ['smiles']"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_excel(path)

    if len(df.values.tolist()) < 300:
        raise ValueError(
            "Training on this data is impossible. The dataset is too small!"
        )

    for column in feature_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )
    for column in target_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )

    # delete molecules eith len more 200
    clear_df = filter_valid_strings(df, feature_column[0])
    if path.split(".")[-1] == "csv":
        clear_df.to_csv(path)
    else:
        clear_df.to_excel(path)

    cmd = [
        sys.executable,
        "-c",
        (
            "from multi_agent_system.MADD_main.tools.automl_tools import ml_dl_training;"
            "ml_dl_training("
            f"case='{case}',"
            f"path='{path}',"
            f"feature_column={feature_column},"
            f"target_column={target_column},"
            f"regression_props={regression_props},"
            f"classification_props={classification_props}"
            ")"
        ),
    ]

    try:
        cwd_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )  # get root dir

        subprocess.Popen(
            cmd,
            stdout=open("/tmp/ml_training.log", "a"),
            stderr=open("/tmp/ml_training.err", "a"),
            cwd=cwd_path,
        )
        time.sleep(50)
        return True
    except Exception as e:
        print(f"Failed to start process: {e}", file=sys.stderr)
        return False


automl_tools = [
    run_ml_dl_training_by_daemon,
    get_case_state_from_server,
    get_state_from_server,
    generate_mol_by_case,
    predict_prop_by_smiles,
]
if __name__ == "__main__":
    # run_ml_dl_training_by_daemon(
    #     "sars_cov",
    #     "/Users/alina/Desktop/ITMO/ChemCoScientist/ChemCoScientist/data_store/datasets/users_dataset.csv",
    #     "smiles",
    #     "IC50",
    #     ["IC50"],
    # )
    ml_dl_training(
        "IC50_prediction",
        "/Users/alina/Desktop/ITMO/MADD-CoScientist/data_cyk_short.csv",
    )
