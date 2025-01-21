import json
from typing import List, Tuple, Union

import requests
import yaml

with open("multi_agents_system/config.yaml", "r") as file:
    config = yaml.safe_load(file)
models_ip_addr = config["models_address"]
models_port = config["models_port"]


def call_for_generation(
    numb_mol: int = 1,
    cuda: bool = True,
    mean_: float = 0.0,
    std_: float = 1.0,
    url: str = f"http://{models_ip_addr}:{models_port}/case_generator",
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


def call_for_random_generation(
    numb_mol: int = 1,
    cuda: bool = True,
    mean_: float = 0.0,
    std_: float = 1.0,
    url: str = f"http://{models_ip_addr}/random_search",
    **kwargs,
) -> Tuple[requests.models.Response, dict]:
    """Function that call Chem server API for random generate molecules with properties.

        This is an example of call random case only without 'case_' argument (it hardcoded on backend as 'RNDM').
                                    ###
        !RECOMMEND USE "call_for_generation" universal call function!
                        !Probably will be suppressed!
                                    ###
    Args:
        numb_mol (int, optional): Number of moluecules that need to generate. Defaults to 1.
        cuda (bool, optional): Cuda usage mode. Defaults to True.
        mean_ (float, optional): mean of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 0.0.
        std_ (float, optional): std of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 1.0.
        url (_type_, optional): URL to API srver. Defaults to 'http://10.32.2.4:80/random_search'.

    Returns:
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
        "model": model,
        "cuda": cuda,
        "mean_": mean_,
        "std_": std_**kwargs,
    }
    try:
        resp = requests.post(url, data=json.dumps(params))
    except requests.exceptions.RequestException as e:
        print(e)

    return resp, json.loads(resp.json())


def call_for_generation_Alzheimer(
    numb_mol: int = 1,
    cuda: bool = True,
    mean_: float = 0.0,
    std_: float = 1.0,
    url: str = f"http://{models_ip_addr}1/search_Alzheimer",
    **kwargs,
) -> Tuple[requests.models.Response, dict]:
    """Function that call Chem server API for Alzheimer disease case generate molecules.

        This is an example of call some specific case. Without 'case_' argument (it hardcoded on backend inside case url.)
        To create some other case call you can just choose interested url from description below.
                                    ###
        !RECOMMEND USE "call_for_generation" universal call function!
                        !Probably will be suppressed!
                                    ###
    Args:
        numb_mol (int, optional): Number of moluecules that need to generate. Defaults to 1.
        cuda (bool, optional): Cuda usage mode. Defaults to True.
        mean_ (float, optional): mean of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 0.0.
        std_ (float, optional): std of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 1.0.
        url (_type_, optional): URL to API srver. Defaults to 'http://10.32.2.4:80/search_Alzheimer'.
                                Can be choose from:
                                .../search_Drug_resist,
                                .../search_Dyslipidemia,
                                .../search_Canser,
                                .../search_Parkinson,
                                .../search_Skleroz



    Returns:
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.

    Example:
        numbs = 4
        params = {'numb_mol': numbs, 'cuda': False, 'mean_': 0,
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
        "case_": "Alzhmr",
        **kwargs,
    }
    try:
        resp = requests.post(url, data=json.dumps(params))
    except requests.exceptions.RequestException as e:
        print(e)

    return resp, json.loads(resp.json())


def call_for_ic50(
    mol_list: Union[requests.models.Response, List[str], dict],
    url: str = f"http://{models_ip_addr}/eva_altz",
) -> Tuple[requests.models.Response, dict]:
    """Function that call Chem server API for analyse molucules for ic_50 score for Alzheimer's disease case.

    Args:
        mol_list (Union[requests.models.Response,List[str]]): May take both response from other api function (if that have a mol_list argument) and just
                                                                list of molecules
        url str: URL for server
    Returns:
        Tuple[requests.models.Response,dict]: Return full respones, or just dict.

    Example:
        response_dict_with_mols = {
                                'mol_list':
                                ['O=C(O)C1=CC([N+](=O)[O-])=CC([N+](=O)[O-])=C1[N+](=O)[O-]',
                                'COC(=O)C1=CC=CN=C1',
                                'O=[N+]([O-])C1=CC=C([N+](=O)[O-])C=C1C1=CC=CC=C1']
                                    }

        responce,mol_dict = call_for_ic50(response_dict_with_mols)
        print(mol_dict)

        >> (<Response [200]>, {"Ic50_results":[8.574580825647168,7.685332428347721,8.427087872921055]})

    """

    if isinstance(mol_list, requests.models.Response):
        resp = requests.post(url, data=json.dumps(json.loads(mol_list.text)))
    elif isinstance(mol_list, dict):
        resp = requests.post(url, data=json.dumps(mol_list))
    else:
        params = {"mol_list": mol_list}
        resp = requests.post(url, data=json.dumps(params))
    return resp, json.loads(resp.text)


def call_for_ic50_sorted(
    mol_list: Union[requests.models.Response, List[str], dict],
    url: str = f"http://{models_ip_addr}/eva_altz_sorted",
) -> Tuple[requests.models.Response, dict]:
    """Function that call Chem server API for analyse molucules for ic_50 score for Alzheimer's disease case with sorted of valid molecules.

    Args:
        mol_list (Union[requests.models.Response,List[str]]): May take both response from other api function (if that have a mol_list argument) and just
                                                                list of molecules
        url str: URL for server
    Returns:
        Tuple[requests.models.Response,dict]: Return full respones, or just dict.

    Example:
        response_dict_with_mols = {
                                'mol_list':
                                ['O=C(O)C1=CC([N+](=O)[O-])=CC([N+](=O)[O-])=C1[N+](=O)[O-]',
                                'COC(=O)C1=CC=CN=C1',
                                'O=[N+]([O-])C1=CC=C([N+](=O)[O-])C=C1C1=CC=CC=C1']
                                    }

        responce,mol_dict = call_for_ic50(response_dict_with_mols)
        print(mol_dict)

        >> (<Response [200]>, {"mol_list_with_ic50_score":[["COC1=CC(CN2CCN(C)CC2)=C2C=C3C4=CC5=C(C=C4CC[N+]3=CC2=C1O)OCO5",3.788580760282815]]})

    """

    if isinstance(mol_list, requests.models.Response):
        resp = requests.post(url, data=json.dumps(json.loads(mol_list.text)))
    elif isinstance(mol_list, dict):
        resp = requests.post(url, data=json.dumps(mol_list))
    else:
        params = {"mol_list": mol_list}
        resp = requests.post(url, data=json.dumps(params))
    return resp, json.loads(resp.text)


def call_for_novelty(
    mol_list: Union[requests.models.Response, List[str], dict],
    url: str = f"http://{models_ip_addr}/eva",
) -> Tuple[requests.models.Response, dict]:
    """Function that call Chem server API for analyse molucules for novelty.

    Args:
        mol_list (Union[requests.models.Response,List[str]]): May take both response from other api function (if that have a mol_list argument) and just
                                                                list of molecules
        url str: URL for server
    Returns:
        Tuple[requests.models.Response,dict]: Return full respones, or just dict.

    Example:
        response_dict_with_mols = {
                                'mol_list':
                                ['O=C(O)C1=CC([N+](=O)[O-])=CC([N+](=O)[O-])=C1[N+](=O)[O-]',
                                'COC(=O)C1=CC=CN=C1',
                                'O=[N+]([O-])C1=CC=C([N+](=O)[O-])C=C1C1=CC=CC=C1']
                                    }

        responce,mol_dict = call_for_novelty(response_dict_with_mols)
        print(mol_dict)

        >> (<Response [200]>, {'novelty_of_CCDC': 33.33333333333333, 'novelty_of_ChEMBL': 33.33333333333333})

        ....

        mols = {'mol_list': ['CS(=O)C1=CC=CC=C1', 'O=[N+]([O-])C1=CC=C([N+](=O)[O-])C=C1[N+](=O)[O-]', 'FC1=CC(I)=CC=C1']}
        resp_eval, eval_dict = call_for_novelty(mols)
        >> (<Response [200]>, {'novelty_of_CCDC': 33.33333333333333, 'novelty_of_ChEMBL': 33.33333333333333})
        ....


        call_for_novelty(mols['mol_list'])
        >> (<Response [200]>, {'novelty_of_CCDC': 33.33333333333333, 'novelty_of_ChEMBL': 33.33333333333333})
    """

    if isinstance(mol_list, requests.models.Response):
        resp = requests.post(url, data=json.dumps(json.loads(mol_list.text)))
    elif isinstance(mol_list, dict):
        resp = requests.post(url, data=json.dumps(mol_list))
    else:
        params = {"mol_list": mol_list}
        resp = requests.post(url, data=json.dumps(params))
    return resp, json.loads(resp.text)


if __name__ == "__main__":
    numbs = 5
    stds = [
        1,
    ]
    for i in stds:
        model = "CVAE"
        params = {"numb_mol": "1", "model": model, "cuda": False, "mean_": 0, "std_": i}
        resp_mol, mols = call_for_generation(**params, hello="world")
        resp_eval, eval_dict = call_for_novelty(resp_mol)
        print("stds", i, "Len=", len(mols["mol_list"]) / numbs, eval_dict)
    means = [0]
    for i in means:
        model = "CVAE"
        params = {
            "numb_mol": numbs,
            "model": model,
            "cuda": False,
            "mean_": i,
            "std_": 0,
        }
        resp_mol, mols = call_for_generation(**params, hello="world")
        resp_eval, eval_dict = call_for_novelty(resp_mol)
        print("mean", i, "Len=", len(mols["mol_list"]) / numbs, eval_dict)
