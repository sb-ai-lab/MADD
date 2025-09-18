import requests
import json
from typing import List, Union, Tuple


def call_for_ic50(mol_list: Union[requests.models.Response, List[str], dict],
                     url: str = 'http://10.32.2.4:80/eva_altz') -> Tuple[requests.models.Response, dict]:
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
        params = {'mol_list': mol_list}
        resp = requests.post(url, data=json.dumps(params))
    return resp, json.loads(resp.text)